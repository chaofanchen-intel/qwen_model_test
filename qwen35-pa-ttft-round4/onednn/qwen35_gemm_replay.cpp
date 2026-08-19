// Replay the COMPLETE Qwen3.5-2B-INT4 decoder GEMM inventory, in model order, with the
// real post-ops and (by default) a distinct weight buffer per projection.
//
// Why this exists
// ---------------
// Round 3 priced ONE shape in isolation and concluded "GEMM is at 96% of the int8 DPAS
// ceiling, no headroom left".  That measurement re-executed a SINGLE primitive 60 times
// against a SINGLE weight buffer, so from iteration 2 onward the weights were served out
// of GPU cache.  The model never gets that: it streams ~686 MB of int4 weights exactly
// once per prefill, and 138 of its 174 GEMMs additionally carry a post-op that reads or
// writes another full-size activation tensor.
//
// This binary reproduces the real conditions so the "achievable floor" it reports can be
// compared like-for-like against the measured in-model gemm_kernel time:
//
//   QWEN_REPLAY_SHARE=0 (default)  distinct weight buffer per projection -> streams like
//                                  the model does
//   QWEN_REPLAY_SHARE=1            every projection of a given shape aliases one weight
//                                  buffer -> reproduces the round-3 (cache-resident) setup
//   QWEN_REPLAY_POSTOPS=0          drop the post-ops, to price them separately
//
// Shapes, counts, post-ops and the M padding (1024 vs 1002) are all transcribed from
// ONEDNN_VERBOSE on a real 1002-token PA prefill; see results/onednn_verbose.log.

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include "oneapi/dnnl/dnnl.hpp"

using namespace dnnl;
using dt = memory::data_type;
using tag = memory::format_tag;

namespace {

enum class PostOp { None, Swish, SwishMul, BinaryAdd, Logistic, Bias };

struct Proj {
    const char *name;
    memory::dim m, k, n;
    PostOp po;
    int count;  // occurrences per decoder pass
};

// One decoder pass = 24 layers (18 linear_attn + 6 self_attn).  Verified against the
// per-descriptor counts in results/onednn_verbose.log.
const std::vector<Proj> kInventory = {
    {"in_proj_qkv", 1024, 2048, 6144, PostOp::None, 18},
    {"up_proj", 1024, 2048, 6144, PostOp::None, 24},
    {"gate_proj", 1002, 2048, 6144, PostOp::SwishMul, 24},
    {"down_proj", 1002, 6144, 2048, PostOp::BinaryAdd, 24},
    {"out/o_proj", 1002, 2048, 2048, PostOp::BinaryAdd, 24},
    {"in_proj_z", 1024, 2048, 2048, PostOp::Swish, 18},
    {"in_proj_b", 1024, 2048, 16, PostOp::Logistic, 18},
    {"in_proj_a", 1024, 2048, 16, PostOp::Bias, 18},
    {"attn_qkv", 1024, 2048, 5120, PostOp::None, 6},
};

constexpr int WEI_MASK_N = 1 << 2;  // per output channel -> ONEDNN_VERBOSE "wei:4"
constexpr int SRC_MASK_MK = 7;      // grouped over K, per M -> "src0:7"

double now_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

void zero(memory &m) {
    if (!m) return;
    size_t bytes = m.get_desc().get_size();
    void *p = m.map_data();
    if (p) std::memset(p, 0, bytes);
    m.unmap_data(p);
}

struct Call {
    matmul prim;
    std::unordered_map<int, memory> args;
};

}  // namespace

int main(int argc, char **argv) {
    const int warmup = 3;
    const int iters = argc > 1 ? std::atoi(argv[1]) : 20;
    const bool share = getenv("QWEN_REPLAY_SHARE") && std::string(getenv("QWEN_REPLAY_SHARE")) == "1";
    const bool postops = !(getenv("QWEN_REPLAY_POSTOPS") && std::string(getenv("QWEN_REPLAY_POSTOPS")) == "0");

    engine eng(engine::kind::gpu, 0);
    stream strm(eng);

    std::vector<Call> calls;
    std::vector<memory> keepalive;
    double total_gflop = 0.0;
    size_t weight_bytes = 0;

    for (const auto &p : kInventory) {
        memory::dims src_dims{p.m, 1, p.k};
        memory::dims wei_dims{1, p.k, p.n};
        memory::dims dst_dims{p.m, 1, p.n};

        auto src_md = memory::desc(src_dims, dt::s8, tag::abc);
        auto wei_md = memory::desc(wei_dims, dt::s4, tag::cab);
        auto dst_md = memory::desc(dst_dims, dt::f16, tag::abc);
        auto bin_md = memory::desc(dst_dims, dt::f16, tag::abc);
        auto bia_md = memory::desc({1, 1, p.n}, dt::f16, tag::abc);

        primitive_attr attr;
        attr.set_scratchpad_mode(scratchpad_mode::user);
        attr.set_scales(DNNL_ARG_SRC, SRC_MASK_MK, {1, p.k}, dt::f16);
        attr.set_scales(DNNL_ARG_WEIGHTS, WEI_MASK_N, {}, dt::f16);

        post_ops po;
        if (postops) {
            switch (p.po) {
                case PostOp::Swish: po.append_eltwise(algorithm::eltwise_swish, 1.f, 0.f); break;
                case PostOp::Logistic: po.append_eltwise(algorithm::eltwise_logistic, 0.f, 0.f); break;
                case PostOp::BinaryAdd: po.append_binary(algorithm::binary_add, bin_md); break;
                case PostOp::SwishMul:
                    po.append_eltwise(algorithm::eltwise_swish, 1.f, 0.f);
                    po.append_binary(algorithm::binary_mul, bin_md);
                    break;
                default: break;
            }
            if (po.len() > 0) attr.set_post_ops(po);
        }
        const bool has_bias = postops && p.po == PostOp::Bias;

        matmul::primitive_desc pd = has_bias
            ? matmul::primitive_desc(eng, src_md, wei_md, bia_md, dst_md, attr)
            : matmul::primitive_desc(eng, src_md, wei_md, dst_md, attr);

        // Activations are genuinely reused across layers in the model, so one buffer per
        // shape is faithful there; weights are what must not be shared.
        auto src = memory(pd.src_desc(), eng);
        auto dstm = memory(pd.dst_desc(), eng);
        auto scratch = memory(pd.scratchpad_desc(), eng);
        auto src_sc = memory({{p.m, p.k / 128}, dt::f16, tag::ab}, eng);
        zero(src); zero(dstm); zero(scratch); zero(src_sc);
        keepalive.insert(keepalive.end(), {src, dstm, scratch, src_sc});

        memory shared_wei, shared_wsc;
        for (int i = 0; i < p.count; ++i) {
            memory wei, wsc;
            if (share && shared_wei) {
                wei = shared_wei;
                wsc = shared_wsc;
            } else {
                wei = memory(pd.weights_desc(), eng);
                wsc = memory({{p.n}, dt::f16, tag::a}, eng);
                zero(wei);
                zero(wsc);
                weight_bytes += wei.get_desc().get_size();
                if (share) {
                    shared_wei = wei;
                    shared_wsc = wsc;
                }
            }
            keepalive.insert(keepalive.end(), {wei, wsc});

            Call c{matmul(pd), {}};
            c.args[DNNL_ARG_SRC] = src;
            c.args[DNNL_ARG_WEIGHTS] = wei;
            c.args[DNNL_ARG_DST] = dstm;
            c.args[DNNL_ARG_SCRATCHPAD] = scratch;
            c.args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC] = src_sc;
            c.args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS] = wsc;
            if (postops && (p.po == PostOp::BinaryAdd || p.po == PostOp::SwishMul)) {
                int idx = (p.po == PostOp::SwishMul) ? 1 : 0;
                auto binm = memory(bin_md, eng);
                zero(binm);
                keepalive.push_back(binm);
                c.args[DNNL_ARG_ATTR_MULTIPLE_POST_OP(idx) | DNNL_ARG_SRC_1] = binm;
            }
            if (has_bias) {
                auto biam = memory(bia_md, eng);
                zero(biam);
                keepalive.push_back(biam);
                c.args[DNNL_ARG_BIAS] = biam;
            }
            calls.push_back(std::move(c));
            total_gflop += 2.0 * p.m * p.k * p.n / 1e9;
        }
    }

    printf("primitives/pass = %zu   weights resident = %.0f MB   share=%d postops=%d\n",
           calls.size(), weight_bytes / 1e6, (int)share, (int)postops);

    for (int w = 0; w < warmup; ++w) {
        for (auto &c : calls) c.prim.execute(strm, c.args);
    }
    strm.wait();

    double t0 = now_ms();
    for (int i = 0; i < iters; ++i) {
        for (auto &c : calls) c.prim.execute(strm, c.args);
    }
    strm.wait();
    double ms = (now_ms() - t0) / iters;

    printf("decoder GEMM pass: %.1f ms   %.1f GFLOP   %.1f TFLOP/s   (iters=%d)\n", ms, total_gflop,
           total_gflop / ms, iters);
    return 0;
}
