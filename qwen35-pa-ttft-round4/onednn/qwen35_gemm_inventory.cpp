// Standalone oneDNN matmul micro-benchmark for the Qwen3.5-2B INT4 GEMMs.
//
// Derived from the GME probe (gme-qwen2-vl-int4-hotspot-optimization/onednn/gemm_probe.cpp),
// but re-parameterised for Qwen3.5, which differs from GME in BOTH axes that matter:
//
//   * shapes      -- K=2048 with N=6144 (N/K=3), where GME had K=1536 with N=8960
//                    (N/K=5.83).  N/K=3 sits inside the gap GME never measured:
//                    GME saw abc win big at 5.83 and lose big at 1.33.
//   * quantization -- the two shipped Qwen3.5-2B INT4 variants use DIFFERENT schemes:
//                       Qwen3.5-2B-INT4       group_size=-1, sym=true
//                                             -> i4 weights, per-output-channel f16
//                                                scale, NO zero point
//                                             (ONEDNN_VERBOSE: wei:s4 ... attr-scales:wei:4:f16)
//                       Qwen3.5-2B-INT4-g128  group_size=128, sym=false
//                                             -> u4 weights, grouped scale + grouped
//                                                u8 zero point
//                    GME only ever ran the second kind in production.  Its own decompose
//                    table shows the layout preference FLIPS between the two
//                    (grouped: abc 1.25x faster; per-OC: abc 0.87x, i.e. slower), so the
//                    layout question has to be re-priced per variant rather than inherited.
//
// Both variants keep the per-token grouped activation scale (attr-scales:src0:7:f16:1x128),
// which is what the plugin's DynamicQuantize emits.
//
// Measurement method is deliberately identical to the GME probe and to how the OpenVINO
// GPU plugin drives the primitive: host wall clock around a batch of submissions, one
// stream wait at the end, no oneDNN profiling interface (benchdnn --mode=P wedges on this
// Xe3 driver; see gme docs/06 6.2).
//
// Build: see build_probe.sh (links oneDNN-ov2026.3 @ babb7375ff, the commit OV 2026.3 vendors)
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "oneapi/dnnl/dnnl.hpp"

using namespace dnnl;
using dt = memory::data_type;
using tag = memory::format_tag;

namespace {

struct Case {
    std::string name;
    dt src_dt, wei_dt;
    bool wei_scales_grouped;  // grouped (K/group x N) weight scales; false = per-output-channel
    int wei_scale_group;      // K group size when grouped
    bool wei_zp;              // asymmetric weights (grouped u8 zero point)
    bool src_scales;          // per-token grouped activation scales, as DynamicQuantize emits
    tag wtag;                 // weight layout under test: cab (K-contiguous) vs abc (N-contiguous)
};

struct Shape {
    std::string name;
    memory::dim m, k, n;
    // Qwen3.5 dispatches every projection in the `M x 1 x K` form (batch=M, M=1) with
    // broadcast weights -- confirmed from ONEDNN_VERBOSE, e.g. `1002x1x2048:1x2048x6144`.
    // That is the batched_m1 form in the GME probe, so it is the default here rather
    // than an option.
    bool batched_m1 = true;
    // How many times this exact shape is executed in one decoder pass. Only used by the
    // `model` mode, which count-weights the per-call cost into a predicted total so the
    // prediction can be compared against the measured gemm_kernel time.
    int count = 0;
};

// Mask conventions for a 3D matmul (batch, M, K) x (batch, K, N): K is dim 1, N is dim 2.
constexpr int WEI_MASK_N = 1 << 2;  // per output channel  -> ONEDNN_VERBOSE "wei:4"
constexpr int WEI_MASK_KN = 7;      // grouped over K, per N -> ONEDNN_VERBOSE "wei:7"
constexpr int SRC_MASK_MK = 7;      // grouped over K, per M -> ONEDNN_VERBOSE "src0:7"

double now_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

// Contents do not change which kernel runs or what it costs, but uninitialised GPU memory
// can seed f16 buffers with NaN/denormal patterns, so everything is zeroed once up front.
void zero(memory &m) {
    if (!m) return;
    size_t bytes = m.get_desc().get_size();
    void *p = m.map_data();
    if (p) std::memset(p, 0, bytes);
    m.unmap_data(p);
}

}  // namespace

int main(int argc, char **argv) {
    const int warmup = 20;
    const int iters = argc > 1 ? std::atoi(argv[1]) : 100;

    engine eng(engine::kind::gpu, 0);
    stream strm(eng);

    // Real Qwen3.5-2B-INT4 prefill shapes, taken from ONEDNN_VERBOSE on a 1002-token
    // PA prefill.  Counts are per decoder pass (24 layers: 18 linear_attn + 6 self_attn).
    std::vector<Shape> shapes = {
        {"mlp gate/up  K2048 N6144", 1002, 2048, 6144},   // x48  -- the dominant group
        {"ssm in_qkv   K2048 N6144", 1002, 2048, 6144},   // x18  (same shape, listed for clarity)
        {"attn qkv     K2048 N5120", 1002, 2048, 5120},   // x6
        {"z/out/o_proj K2048 N2048", 1002, 2048, 2048},   // x24
        {"mlp down     K6144 N2048", 1002, 6144, 2048},   // x24
    };

    // QWEN_PROBE_SHAPES=nsweep : hold K=2048 and walk N, to locate the cab/abc crossover
    //                            for THIS K rather than reusing GME's N>=4K threshold,
    //                            which was fitted at K=1536.
    // QWEN_PROBE_SHAPES=msweep : hold the dominant shape and walk the token count, since
    //                            GME found the layout preference also flips below ~128 rows.
    if (const char *sel = getenv("QWEN_PROBE_SHAPES")) {
        std::string s(sel);
        if (s == "nsweep") {
            shapes.clear();
            for (memory::dim n : {2048, 3072, 4096, 5120, 6144, 8192, 10240, 12288})
                shapes.push_back({"K2048 N" + std::to_string(n), 1002, 2048, n});
        } else if (s == "msweep") {
            shapes.clear();
            for (memory::dim m : {16, 32, 64, 128, 256, 512, 1002})
                shapes.push_back({"M" + std::to_string(m) + " K2048 N6144", m, 2048, 6144});
        } else if (s == "model") {
            // The COMPLETE decoder GEMM inventory, read straight out of
            // openvino_language_model.xml (every `*.weight` Const of a projection, with
            // its [N, K] and its multiplicity across the 24 layers).  Round 3 priced only
            // the dominant shape; this mode prices all of them and count-weights them, so
            // the sum can be compared against the measured gemm_kernel total.  If the two
            // agree, the shape MIX -- not the kernel -- explains the aggregate TFLOP/s,
            // and there is no plugin-side headroom left in GEMM.
            shapes.clear();
            shapes.push_back({"gate/up      K2048 N6144", 1002, 2048, 6144, true, 48});
            shapes.push_back({"in_proj_qkv  K2048 N6144", 1002, 2048, 6144, true, 18});
            shapes.push_back({"down_proj    K6144 N2048", 1002, 6144, 2048, true, 24});
            shapes.push_back({"z/out/o_proj K2048 N2048", 1002, 2048, 2048, true, 42});
            shapes.push_back({"q_proj       K2048 N4096", 1002, 2048, 4096, true, 6});
            shapes.push_back({"k_proj/v_proj K2048 N512", 1002, 2048, 512, true, 12});
            shapes.push_back({"in_proj_b/a   K2048 N16", 1002, 2048, 16, true, 36});
        } else if (s == "dispatch") {
            // Qwen3.5 presents EVERY projection as `M x 1 x K` (batch=M, M=1, weights
            // broadcast across the batch) -- unlike GME, where only some projections took
            // that form.  If the conventional `1 x M x K` form is cheaper, that would be a
            // graph-level lever on every GEMM in the model, so it gets priced explicitly.
            shapes.clear();
            for (auto nk : std::vector<std::pair<memory::dim, memory::dim>>{
                     {2048, 6144}, {2048, 5120}, {2048, 2048}, {6144, 2048}}) {
                std::string tagname = "K" + std::to_string(nk.first) + " N" + std::to_string(nk.second);
                shapes.push_back({tagname + " Mx1xK", 1002, nk.first, nk.second, true});
                shapes.push_back({tagname + " 1xMxK", 1002, nk.first, nk.second, false});
            }
        }
    }

    std::vector<Case> cases = {
        // --- what Qwen3.5-2B-INT4 actually executes today (group_size=-1, sym=true) ---
        {"perOC_s4_cab",  dt::s8, dt::s4, false, 0,   false, true, tag::cab},
        {"perOC_s4_abc",  dt::s8, dt::s4, false, 0,   false, true, tag::abc},
        {"perOC_s4_any",  dt::s8, dt::s4, false, 0,   false, true, tag::any},
        // --- what Qwen3.5-2B-INT4-g128 actually executes today (group_size=128, sym=false) ---
        {"g128_u4_cab",   dt::s8, dt::u4, true,  128, true,  true, tag::cab},
        {"g128_u4_abc",   dt::s8, dt::u4, true,  128, true,  true, tag::abc},
        {"g128_u4_any",   dt::s8, dt::u4, true,  128, true,  true, tag::any},
        // --- ceilings, to say whether any remaining gap is dequant or the DPAS path ---
        {"ref_s8_dense",  dt::s8, dt::s8, false, 0,   false, false, tag::any},
        {"ref_f16_dense", dt::f16, dt::f16, false, 0, false, false, tag::any},
    };

    printf("%-16s", "case");
    for (auto &s : shapes) printf("%30s", s.name.c_str());
    printf("\n%s\n", std::string(16 + 30 * shapes.size(), '-').c_str());

    std::map<std::string, double> baseline;
    const char *case_filter = getenv("QWEN_PROBE_CASES");
    const bool model_mode = getenv("QWEN_PROBE_SHAPES") && std::string(getenv("QWEN_PROBE_SHAPES")) == "model";
    std::vector<std::pair<std::string, std::pair<double, double>>> model_totals;  // case -> {ms, gflop}

    for (auto &c : cases) {
        if (case_filter && std::string(case_filter).find(c.name) == std::string::npos) continue;
        printf("%-16s", c.name.c_str());
        fflush(stdout);
        double total_ms = 0.0, total_gflop = 0.0;

        for (auto &s : shapes) {
            memory::dims src_dims = s.batched_m1 ? memory::dims{s.m, 1, s.k} : memory::dims{1, s.m, s.k};
            memory::dims wei_dims = {1, s.k, s.n};
            memory::dims dst_dims = s.batched_m1 ? memory::dims{s.m, 1, s.n} : memory::dims{1, s.m, s.n};

            if (c.wei_scales_grouped && s.k % c.wei_scale_group != 0) {
                printf("%30s", "n/a (K%group)");
                fflush(stdout);
                continue;
            }

            auto src_md = memory::desc(src_dims, c.src_dt, tag::abc);
            auto wei_md = memory::desc(wei_dims, c.wei_dt, c.wtag);
            auto dst_md = memory::desc(dst_dims, dt::f16, tag::abc);

            primitive_attr attr;
            attr.set_scratchpad_mode(scratchpad_mode::user);
            if (c.src_scales) attr.set_scales(DNNL_ARG_SRC, SRC_MASK_MK, {1, s.k}, dt::f16);
            if (c.wei_scales_grouped)
                attr.set_scales(DNNL_ARG_WEIGHTS, WEI_MASK_KN, {c.wei_scale_group, 1}, dt::f16);
            else if (c.wei_dt == dt::u4 || c.wei_dt == dt::s4)
                attr.set_scales(DNNL_ARG_WEIGHTS, WEI_MASK_N, {}, dt::f16);
            if (c.wei_zp)
                attr.set_zero_points(DNNL_ARG_WEIGHTS, WEI_MASK_KN, {c.wei_scale_group, 1}, dt::u8);

            matmul::primitive_desc pd;
            try {
                pd = matmul::primitive_desc(eng, src_md, wei_md, dst_md, attr);
            } catch (const dnnl::error &e) {
                printf("%30s", "unsupported");
                fflush(stdout);
                if (getenv("PROBE_EXPLAIN")) fprintf(stderr, "\n  %s: %s\n", c.name.c_str(), e.what());
                continue;
            }

            matmul prim(pd);
            auto src = memory(pd.src_desc(), eng);
            auto wei = memory(pd.weights_desc(), eng);
            auto dstm = memory(pd.dst_desc(), eng);
            auto scratch = memory(pd.scratchpad_desc(), eng);

            std::unordered_map<int, memory> args = {{DNNL_ARG_SRC, src},
                                                    {DNNL_ARG_WEIGHTS, wei},
                                                    {DNNL_ARG_DST, dstm},
                                                    {DNNL_ARG_SCRATCHPAD, scratch}};

            memory src_sc, wei_sc, wei_zpm;
            if (c.src_scales) {
                src_sc = memory({{s.m, s.k / 128}, dt::f16, tag::ab}, eng);
                args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC] = src_sc;
            }
            if (c.wei_scales_grouped) {
                wei_sc = memory({{1, s.k / c.wei_scale_group, s.n}, dt::f16, tag::abc}, eng);
                args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS] = wei_sc;
            } else if (c.wei_dt == dt::u4 || c.wei_dt == dt::s4) {
                wei_sc = memory({{s.n}, dt::f16, tag::a}, eng);
                args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS] = wei_sc;
            }
            if (c.wei_zp) {
                wei_zpm = memory({{1, s.k / c.wei_scale_group, s.n}, dt::u8, tag::abc}, eng);
                args[DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS] = wei_zpm;
            }

            zero(src); zero(wei); zero(dstm); zero(scratch);
            zero(src_sc); zero(wei_sc); zero(wei_zpm);

            for (int i = 0; i < warmup; ++i) prim.execute(strm, args);
            strm.wait();

            double t0 = now_ms();
            for (int i = 0; i < iters; ++i) prim.execute(strm, args);
            strm.wait();
            double ms = (now_ms() - t0) / iters;

            double gflop = 2.0 * s.m * s.k * s.n / 1e9;
            total_ms += ms * s.count;
            total_gflop += gflop * s.count;
            char buf[64];
            std::string key = s.name;
            if (baseline.find(key) == baseline.end()) baseline[key] = ms;
            snprintf(buf, sizeof(buf), "%.3fms %.1fT %.2fx", ms, gflop / ms, baseline[key] / ms);
            printf("%30s", buf);
            fflush(stdout);
        }
        printf("\n");
        if (model_mode) model_totals.push_back({c.name, {total_ms, total_gflop}});
    }

    printf("\nms = per-call wall time, T = TFLOP/s, Nx = speedup vs the first case listed\n");
    printf("iters=%d warmup=%d\n", iters, warmup);

    if (model_mode) {
        printf("\n--- count-weighted decoder total (one prefill pass, M=1002) ---\n");
        printf("%-16s %14s %14s %12s\n", "case", "predicted ms", "GFLOP", "eff TFLOP/s");
        for (auto &t : model_totals)
            printf("%-16s %14.1f %14.1f %12.1f\n", t.first.c_str(), t.second.first, t.second.second,
                   t.second.second / t.second.first);
        printf("\nCompare `predicted ms` against the measured gemm_kernel time minus lm_head.\n");
    }
    return 0;
}
