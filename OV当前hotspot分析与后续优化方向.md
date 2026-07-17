# OpenVINO 当前优化态 hotspot 分析与后续优化方向（Qwen3 / Qwen3.5）

> 日期：2026-06-29（含同日**二次复核更正**：修正了 `Transpose_1` 的错误归因，并用实测排除了 GDN V_BLOCK_SIZE / DQ 进一步去重 / transpose kernel 级修法三条方向。更正点见 §0、§5、§7.4、§8、§12、§15。）  
> 目标：在已经落地 opt-1/opt-2/opt-3 的本地 OpenVINO GPU 插件基础上，用 `model_bench --profile` 和 VTune 重新分析 Qwen3/Qwen3.5 当前剩余 hotspot，提出下一批具体可优化点、源码入口和验证方法。  
> 环境：`source /home/user/envs/model_test_env/bin/activate`，`source /opt/intel/oneapi/setvars.sh`，本地 OpenVINO/GenAI install：`/home/user/chaofan/model_test/benchmark/openvino/install`。
> 结论先看 §0：**本轮无新增已落地优化**（3-opt 态已是 md5-无损约束下的本机 prefill 最优）；价值在更正归因 + 实测排除三条无效方向 + 把唯一高价值方向（conv-sandwich fused op）定清楚。

---

## 0. 当前结论

当前源码已经处于上一轮三项优化后的状态：

- `layout_optimizer.cpp`：depthwise conv 保持 planar，去掉 Qwen3.5 SSM conv 前后的大块 Reorder。
- `permute_kernel_ref.cpp`：修复 `permute_ref` 在 F 轴连续写场景下的 LWS 选择。
- `dynamic_quantize_fully_connected.cpp` + `dynamic_quantize.cpp`：共享激活的多个 FC 复用同一个 `DynamicQuantize`，并保留 decode skip 行为。

重新 profile 后，Qwen3.5 的大 Reorder 已经不再是问题；当前剩余 hotspot 变成两类：

1. **Qwen3.5 hybrid/SSM 专属的 layout 与状态搬运**：`linear_attn` 子图里的 `Transpose_1`、`Slice_2`、`Concat`、depthwise `GroupConvolution`、`GatedDeltaNet` 周边 kernel。
2. **Qwen3/Qwen3.5 共有的 transformer 本体**：`FullyConnectedCompressed/gemm_kernel`、`lm_head`、`DynamicQuantize`、SDPA/PA、RMS。

最值得继续做的不是再碰已消掉的 Reorder，而是：

| 优先级 | 方向 | 主要命中模型 | 预期收益 | 风险 |
|---|---|---|---|---|
| P0 | **把 causal depthwise conv + 前后 transpose + state concat/slice 做成一个 layout-aware fused op**（唯一能消掉那 3.57 ms `Transpose_1` 的途径） | Qwen3.5 | 中到高 | 高 |
| P2 | **greedy-only GPU ArgMax/TopK 输出 token id，减少 host logits 搬运** | Qwen3/Qwen3.5 | e2e 小到中，infer 小 | 中高 |
| P3 | Qwen3 dense 的 SDPA/PA backend A/B（需先解决 `-lc` 的 `stod` 报错） | Qwen3 | 小 | 中 |

> **2026-06-29 二次复核更正了上一版的方向表。** 经源码级追踪 + 实测，上一版列为 P0/P1 的几个方向已被证伪或证明已到floor：
> - ~~「改 GDN fusion 边界消除 GDN transpose」~~：**归因错误**。当前最大的 `linear_attn/Transpose_1`（3.57 ms）是 **conv 路径的「转回」transpose**（`Swish[B,6144,S]→[B,S,6144]→VariadicSplit`），不是 `FuseGDNLoop` 的 `out_transposed`。改 GDN layout 契约对它无效。详见 §5。
> - ~~「调优 `gated_delta_net_ref` kernel（V_BLOCK_SIZE）」~~：**实测死路**。VBS=4 已最优；VBS=8 的「加速」是 kernel 把累加器写死 `float4` 导致漏算一半 V 列的假象，正确实现 VBS=8 反而净回退。详见 §7.4。
> - ~~「进一步系统化 DynamicQuantize 去重」~~：**已到 floor**。opt-3 已把 Qwen3 dense 的 84 处共享冗余全部消掉（197 MatMul→113 DQ 恰好对上），剩余 DQ 都是不同 config 或不共享输入，通用 CSE 也救不了。详见 §8。
>
> 唯一仍有可观收益、但属高风险的方向是把 conv-sandwich 整段融成一个有状态的 layout-aware kernel（§5.3 + §6 合并）。这需要新增 internal op + 改 transformation + 写 kernel，且要同时保住 prefill **和** decode 的 md5（opt-3 已经证明动 SSM 状态路径极易破 decode 数值），属于需要立项的工作，不在一次小改的 md5-无损范围内。

`gemm_kernel` / `FullyConnectedCompressed` 仍是最大绝对时间项，但它大多是 INT4/XMX 矩阵乘本体；除非改模型输出策略或重写 matmul 路径，否则局部源码优化空间有限。

> **本轮（2026-06-29 第二次）结论**：在 md5 逐字节无损这条硬约束下，**当前优化态（opt-1/2/3）已无可安全落地的新局部优化点**；剩余 hotspot 要么已到算力/带宽 floor，要么需要高风险的图级 fused-op 立项。本轮没有新增已落地的优化，价值在于更正了上一版的错误归因、并用实测排除了三条看似可行实则无效/有害的方向（见下文各节）。

---

## 1. 分析方法

### 1.1 确认当前二进制使用本地 OV

```bash
cd /home/user/chaofan/model_test
source /home/user/envs/model_test_env/bin/activate
source benchmark/openvino/install/setupvars.sh >/dev/null 2>&1
export LD_LIBRARY_PATH="$PWD/benchmark/openvino/install/runtime/lib/intel64:${LD_LIBRARY_PATH:-}"

ldd model_bench/build-local/model_bench | grep -i openvino
git -C benchmark/openvino status --short
```

确认 `model_bench/build-local/model_bench` 链接到：

```text
/home/user/chaofan/model_test/benchmark/openvino/install/runtime/lib/intel64/libopenvino*.so
```

当前 OpenVINO 源码改动集中在：

```text
benchmark/openvino/src/plugins/intel_gpu/src/graph/dynamic_quantize.cpp
benchmark/openvino/src/plugins/intel_gpu/src/graph/layout_optimizer.cpp
benchmark/openvino/src/plugins/intel_gpu/src/kernel_selector/kernels/permute/permute_kernel_ref.cpp
benchmark/openvino/src/plugins/intel_gpu/src/plugin/transformations/dynamic_quantize_fully_connected.cpp
```

重建并刷新 install 中的 GPU plugin：

```bash
cd /home/user/chaofan/model_test
bash benchmark/openvino/rebuild_gpu_plugin.sh
```

### 1.2 OpenVINO code profile：真实 prompt、分阶段逐算子

Qwen3.5-2B：

```bash
cd /home/user/chaofan/model_test/model_bench
source /home/user/envs/model_test_env/bin/activate
source ../benchmark/openvino/install/setupvars.sh >/dev/null 2>&1
export LD_LIBRARY_PATH="/home/user/chaofan/model_test/benchmark/openvino/install/runtime/lib/intel64:${LD_LIBRARY_PATH:-}"

./build-local/model_bench vlm \
  -m ../benchmark/models/Qwen3.5-2B-INT4 \
  -d GPU -n 1 -ic 128 \
  -pf ./prompts/llama-2-7b-chat_l.jsonl \
  --no-monitor \
  --profile --profile-iters 32 --profile-warmup 3 --profile-topk 80 \
  --profile-csv results/hotspot_20260629_q35_2b_gpu.csv \
  -r /tmp/hotspot_q35_2b.csv -rj /tmp/hotspot_q35_2b.json
```

Qwen3-1.7B：

```bash
./build-local/model_bench llm \
  -m ../benchmark/models/Qwen3-1.7B-INT4 \
  -d GPU -n 1 -ic 128 \
  -pf ./prompts/llama-2-7b-chat_l.jsonl \
  --no-monitor \
  --profile --profile-iters 32 --profile-warmup 3 --profile-topk 80 \
  --profile-csv results/hotspot_20260629_q3_1_7b_gpu.csv \
  -r /tmp/hotspot_q3_1_7b.csv -rj /tmp/hotspot_q3_1_7b.json
```

注意：`--profile` 的 per-op `real_time` 是 `get_profiling_info()` 的逐算子统计，适合看相对结构和具体 node，不等于 wall-clock TTFT。decode 中 `DynamicQuantize` 还有一个额外 caveat：raw `--profile` 会列出 DQ 行，但 VTune 的真实 pipeline 采样显示 decode 阶段 DQ kernel 基本被 runtime skip，见 §3.3。

### 1.3 VTune GPU hotspots：真实 GenAI pipeline

128-token 全流程（观察 decode）：

```bash
cd /home/user/chaofan/model_test
source /home/user/envs/model_test_env/bin/activate
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
source benchmark/openvino/install/setupvars.sh >/dev/null 2>&1
export LD_LIBRARY_PATH="$PWD/benchmark/openvino/install/runtime/lib/intel64:${LD_LIBRARY_PATH:-}"

vtune -collect gpu-hotspots \
  -r /tmp/vtune_hotspot_20260629_q35_2b_opt -- \
  model_bench/build-local/model_bench vlm \
    -m benchmark/models/Qwen3.5-2B-INT4 -d GPU -n 1 -ic 128 \
    -pf model_bench/prompts/llama-2-7b-chat_l.jsonl --no-monitor \
    -r /tmp/vtune_q35_2b.csv -rj /tmp/vtune_q35_2b.json

vtune -collect gpu-hotspots \
  -r /tmp/vtune_hotspot_20260629_q3_1_7b_opt -- \
  model_bench/build-local/model_bench llm \
    -m benchmark/models/Qwen3-1.7B-INT4 -d GPU -n 1 -ic 128 \
    -pf model_bench/prompts/llama-2-7b-chat_l.jsonl --no-monitor \
    -r /tmp/vtune_q3_1_7b.csv -rj /tmp/vtune_q3_1_7b.json
```

prefill-only / TTFT 视角：

```bash
vtune -collect gpu-hotspots \
  -r /tmp/vtune_hotspot_20260629_q35_2b_prefill -- \
  model_bench/build-local/model_bench vlm \
    -m benchmark/models/Qwen3.5-2B-INT4 -d GPU -n 1 -ic 1 \
    -pf model_bench/prompts/llama-2-7b-chat_l.jsonl --no-monitor \
    -r /tmp/vtune_q35_2b_prefill.csv -rj /tmp/vtune_q35_2b_prefill.json

vtune -collect gpu-hotspots \
  -r /tmp/vtune_hotspot_20260629_q3_1_7b_prefill -- \
  model_bench/build-local/model_bench llm \
    -m benchmark/models/Qwen3-1.7B-INT4 -d GPU -n 1 -ic 1 \
    -pf model_bench/prompts/llama-2-7b-chat_l.jsonl --no-monitor \
    -r /tmp/vtune_q3_1_7b_prefill.csv -rj /tmp/vtune_q3_1_7b_prefill.json
```

导出 computing-task 报告：

```bash
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1

vtune -report hotspots -group-by computing-task \
  -r /tmp/vtune_hotspot_20260629_q35_2b_prefill \
  -format csv -csv-delimiter comma \
  > /tmp/vtune_hotspot_20260629_q35_2b_prefill_tasks.csv

vtune -report hotspots -group-by computing-task \
  -r /tmp/vtune_hotspot_20260629_q3_1_7b_prefill \
  -format csv -csv-delimiter comma \
  > /tmp/vtune_hotspot_20260629_q3_1_7b_prefill_tasks.csv
```

VTune 本次每轮结束都打印了：

```text
vtune: Error: Cannot stop collection of GPU events
```

但随后仍正常生成 result path、summary 和 computing-task CSV，所以本次报告可用；这个错误更像采集停止阶段的 GPU events 清理警告。

---

## 2. 当前 benchmark 口径

本轮不是重新做 n=8 正式 A/B，而是 hotspot 定位。代表性 wall-clock 数据如下：

| 模型 | 命令 | TTFT | first-token infer | 说明 |
|---|---:|---:|---:|---|
| Qwen3.5-2B-INT4 | `--profile -n 1 -ic 128` | 176.98 ms | 132.27 ms | 当前优化态，真实 prompt 950 token |
| Qwen3-1.7B-INT4 | `--profile -n 1 -ic 128` | 62.29 ms | 62.23 ms | 当前优化态，真实 prompt 937 token |
| Qwen3.5-2B-INT4 | VTune `-ic 1` | 179.41 ms | 133.74 ms | VTune 有采集开销，但 kernel 排名可用 |
| Qwen3-1.7B-INT4 | VTune `-ic 1` | 62.22 ms | 61.93 ms | VTune prefill-only |

和上一份复现文档的 n=8 结果一致：Qwen3.5-2B 当前优化态 infer-TTFT 约 132 ms，Qwen3-1.7B dense 约 62 ms。

---

## 3. `model_bench --profile` 结果

### 3.1 Qwen3.5-2B：LM prefill

CSV：`model_bench/results/hotspot_20260629_q35_2b_gpu.csv`  
Stage：`vlm_lm_prefill`，op-sum = 24.884 ms。

| node_type | op-time | 占 op-sum | count | 解释 |
|---|---:|---:|---:|---|
| FullyConnectedCompressed | 12.486 ms | 50.18% | 175 | INT4 FC，本体 + `lm_head` |
| Transpose | 4.656 ms | 18.71% | 60 | 主要来自 `linear_attn/Transpose_1` |
| DynamicQuantize | 4.030 ms | 16.20% | 97 | opt-3 后剩余 DQ |
| GatedDeltaNet | 1.079 ms | 4.34% | 18 | SSM recurrence fused op |
| StridedSlice | 0.609 ms | 2.45% | 65 | 主要来自 linear_attn state/slice |
| IndirectSDPA | 0.504 ms | 2.03% | 6 | 6 个 dense attention 层 |
| Convolution | 0.493 ms | 1.98% | 18 | SSM depthwise conv |
| RMS | 0.369 ms | 1.48% | 79 | RMSNorm |
| Concat | 0.331 ms | 1.33% | 41 | 主要来自 linear_attn state concat |

最重的单节点：

| node | op-time | 说明 |
|---|---:|---|
| `__module.model.lm_head/ov_ext::linear/MatMul.0` | 5.108 ms | 单个 `lm_head` 占 prefill op-sum 20.5% |
| `linear_attn/aten::transpose/Transpose_1` | 3.617 ms / 18 层合计 | Qwen3.5 当前最大的非 GEMM 可优化搬运 |
| `linear_attn/aten::slice/Slice_2` | 0.481 ms / 18 层合计 | state/casual conv 周边搬运 |
| `linear_attn/aten::cat/Concat` | 0.304 ms / 18 层合计 | state 拼接 |
| `linear_attn/aten::_convolution/GroupConvolution` | 0.493 ms / 18 层合计 | depthwise causal conv |

按去掉 layer id 后的重复模式：

| 模式 | rows | op-time |
|---|---:|---:|
| `linear_attn/aten::transpose/Transpose_1` | 18 | 3.617 ms |
| `linear_attn/aten::transpose/Transpose` | 18 | 0.464 ms |
| `linear_attn/aten::slice/Slice_2` | 18 | 0.481 ms |
| `linear_attn/aten::slice/Slice_7` | 18 | 0.072 ms |
| `linear_attn/aten::cat/Concat` | 18 | 0.304 ms |
| `linear_attn/aten::_convolution/GroupConvolution` | 18 | 0.493 ms |
| `Loop_*` / GatedDeltaNet | 18 | 1.079 ms |

结论：Qwen3.5 prefill 的可优化空间已经高度集中到 **18 个 linear_attn/SSM 层的重复结构**。

### 3.2 Qwen3.5-2B：LM decode

Stage：`vlm_lm_decode`，`--profile-iters 32`，op-sum = 781.173 ms。

| node_type | op-time / 32 steps | 占 op-sum | count | 解释 |
|---|---:|---:|---:|---|
| FullyConnectedCompressed | 393.462 ms | 50.37% | 5600 | FC + 每步 `lm_head` |
| Transpose | 147.049 ms | 18.82% | 1920 | raw profile 中 residual permute 很明显 |
| DynamicQuantize | 128.960 ms | 16.51% | 3104 | 见 §3.3 caveat |
| GatedDeltaNet | 31.978 ms | 4.09% | 576 | SSM recurrence |
| StridedSlice | 17.549 ms | 2.25% | 2080 | state/slice |
| IndirectSDPA | 16.859 ms | 2.16% | 192 | attention decode |
| Convolution | 15.234 ms | 1.95% | 576 | depthwise conv decode |
| RMS | 10.947 ms | 1.40% | 2528 | RMSNorm |
| Concat | 9.663 ms | 1.24% | 1312 | state concat |

raw `--profile` 的 decode 结构说明：如果后续关注 TPOT，Qwen3.5 的 `linear_attn` 周边 transpose/slice/concat/conv/GDN 仍然是重复执行的真实结构。但其中 `DynamicQuantize` 的实际 kernel 时间要以 VTune 为准。

### 3.3 `--profile` 与 VTune 对 DynamicQuantize decode 的差异

`--profile` 中 Qwen3/Qwen3.5 decode 都列出了大量 `DynamicQuantize` 行，例如 Qwen3.5 decode 32 step 中 DQ op-time 128.960 ms。

但 VTune 128-token 真实 GenAI pipeline 显示：

- Qwen3.5：`dynamic_quantize_gpu_opt` 只有 0.009569 s，占 GPU task time 0.27%，instance count 192。
- Qwen3：`dynamic_quantize_gpu_opt` 只有 0.009130 s，占 GPU task time 0.31%，instance count 224。

这个 instance count 基本等于 **warmup + measured 两次 prefill** 的 DQ 数量，而不是 128-token decode 的 DQ 数量。结论：真实 pipeline decode 中 DQ 仍然被 runtime skip，`--profile` 的 decode DQ 行更适合看图结构/潜在路径，不适合当实际 kernel 时间预算。

后续优化 DQ 时，应该主要以 prefill DQ 为目标；decode 侧不能因为 `--profile` 行数多就误判为最大真实 kernel。

### 3.4 Qwen3-1.7B：LLM prefill

CSV：`model_bench/results/hotspot_20260629_q3_1_7b_gpu.csv`  
Stage：`llm_prefill`，op-sum = 18.622 ms。

| node_type | op-time | 占 op-sum | count | 解释 |
|---|---:|---:|---:|---|
| FullyConnectedCompressed | 10.595 ms | 56.90% | 141 | dense 模型主耗时 |
| DynamicQuantize | 4.341 ms | 23.31% | 113 | prefill 激活量化 |
| IndirectSDPA | 1.978 ms | 10.62% | 28 | attention prefill |
| Transpose | 0.936 ms | 5.03% | 56 | attention q/k/v 相关 permute |
| RMS | 0.455 ms | 2.44% | 113 | RMSNorm |

最重单节点：

| node | op-time | 说明 |
|---|---:|---|
| `__module.lm_head/ov_ext::linear/MatMul.0` | 3.086 ms | 单个 `lm_head` 占 prefill op-sum 16.6% |
| 多个 `DynamicQuantize_*` | 单个约 0.105 ms | 每层重复，但已是 FC 量化路径的一部分 |

### 3.5 Qwen3-1.7B：LLM decode

Stage：`llm_decode`，`--profile-iters 32`，op-sum = 585.141 ms。

| node_type | op-time / 32 steps | 占 op-sum | count | 解释 |
|---|---:|---:|---:|---|
| FullyConnectedCompressed | 332.675 ms | 56.85% | 4512 | FC + `lm_head` |
| DynamicQuantize | 138.912 ms | 23.74% | 3616 | 实际 kernel 见 §3.3 caveat |
| IndirectSDPA | 59.945 ms | 10.24% | 896 | attention decode |
| Transpose | 29.890 ms | 5.11% | 1792 | attention permute |
| RMS | 14.210 ms | 2.43% | 3616 | RMSNorm |

Qwen3 dense 结构里没有 Qwen3.5 那种 SSM conv/layout 浪费。它的下一步优化重点更偏 SDPA/PA、DQ、RMS，以及 `lm_head`/sampling 输出策略。

---

## 4. VTune 结果

VTune 的 computing-task 口径是真实 GenAI pipeline 的 GPU kernel 视角。下面的 total time 包含 warmup + measured 两次运行，因此更适合看比例和 kernel 排名，不直接等同单次 TTFT。

### 4.1 Qwen3.5-2B prefill-only (`-ic 1`)

VTune result：`/tmp/vtune_hotspot_20260629_q35_2b_prefill`  
聚合后的 GPU task total = 0.245573 s。

| kernel family | time | 占比 | instances | avg occupancy | 解释 |
|---|---:|---:|---:|---:|---|
| `gemm_kernel` | 0.079445 s | 32.35% | 350 | FC/GEMM 本体 |
| `gated_delta_net` | 0.036565 s | 14.89% | 36 | 18 层 × warmup/measured |
| `permute_ref` | 0.032404 s | 13.20% | 120 | residual transpose |
| `strided_slice_ref` | 0.025350 s | 10.32% | 128 | linear_attn/state 周边 slice |
| `concatenation_gpu_simple_ref` | 0.014113 s | 5.75% | 148 | linear_attn/state concat |
| `rms_gpu_bfyx_opt` | 0.009896 s | 4.03% | 158 | RMSNorm |
| `sdpa_micro__prefill` | 0.009783 s | 3.98% | 12 | 6 个 attention 层 ×2 |
| `dynamic_quantize_gpu_opt` | 0.009317 s | 3.79% | 192 | prefill DQ |
| `gen_conv` | 0.009109 s | 3.71% | 36 | depthwise conv |
| `generic_eltwise_ref` | 0.006300 s | 2.57% | 116 | 小 eltwise |
| `conv_reorder` | 0.006172 s | 2.51% | 36 | conv 内/周边 reorder-like kernel |

这个表比 `--profile` 更强调 `GatedDeltaNet`、`Slice`、`Concat` 和 conv 周边 kernel，说明 TTFT 里 SSM 子图的实际 GPU 时间仍很可观。

### 4.2 Qwen3-1.7B prefill-only (`-ic 1`)

VTune result：`/tmp/vtune_hotspot_20260629_q3_1_7b_prefill`  
聚合后的 GPU task total = 0.125550 s。

| kernel family | time | 占比 | instances | avg occupancy | 解释 |
|---|---:|---:|---:|---:|---|
| `gemm_kernel` | 0.081577 s | 64.98% | 282 | 44.2% | FC/GEMM 本体 |
| `sdpa_micro__prefill` | 0.017310 s | 13.79% | 56 | 49.0% | attention prefill |
| `rms_gpu_bfyx_opt` | 0.012173 s | 9.70% | 226 | 64.0% | RMSNorm |
| `dynamic_quantize_gpu_opt` | 0.009135 s | 7.28% | 224 | 100.0% | prefill DQ |
| `pa_kv_cache_update_ref` | 0.002727 s | 2.17% | 56 | 49.2% | KV update |
| `rope_opt` | 0.002281 s | 1.82% | 112 | 99.2% | RoPE |

Qwen3 dense 已经主要是 GEMM + attention 本体。和 Qwen3.5 不同，它没有明显的 SSM layout 搬运大块。

### 4.3 128-token 真实 pipeline：decode 权重

Qwen3.5-2B，VTune result：`/tmp/vtune_hotspot_20260629_q35_2b_opt`，GPU task total = 3.582744 s。

| kernel family | 占比 | 说明 |
|---|---:|---|
| `gemm_kernel` | 80.70% | 128-token 下 decode 的 FC/GEMM 绝对主导 |
| `sdpa_micro__generate` | 3.99% | attention generate |
| `gated_delta_net` | 3.52% | SSM recurrence |
| `rms_gpu_bfyx_opt` | 2.09% | RMSNorm |
| `strided_slice_ref` | 2.07% | SSM/state slice |
| `clEnqueueMemcpyINTEL` | 1.45% | host/device copy |
| `concatenation_gpu_simple_ref` | 1.20% | SSM/state concat |
| `permute_ref` | 1.13% | residual transpose |
| `dynamic_quantize_gpu_opt` | 0.27% | 主要是 prefill；decode 已 skip |

Qwen3-1.7B，VTune result：`/tmp/vtune_hotspot_20260629_q3_1_7b_opt`，GPU task total = 2.902840 s。

| kernel family | 占比 | 说明 |
|---|---:|---|
| `gemm_kernel` | 84.42% | dense decode 绝对主导 |
| `paged_attention_opt__single_token` | 8.30% | PA decode attention |
| `rms_gpu_bfyx_opt` | 3.31% | RMSNorm |
| `pa_kv_cache_update_ref` | 1.34% | KV update |
| `rope_opt` | 1.01% | RoPE |
| `dynamic_quantize_gpu_opt` | 0.31% | 主要是 prefill；decode 已 skip |

---

## 5. 可优化点 1：Qwen3.5 `linear_attn` 残余 Transpose（已更正归因）

> **重要更正（2026-06-29 二次复核）**：本节早先版本把最大的 `linear_attn/aten::transpose/Transpose_1` 归因为 `FuseGDNLoop` 插入的 `out_transposed`（GDN 输出 transpose），并据此把「改 GDN layout 契约」列为 P0。**经源码级追踪，这个归因是错的**，下面给出更正后的事实与据此调整的方向。

### 5.1 证据 + 更正后的根因

Qwen3.5 prefill 中：

- `Transpose` op-sum：~5.76 ms（当前态），占 16.1%。
- 其中 `linear_attn/aten::transpose/Transpose_1`：18 层合计 ~3.57 ms（198 µs/层），是最大的非 GEMM 搬运。
- 同形状的 `linear_attn/aten::transpose/Transpose`（pre）只有 88.9 µs/层 —— 同样 `[0,2,1]`、同样 6144×S 数据量，却差 2.2×。
- VTune prefill-only：`permute_ref` 占 13.20%。

**更正后的根因（用 IR 边追踪 producer/consumer 确认）**：`Transpose_1`（IR id=71）的 **输入是 `Swish`（conv→SiLU 的输出，`[B,6144,S]`）、输出 `[B,S,6144]` 喂给 `VariadicSplit`**。它是 **因果 depthwise conv 的「转回」transpose**，属于 conv 路径，**不是** GDN 的 q/k/v/output transpose。完整链路：

```
in_proj_qkv(MatMul) → Transpose(pre, [B,S,6144]→[B,6144,S]) → Concat(state) →
GroupConvolution(depthwise,k=4) → Slice_2(因果) → Swish → Transpose_1([B,6144,S]→[B,S,6144]) → VariadicSplit → ...
```

两个 transpose 只为把张量在 conv 需要的「通道在前」`[B,6144,S]` 与下游 reshape/split 需要的 `[B,S,6144]` 之间来回翻。`FuseGDNLoop` 的 `out_transposed`（下方代码）确实存在，但它在 GDN 的 q/k/v/输出路径上、且大概率已被随后的 `TransposeFuse` 折叠，**不是当前 profile 里这个 3.57 ms 的来源**。

```cpp
// fuse_gated_delta_net.cpp FuseGDNLoop —— 这些是 GDN 路径的 transpose，不是 Transpose_1
auto out_transposed = std::make_shared<v1::Transpose>(linear_attn->output(0), perm_bhls_to_blhs);
```

### 5.2 为什么没有简单的 kernel 级修法（已实测排除）

- **`permute_ref` 已被 opt-2 调过**：实测 in-model GWS/LWS 显示 opt-2 已对 `Transpose_1` 生效（gws=[1,950,6144]、lws=[1,1,1024] 落在合并写的 F 轴）。pre-conv 那个反而没触发 opt-2（gws_f=950<gws_y=6144），但它本来就快。2.2× 的差异来自 **post-conv 的输入是 `Slice_2` 的非紧凑视图**（沿内层时间轴切片，读访问带 stride/offset），standalone 紧凑 microbench 量到 143 µs、in-model 198 µs，差额就是这个 pitch 惩罚。
- **tiled SLM kernel（`permute_tile_8x8_4x4`，FORCE_PRIORITY_1，能同时合并读写）不适用**：它的 `Validate` 要求 `is_rotating_except_batch`（把 F 旋到最内轴，如 `[0,3,1,2]`），而我们的是 F↔Y 两轴 swap（X=1），模式不匹配，强行启用会算错。
- 结论：**这个 transpose 没有 md5 无损的局部 kernel 修法**。要再降只能图级消除整段 conv-sandwich（§6），属高风险。

### 5.3 真正的优化方向（图级，高风险，需立项）

目标：消除 `Transpose(pre)` + `Slice_2` + `Transpose_1` 这段 conv-sandwich，让 depthwise causal conv 直接在原生 layout 上做。OpenVINO `Convolution` 算子语义把通道固定在 axis=1、空间在末轴，**无法直接吃 `[B,S,C]`**，所以必须新增一个 Qwen3.5 SSM 专用的 fused op（见 §6），把 transpose/concat/slice/conv 收进一个 layout-aware kernel。这与 §6 是同一件事，合并推进。

> 不再建议「改 GDN layout 契约去掉 out_transposed」作为消除 `Transpose_1` 的手段——因为 `Transpose_1` 根本不是 out_transposed。若将来要顺手清理 GDN 路径上可能残留的 transpose，可作为独立小项，但收益不对应当前这 3.57 ms。

---

## 6. 可优化点 2（合并 §5.3）：Qwen3.5 causal depthwise conv-sandwich 融合

> 经 §5 更正后，这一节升级为**唯一的性能立项**：它要融的 conv-sandwich 不只含 concat/slice/conv，还包含 §5 那个 3.57 ms 的 `Transpose_1`（以及 1.6 ms 的 pre-conv `Transpose`）。把这整段做成一个有状态、layout-aware 的 fused op，才能同时拿下 transpose + slice + concat。

### 6.1 证据

Qwen3.5 prefill-only VTune 中，和 linear_attn conv/state 相关的 kernel 加起来已经不小：

| kernel | 占比 |
|---|---:|
| `strided_slice_ref` | 10.32% |
| `concatenation_gpu_simple_ref` | 5.75% |
| `gen_conv` | 3.71% |
| `conv_reorder` | 2.51% |

`--profile` 中对应节点：

| 模式 | rows | op-time |
|---|---:|---:|
| `linear_attn/aten::slice/Slice_2` | 18 | 0.481 ms |
| `linear_attn/aten::slice/Slice_7` | 18 | 0.072 ms |
| `linear_attn/aten::cat/Concat` | 18 | 0.304 ms |
| `linear_attn/aten::_convolution/GroupConvolution` | 18 | 0.493 ms |

opt-1 已经去掉了 depthwise conv 前后的大 layout Reorder，但当前仍然存在**逻辑上的 state 搬运**：为了 causal conv，把历史 state 和当前输入 concat，再通过 slice 更新 state/取窗口。这些都是纯搬运/小算子，和 conv 本体计算量不匹配。

### 6.2 优化方案

新增一个 Qwen3.5 SSM 专用的 internal op，例如 `CausalDepthwiseConv1DState`：

输入：

- 当前 hidden/conv input：`x`
- conv state：`state`
- depthwise weights/bias

输出：

- causal conv result
- updated state

在 transformation pass 中匹配以下模式：

```text
state + x -> Concat -> GroupConvolution -> ...
Concat/Slice -> updated_state
```

或者更保守一点：只 fuse `Concat + Slice state update`，先不动 `GroupConvolution`。

GPU kernel 侧：

- 不实际 materialize concat buffer。
- 对 kernel=4 的 depthwise conv，按逻辑索引从 `state` 和 `x` 读取窗口。
- 同一个 kernel 写 conv output 和 new state。
- decode 下 seq=1 时收益尤其直接：避免每 token 每层的 concat/slice 小 kernel。

### 6.3 源码入口

- 现有 conv/layout：`benchmark/openvino/src/plugins/intel_gpu/src/graph/layout_optimizer.cpp`
- GroupConvolution 转 primitive：`benchmark/openvino/src/plugins/intel_gpu/src/plugin/transformations/convert_convolution.cpp`
- 可以新增 pass 到 GPU transformation pipeline，位置靠近 `GatedDeltaNetFusion` 后或 convolution conversion 前。

### 6.4 验证方法

检查 `--profile`：

- `linear_attn/aten::cat/Concat` 18 行下降。
- `linear_attn/aten::slice/Slice_2` 18 行下降。
- `GroupConvolution` count 可能变成新 internal op；总 op-sum 应下降。

检查 VTune：

- `strided_slice_ref`、`concatenation_gpu_simple_ref`、`gen_conv`/`conv_reorder` 合计下降。

正确性：

- `-ic 1` md5 一致。
- `-ic 128` md5 一致，因为 state update 最容易影响 decode。

---

## 7. 可优化点 3：`gated_delta_net_ref` kernel 本体调优

### 7.1 证据

Qwen3.5 prefill-only VTune：`gated_delta_net` 占 14.89%，avg occupancy 53.3%。

当前 kernel 已有一个特化路径：

```cpp
constexpr size_t v_block_size = 4;
...
if (K_HEAD_DIM == 128 && (V_HEAD_DIM % V_BLOCK_SIZE == 0))
```

dispatch：

```cpp
v_blocks = ceil(V_HEAD_DIM / 4)
wgs.global = {batch, head_nums, v_blocks * subgroup_size}
wgs.local = {1, 1, subgroup_size}
```

kernel 内部每个 workgroup 按一个 head 的若干 V columns 顺序扫完整 `seq_len`，recurrence 限制了时间维并行度，但 V/head 维仍可调。

### 7.2 优化方案

> **已实测结论（2026-06-29）：`V_BLOCK_SIZE` 调参是死路，VBS=4 已是最优。** 详见 §7.4 的实测记录。其余方向（输出 transpose 融合、native_exp、state I/O 向量化）未逐一实测，留作后续。

1. **调 `V_BLOCK_SIZE`（已实测，不可行）**
   - 当前为 4。直接改成 8/16 时 GDN op-time 看似从 3.0 ms 降到 1.7 ms，但 **md5 变了**——因为 kernel 快路径把每-V-列的累加器**写死成 `float4`**（`b_v_vec`/`dot_part_k_vec`/`dot_part_q_vec`/`h_k_vec`/`b_output_vec`），却用 `v_idx < V_BLOCK_SIZE` 循环，VBS>4 时第 4–7 列越界且归约只覆盖 `.s0..s3`，**少算一半 V 列**，那个「加速」是漏算出来的假象。
   - 我把这些 `float4` 改成 `float[V_BLOCK_SIZE]` 数组（VBS=4 时与基线 **md5 逐字节一致**，确认重构无损），再设 VBS=8 让 8 列**正确**计算——此时 GDN op-time 反而**升到 6.48 ms**（每 subgroup 持有 8×float8 状态→寄存器翻倍/spill，work-group 数减半→occupancy 更差）。即正确实现下 VBS=8 是**净回退**。
   - 故 VBS=4 已最优，此 knob 关闭。

2. **输出 transpose 融合**
   - 注意：§5 已更正，`Transpose_1` 不是 GDN 输出 transpose，所以「GDN 直接写消费者 layout 以消掉 permute_ref」对当前那 3.57 ms 不成立。此条仅在清理 GDN 路径自身可能残留的 transpose 时才相关，收益另算。

3. **exp/l2norm 细节**
   - `g` 每 token 做 `exp`。可测试 `native_exp` 或近似路径，但这会改变数值，必须用 md5 和 logits 误差严格验证；默认不建议作为第一 POC。

4. **state load/write 优化**
   - 当前 state 存在 `h_state[V_BLOCK_SIZE][K_CHUNKS]`，可测 block read/write 或更大向量化，但要关注 register spill（VBS 实验已显示这个 kernel 对寄存器压力很敏感）。

### 7.3 验证方法

单独做 `gated_delta_net_ref.cl/.cpp` variant 后，用 prefill-only VTune 对比：

```bash
vtune -collect gpu-hotspots \
  -r /tmp/vtune_gdn_variant -- \
  model_bench/build-local/model_bench vlm \
    -m benchmark/models/Qwen3.5-2B-INT4 -d GPU -n 1 -ic 1 \
    -pf model_bench/prompts/llama-2-7b-chat_l.jsonl --no-monitor \
    -r /tmp/gdn_variant.csv -rj /tmp/gdn_variant.json
```

看：

- `gated_delta_net` time 是否下降。
- `Peak XVE Threads Occupancy` 是否改善。
- 是否出现 spill memory。
- md5 是否一致。

### 7.4 实测记录：V_BLOCK_SIZE 已验证为死路（2026-06-29）

| 配置 | GDN op-time（prefill） | md5 | 结论 |
|---|---:|---|---|
| VBS=4（基线，float4 写死） | 3.01 ms | `fd4d5e3c…` | 当前态 |
| VBS=2（原 float4） | 4.43 ms | `fd4d5e3c…`（同基线，因 2<4 不越界） | 更慢 |
| VBS=8（原 float4，**有 bug**） | 1.72 ms | `a41ab4bc…`（**变了！**） | 假加速：少算 4 列 |
| VBS=16（原 float4，**有 bug**） | 2.29 ms | `a41ab4bc…`（**变了！**） | 同上 |
| VBS=4（float4→float[VBS] 重构） | 3.29 ms | `fd4d5e3c…`（**与基线逐字节一致**） | 重构无损 ✓ |
| VBS=8（重构后，**正确**） | 6.48 ms | `fd4d5e3c…`（正确） | 正确实现下**净回退** |

**机制**：kernel 快路径的每-V-列累加器写死 `float4`，循环却跑 `v_idx<V_BLOCK_SIZE`。VBS>4 时第 4–7 列越界、归约只覆盖 `.s0..s3`，于是「加速」其实是漏算一半 V 列。把 `float4` 改成 `float[V_BLOCK_SIZE]` 数组后 VBS=4 与基线 md5 逐字节一致（证明重构无损），但 VBS=8 正确计算时寄存器翻倍/spill、work-group 数减半，GDN 反而 6.48 ms。**故 VBS=4 已最优，knob 关闭。** 这个 float4-vs-VBS 的潜在 bug 值得作为独立 correctness 修复上报（VBS≠4 时会算错），但它不是 TTFT 优化点。

> 工作区已回滚到 3-opt 干净态（GDN 文件无改动）；上述 variant 仅用于实测验证，未保留。

---

## 8. 可优化点 4：DynamicQuantize 后续优化

### 8.1 当前状态

opt-3 后 Qwen3.5-2B prefill DQ 已降到 97 个节点：

| 来源 | DQ 现状 |
|---|---:|
| Qwen3.5 `--profile` prefill | 4.030 ms，97 nodes，占 op-sum 16.20% |
| Qwen3.5 VTune prefill-only | 0.009317 s，占 GPU task 3.79% |
| Qwen3 `--profile` prefill | 4.341 ms，113 nodes，占 op-sum 23.31% |
| Qwen3 VTune prefill-only | 0.009135 s，占 GPU task 7.28% |

Qwen3 dense 中 DQ 仍有 113 个节点。**2026-06-29 用 IR 边分析复核：这 113 个已经是 opt-3 去重后的 floor，无进一步空间。** 证据：Qwen3-1.7B `openvino_model.xml` 共 198 个 MatMul、56 个「共享同一激活」的组，按 opt-3 口径可消的冗余 = Σ(组大小−1) = **84**；198 − 84 = 114 ≈ 实测 113（差 1 在舍入/lm_head 边界内）。也就是说：

- dense 的 q/k/v 是**三个独立投影**共享 input_layernorm 输出（实测 layer-0 组 = `[2048×2048, 1024×2048, 1024×2048]`，K 均=2048，config 相同），gate/up 共享 post-attn 输出——**这些 opt-3 已经全部合并**。
- 剩下的 113 个 DQ 分属 o_proj / down_proj / lm_head 等**不共享输入**的 FC，每个 FC 一份是必须的，没有冗余可去。
- 通用 `SharedOpOptimization` 即使补 `visit_attributes` 后再跑一轮，也只能合并「同输入+同 config」，而这部分 opt-3 已经覆盖——所以**系统化 CSE 不会带来额外的 DQ 减少**，只是把 opt-3 的点查换成更通用的实现（工程整洁度收益，非性能收益）。

### 8.2 优化方案

1. **把当前 pass 内去重系统化**
   - 给 `ov::op::internal::DynamicQuantize` 实现 `visit_attributes`。
   - 在 `DynamicQuantizeFullyConnected` 插入后，补跑一轮安全的 `SharedOpOptimization` / CSE。
   - 这样比当前只在 sibling FC target_inputs 中点查更通用。
   - 必须保证 config 不同的 DQ 不会被错误合并。

2. **分析 Qwen3 dense 剩余 113 个 DQ 的来源**
   - 打开 GPU graph dump，统计每个 DQ 的 consumer FC friendly name、group size、是否 precomputed reduction。
   - 确认是否还有“同一 activation + config 相同”但未被 opt-3 命中的 case。

3. **kernel 级调优**
   - 源码：`benchmark/openvino/src/plugins/intel_gpu/src/kernel_selector/kernels/dynamic_quantize/dynamic_quantize_kernel_opt.cpp`
   - 当前 dispatch 对 `LARGE_GS`：`gws={simd,total_block_num,batch}`，`lws={simd,block_num,1}`。
   - VTune DQ occupancy 100%，说明不是简单 occupancy 问题；更可能是内存/归约/量化本体。kernel 级收益预期小于图级减少 DQ 数量。

4. **长线：DQ + FC tile 融合**
   - 当前 DQ 独立生成 i8 activation 和 scale，再由 oneDNN FC 读取。
   - 如果能在 FC tile 内完成 activation quantization，可减少一次写回/读回。
   - 这是中高风险方向，因为会绕开 oneDNN 现有 compressed FC 接口。

### 8.3 验证方法

- Qwen3.5/Qwen3 都跑 `--profile`，检查 DQ node count 和 op-time。
- VTune prefill-only 看 `dynamic_quantize_gpu_opt` instances/time。
- 必须验证 decode md5，因为 DQ skip 行为曾经踩过坑。

---

## 9. 可优化点 5：Qwen3 dense SDPA/PA 路径

### 9.1 证据

Qwen3-1.7B prefill-only VTune：

- `sdpa_micro__prefill`：13.79%，instances=56，avg occupancy 49.0%。
- `pa_kv_cache_update_ref`：2.17%。

128-token VTune：

- `paged_attention_opt__single_token`：8.30%。
- `pa_kv_cache_update_ref`：1.34%。
- `paged_attention_opt__single_token_finalization`：0.56%。

`model_bench` 的 Qwen3 LLM 默认注入：

```cpp
ov::AnyMap plugin_cfg = build_plugin_config(args, device, "PA");
```

所以当前 Qwen3 的 attention 结果是默认 `ATTENTION_BACKEND=PA` 路径。

### 9.2 已做的小验证

尝试用 load config 覆盖：

```bash
printf '{"ATTENTION_BACKEND":"SDPA"}\n' > /tmp/ov_cfg_sdpa.json
model_bench/build-local/model_bench llm \
  -m benchmark/models/Qwen3-1.7B-INT4 -d GPU -n 3 -ic 1 \
  -pf model_bench/prompts/llama-2-7b-chat_l.jsonl --no-monitor \
  -lc /tmp/ov_cfg_sdpa.json \
  -r /tmp/q3_sdpa_smoke.csv -rj /tmp/q3_sdpa_smoke.json
```

当前直接失败：

```text
ERROR: stod
```

所以本轮不能把 `SDPA` backend 作为已验证替代方案。后续需要先确认 OpenVINO GPU plugin 当前支持的 `ATTENTION_BACKEND` 枚举值或 GenAI 传参格式。

### 9.3 优化方案

1. **找出合法 backend 枚举并做 PA vs 非 PA A/B**
   - 如果有非 PA 的 prefill 更快路径，可能降低 Qwen3 TTFT。
   - 注意 decode 可能更依赖 PA，不能只看 prefill。

2. **调优 PA single-token kernel**
   - 128-token 下 `paged_attention_opt__single_token` 占 8.30%。
   - 这是 TPOT 方向，不是 TTFT 主方向。

3. **优化 KV update / finalization 小 kernel**
   - `pa_kv_cache_update_ref` 和 finalization 占比不大，但 instance 多、occupancy 偏低。
   - 可看是否能 fuse 到 attention 主 kernel或减少单独 launch。

---

## 10. 可优化点 6：`lm_head` 与 greedy sampling 输出策略

### 10.1 证据

`lm_head` 是两个模型里最重的单个 FC：

| 模型 | stage | node | op-time |
|---|---|---|---:|
| Qwen3.5-2B | prefill | `lm_head/MatMul.0` | 5.108 ms |
| Qwen3.5-2B | decode 32 steps | `lm_head/MatMul.0` | 161.444 ms |
| Qwen3-1.7B | prefill | `lm_head/MatMul.0` | 3.086 ms |
| Qwen3-1.7B | decode 32 steps | `lm_head/MatMul.0` | 97.443 ms |

GenAI greedy sampling 当前会从 logits tensor 获取完整 vocab 行，然后 CPU 侧扫描最大值：

```cpp
Logits Sampler::_get_logit_vector(ov::Tensor logits, size_t batch_idx, size_t token_idx) {
    ...
    float* logits_data = logits.data<float>() + batch_offset + sequence_offset;
    return Logits{logits_data, vocab_size};
}

Token Sampler::_greedy_sample(const Logits& logits, size_t top_logprobs) const {
    for (size_t i = 0; i < logits.m_size; ++i) {
        ... // scan all vocab
    }
}
```

### 10.2 能优化什么，不能优化什么

不能轻易优化掉 `lm_head` GEMM 本体：精确 greedy argmax 需要知道全 vocab logits，除非接受近似 shortlist 或改变模型结构。

可以优化的是 **e2e host overhead**：

- 当 `do_sample=false` 且 `logprobs==0` 且没有需要全 logits 的 processor 时，在 graph 里接一个 GPU `TopK(k=1)` / `ArgMax`。
- pipeline 只从 GPU 拿 token id，而不是完整 vocab logits。
- 下一步 input token 直接用这个 id。

这个方向主要降低 e2e-TTFT/TPOT 中 logits copy + CPU scan 的开销，对 `first_token_infer_mean_ms` 中的 GPU GEMM 不会有大影响。

### 10.3 风险

- 采样配置复杂：logprobs、grammar、penalty、top_k/top_p、beam search 都可能需要 logits。
- 只能先做 greedy-only fast path。
- 需要保证与当前 CPU greedy 完全一致。

---

## 11. 不建议优先做的方向

### 11.1 继续优化已消失的 Reorder

Qwen3.5 的大块 conv 前后 Reorder 已经被 opt-1 消掉。当前 VTune prefill-only 的 `reorder_data` 只有 0.18%，128-token 也只有 0.12%。不值得作为主线。

### 11.2 直接重写 `gemm_kernel` / FC compressed

Qwen3/Qwen3.5 的最大绝对项都是 `gemm_kernel`，但这是 INT4 compressed FC 的 XMX 路径。除非有 oneDNN/driver 级问题，否则本地小改很难赢过现有 kernel。当前更现实的是减少 GEMM 周边的搬运、DQ、host copy，而不是重写 GEMM。

### 11.3 把 `--profile` decode DQ 当作 TPOT 最大优化点

真实 VTune pipeline 已显示 decode DQ kernel 基本被 skip。优化 DQ 应主要看 prefill；decode DQ 不能按 `--profile` 里的 16%-24% 直接估算收益。

---

## 12. 建议实施顺序（2026-06-29 复核后修订）

> 上一版的 Phase 1/2/3（GDN layout、conv/state fused op、DQ 系统化）已据本轮实测重排：Phase「GDN layout 消 Transpose_1」作废（归因错误，§5）；DQ 系统化降级为「整洁度而非性能」（§8）。当前只有一个值得立项的高风险大项，外加两个 dense 小项。

### Phase 1（唯一性能立项，高风险）：Qwen3.5 conv-sandwich fused op

目标：把 `Transpose(pre) + Concat(state) + GroupConvolution + Slice_2 + Transpose_1` 整段融成一个**有状态、layout-aware** 的 internal op，从根上消掉那 3.57 ms 的 `Transpose_1` + 1.65 ms 的 `Slice_2` + concat。

改动范围：

- 新增 internal op（如 `CausalDepthwiseConv1DState`）+ 其 shape/layout 契约。
- transformation pass：匹配 `transpose→concat→groupconv→slice→swish→transpose` 模式并替换。
- `graph/impls/ocl_v2/` 新 kernel：按逻辑索引从 state 和 x 读 conv 窗口、同 kernel 写 conv 输出和 new state，不 materialize concat/transpose 中间 buffer。

成功判据：

- `--profile` 中 `linear_attn/aten::transpose/Transpose_1`、`aten::slice/Slice_2`、`aten::cat/Concat` 18 行显著下降或消失。
- VTune prefill-only 中 `permute_ref`、`strided_slice_ref`、`concatenation_gpu_simple_ref` 合计下降。
- **正确性是最大风险**：必须 Qwen3.5 0.8B/2B/4B 的 `-ic 1`（prefill）**和** `-ic 128`（decode）md5 都与当前态逐字节一致。opt-3 的踩坑（动 SSM 状态路径破了 decode md5）说明这里 decode 极易出错，需双重校验。

> 这是一个需要单独立项、反复验证的工作，不是一次 ~50 行的小改。在它落地前，当前 3-opt 态就是本机 md5-无损约束下的 prefill 最优。

### Phase 2（dense 小项，e2e）：greedy-only GPU ArgMax 输出 token id

见 §10。只降 e2e 的 logits copy + CPU scan，不动 GEMM 本体；需保证与 CPU greedy 逐 token 一致。

### Phase 3（dense 小项，需先修工具）：attention backend A/B

见 §9。`-lc '{"ATTENTION_BACKEND":"SDPA"}'` 当前报 `stod` 错，需先确认 GenAI/GPU plugin 接受的传参格式，才能做 PA vs SDPA 的 prefill A/B。

---

## 13. 回归 checklist

每个 POC 都建议跑以下最小闭环：

```bash
cd /home/user/chaofan/model_test/benchmark/openvino
bash rebuild_gpu_plugin.sh

cd /home/user/chaofan/model_test/model_bench
source /home/user/envs/model_test_env/bin/activate
source ../benchmark/openvino/install/setupvars.sh >/dev/null 2>&1
export LD_LIBRARY_PATH="/home/user/chaofan/model_test/benchmark/openvino/install/runtime/lib/intel64:${LD_LIBRARY_PATH:-}"

# Qwen3.5 prefill correctness + TTFT
./build-local/model_bench vlm \
  -m ../benchmark/models/Qwen3.5-2B-INT4 -d GPU -n 8 -ic 1 \
  -pf ./prompts/llama-2-7b-chat_l.jsonl --no-monitor \
  -r /tmp/q35_ic1.csv -rj /tmp/q35_ic1.json

# Qwen3.5 decode correctness
./build-local/model_bench vlm \
  -m ../benchmark/models/Qwen3.5-2B-INT4 -d GPU -n 8 -ic 128 \
  -pf ./prompts/llama-2-7b-chat_l.jsonl --no-monitor \
  -r /tmp/q35_ic128.csv -rj /tmp/q35_ic128.json

# Qwen3 dense 对照
./build-local/model_bench llm \
  -m ../benchmark/models/Qwen3-1.7B-INT4 -d GPU -n 8 -ic 128 \
  -pf ./prompts/llama-2-7b-chat_l.jsonl --no-monitor \
  -r /tmp/q3_ic128.csv -rj /tmp/q3_ic128.json
```

每次必须检查：

- `first_token_infer_mean_ms`：纯推理 TTFT。
- `ttft_mean_ms`：e2e TTFT。
- `iterations[].result_md5`：输出 md5。Qwen3.5 permutation OFF 应单一 md5 一致；Qwen3 permutation ON 时比较 md5 多重集。
- `--profile-csv` 中目标 node/kernel 是否真的减少。
- VTune 中对应 computing-task 是否下降。

---

## 14. 本轮产物

| 类型 | 路径 |
|---|---|
| Qwen3.5 code profile CSV | `model_bench/results/hotspot_20260629_q35_2b_gpu.csv` |
| Qwen3 code profile CSV | `model_bench/results/hotspot_20260629_q3_1_7b_gpu.csv` |
| Qwen3.5 profile JSON | `/tmp/hotspot_q35_2b.json` |
| Qwen3 profile JSON | `/tmp/hotspot_q3_1_7b.json` |
| Qwen3.5 VTune 128-token | `/tmp/vtune_hotspot_20260629_q35_2b_opt` |
| Qwen3 VTune 128-token | `/tmp/vtune_hotspot_20260629_q3_1_7b_opt` |
| Qwen3.5 VTune prefill-only | `/tmp/vtune_hotspot_20260629_q35_2b_prefill` |
| Qwen3 VTune prefill-only | `/tmp/vtune_hotspot_20260629_q3_1_7b_prefill` |
| VTune exported CSV | `/tmp/vtune_hotspot_20260629_*_tasks.csv` |

---

## 15. 一句话总结（2026-06-29 复核版）

本轮没有新增已落地的优化：在 md5 逐字节无损这条硬约束下，**当前 3-opt 态已是本机可安全落地的 prefill 最优**。复核更正了上一版的关键错误归因（最大的 `Transpose_1` 是 conv 路径的转回 transpose，不是 GDN 输出 transpose），并用实测排除了三条看似可行的方向：GDN `V_BLOCK_SIZE` 调参（VBS=4 已最优，VBS=8 的加速是 float4 漏算的假象）、DQ 进一步去重（opt-3 已到 floor）、transpose 的 kernel 级修法（permute_ref 已调过、tiled kernel 模式不匹配）。**唯一仍有可观收益的是把 Qwen3.5 的 conv-sandwich（transpose+concat+conv+slice）融成一个有状态的 layout-aware fused op**——这能消掉 ~3.57 ms 的 `Transpose_1` + ~1.65 ms 的 `Slice_2`，但属高风险、需单独立项并同时严格校验 prefill 与 decode 的 md5（opt-3 已证明动 SSM 状态路径极易破 decode 数值）。Qwen3 dense 已接近 GEMM/attention 本体，只剩 greedy-only 输出 token id（降 e2e）这类小项。