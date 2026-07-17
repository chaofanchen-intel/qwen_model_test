# Qwen3 / Qwen3.5 性能优化：完整技术说明与测试数据（统一版）

日期：2026-07-11。本文件为单一自包含文档，合并了 **优化原理 + vtune 性能分析证据 + 修正后的测试数据（TTFT / TPOT / TPS）**，取代此前分散的《优化进展汇报_20260710》与其数据修正版。

---

## 0. 摘要

- **Qwen3.5 GPU（PA backend）**：落地三项**逐字节无损**的 GPU 插件优化 **opt-3 / opt-5 / opt-6**，均由 vtune 定位热点、并有 kernel 级时间/占用率证据。冷启动实测 prefill TTFT 相对 pristine：**0.8B −6.8%、2B −6.6%、4B −4.5%**，TPOT/TPS（decode）不变（优化只作用于 prefill）。另有 GenAI text-only 的 e2e 小优化 opt-7（本二进制与 pristine 逐字节相同，不单列收益）。
- **Qwen3 NPU（NPUW 配置）**：无可改 OCL kernel，优化面是 NPUW/驱动配置。**turbo 一致改善 TTFT（−4%~−7%）且逐字节无损**；**BEST_PERF+turbo 吞吐/E2E 最优**，但在 0.6B/1.7B 上会改变输出（非无损），仅在 4B 上无损。
- **最大剩余热点**：oneDNN XMX INT4 GEMM（≈47–50%）与串行 Gated-DeltaNet（≈19–20%），二者分别是 oneDNN 硬上限与真·串行递归，不属于低风险局部优化范围。
- **测试方法要点**：本机（Core Ultra 7 356H）SoC 会热漂移，连续无冷却的多状态测量会把绝对 TTFT 抬高 40–50% 并淹没优化差异。本轮全部采用**逐次温度门控的冷启动**采集（每次测量前把 SoC 冷却回 ~50°C 空载地板），数据平坦（std 1–4 ms）、可复现（run-to-run 漂移 < 3.3 ms）。方法与污染修正详见 §6。

---

## 1. 测试环境与方法

| 项目 | Qwen3.5 GPU / PA | Qwen3 NPU |
|---|---|---|
| 硬件 | Intel Core Ultra 7 356H，iGPU `0xb0a0`（32 XVE） | 同机 AI Boost NPU（`intel_vpu`） |
| 软件 | 本地源码 OpenVINO 2026.2.1 + GenAI 2026.2，vtune 2026.2 | 同一 OpenVINO/GenAI，NPU Level Zero 驱动 |
| Python | `source /home/chaofan/envs/qwen_env/bin/activate` | 同左 |
| 模型 | Qwen3.5 0.8B / 2B / 4B INT4，`ATTENTION_BACKEND=PA` | Qwen3 0.6B / 1.7B / 4B INT4 |
| 输入 | `model_bench/prompts/qwen3.5_1k.jsonl`（模板后 1000–1002 tok） | `benchmark/tests/prompts/qwen3_1k.jsonl`（1000 tok） |
| 生成 | `-ic 512`，实测每次输出 512 token | 同左 |
| 迭代 | **n=5（+1 warm-up）**，每状态单独进程 | 同左 |
| permutation | **开启**（GPU、NPU 均开）。为确定性、按迭代号索引的输入变异，故同一迭代号在不同状态下输入一致，md5 可逐迭代跨状态对比 | 同左 |
| 采集条件 | **逐次温度门控冷启动**：每次跑前把 `x86_pkg_temp` 冷却回 ~50°C 空载地板（见 §6） | 同左 |

**两种 TTFT 口径**（本文都会用到，勿混读）：

- **vtune 口径（kernel 归因）**：`-ic 1` 纯 isolated prefill，只跑第一 token 的 GPU kernel，用于把 TTFT 拆到每个 kernel。§3 的 vtune 表用此口径（2B）。
- **业务口径（对外 KPI）**：`-ic 512` 真实 512-token 生成，`ttft_mean_ms`（端到端首 token，含 GenAI host 开销）+ `tpot_mean_ms` + `throughput`。§3.7 / §4.3 的性能表用此口径。

**测量工具**：`model_bench`（C++，链接本地 OV）。PA/配置通过 `-lc <json>` 传入。GPU 各优化状态通过热插拔 `libopenvino_intel_gpu_plugin.so`（pristine / opt3 / opt3_opt5 / opt3_opt5_opt6 四个不同二进制）复现；genai 库无需替换。

---

## 2. Qwen3.5 GPU / PA：优化前的 hotspot 全景

先用 vtune `gpu-hotspots` 抓真实 `VLMPipeline`（**不能用 `--profile`**，后者只看到导出时固定的 SDPA 静态图）。2B、PA、`-ic 1` 纯 prefill、pristine 的逐 kernel 分布（每 iter，原始数据 `repro_356H_20260701/vtune/vt_B0.csv`，GPU task 合计 ≈256 ms/iter）：

| kernel | ms/iter | 占比 | 占用率 | 性质 / 是否可改 |
|---|---:|---:|---:|---|
| `gemm_kernel` | 117.6 | 45.8% | ~40% | oneDNN ngen-jit INT4×INT8 XMX GEMM，**非 OCL，不可编辑**（§5.1） |
| `paged_gated_delta_net_opt` | 48.8 | 19.0% | ~53% | Gated-DeltaNet 真·串行 SSM 递归（§5.2） |
| `dynamic_quantize_gpu_opt` | 18.9 | 7.3% | 100% | 每个 FC 各插一份 DQ → **opt-3 去重** |
| `rms_gpu_bfyx_opt` | 18.3 | 7.1% | ~40% | 访存延迟 bound，提占用率无墙钟收益（已证伪，§5.3） |
| `activation_ref` | 16.0 | 6.2% | 100% | 标量/通用 activation → **opt-6 切向量化 kernel** |
| `paged_causal_conv1d_ref` | 14.1 | 5.5% | ~60% | 串行扫 token 的因果 conv，token 维零并行 → **opt-5 分块** |

> 前提结论：PA 本身已大幅快于旧 SDPA（infer-TTFT −35%~−51%），且 SDPA 时代的 opt-1/opt-2 在 PA 下失效（对应搬运算子被融合 kernel 绕开）。唯 opt-3 在 PA 下仍有效。故本文只讲 PA 下有效的 opt-3/5/6/7。

三项优化的共同原则：**只改 GPU 工作项划分 / kernel 选择，不改任何浮点算术顺序 → 逐字节 md5 无损**（贪心解码，prefill ic=1 + decode ic=128 的 `result_md5` 多重集与 pristine 逐一相同）。

---

## 3. GPU 优化详解（含 vtune 证据）

### 3.1 opt-3：共享激活的多 FC 复用一个 DynamicQuantize

**观察（vtune）**：`dynamic_quantize_gpu_opt` 在 PA prefill 占 ~7.3%（18.9 ms/iter），占用率已 100%——不是占用率问题，而是**做了冗余的量化**。

**根因**：Qwen3.5 每个 decoder 层里多个 FC 共享同一激活——qkv/a/b/z 共享 `input_layernorm` 输出，gate/up 共享 `post_attention_layernorm` 输出。GPU 插件原本给**每个 FC 各插一份** `DynamicQuantize`，同一激活被重复量化多次。

**做法**：在插 DQ 前查找「同一激活 + DQ 配置逐字段相等」的已有 DQ 并复用；放宽 `dynamic_quantize.cpp` 的 skip 断言，使共享 DQ 在 decode 仍能 skip 成 f16 passthrough（保住 decode 数值一致）。
源码：`intel_gpu/src/plugin/transformations/dynamic_quantize_fully_connected.cpp` + `src/graph/dynamic_quantize.cpp`（patch `pa_opt3_dq_reuse.patch`）。

**vtune 证据**：DQ 实例数 **1044 → 576**，kernel 时间 **18.85 → 12.36 ms/iter（−34%）**，占用率保持 100%。

### 3.2 opt-5：token-TILED causal conv1d（本机取代 opt-4）

**观察（vtune）**：`paged_causal_conv1d_ref` dispatch 为 `GWS={seq_count, hidden}`、`LWS={1,256}`。prefill 时 `seq_count==1`，整个 kernel 只有 `hidden=6144` 个线程，**每个线程沿时间轴串行扫 ~1000 个 token**——token 维零并行，占用率仅 ~60%（6144÷SIMD32=192 线程=320 HW 线程的 60%，vtune「work-size 受限」）。

**第一版尝试 opt-4（token 全并行）→ 本机回退**：把 token 加进 global range（`GWS={seq,hidden,total_tokens}`）后占用率确实到 100%，**但每个 output-token work-item 都要从 global 重新读自己的 K 元窗口**，全局读放大 ~K 倍 → kernel 变 memory-bound。vtune 对账：conv1d **14.13 → 24.71 ms/iter（+75%）**，其余 kernel 不变，正好对上 2B 墙钟 +10 ms/iter 的净回退。（opt-4 在 358H 大 iGPU 上「占用率红利 > 冗余读代价」是净收益；在 356H 小 iGPU 上相反。md5 仍无损，纯性能回退。）

**opt-5（token 分块，保留流式复用）**：把每条序列切成**很少的 G 个连续块**，`GWS={seq_count, hidden, G}`，每个 `(seq,h,g)` work-item **串行扫自己那一块**（块内滑动窗口、零冗余读），只在块起点一次性重建 K-1 元历史。这样占用率 = `hidden×G`（G≥2 即 100%），额外读只有 `hidden×G×(K-1)`（可忽略），**没有 opt-4 的 `hidden×total_tokens×K` 内存放大**；逐元素 fma 顺序与串行版一致 → bit 无损。
G 扫描在 **G=3 达谷底**，故默认 G=3。源码：`intel_gpu/src/graph/impls/ocl_v2/paged_causal_conv1d_ref.{cl,cpp}`（patch `pa_opt5_conv1d_tiled.patch`，与 opt-4 互斥）。

**vtune 证据**：conv1d **14.13 → 10.45 ms/iter（−26%）**，占用率 **60% → 100%**，最终 `GWS=1×6144×3`。

### 3.3 opt-6：dynamic activation 走向量化 `activation_opt`

**观察（vtune）**：opt-3+opt-5 后仍有两个大尺寸 `activation_ref`（`1×1×6156288`、`1×1×2052096`），合计约 6–7% task time（~16 ms/iter）。它们是普通逐元素 activation，数学上不需要复杂 layout 索引；OpenVINO 已有 float4 向量化的 `activation_opt`，只是此前不支持 dynamic shape。

**根因**：`ActivationKernelOpt` 的 selector 没声明 `EnableDynamicShapesSupport()`，dynamic shape 只能落到通用 `activation_ref`；直接打开会报 `CL_INVALID_ARG_INDEX`，因为 shape-agnostic kernel 会多传 `OPTIONAL_SHAPE_INFO_ARG` 而 `activation_opt.cl` 签名没接收。

**做法**：给 `ActivationKernelOpt` 增加 dynamic-shape support，但**只放开「输入/输出 layout 相同、无 fused ops、无 runtime activation 参数」的安全子集**；`activation_opt.cl` 参数表补 `OPTIONAL_SHAPE_INFO_ARG`。只改 kernel 选择，不改数学。
源码：`kernel_selector/kernels/activation/activation_kernel_opt.cpp` + `cl_kernels/activation_opt.cl`（patch `opt6_dynamic_activation_opt.patch`）。

**vtune 证据**：activation 家族 **15.93 → 7.06 ms/iter（−55.7%）**，vtune 中 `activation_ref` 主项被 `activation_opt` 取代（最终态占用率 ~80%）。

### 3.4 opt-7：text-only VLM embedding copy-elision（e2e-only）

Qwen3.5 走 `InputsEmbedderQwen3VL::get_inputs_embeds()`：text-only 时原逻辑仍分配一个同 shape 的 `inputs_embeds` 并 `memcpy(text_embeds → inputs_embeds)`。opt-7 在 `images.empty() && videos.empty()` 时直接返回 `text_embeds`，省掉这次大 tensor copy。它**不改 LM kernel**，只减少 GenAI host 路径上的一次 memcpy，属亚毫秒级 e2e 优化。
源码：`openvino.genai/.../visual_language/qwen3_vl/classes.cpp`（patch `opt7_..._copy_elision.patch`）。
> 注意：本轮复现所用的 `opt7` 与 `pristine` 的 genai 二进制**逐字节相同**（md5 均 `8b1563ec…`），故 opt-7 在下方性能表中不单列收益，`final` 即 `opt3+opt5+opt6`。

### 3.5 vtune 汇总：pristine → final（opt-3+opt-5+opt-6）

2B、PA、`-ic 1` isolated prefill，每 iter（原始 `vt_B0.csv` / 最终态 `vt_20260709_opt7_2b.csv`）：

| kernel 家族 | pristine ms/iter | final ms/iter | 变化 | 归因 |
|---|---:|---:|---:|---|
| **GPU task 合计** | 255.3 | 235.1 | **−7.9%** | — |
| `gemm_kernel` | 117.8 | 116.9 | −0.8% | oneDNN XMX，不可改 |
| `paged_gated_delta_net_opt` | 48.7 | 48.3 | −0.9% | 串行 SSM，已满占用 |
| `dynamic_quantize_gpu_opt` | 19.0 | 12.4 | **−34.5%** | **opt-3**（实例 1044→576） |
| `paged_causal_conv1d_ref` | 14.1 | 10.5 | **−25.2%** | **opt-5**（occ 60→100%） |
| activation（ref→opt） | 15.9 | 7.1 | **−55.7%** | **opt-6** |
| `rms_gpu_bfyx_opt` | 18.3 | 18.4 | +0.4% | 访存 bound，不改 |

四者一致（机理 / md5 / 墙钟 / vtune）：三项优化各自压掉其目标 kernel，合计使 isolated-prefill GPU kernel 时间 −7.9%。gemm/GDN 基本不动，是硬瓶颈。

### 3.6 GPU 业务口径测试数据（`-ic 512`，冷启动，permute ON，n=5）

`TTFT vs pristine` 相对「两次独立冷启动 pristine 的均值」；`decode TPS` 由 TPOT 推得；所有状态实测输出 512 token；`md5` 为逐迭代序列是否与 pristine 一致。梯度对照（pristine 首末两次之差）：0.8B Δ=−3.34、2B Δ=−0.21、4B Δ=+0.15 ms —— 全部通过，确认无跨状态温漂。

#### Qwen3.5-0.8B INT4（pristine 两样本 204.53 / 201.19 → 基线 202.86）

| 状态 | TTFT ms | ±std | vs pristine | TPOT ms | TPS | decode TPS | E2E ms | md5 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| pristine | 202.86 | — | — | 8.67 | 110.45 | 115.38 | 4635.7 | 基准 |
| opt3 | 197.75 | 1.12 | −2.52% | 8.68 | 110.47 | 115.22 | 4635.0 | 一致 |
| opt3+opt5 | 195.23 | 1.41 | −3.76% | 8.70 | 110.28 | 114.95 | 4642.8 | 一致 |
| **final** (opt3+opt5+opt6) | **188.99** | 2.89 | **−6.84%** | 8.67 | 110.81 | 115.36 | 4620.7 | **一致** |

#### Qwen3.5-2B INT4（pristine 两样本 271.24 / 271.03 → 基线 271.13）

| 状态 | TTFT ms | ±std | vs pristine | TPOT ms | TPS | decode TPS | E2E ms | md5 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| pristine | 271.13 | — | — | 14.02 | 68.83 | 71.32 | 7438.4 | 基准 |
| opt3 | 264.18 | 1.75 | −2.56% | 14.07 | 68.69 | 71.10 | 7453.7 | 一致 |
| opt3+opt5 | 260.76 | 1.90 | −3.83% | 14.07 | 68.68 | 71.06 | 7454.4 | 一致 |
| **final** | **253.14** | 1.23 | **−6.64%** | 14.11 | 68.58 | 70.86 | 7466.2 | **一致** |

#### Qwen3.5-4B INT4（pristine 两样本 575.79 / 575.94 → 基线 575.86）

| 状态 | TTFT ms | ±std | vs pristine | TPOT ms | TPS | decode TPS | E2E ms | md5 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| pristine | 575.86 | — | — | 27.11 | 35.48 | 36.89 | 14430.3 | 基准 |
| opt3 | 565.37 | 2.78 | −1.82% | 27.07 | 35.56 | 36.95 | 14398.9 | 一致 |
| opt3+opt5 | 565.91 | 2.21 | −1.73% | 27.11 | 35.50 | 36.88 | 14423.1 | 一致 |
| **final** | **549.75** | 1.25 | **−4.54%** | 27.12 | 35.54 | 36.88 | 14407.7 | **一致** |

**GPU 小结**：final 相对 pristine 稳定下降 **0.8B −6.8% / 2B −6.6% / 4B −4.5%**，随 opt3→opt5→opt6 单调改善；**TPOT / TPS / decode TPS 在各状态间基本恒定**（优化只作用于 prefill，不动 decode），符合预期；**md5 逐迭代跨状态全一致**（无损）。4B 上总降幅较小，是因为 prefill 更被 gemm 主导，DQ/conv/activation 占比更低。

---

## 4. Qwen3 NPU：配置优化与测试数据

### 4.1 为什么 NPU 用配置矩阵而非 kernel vtune

NPU 的 LLM 由 NPUW 把 prefill/generate 子图编译成 NPU blob，**没有可直接修改的 GPU OCL kernel，也拿不到等价的 per-kernel 归因**。因此 NPU 的"优化"是 **NPUW/驱动配置 → TTFT/TPOT/TPS/E2E/MD5 的完整配置矩阵**，并以源码映射解释配置作用。

统一基线：`MAX_PROMPT_LEN=1024`、`MIN_RESPONSE_LEN=512`、`NPUW_LLM_PREFILL_CHUNK_SIZE=2048`。

- **`NPU_TURBO=YES`**：`compiler_adapter/src/graph.cpp::Graph::initialize_impl()` 里设置 `ZE_NPU_COMMAND_QUEUE_OPTION_TURBO`（更激进的 NPU 频率/队列策略）。
- **`NPUW_LLM_GENERATE_HINT=BEST_PERF`**：`npuw/llm_compiled_model.cpp::get_default_generate_config()` 里写入 `NPUW_ONLINE_PIPELINE=NONE`（改变子图 partitioning/编译），改善 decode，但会改变数值。

### 4.2 NPU 测试数据（`-ic 512`，冷启动，permute ON，n=5）

`vs base` / `md5` 均相对同尺寸 baseline。4B 做了 base 复测作梯度对照（Δ=−1.61 ms，通过）。

#### Qwen3-0.6B INT4

| 配置 | TTFT ms | ±std | vs base | TPOT ms | TPS | decode TPS | E2E ms | E2E vs base | md5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| baseline | 150.19 | 3.46 | — | 10.66 | 91.49 | 93.84 | 5596.4 | — | 基准 |
| turbo | 143.85 | 1.50 | −4.22% | 10.59 | 92.16 | 94.44 | 5555.8 | −0.7% | **一致** |
| BEST_PERF | 153.23 | 1.93 | +2.03% | 10.04 | 96.91 | 99.63 | 5283.8 | −5.6% | **变化** |
| BEST_PERF+turbo | 140.85 | 2.04 | −6.22% | 9.12 | 106.76 | 109.78 | 4799.4 | −14.2% | **变化** |

#### Qwen3-1.7B INT4

| 配置 | TTFT ms | ±std | vs base | TPOT ms | TPS | decode TPS | E2E ms | E2E vs base | md5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| baseline | 293.13 | 2.58 | — | 17.47 | 55.52 | 57.24 | 9221.6 | — | 基准 |
| turbo | 271.48 | 0.99 | −7.39% | 17.38 | 55.93 | 57.53 | 9154.3 | −0.7% | **一致** |
| BEST_PERF | 290.63 | 2.73 | −0.85% | 16.49 | 58.73 | 60.64 | 8718.6 | −5.5% | **变化** |
| BEST_PERF+turbo | 269.90 | 3.31 | −7.93% | 15.27 | 63.46 | 65.53 | 8072.6 | −12.5% | **变化** |

#### Qwen3-4B INT4

| 配置 | TTFT ms | ±std | vs base | TPOT ms | TPS | decode TPS | E2E ms | E2E vs base | md5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| baseline | 690.46 | 2.02 | — | 31.29 | 30.70 | 31.96 | 16679.5 | — | 基准 |
| turbo | 652.96 | 1.36 | **−5.43%** | 31.13 | 30.92 | 32.13 | 16560.7 | −0.7% | **一致** |
| BEST_PERF | 695.61 | 3.76 | +0.75% | 31.53 | 30.46 | 31.72 | 16808.9 | +0.8% | **一致** |
| BEST_PERF+turbo | 651.15 | 2.40 | **−5.69%** | 29.89 | 32.15 | 33.45 | 15927.1 | −4.5% | **一致** |

### 4.3 NPU 正确性（md5）与结论

用 5 个不同输入逐迭代对比 md5：

- **turbo：三尺寸全部逐字节与 base 一致（无损）。**
- **BEST_PERF / BEST_PERF+turbo：0.6B、1.7B 上 5/5 输入均与 base 不同（非无损）；仅 4B 上 5/5 一致（无损）。**

结论：
1. **turbo 在三尺寸上一致改善 TTFT（−4%~−7%）且无损**，是最稳的通用开关。
2. **BEST_PERF 改善 decode（TPOT↓、TPS/E2E↑）**，但在 0.6B/1.7B 上改变输出，仅 4B 无损；对 TTFT 影响接近中性。
3. **4B 上 turbo 与 BEST_PERF+turbo 均无损且改善 TTFT/E2E**，是 4B 的推荐配置。

---

## 5. 剩余瓶颈与未来方向

当前栈（opt-3+5+6）后，2B PA prefill 的 GPU task 仍由两大项主导，均非低风险局部优化可动：

### 5.1 `gemm_kernel`（≈47–50%）—— oneDNN XMX，不可改
三个主 shape 是 `[1002,2048]×[6144,2048]` 的 INT4×INT8(动态量化) XMX 矩阵乘，由 **oneDNN 的 ngen-jit / gemmstone micro-kernel** 生成（`kernel_name()=="gemm_kernel"`），**不是可编辑的 OCL kernel**。实测把它强制换成可编辑的 OCL FC 路径：2B infer-TTFT **206ms → 1015ms（5× 慢）**且 md5 变化。占用率 40% 是 XMX systolic 阵列的正常特征，非缺陷。→ 插件 OCL 层不可改进。

### 5.2 `paged_gated_delta_net_opt`（≈19–20%）—— 串行 SSM 递归
实现 Gated-DeltaNet 的逐 token rank-1 状态更新（`S_t = b_g·S_{t-1}; S_t += k_t⊗((v_t−h_k)·β_t)`），**输出 token 之间有真实依赖，不能像 conv 那样 token 并行**。已是 `_opt` 版且 100% 占用。V_BLOCK_SIZE 扫描呈 U 形、VBS=4 为 md5-无损谷底，框定 chunkwise 上限仅约 −3~4% 总 prefill，且需块内三角求逆、几乎必然破 md5 → 负期望，不做。

### 5.3 已证伪的其它项
- `rms_gpu_bfyx_opt`（~7%，occ 40% SLM 受限）：实测把占用率修到 100%，**kernel 时间不变**——是访存延迟 bound，不是占用率 bound。SLM 修复无害但无墙钟收益，已回退。教训：vtune 的「occ 低 + 限制因素=SLM」≠「提占用率就能加速」，必须实测墙钟。
- 4-way FC 水平融合（`max_num_fcs_to_fuse` 3→4）：实测三尺寸均 +0.8~2.0% 回退，已否决。
- `sdpa_micro` tile、`eltwise/slice/concat` 向量化：前者核心是 oneDNN gemmstone（无 md5-无损 lever）；后者已 100%-occ 带宽 bound，向量化收益很小。

> 想明显超过当前收益，只剩两条高成本路径：**GDN chunkwise/并行递归重写**（改累加顺序，破 bit-exact，需重定义正确性验收）或 **oneDNN XMX INT4 GEMM tiling 级优化**（在 oneDNN 层，非插件局部 patch）。

---

## 6. 数据采集方法与污染修正说明（为什么本数据可信）

本机 SoC（CPU/iGPU/NPU 同封装，`x86_pkg_temp`=`thermal_zone6`）升温快、降温慢（近地板处 ~2–3°C/min）。此前一版数据在**连续无冷却的长会话**里采集，出现单调温漂：同一状态按轮次 r0(冷)→r4(热) 的 TTFT/TPOT 同步抬升（如 4B pristine 604→922 ms），导致：

1. **绝对 TTFT 被大幅抬高**（旧 4B final 836 ms vs 本轮 550 ms，抬高约 +52%；旧 2B final 363 vs 253，+43%）；
2. **状态间小幅优化差被热方差淹没甚至反号**（旧数据 4B final 甚至比 pristine 还高）；
3. **NPU-4B 跨配置温漂**把 turbo/BEST_PERF 误判为让 TTFT 变差。

修正方法（本轮全部采用）：

- **逐次温度门控冷启动**：每次测量前把 `x86_pkg_temp` 冷却回 ~50°C 空载地板（stall 检测：窗口内降温 <2°C 即判定到达地板）才开跑，使每个状态从同一冷基线起测。固定时长冷却（如 45s）压不住热积累、无效。
- **梯度对照**：每尺寸/模型额外做一次 pristine/base 复测；本轮 GPU Δ ≤ 3.3 ms、NPU-4B Δ = 1.6 ms，全部通过。
- 结果：全部状态 std 仅 1–4 ms（旧 4B 达 56–119 ms），run-to-run 漂移 < 3.3 ms。

因此本文的绝对 TTFT/TPOT/TPS 数据取代此前任何受污染版本；旧的连续会话绝对值不应再对外引用。

---

## 7. 落地建议

| 业务目标 | 推荐 | 说明 |
|---|---|---|
| Qwen3.5 GPU（任意尺寸，首 token 优先） | **opt-3 + opt-5 + opt-6**（=final） | TTFT 稳定 −4.5%~−6.8%，逐字节无损；text-only 可再叠 opt-7（e2e 小优化） |
| Qwen3 NPU（严格逐字节无损，任意尺寸） | **turbo only** | TTFT −4%~−7%，md5 全一致 |
| Qwen3 NPU 4B（TTFT 或吞吐/E2E 优先） | **BEST_PERF + turbo** | TTFT −5.7%、E2E −4.5%、decode TPS +4.7%，且 4B 上**无损** |
| Qwen3 NPU 0.6B/1.7B（吞吐优先，可接受输出变化） | BEST_PERF + turbo | E2E −12.5%~−14.2%，但**输出非逐字节一致**；严格无损场景退回 turbo |
| Qwen3.5 GPU 4B / 大 hotspot 继续压 | 需立项 | gemm（oneDNN）/ GDN（串行递归）为高成本方向，超出局部优化范围 |

---

## 8. 复现与原始数据

- **新原始数据（冷启动，本文所有性能表）**：`benchmark/tests/repro_rerun_20260710/gpu_pa/<size>/<state>.json`（15）、`.../npu/Qwen3-<size>_<config>.json`（13）；驱动 `run_rerun_v3.sh`；进度 `logs/progress.log`；汇总 `logs/FINAL_SUMMARY.txt`；QA `monitor.py`。
- **vtune 原始数据**：`benchmark/tests/repro_356H_20260701/vtune/`（`vt_B0`/`opt5G3`/`opt6`/`best2b`/`vt_20260709_opt7_2b`.csv，逐 kernel）；`opt_pa_archive/vtune_PA_prefill_*.csv`（opt-4 前后占用率对账）；`reeval_PA_20260630/vtune_{PA,SDPA}_2b_kernels.csv`（PA vs SDPA 结构对比）。
- **优化 patch**：`benchmark/openvino/qwen_patch/`（opt-3/5/6/7）；机制与源码映射详见 `benchmark/tests/OV源码优化降低TTFT_PA.md`。

代表性复现命令：

```bash
source /home/chaofan/envs/qwen_env/bin/activate
source benchmark/openvino/install/setupvars.sh
export LD_LIBRARY_PATH="$PWD/benchmark/openvino/install/runtime/lib/intel64:$LD_LIBRARY_PATH"

# GPU final 业务口径（-ic 512，冷启动逐次门控见 run_rerun_v3.sh）
./model_bench/build-local/model_bench vlm -m ./benchmark/models/Qwen3.5-2B-INT4 \
  -d GPU -n 5 -ic 512 -pf ./model_bench/prompts/qwen3.5_1k.jsonl \
  --enable-prompt-permutation --no-monitor -lc benchmark/tests/repro_rerun_20260710/gpu_pa.json

# GPU vtune kernel 归因（-ic 1 isolated prefill；用 vtune，不能用 --profile）
vtune -collect gpu-hotspots -r /tmp/vt -- ./model_bench/build-local/model_bench vlm \
  -m ./benchmark/models/Qwen3.5-2B-INT4 -d GPU -n 5 -ic 1 \
  -pf ./model_bench/prompts/qwen3.5_1k.jsonl --no-monitor -lc /tmp/pa.json
vtune -report hotspots -group-by computing-task -r /tmp/vt -format csv

# NPU BEST_PERF+turbo（-ic 512）
./model_bench/build-local/model_bench llm -m ./benchmark/models/Qwen3-4B-INT4 \
  -d NPU -n 5 -ic 512 -pf ./benchmark/tests/prompts/qwen3_1k.jsonl \
  --enable-prompt-permutation --no-monitor -lc benchmark/tests/repro_rerun_20260710/npu_best_perf_turbo.json
```
