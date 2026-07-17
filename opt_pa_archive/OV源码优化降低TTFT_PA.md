# 在 PA backend 下分析并优化 Qwen3.5 TTFT —— 复现文档

> **适用范围：`ATTENTION_BACKEND=PA`（PagedAttention，OpenVINO 现默认）。**
> 硬件：Intel Core Ultra X7 358H（Panther Lake）+ Arc B390 iGPU + AI Boost NPU，Ubuntu 24.04。
> 软件：本地源码构建的 OpenVINO 2026.2.1 + OpenVINO GenAI 2026.2（Release，HEAD `ede283a88e`），vtune 2026.2。
> Python 环境：`source /home/user/envs/qwen_env/bin/activate`（与 stock pip OV 链接；C++ 路径用 `model_bench` 链接本地 OV）。
> 测量工具：`model_bench`（C++，链接本地 OV）；PA 通过 `-lc <file>`（model_bench 只接受文件路径）传 `{"ATTENTION_BACKEND":"PA"}`。
> 撰写日期：2026-06-30。所有 TTFT/op-sum 数值均为 **PA 口径**，且在 **pristine HEAD（不含 SDPA 时代 opt-1/2/3）** 的干净基础上测得。
>
> 配套文档：`OV源码优化降低TTFT_SDPA.md`（SDPA 时代的 opt-1/2/3，已被 PA 取代）；其 patch 在 `opt_sdpa_archive/`。本文的 opt-4 patch 在 `opt_pa_archive/pa_opt4_conv1d.patch`。
>
> **【356H 补充数据，2026-07-01】** 以下正文表格的原始数据来自 358H；本次在 **Intel Core Ultra 7 356H**（同为 Panther Lake，本机 `python` 环境 `source /home/chaofan/envs/qwen_env/bin/activate`）上按相同方法（`model_bench` C++ + 本地源码 OV 同一 HEAD、GPU、n=8、真实 1000-tok prompt）补测。356H 的行/列均以「**[356H]**」标注，358H 原数据保留不动。原始测量见 `repro_356H_20260701/`（`SUMMARY_356H.md` + `results.tsv` / `pa_sweep*.tsv`）。
> **356H 结论差异（重要）：opt-3 在 356H 上依然是有效收益（PA infer-TTFT −2~3.2%），但 opt-4（token-parallel conv1d）在 356H 上是净回退（+4~7%，与 358H 的 −1.3~4.4% 收益相反），尽管 md5 仍逐字节无损。故 356H 上 PA 的最优实用配置是「仅 opt-3」，不启用 opt-4。** 详见各节 [356H] 表（opt-4 回退详情见 §3.4，opt-3 净收益见 §2.1，双口径汇总见 §4.1）。

---

## 0. 结论先行

1. **PA 本身就大幅快于 SDPA。** 同模型、stock OV、真实 1000-token prompt，GPU infer-TTFT：PA 比 SDPA 快 **34–51%**（0.8B −51%、2B −40%、4B −34%）。OpenVINO 已默认 PA，应优先用 PA。**[356H]** 356H 上这一结论完全成立且量级几乎一致：PA vs SDPA infer-TTFT 0.8B −50.5%、2B −38.6%、4B −35.6%（见 §1）。
2. **SDPA 时代的 opt-1/opt-2 在 PA 下完全失效**（PA 融合 SSM kernel 绕开了它们针对的 Reorder/Transpose 搬运，§2）。**但 opt-3（共享激活的 FC 复用 DynamicQuantize）在 PA 下仍然有效**——`dynamic_quantize_gpu_opt` 在 PA prefill 仍占 ~6.6%，去重照样省。
3. **PA 下落地两个有效优化，可叠加、均 md5 逐字节无损：**
   - **opt-4**：token-parallel `paged_causal_conv1d`（§3）——把串行扫 token 的 conv kernel 改成 token 并行，占用率 20%→100%。
   - **opt-3**：共享激活的多 FC 复用一个 DynamicQuantize（§4，源自 SDPA 时代但在 PA 下重新验证有效）。

   **叠加总效果（干净 pristine 基础，n=8，真实 1000-tok prompt）：**

   | 模型 | infer-TTFT 总降幅 | e2e-TTFT 总降幅 |
   |---|---:|---:|
   | Qwen3.5-0.8B-INT4 | **−6.7%** | −3.9% |
   | Qwen3.5-2B-INT4 | **−6.5%** | −4.4% |
   | Qwen3.5-4B-INT4 | **−3.9%** | −3.6% |

   （infer-TTFT = `first_token_infer_mean_ms` 纯推理 prefill；e2e-TTFT = `ttft_mean_ms` 端到端含 GenAI host 开销。e2e 降幅小是因为分母含 ~40ms 固定 host 开销。）

   **[356H]** 在 356H 上「opt-4+opt-3」叠加并**不是净收益**——因为 opt-4 本机回退，抵消了 opt-3 的收益（infer-TTFT：0.8B +4.2%、2B +1.0%、4B +1.8%）。**356H 上应只启用 opt-3**，其单独收益（PA，交错 A/B，n=8，ic=1）为：

   | 模型 | opt-3 单独 infer-TTFT | opt-3 单独 e2e-TTFT |
   |---|---:|---:|
   | Qwen3.5-0.8B-INT4 | **−2.1%** (150.8→148.5 ms) | −1.6% (196.3→193.9 ms) |
   | Qwen3.5-2B-INT4 | **−3.2%** (214.0→207.6 ms) | −2.4% (259.8→253.7 ms) |
   | Qwen3.5-4B-INT4 | **−2.1%** (506.1→495.9 ms) | −2.0% (552.2→542.4 ms) |
4. **次大项 `paged_gated_delta_net_opt`（prefill 19%）已分析为高风险、暂不动**——它是真·串行 SSM 递归，无法简单并行，需 chunkwise 重写（§6）。

---

## 1. PA vs SDPA：先量化差距

口径：官方 runner（stock pip OV），真实 1000-token prompt，GPU，`1st_infer_latency_avg`：

| 模型 | PA | SDPA | PA 相对 SDPA |
|---|---:|---:|---:|
| Qwen3.5-0.8B-INT4 | 70.5 ms | 143.8 ms | **−51.0%** |
| Qwen3.5-2B-INT4 | 99.7 ms | 165.8 ms | **−39.9%** |
| Qwen3.5-4B-INT4 | 229.6 ms | 350.3 ms | **−34.5%** |

> 参考：SDPA 文档里「SDPA + 全部三优化」的 2B 是 132.3ms——**单是 PA（99.7ms）就比它还快 33ms**。这就是「PA 取代了 SDPA 优化」的直接证据。

**[356H]** 356H 上同口径（`results/*.csv`，stock pip OV，官方 runner，n=10，ic=512，真实 1000-tok prompt，`1st_infer_latency_avg`）：

| 模型 | PA | SDPA | PA 相对 SDPA |
|---|---:|---:|---:|
| Qwen3.5-0.8B-INT4 | 151.3 ms | 305.8 ms | **−50.5%** |
| Qwen3.5-2B-INT4 | 215.5 ms | 351.0 ms | **−38.6%** |
| Qwen3.5-4B-INT4 | 510.0 ms | 791.7 ms | **−35.6%** |

> 356H 绝对值约为 358H 的 2× 慢（同为 Panther Lake 但 iGPU 规格更低），但 **PA 相对 SDPA 的加速比几乎一致**（−50.5/−38.6/−35.6% vs 358H −51/−39.9/−34.5%），佐证「PA 优先」在两台机器上都成立。e2e-TTFT 同口径：0.8B 193.9/348.3 (−44.3%)、2B 263.8/398.5 (−33.8%)、4B 563.7/843.6 (−33.2%)。

---

## 2. SDPA 时代的 opt-1/opt-2 在 PA 下失效（opt-3 例外，见 §4）

### 2.1 A/B 实测（早期把 opt-1+2+3 一起叠在 PA 上测）

| 模型 | PA 基线（无 opt） | PA +opt-1+2+3 | PA 下「三优化合计」降幅 | （对照）SDPA 下总降幅 |
|---|---:|---:|---:|---:|
| Qwen3.5-0.8B-INT4 | 70.17 ms | 69.00 ms | −1.7% | −20.8% |
| Qwen3.5-2B-INT4 | 99.01 ms | 96.78 ms | −2.3% | −16.1% |
| Qwen3.5-4B-INT4 | 225.55 ms | 221.44 ms | −1.8% | −12.6% |

**[356H]** 在 356H 上「PA 基线 vs PA+opt1+2+3」（交错热插拔 A/B，n=8，ic=1，2 reps 均值）。因 opt-1/opt-2 在 PA 下不触发，这里的降幅**本质就是 opt-3 单独贡献**：

| 模型 | PA 基线（pristine） | PA +opt-1+2+3 | PA 下降幅（=opt-3） |
|---|---:|---:|---:|
| Qwen3.5-0.8B-INT4 | 151.4 ms | 148.5 ms | **−2.1%** |
| Qwen3.5-2B-INT4 | 214.4 ms | 207.6 ms | **−3.2%** |
| Qwen3.5-4B-INT4 | 506.5 ms | 495.9 ms | **−2.1%** |

> 356H 与 358H 一致：opt-1/opt-2 归零、opt-3 单独有效（356H −2.1~3.2%，358H −2.3~3.4%）。md5 多重集逐一相同。

> ⚠️ 这个「合计 ~2%」一度误导我们以为三优化整体在 PA 下都没用。后来在干净基础上**逐项**拆开测（§4）才发现：opt-1/opt-2 确实归零，但 **opt-3 单独贡献 −2.3~3.4%**——被无效的 opt-1/2 在合并测量里平均稀释了。所以 §2.2 的「失效」只针对 opt-1/opt-2。

### 2.2 根因（vtune gpu-hotspots，kernel 级，2B，PA vs SDPA 各一次）

PA 走一套全新的融合 SSM/attention kernel，把 SDPA 路径里那些被 opt-1/2/3 针对的独立搬运算子直接融掉或绕开了：

| 优化 | SDPA 下针对的 kernel（占比） | PA 下的情况 | 是否还有收益 |
|---|---|---|---|
| **opt-1**（去 depthwise conv 的 Reorder） | `conv_reorder` 5.0% + `gen_conv`(GroupConv) 2.3% | PA 用融合的 `paged_causal_conv1d_ref`(6.6%)，**不插 reorder** | ❌ 无对象 |
| **opt-2**（修 permute_ref 的 LWS） | `permute_ref` 7.6%（SSM 的两个独立 transpose） | PA 把转置融进 `paged_gated_delta_net_opt`(10.4%)，**无独立 permute_ref** | ❌ 不触发 |
| **opt-3**（共享激活的 FC 复用 DQ） | `dynamic_quantize_gpu_opt` 2.2% | PA 下仍在(prefill ~6.6%)，FC 去重照样有效 | ✅ **PA 下有效**，见 §4 |

> 原始数据：`reeval_PA_20260630/vtune_{PA,SDPA}_2b_kernels.csv`。结论：opt-1/2/3 是 SDPA 路径专属的，在 PA 默认路径下应视为失效；代码已从工作树撤回 pristine（仅以 patch 形式存档于 `opt_sdpa_archive/`）。

### 2.3 方法论坑（重要）
`model_bench --profile`（deep-profile）直接 `compile_model` 已导出的静态 LM 子图 IR，而 `ATTENTION_BACKEND` 是 **GenAI pipeline 级**属性——给 deep-profile 传 `-lc {ATTENTION_BACKEND:PA}` 会报 `Option not found: ATTENTION_BACKEND`，且即便不报错它看到的也永远是**导出时固定的 SDPA 结构图**（profile 里出现 `IndirectSDPA` 即为证）。**要看 PA 的真实 kernel 结构，必须用 vtune 抓真实 VLMPipeline，不能用 `--profile`。**

---

## 3. opt-4：token-parallel paged_causal_conv1d（PA 时代首个有效优化）

### 3.1 找 hotspot（vtune，纯 prefill ic=1，2B，pristine）

```
gemm_kernel                 45.7%  occ 46.5%   <- INT4 矩阵乘，XMX 已最优
paged_gated_delta_net_opt   18.7%  occ 53%     <- SSM 核心（次大，已是 _opt）
paged_causal_conv1d_ref     11.5%  occ 20%     <- _ref + 占用率最低  → opt-4 目标
dynamic_quantize_gpu_opt     5.0%  occ 100%    <- 已打满
```

### 3.2 根因
`paged_causal_conv1d_ref` 的 dispatch 是 `GWS={seq_count, hidden_size}`、`LWS={1,256}`（`impls/ocl_v2/paged_causal_conv1d_ref.cpp`）。prefill 时 `seq_count==1`，于是整个 kernel 只有 `hidden_size=6144` 个线程，**每个线程沿时间轴串行扫 ~1000 个 token**（`.cl` 里 `for t in seq_tokens`）—— token 维零并行，占用率仅 20%。

### 3.3 改动（+74/−43 行，纯 dispatch + kernel 改写，不改数学）
这是 kernel_size=K 的因果 conv，`out[t]=bias+Σ_k w[k]·in_window[t][k]`，**输出 token 之间互相独立**。把 token 加进 global range → `GWS={seq_count, hidden_size, total_tokens}`；每个 `(seq,h,t)` work-item 直接从 `input_embeds` 重建自己的 K 元窗口（前 K-1 个 token 的历史部分从初始 `conv_state_table` 读），算出 `out[t]`，并仅在串行版同样写 state 的边界 token 上写回 KV-cache（不同 t 写不同 slot，天然无竞争）。每个输出元素的 fma 顺序与串行版逐位一致 → 数值无损。

文件：`src/plugins/intel_gpu/src/graph/impls/ocl_v2/paged_causal_conv1d_ref.{cl,cpp}`。patch：`opt_pa_archive/pa_opt4_conv1d.patch`。

### 3.4 效果（model_bench C++ + 本地 OV，PA，GPU，n=8、ic=1 纯 prefill、真实 1000-tok prompt，**干净 pristine 基础**）

| 模型 | opt-4 前（pristine） | opt-4 后 | infer-TTFT 降幅 | md5 (prefill+decode) |
|---|---:|---:|---:|:---:|
| Qwen3.5-0.8B-INT4 | 69.23 ms | 66.16 ms | **−3.07 ms (−4.4%)** | ✅ 逐字节一致 |
| Qwen3.5-2B-INT4 | 98.14 ms | 94.98 ms | **−3.16 ms (−3.2%)** | ✅ 逐字节一致 |
| Qwen3.5-4B-INT4 | 224.26 ms | 221.37 ms | **−2.90 ms (−1.3%)** | ✅ 逐字节一致 |

**[356H]** ⚠️ **opt-4 在 356H 上净回退**（交错热插拔 A/B，n=8，ic=1，2 reps 均值，rep 间抖动 <3ms）：

| 模型 | opt-4 前（pristine） | opt-4 后 | infer-TTFT 变化 | md5 (prefill+decode) |
|---|---:|---:|---:|:---:|
| Qwen3.5-0.8B-INT4 | 150.8 ms | 161.4 ms | **+10.6 ms (+7.0%)** | ✅ 逐字节一致（多重集） |
| Qwen3.5-2B-INT4 | 214.0 ms | 224.0 ms | **+10.0 ms (+4.7%)** | ✅ 逐字节一致（多重集） |
| Qwen3.5-4B-INT4 | 506.1 ms | 525.7 ms | **+19.6 ms (+3.9%)** | ✅ 逐字节一致（多重集） |

> 这是与 358H **方向相反**的结果，但已用交错热插拔严格排除热漂移（同一 2B 上 pristine 三 reps 214.4/214.1/214.3ms、opt4 三 reps 223.5/224.6/222.7ms，几乎无抖动，见 `pa_sweep.tsv`）。opt-4 的 token-parallel dispatch（`GWS={seq,hidden,total_tokens}`）在 356H 这块 iGPU 上没有带来 358H 那样的占用率红利、反而更慢（推测与 356H iGPU 的 EU 数量/调度特性不同、`total_tokens` 维过度切分带来额外开销有关；本次未单独做 356H 的 vtune 占用率对账）。**数值完全无损（md5 多重集一致）——纯性能回退，不是错误。结论：356H 上不要启用 opt-4。**

### 3.5 vtune 对账（2B prefill，clean-base）
`paged_causal_conv1d_ref` 占用率 **20% → 100%**，kernel op-time **65.98ms → 45.10ms（−32%）**，墙钟 prefill −3.1ms。机理 / md5 / 墙钟 / vtune 四者一致。原始数据：`opt_pa_archive/vtune_PA_prefill_{B0_pristine,B1_opt4}_2b.csv`。

### 3.6 正确性
贪心解码（`do_sample=false`），128-token decode + 1-token prefill 的 `result_md5` 多重集与 pristine 基线逐一相同（三尺寸全部）。opt-4 只改 GPU 工作项划分 + 等价的窗口重建，不动任何算术，故 bit 级一致。

---

## 4. opt-3：共享激活的多 FC 复用一个 DynamicQuantize（PA 下仍有效）

opt-3 源自 SDPA 时代（详细根因/踩坑见 SDPA 文档 §5），但与 opt-1/opt-2 不同，它在 PA 下**重新验证仍然有效**——因为 `dynamic_quantize_gpu_opt` 在 PA prefill 仍占 ~6.6%，FC 去重照样减少冗余量化。

**做法**：Qwen3.5 每个 decoder 层里多个 FC（qkv/a/b/z 共享 `input_layernorm` 输出；gate/up 共享 `post_attention_layernorm` 输出）共用同一激活；GPU 插件原本给每个 FC 各插一份 `DynamicQuantize`。opt-3 在插 DQ 前查找「同一激活 + 配置逐字段相等」的现有 DQ 复用，并放宽 `dynamic_quantize.cpp` 的 skip 断言让共享 DQ 在 decode 仍可 skip 成 f16 passthrough（保住 decode 数值一致）。改 2 文件：`dynamic_quantize_fully_connected.cpp` + `dynamic_quantize.cpp`。patch：`opt_pa_archive/pa_opt3_dq_reuse.patch`。

**在干净 PA 基础上、叠加在 opt-4 之上的单独增量（n=8、ic=1 纯 prefill）：**

| 模型 | +opt-4（infer-TTFT） | +opt-4+opt-3 | opt-3 单独增量 | md5 (prefill+decode) |
|---|---:|---:|---:|:---:|
| Qwen3.5-0.8B-INT4 | 66.16 ms | 64.61 ms | **−2.3%** | ✅ 逐字节一致 |
| Qwen3.5-2B-INT4 | 94.98 ms | 91.73 ms | **−3.4%** | ✅ 逐字节一致 |
| Qwen3.5-4B-INT4 | 221.37 ms | 215.44 ms | **−2.7%** | ✅ 逐字节一致 |

**[356H]** 在 356H 上「opt-3 叠加在 opt-4 之上」的单独增量（交错热插拔 A/B，n=8，ic=1，2 reps 均值）—— opt-3 仍是收益，且量级与 358H 接近：

| 模型 | +opt-4（infer-TTFT） | +opt-4+opt-3 | opt-3 单独增量 | md5 (prefill+decode) |
|---|---:|---:|---:|:---:|
| Qwen3.5-0.8B-INT4 | 161.4 ms | 157.1 ms | **−2.7%** | ✅ 逐字节一致（多重集） |
| Qwen3.5-2B-INT4 | 224.0 ms | 216.2 ms | **−3.5%** | ✅ 逐字节一致（多重集） |
| Qwen3.5-4B-INT4 | 525.7 ms | 515.3 ms | **−2.0%** | ✅ 逐字节一致（多重集） |

> ⚠️ 注意：356H 上 opt-3 只是把 opt-4 造成的回退补回来一部分——「opt-4+opt-3」相对 pristine 仍是净回退（见 §4.1 [356H]）。**356H 上正确做法是只叠 opt-3、不叠 opt-4**（此时 opt-3 单独 vs pristine 见 §2.1 [356H]，为 −2.1~3.2% 的净收益）。

### 4.1 opt-4 + opt-3 叠加总效果（双口径，vs pristine B0）

| 模型 | **infer-TTFT** pristine→最终 | 降幅 | **e2e-TTFT** pristine→最终 | 降幅 |
|---|---|---:|---|---:|
| Qwen3.5-0.8B-INT4 | 69.23 → 64.61 ms | **−6.7%** | 112.36 → 107.95 ms | −3.9% |
| Qwen3.5-2B-INT4 | 98.14 → 91.73 ms | **−6.5%** | 142.03 → 135.81 ms | −4.4% |
| Qwen3.5-4B-INT4 | 224.26 → 215.44 ms | **−3.9%** | 270.08 → 260.36 ms | −3.6% |

> infer-TTFT（`first_token_infer_mean_ms`）是纯推理 prefill；e2e-TTFT（`ttft_mean_ms`）含 GenAI pipeline 的 host 固定开销（~40ms），故 e2e 降幅小于 infer。两个状态在相同 iter 用相同 prompt，md5 多重集逐一相同 → 全程数值无损。原始数据：`opt_pa_archive/opt4_opt3_combined_AB.txt`。

**[356H]** 356H 上 opt-4 是回退，所以「opt-4+opt-3」叠加相对 pristine **不是净收益**；而**只用 opt-3** 才是净收益。两条口径都列出（交错热插拔 A/B，n=8，ic=1，2 reps 均值）：

| 模型 | **infer-TTFT** pristine→opt4+opt3 | 变化 | pristine→**仅 opt-3** | 变化 |
|---|---|---:|---|---:|
| Qwen3.5-0.8B-INT4 | 150.8 → 157.1 ms | +4.2% | 150.8 → 148.5 ms | **−2.1%** |
| Qwen3.5-2B-INT4 | 214.0 → 216.2 ms | +1.0% | 214.0 → 207.6 ms | **−3.2%** |
| Qwen3.5-4B-INT4 | 506.1 → 515.3 ms | +1.8% | 506.1 → 495.9 ms | **−2.1%** |

| 模型 | **e2e-TTFT** pristine→opt4+opt3 | 变化 | pristine→**仅 opt-3** | 变化 |
|---|---|---:|---|---:|
| Qwen3.5-0.8B-INT4 | 196.3 → 201.9 ms | +2.8% | 196.3 → 193.9 ms | **−1.6%** |
| Qwen3.5-2B-INT4 | 259.8 → 261.7 ms | +0.7% | 259.8 → 253.7 ms | **−2.4%** |
| Qwen3.5-4B-INT4 | 552.2 → 561.2 ms | +1.6% | 552.2 → 542.4 ms | **−2.0%** |

> **356H 落地建议：只启用 opt-3（`pa_opt3_dq_reuse.patch`），不启用 opt-4。** 全程 md5 多重集逐一相同（见 §4 各表 md5 列与本文档 §3.6）。

---

## 5. 复现 checklist

```bash
# 0) 环境
cd /home/user/chaofan/model_benchmark
source benchmark/openvino/install/setupvars.sh
export LD_LIBRARY_PATH="$PWD/benchmark/openvino/install/runtime/lib/intel64:$LD_LIBRARY_PATH"

# 1) PA profiling（vtune，NOT --profile）找 hotspot
source /opt/intel/oneapi/setvars.sh
echo '{"ATTENTION_BACKEND":"PA"}' > /tmp/pa.json
cd model_bench
vtune -collect gpu-hotspots -r /tmp/vt_pa -- ./build-local/model_bench vlm \
  -m ../benchmark/models/Qwen3.5-2B-INT4 -d GPU -n 5 -ic 1 \
  -pf ./prompts/qwen3.5_1k.jsonl --no-monitor -lc /tmp/pa.json -r /tmp/x.csv -rj /tmp/x.json
vtune -report hotspots -group-by computing-task -r /tmp/vt_pa -format csv

# 2) 应用 PA 有效优化（opt-4 + opt-3，2 个独立 patch，文件不重叠）+ 重建（~12s）
cd ../benchmark/openvino
git apply ../tests/opt_pa_archive/pa_opt4_conv1d.patch
git apply ../tests/opt_pa_archive/pa_opt3_dq_reuse.patch
bash rebuild_gpu_plugin.sh

# 3) A/B（n=8，PA，对比 first_token_infer_mean_ms + iterations[].result_md5；prefill ic=1 + decode ic=128）
cd ../../model_bench
./build-local/model_bench vlm -m ../benchmark/models/Qwen3.5-2B-INT4 -d GPU -n 8 -ic 1 \
  -pf ./prompts/qwen3.5_1k.jsonl --no-monitor -lc /tmp/pa.json -r /tmp/x.csv -rj /tmp/out.json
```

---

## 6. 还能继续做什么

opt-4+opt-3 落地后，最终态（pristine+opt-4+opt-3）的 PA prefill 逐 kernel + 占用率限制因素（2B，ic=1）：

```
gemm_kernel                 47.5%  occ 40%  work-size 受限  <- oneDNN jit INT4 GEMM（不可改，§6.2）
paged_gated_delta_net_opt   19.6%  occ 53%  work-size 受限  <- SSM 递归，次大（§6.1）
paged_causal_conv1d_ref      8.4%  occ 100%                <- opt-4 已打满
activation_ref               6.0%  occ 100%
rms_gpu_bfyx_opt             5.5%  occ 40%  SLM 受限        <- 小项 ~15ms（§6.2）
dynamic_quantize_gpu_opt     5.2%  occ 100%                <- opt-3 已打满
```

| 方向 | prefill 占比 / 现状 | 结论 |
|---|---|---|
| **`paged_gated_delta_net_opt`** | 19.6%，occ 53%，work-size 受限 | **已深挖、暂不做**：见 §6.1。真·串行 SSM 递归；VBS 扫描证明 VBS=4 已是 md5-无损最优；chunkwise 上限仅 ~−3-4% 总 prefill 且数天+大概率破 md5，负期望。 |
| `gemm_kernel`（oneDNN INT4 GEMM） | 47.5%，occ 40% | 不做。是 oneDNN ngen-jit micro-kernel，非可编辑 OCL；40% 占用对 XMX 正常。见 §6.2。 |
| `dynamic_quantize_gpu_opt` | 5.2%，occ 100% | opt-3 已覆盖，已打满。 |
| `rms_gpu_bfyx_opt` | 5.5%，occ 40% SLM 受限 | 小项（~15ms，<1% 总 prefill），性价比低。见 §6.2。 |
| opt-3 / opt-4 整理成上游 PR + 单测 | 已落地、md5 无损 | 做。opt-4 通用于任何 PA causal-conv1d 模型；opt-3 通用于任何「多 FC 共享激活」模型。 |

### 6.1 `paged_gated_delta_net_opt`：已深挖，结论是暂不做（2026-06-30 第二轮）
读 kernel（`impls/ocl_v2/paged_gated_delta_net_opt.cl`）确认它实现的是 Gated-DeltaNet 的**串行状态递归**：

```
S_t = b_g · S_{t-1};  h_k = <S_t, k_t>;  S_t += k_t ⊗ ((v_t − h_k)·β_t);  out_t = <S_t, q_t>
```

第 209–243 行的 `state[v_idx][kc] *= b_g` → `fma(k_norm, update, state)` 是 token-by-token 的真实依赖（`while token<token_end` 循环，第 151 行），**与 conv 不同，输出 token 之间不独立，不能用 opt-4 那种 token 并行**。它已经是 `_opt` 版（subgroup 向量化 + V_BLOCK 分块 + block read/write）。

**vtune 诊断**：53% 占用率的限制因素是 **Work Size（53%）**，不是 SLM/barrier/寄存器（都 100%）。即 work-item 数量不足（`head_nums(16) × v_blocks(32) × subgroup(16)`，prefill 时 sequences=1 不贡献并行）。

**V_BLOCK_SIZE 扫描实测（2B，infer-TTFT）——画出 U 形曲线，VBS=4 在谷底：**

| VBS | v_blocks | 并行 sub-groups | q/k 冗余 | infer-TTFT | vs VBS=4 |
|---|---:|---:|---|---:|---:|
| 2 | 64 | 1024 | 2× | 107.0 ms | **+16.7%** |
| **4** | 32 | 512 | 1× | **91.7 ms** | — (最优) |
| 8 | 16 | 256 | 0.5× | 99.9 ms | **+8.9%** |

减小 VBS → 并行度↑但 q/k 归一化被各 v_block 重复计算（冗余↑）；增大 VBS → 冗余↓但并行度↓。**两端都回退 → VBS=4 是 md5-无损下的最优点。** 这条 U 曲线同时框定了 chunkwise 的上限：

- chunkwise（块内矩阵乘 + 块间递归）的全部价值就是同时往「并行度↑」「冗余↓」两个好方向走；但 VBS 扫描显示这两块的弹性都很小，说明 GDN 的 105ms **主体是真实递归计算**（state `128×4` 逐 token rank-1 更新 ×~1000 token），chunkwise 也省不掉。
- **估算上限：GDN 105ms → 最好 ~85ms，对总 prefill（536ms）约 −3~4%。**
- **代价**：重写需块内三角求逆 `(I − tril(β·KKᵀ))⁻¹`，数天工作量，且改浮点累加顺序 → 几乎必然破逐字节 md5、大概率多轮调试失败。

**结论：负期望投入，本轮不做。** 留作「若放宽精度验收 + 立项」的未来方向。

### 6.2 其他已排除项
- **`gemm_kernel`（47.5%，最大块）**：经确认是 **oneDNN 的 ngen-jit INT4 GEMM micro-kernel**（`thirdparty/onednn_gpu/.../gemm/jit`，`kernel_name()=="gemm_kernel"`），**不是可编辑的 OCL kernel**。occ 40% 对 XMX systolic 阵列是正常的（XMX 忙时 EU 线程本就不满），是算力本体非瓶颈。三个最大实例（GWS 768×8）合计 225ms / 占 gemm 88%，dispatch 由 oneDNN 内部 heuristic 决定，插件层改不动。
- **`rms_gpu_bfyx_opt`（5.5%，occ 40% SLM 受限）**：仅 ~15ms，占总 prefill <1%，性价比低。

> 本轮深挖的完整数据（含 VBS 扫描、kernel 占用率限制因素、gemm 来源认定）见 `opt_pa_archive/gdn_gemm_investigation.txt` 与 `vtune_PA_prefill_final_opt4opt3_2b.csv`。
>
> **总结：低风险 dispatch 类优化已挖到底。opt-4（conv1d）+ opt-3（DQ）是不重写算法前提下的 PA 实用上限（2B infer-TTFT −6.5% / e2e −4.4%）。**

---

## 附：产物清单

| 文件 | 说明 |
|---|---|
| `src/plugins/intel_gpu/src/graph/impls/ocl_v2/paged_causal_conv1d_ref.{cl,cpp}` | opt-4：token 并行 conv1d（kernel 窗口重建 + GWS 加 token 维） |
| `src/plugins/intel_gpu/src/plugin/transformations/dynamic_quantize_fully_connected.cpp` + `src/graph/dynamic_quantize.cpp` | opt-3：共享激活的 FC 复用 DynamicQuantize（PA 下仍有效） |
| `opt_pa_archive/pa_opt4_conv1d.patch` | opt-4 git patch（基于 pristine HEAD `ede283a88e`，2 文件） |
| `opt_pa_archive/pa_opt3_dq_reuse.patch` | opt-3 git patch（基于 pristine HEAD，2 文件；与 opt-4 文件不重叠，可一起 apply） |
| `opt_pa_archive/opt4_conv1d_AB_cleanbase.txt` | opt-4 单独 A/B + md5 |
| `opt_pa_archive/opt4_opt3_combined_AB.txt` | opt-4+opt-3 叠加双口径（infer/e2e TTFT）+ md5 汇总 |
| `opt_pa_archive/vtune_PA_prefill_{B0_pristine,B1_opt4}_2b.csv` | opt-4 前后的 PA prefill vtune（占用率对账） |
| `reeval_PA_20260630/vtune_{PA,SDPA}_2b_kernels.csv` | PA vs SDPA kernel 结构对比（§2 根因） |

---
*基于本地源码构建的 OpenVINO 2026.2.1 / model_bench(C++) / vtune 2026.2，Panther Lake + Arc B390。opt-4 已实测、正确性以 128-token 贪心 md5（含 prefill ic=1）校验逐字节一致；所有数据均在 pristine（不含 SDPA opt-1/2/3）基础上、PA backend、n=8、real prompt(~1000 tok)、GPU 测得。*
