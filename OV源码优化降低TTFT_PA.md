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
> **356H 结论差异（重要）：opt-3 在 356H 上依然是有效收益（PA infer-TTFT −2~3.2%），但 opt-4（token-parallel conv1d）在 356H 上是净回退（+4~7%，与 358H 的 −1.3~4.4% 收益相反），尽管 md5 仍逐字节无损。** 详见各节 [356H] 表（opt-4 回退详情见 §3.4，opt-3 净收益见 §2.1，双口径汇总见 §4.1）。
> **【新增 opt-5，2026-07-01】** 用 vtune 归因 opt-4 回退（占用率虽 60→100% 但冗余 K-窗口读使 conv1d 变 memory-bound、2× 慢）后，设计出 **opt-5：token-TILED conv1d**——只切成很少的 G 个连续块、块内保留串行滑动窗口（无冗余读），既把占用率打到 100% 又不放大内存。opt-5 在 356H 上 conv1d kernel 14.1→10.5ms/iter（−26%），叠加 opt-3 后 infer-TTFT 相对 pristine：**0.8B −4.0% / 2B −4.4% / 4B −1.7%**，md5 全尺寸 prefill+decode 逐一无损。**opt-5 阶段的 356H 建议：opt-3 + opt-5（不用 opt-4）；当前最终建议已由 opt-6 更新为 opt-3 + opt-5 + opt-6。** 详见 §4.5/§4.7，patch `opt_pa_archive/pa_opt5_conv1d_tiled.patch`。
> **【新增 opt-6，2026-07-02】** 继续 profiling opt-3+opt-5 态后发现大尺寸 dynamic-shape `activation_ref` 仍是可编辑热点（0.8B/2B prefill 中约 6–7% task time）。OpenVINO 已有 float4 向量化的 `activation_opt`，但此前不支持 dynamic shape，且 kernel 签名缺少 shape-info 参数；**opt-6：dynamic activation 走 activation_opt** 只放开「同布局、无 fused ops、无 runtime activation 参数」的安全子集。356H 上叠加 opt-3+opt-5 后，infer-TTFT 相对 pristine：**0.8B −9.2% / 2B −8.1% / 4B −4.7%**，e2e-TTFT：**0.8B −5.6% / 2B −6.1% / 4B −4.1%**。356H 当前最终建议：**opt-3 + opt-5 + opt-6**。详见 §4.7，patch `benchmark/openvino/qwen_patch/opt6_dynamic_activation_opt.patch`。
> **【新增 opt-7，2026-07-03】** 重新 profiling/审视 e2e gap 后确认：opt-3/opt-5/opt-6 后 2B 当前 GPU kernel 热点已主要是 `gemm_kernel`(~49%)、`paged_gated_delta_net_opt`(~20%)、`rms`(~7%)、DQ(~5%)、conv1d(~4%)、activation_opt(~3%)；opt-3/5/6 均已按预期生效。e2e 仍比 `first_token_infer` 多约 45ms，其中一部分来自 GenAI text-only VLM 的 embedding host/memory path。**opt-7：Qwen3.5 text-only 时去掉 text embedding 输出到 `inputs_embeds` 的整块 memcpy**，是 e2e-only 小优化：叠加后 356H e2e-TTFT 0.8B/2B/4B 分别为 **185.1 / 243.9 / 527.7 ms**（相对 pristine **−5.9% / −6.4% / −4.4%**）。详见 §4.8，patch `benchmark/openvino/qwen_patch/opt7_genai_qwen35_text_embedding_copy_elision.patch`。
> **【更新 opt-5 自适应分块，2026-07-22】** opt-5 的分块因子 **G 原为硬编码 3**（356H 手工调优值），现改为**自适应**（`.cl` kernel 不变，仅 `.cpp` 的 dispatch 改为按设备+序列计算 G）。公式：`G = clamp( ceil( HW线程 × SIMD32 × 3/2 / (seq_count × hidden) ), 1, min( 平均序列长度/16, 64, token总数 ) )`，其中 `HW线程 = execution_units_count × num_threads_per_eu`。含义：（a）**按 GPU 规模**取 G 使 `seq_count×hidden×G` 个 work-item 以 ~1.5× 过订阅填满硬件线程——大 iGPU 自动取更大 G、小 iGPU 更小，无需按机器重调；（b）**按序列长度**保证每块 ≥16 token 且块数 ≤ token 总数（decode 每序列 1 token → G=1，退化为原串行 kernel）。`OV_CONV1D_TOKEN_GROUPS=<n>` 仍可**强制**覆盖（unset/0=自动），`OV_CONV1D_DEBUG=1` 打印一次选中的 G。**本次在 HEAD `b0fbbcafa7` 重建 GPU 插件复测**：356H（EU=32×threads/eu=10=320 线程、hidden=6144、1000-tok prefill）自动选到 **G=3**——与原手工值一致；A/B（`OV_CONV1D_TOKEN_GROUPS=1` 串行 ≡ pristine vs 自动 G=3）在 0.8B/2B ic=128 n=8 下**逐迭代 md5 完全一致（bit-exact）**；infer-TTFT 保持 opt-5 收益（2B whole-model G 扫描：串行 G=1 211.3 → 自动 G=3 207.8ms，见 §4.5.2）。358H（Arc B390，pristine conv1d 占用率 20% ⇒ ~960 线程）按同式**预计 G≈8**，即这版自适应 opt-5 预期在 358H 上也有效（仍是分块、无 opt-4 的内存放大），从而有望**统一替代 opt-4**（待 358H 实测确认）。4B 本机已无模型，逻辑按同式覆盖。详见 §4.5.2，patch 已更新（`opt5_conv1d_token_tiled.patch` / `pa_opt5_conv1d_tiled.patch`）。

---

## 0. 结论先行

### 0.1 按平台的推荐栈（避免混读 358H / 356H）

| 平台 | PA 推荐 patch | 已测结果摘要 | 备注 |
|---|---|---|---|
| **Core Ultra X7 358H + Arc B390 iGPU** | opt-3 + opt-4（历史数据）；**自适应 opt-5 后建议改用 opt-3 + opt-5(+opt-6)** | infer-TTFT：0.8B −6.7%、2B −6.5%、4B −3.9%（opt-3+opt-4 历史数据） | opt-4 在大 iGPU 上有效；**自 2026-07-22 起 opt-5 的 G 自适应（358H 预计 G≈8），预期在 358H 上取代 opt-4 —— 待 358H 实测确认后即可全平台统一用 opt-5**；opt-6 尚未在 358H 补测，不外推 |
| **Core Ultra 7 356H + 32 XVE iGPU** | **opt-3 + opt-5 + opt-6**；text-only GenAI e2e 再叠 **opt-7** | infer-TTFT：0.8B −9.0%、2B −7.9%、4B −4.8%；e2e-TTFT：0.8B −5.9%、2B −6.4%、4B −4.4% | opt-4 在 356H 回退，必须用 opt-5 取代（自适应 G 在 356H 自动=3）；opt-6 是 GPU kernel 优化；opt-7 是 GenAI text-only e2e 小优化 |

后续正文保留了原始时间线（358H → 356H 补测 → opt-5 → opt-6），但读结论时优先看本表。

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
4. **[356H] opt-6 打开了旧结论里漏掉的 activation_ref 空间。** 旧 §4.6 里的「opt-3+opt-5 已是本机上限」是 opt-6 之前的阶段性判断；本轮发现 `activation_ref` 可通过选择已有 `activation_opt` 向量化 kernel 继续降低 TTFT（§4.7）。
5. **[356H] opt-7 是 e2e-only 小优化，不改变 GPU LM infer。** 它去掉 text-only VLM embedding 输出的冗余 CPU tensor copy，对 `first_token_infer` 基本无影响，但 e2e 再降约 0.3%（§4.8）。
6. **次大项 `paged_gated_delta_net_opt`（prefill 19%）仍分析为高风险、暂不动**——它是真·串行 SSM 递归，无法简单并行，需 chunkwise 重写（§6）。
7. **[356H] 第四轮 review（2026-07-03，§4.9）：当前栈已接近本机 md5-无损局部优化上限。** fresh vtune 复核 opt-3/5/6 全部生效；本轮新试的角度中——**4-way FC horizontal fusion（把 SSM in_proj q/k/v/z 融成一个大 gemm）实测在三个尺寸上都是负收益（+0.8~2.0%），已否决**；sdpa_micro（oneDNN gemmstone）dead-end；eltwise/slice/concat 向量化因已 100% 占用（带宽 bound）marginal。gemm(50%)/GDN(20%) 仍是 oneDNN 硬上限 / 串行递归，只剩高成本立项路径。
8. **[356H] 第五轮 fresh profiling（2026-07-09，§4.10）：再次复核当前二进制后未发现新的低风险落地点。** 当前 2B `first_token_infer≈198ms`、e2e≈250ms；GPU task 仍由 gemm(~49%)+GDN(~20%) 主导。新实测的 `eltwise_simple_vload8` dynamic-shape probe 没有命中目标（vtune 仍显示 `generic_eltwise_ref`），已回退；md5 一致但不落地。结论维持：若要继续大幅提升，需要 GDN chunkwise/oneDNN GEMM 级别立项或放宽 bit-exact 约束。

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

## 4.5 【356H】opt-4 回退的 vtune 归因 + opt-5：token-TILED conv1d（自适应分块，取代 opt-4；356H 已测、358H 预计通用）

> 2026-07-01 在 356H 上用 vtune gpu-hotspots（xe 驱动，`Display controller 0xb0a0`，**32 XVE × 10 threads/XVE = 320 HW threads**）对 opt-4 做占用率对账，定位回退根因，并据此设计出 356H 上真正有效的 opt-5。原始数据：`repro_356H_20260701/vtune/`（`vt_B0/opt4/opt5G3/opt5G4/best2b.csv`）。

### 4.5.1 opt-4 为什么在 356H 变慢（vtune 逐 kernel 对账，2B PA prefill，n=5 → /5 为每 iter）

| conv1d 状态 | GWS | occ | occ 限制因素 | conv1d kernel 时间 |
|---|---|---:|---|---:|
| **pristine（串行）** | `1×6144×1` | 60% | Work Size | 14.13 ms/iter |
| **opt-4（全 token 并行）** | `1×6144×1002` | 100% | — | **24.71 ms/iter** |
| **opt-5（token 分块，356H 自动 G=3）** | `1×6144×3` | 100% | — | **10.45 ms/iter** |

- 串行版占用率 = 6144 work-items ÷ SIMD32 = **192 threads = 320 的 60%**——与 vtune 实测 60% 完全对上，即「work-size 受限」。
- opt-4 把 GWS 撑到 `6144×1002`（每个输出 token 一个 work-item），占用率确实到 100%，**但每个 work-item 都要从 global 重新读自己的 K=4 元窗口**，全局读放大 ~K 倍 → kernel 变 memory-bound → 14.1→24.7ms/iter（**+75%**）。其余 kernel 全部不变，op-sum +9.3ms/iter，正好对上墙钟 2B +10ms/iter 的回退。
- **根因一句话**：opt-4 在 358H 的大 iGPU（Arc B390，EU 多）上「占用率红利 > 冗余读代价」，故净收益；在 356H 的小 iGPU（32 XVE）上「冗余读代价 > 占用率红利」，故净回退。

### 4.5.2 opt-5：token-TILING（分块，保留流式复用）

把每条序列的 token 切成**很少的 G 个连续块**，GWS=`{seq_count, hidden, G}`，每个 `(seq,h,g)` work-item **串行扫自己那一块**（滑动窗口，块内零冗余读），只在块起点一次性从 `input_embeds`/初始 state 重建 K-1 元历史。这样：
- 占用率：`hidden×G` work-items。G≥2 即 12288 wi > 320 threads → **100% 占用**。
- 额外读只有 `hidden×G×(K-1)`（可忽略），**没有 opt-4 的 `hidden×total_tokens×K` 内存放大**。
- 数学逐元素 fma 顺序与串行版一致 → **bit 级无损**；状态写回被唯一块拥有（连续不重叠）→ 无竞争。

**G 自适应选取（2026-07-22 更新，取代原「硬编码默认 3」）。** dispatch 时按公式
`G = clamp( ceil( HW线程 × SIMD32 × 3/2 / (seq_count × hidden) ), 1, min( 平均序列长度/16, 64, token总数 ) )`
自动计算 G（`HW线程 = execution_units_count × num_threads_per_eu`；SIMD32 为该 scalar kernel 实测 pack 宽度）：
- **按 GPU 规模填满线程**：取 G 使 `seq_count×hidden×G` 个 work-item 以 ~1.5× 过订阅硬件线程数。大 iGPU（EU 多）自动取更大 G，小 iGPU 更小 —— 这是让 opt-5 跨 iGPU 规模无需重调的关键。
- **按序列长度收敛**：每块保持 ≥16 token（使每块一次性的 K-1 priming 读相对流式工作量可忽略），块数不超过 token 总数（decode 每序列 1 token → **G=1，退化为原串行 kernel**）。
- **356H**（320 线程、hidden=6144、~1000-tok prefill）自动得 **G=3**（`target=320×32×1.5=15360`, `ceil(15360/6144)=3`）——与之前手工值一致，`OV_CONV1D_DEBUG=1` 已实测确认。**358H**（~960 线程）预计 **G≈8**。

`OV_CONV1D_TOKEN_GROUPS=<n>` 可**强制**指定 G（unset/0=自动）。实测 whole-model G 扫描（2B PA ic=1 n=8，1st-infer）：G=1 211.3 / G=2 211.2 / **G=3 207.8（自动）** / G=4 208.2 / G=5 206.0 / G=8 206.5 ms —— 自动的 G=3 已在收益「膝点」，更大 G 仅亚毫秒差异；vtune 口径 conv1d kernel 仍在 G=3 附近达 10.45ms/iter 谷底（G=4 11.54ms）。改 2 文件（`paged_causal_conv1d_ref.{cl,cpp}`），patch：`opt_pa_archive/pa_opt5_conv1d_tiled.patch`（与 opt-4 patch 互斥，二者只能取一；356H 用 opt-5，自适应后 358H 亦应用 opt-5 取代 opt-4，待实测）。

### 4.5.3 opt-5 效果（356H，交错热插拔 A/B，n=8，ic=1，3 reps 均值）

对比基线取「opt-3 单独态」（356H 当前最优）与 pristine：

| 模型 | pristine | opt-3 | **opt-3+opt-5** | opt-3 vs pristine | **opt-3+5 vs pristine** | opt-5 增量(vs opt-3) |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B | 151.3 | 149.0 | **145.3** | −1.5% | **−4.0%** | −2.5% |
| Qwen3.5-2B | 214.3 | 207.3 | **204.8** | −3.2% | **−4.4%** | −1.2% |
| Qwen3.5-4B | 505.0 | 494.5 | 496.2 | −2.1% | −1.7% | +0.3%（噪声内） |

e2e-TTFT 同口径：0.8B 196.7→190.6（**−3.1%**）、2B 260.7→250.6（**−3.9%**）、4B 551.9→542.1（−1.8%）。

- **conv1d kernel 本身 14.1→10.5ms/iter（−26%，占用率 60→100%）是确定无疑的改善**；whole-model 在小/中模型上显现（conv1d 占比更高），4B 上因 conv1d 占比小而落在噪声内。
- **正确性**：三尺寸 prefill(ic=1) 与 128-token decode(ic=128) 的 md5 多重集与 opt-3 基线**逐一相同**（0.8B 5 元、2B 8 元/decode 7 元、4B 3 元多重集全部一致）。
- **opt-5 阶段 356H 落地建议**：`pa_opt3_dq_reuse.patch` + `pa_opt5_conv1d_tiled.patch`（**不要** opt-4）。infer-TTFT 相对 pristine：0.8B −4.0% / 2B −4.4% / 4B −1.7%。当前最终建议见 §4.7：opt-3 + opt-5 + opt-6。

### 4.5.4 356H PA prefill 剩余 hotspot（opt-3 态，2B，op-sum≈248ms/iter）与后续方向

| kernel | 占比 | occ | 限制因素 | 结论 |
|---|---:|---:|---|---|
| `gemm_kernel`（oneDNN INT4 GEMM） | 47.3% | 40% | Work Size | 不可改（ngen-jit micro-kernel，非 OCL），40% 对 XMX 正常 |
| `paged_gated_delta_net_opt` | 19.5% | 100% | — | 真·串行 SSM 递归，已满占用，chunkwise 负期望（见 §6.1） |
| `rms_gpu_bfyx_opt` | 7.4% | **40%** | **SLM** | **已识别、暂缓**：主实例 `GWS=16×16032 LWS=16` SLM 受限 40%；`SLM_SIZE` 按 `maxSlmSize` 过量分配（实际只需 `num_subgroups` 元）。但该 kernel 被**所有模型共享**（dense/embedding/reranker），改动 blast radius 大、上限仅 ~9.5ms/iter，需单独立项 + 全模型回归后再动 |
| `paged_causal_conv1d_ref`（opt-5 后） | ~4% | 100% | — | opt-5 已打满 |
| `dynamic_quantize_gpu_opt` | 4.9% | 100% | — | opt-3 已覆盖、已满 |

> **结论：356H 上低风险 dispatch 类优化，conv1d 已由 opt-5 挖到 100% 占用。下一个可编辑候选是 `rms_gpu_bfyx_opt` 的 SLM 过量分配（占用率 40%→更高），但因跨模型共享需谨慎立项。** gemm/GDN 与 358H 结论一致（不可改 / 负期望）。

## 4.6 【356H】第三轮深度 profiling：gemm / rms / GDN 三大项逐一证伪（opt-6 前的阶段性结论）

> 2026-07-01 应「收益仍太小、要更大 hotspot」的要求，对最优态（opt-3+opt-5）做了第三轮深挖，把三个大项**逐一动手验证**（不再只引用 358H 结论）。全部为负结果——记录下来避免重复投入。

**目标排序（opt-3+opt-5 态，2B PA prefill，vtune，op-sum≈244ms/iter）：** gemm 47.9% / GDN 19.8% / rms 7.5% / activation 6.6% / DQ 5.1% / conv1d 4.3%（已由 opt-5 打满）。

### 4.6.1 gemm（47.9%，占用率 40%）——实测证伪：oneDNN XMX 远优于任何 OCL FC，40% 非缺陷
- gemm 三个主 shape 均 `GWS=256×8 LWS=32×8 occ40%（work-size 受限）`，约 79 次/iter（≈24 层 ×3 大 FC），是 `[1002,2048]×[6144,2048]` 的 INT4×INT8(动态量化) XMX 矩阵乘。
- **关键实验**：临时改 `execution_config.cpp` 用环境变量强制 `m_use_onednn=false`（走可编辑的 OCL FC 路径 `fully_connected_gpu_bf_tiled` 等），2B PA infer-TTFT **206ms → 1015ms（5× 慢）**，且 md5 变化（OCL INT4 FC 数值路径不同）。
- **结论**：oneDNN 的 ngen-jit XMX systolic micro-kernel 尽管占用率只有 40%（EU 线程本就不必填满 XMX 阵列），**每线程效率远高于任何 OCL kernel**；40% 占用是 XMX 的正常特征、不是可修的缺陷。gemm 在插件 OCL 层**根本不可改进**。（`GPU_USE_ONEDNN` / force-implementations 均被 `ENABLE_DEBUG_CAPS`（本 build = OFF）门控，`-lc` 无法运行时切换；本次靠临时打桩验证，验证后已还原。）

### 4.6.2 rms_gpu_bfyx_opt（7.5%，占用率 40% SLM 受限）——实测证伪：占用率是伪信号，kernel 是访存延迟 bound
- 根因定位准确：dynamic-shape 路径把 `SLM_SIZE` 设成 `maxSlmSize`(=1024)，而 `slm_buf` 只按 `get_sub_group_id()` 索引、实际只需 `num_subgroups=LWS/16` 个槽（主实例 LWS=16 → 只需 1 槽）。4KB/WG 的 SLM 过量分配把每 XVE 常驻 WG 数压到 40% 占用。
- **实测**（改 `SLM_SIZE = maxSlmSize/SUB_GROUP_SIZE`，md5 逐字节一致）：主实例 `16×16032` 占用率 **40%→100%**（vtune 确认 SLM 限制解除），但 **kernel 时间 9.47ms→9.47ms 不变**，whole-model 交错 A/B **+0.28%（噪声内）**。
- **结论**：该 rms 实例（16 workers 各读 128 连续元素）是**访存延迟 bound，不是占用率 bound**——提高占用率不改善墙钟。SLM 修复本身正确无害（省 SLM、md5 不变），但**不是 356H 上的 TTFT 收益，已回退**。教训：**vtune 的「occ 40% + 限制因素=SLM」不等于「提占用率就能加速」，必须实测墙钟验证。**

### 4.6.3 GDN（19.8%，占用率 100%）——维持 358H 结论：已满占用、串行递归、无 dispatch 空间
- 356H 上 GDN 占用率就是 100%（vs 358H 报的 53%——因 VBS/shape 差异），已无「work-size 受限」余地；核心是逐 token rank-1 状态更新的真·串行递归，与 §6.1 一致，chunkwise 负期望。

### 4.6.4 本轮总结（诚实的天花板）
| 大项 | 占比 | 可编辑? | 本轮实测结论 |
|---|---:|---|---|
| gemm | 47.9% | 否（oneDNN） | OCL FC 替代 5× 慢 → 不可改进 |
| GDN | 19.8% | 是（OCL） | 已 100% 占用、串行递归 → 无低风险空间 |
| rms | 7.5% | 是（OCL） | 占用率可修但延迟 bound → 无墙钟收益（已回退） |
| conv1d | 4.3% | 是（OCL） | **opt-5 已打满（已落地）** |
| DQ | 5.1% | 是 | **opt-3 已覆盖（已落地）** |

> **阶段性结论（2026-07-01）：在当时已经验证过的候选里，opt-3 + opt-5 是 356H PA prefill 的实用上限（2B infer-TTFT 相对 pristine −4.4%）。** 这句话后来被 §4.7 的 opt-6 更新：`activation_ref` 虽然不是 gemm/GDN/rms 三大项，但它是可低风险切换到已有向量化 kernel 的剩余热点。gemm/GDN/rms 的负结论仍成立。

## 4.7 【356H】opt-6：dynamic activation 选择向量化 activation_opt（本轮新增）

> 2026-07-02 继续分析 opt-3+opt-5 后的 0.8B/2B PA prefill。`gemm`、`GDN`、`rms` 的大方向已在 §4.6 证伪，但 vtune 旧报告里仍能看到两个大尺寸 `activation_ref`：`1×1×6156288` 与 `1×1×2052096`，合计约 6–7% task time。它们是普通逐元素 activation，数学上不需要复杂 layout 索引；OpenVINO 已有 float4 向量化的 `activation_opt`，只是此前没有 dynamic-shape support。

### 4.7.1 根因

`activation_opt.cl` 本身是线性 float4 读写：每个 work-item 处理 4 个连续元素，适合这类大 contiguous activation。但 selector 的 `ActivationKernelOpt` 没有声明 `EnableDynamicShapesSupport()`，因此 dynamic shape 下只能落到通用 `activation_ref`。直接打开 dynamic 后第一次运行报 `CL_INVALID_ARG_INDEX`，原因是 shape-agnostic kernel 会额外传 `OPTIONAL_SHAPE_INFO_ARG`，而 `activation_opt.cl` 签名没有接收该参数。

### 4.7.2 改动

改 2 个文件，patch 见 `benchmark/openvino/qwen_patch/opt6_dynamic_activation_opt.patch`：

| 文件 | 改动 |
|---|---|
| `src/plugins/intel_gpu/src/kernel_selector/kernels/activation/activation_kernel_opt.cpp` | 给 `ActivationKernelOpt` 增加 dynamic-shape support；同时限制 dynamic 路径只用于「输入/输出 layout 相同、无 fused ops、无 `inputActivationParams`」的安全子集 |
| `src/plugins/intel_gpu/src/kernel_selector/cl_kernels/activation_opt.cl` | kernel 参数列表加入 `OPTIONAL_SHAPE_INFO_ARG`，使 shape-agnostic dispatch 能正确绑定参数 |

这个优化不改 activation 的数学表达式，只改变 kernel 选择：从通用坐标索引的 scalar/ref kernel 换成已有的线性 float4 opt kernel。

### 4.7.3 效果（356H，PA，GPU，真实 1000-token prompt，ic=1）

对比基线取 pristine、opt-3+opt-5、以及最终 opt-3+opt-5+opt-6。infer-TTFT = `first_token_infer_mean_ms`，e2e-TTFT = `ttft_mean_ms` / `Prefill`（含 GenAI host 固定开销）。0.8B 最终态用 n=8 复测；2B/4B 最终态用 n=5。

| 模型 | pristine infer | opt-3+5 infer | **opt-3+5+6 infer** | final vs pristine | opt-6 增量 | pristine e2e | opt-3+5 e2e | **opt-3+5+6 e2e** | final e2e vs pristine |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B | 151.3 | 145.3 | **137.3** | **−9.2%** | −5.5% | 196.7 | 190.6 | **185.7** | **−5.6%** |
| Qwen3.5-2B | 214.3 | 204.8 | **196.9** | **−8.1%** | −3.8% | 260.7 | 250.6 | **244.8** | **−6.1%** |
| Qwen3.5-4B | 505.0 | 496.2 | **481.2** | **−4.7%** | −3.0% | 551.9 | 542.1 | **529.5** | **−4.1%** |

原始运行输出：`/tmp/activation_opt_final_08b.{csv,json}`、`/tmp/activation_opt_2B-INT4.{csv,json}`、`/tmp/activation_opt_4B-INT4.{csv,json}`。本机 vtune 在 opt-6 后重采时两次触发 collector 内部断言并生成损坏结果，因此 opt-6 的 post-profile 不作为证据；target 来自 opt-3+opt-5 的既有 vtune hotspot，收益以 A/B benchmark 和 md5 为准。

### 4.7.4 正确性

0.8B opt-3+5 基线与 opt-6 最终态 md5 多重集逐一相同：`cae8d14e×7 + 514faf5a×1 + dc27db97×1`（含 warmup/8 measured 输出）。2B/4B 最终态分别为稳定单值 `dd753679...`、`e067b6f4...`，与此前 PA 基线/opt-3/opt-5 的同尺寸 md5 一致。opt-6 只切换等价 activation kernel，未观察到 decode/prefill 数值差异。

### 4.7.5 当前 356H 最终建议

**356H PA GPU kernel 最终落地栈：`opt3_dynamic_quantize_reuse.patch` + `opt5_conv1d_token_tiled.patch` + `opt6_dynamic_activation_opt.patch`。** text-only VLM e2e 可额外叠加 §4.8 的 GenAI opt-7。不启用 opt-4。若要补 358H 数据，应单独在 358H 上测 opt-5/opt-6/opt-7；本文不把 356H 的收益外推到 358H。

## 4.8 【356H】重新 profiling 后的 e2e 小优化 opt-7：去掉 text-only embedding memcpy

### 4.8.1 重新 profiling 结论

在 opt-3+opt-5+opt-6 当前栈上重新跑 2B PA vtune（GPU，`n=3`，真实 1000-token prompt），`activation_ref` 已按预期变成 `activation_opt`，opt-5 的 conv1d 仍为 `GWS=1×6144×3` 且 100% occupancy。当前 GPU task 主要分布：

| kernel | 占比 | 结论 |
|---|---:|---|
| `gemm_kernel` | ~49% | oneDNN XMX GEMM，仍不可在 OCL 层低风险优化 |
| `paged_gated_delta_net_opt` | ~20% | 真串行 SSM 递归，100% occupancy，仍不适合局部 dispatch 优化 |
| `rms_gpu_bfyx_opt` | ~7% | 前文已证伪：提高 occupancy 不降墙钟 |
| `dynamic_quantize_gpu_opt` | ~5% | opt-3 已减少重复 DQ，剩余为必要量化 |
| `paged_causal_conv1d_ref` | ~4% | opt-5 生效；G 自适应在 356H 选到 3（G=3/5/8 whole-model 仅亚毫秒差异，3 已在膝点） |
| `activation_opt` | ~3% | opt-6 生效；尝试 float8 vector width 触发 `CL_OUT_OF_RESOURCES`，已回退 |

所以继续压 GPU infer 的低风险空间已经很小。e2e 角度看，2B 当前 `first_token_infer≈197.6ms`、e2e≈244.8ms，中间约 47ms 是 GenAI/VLM host 和 embedding 准备路径。

### 4.8.2 opt-7 改动

Qwen3.5 复用 `InputsEmbedderQwen3VL::get_inputs_embeds()`。text-only 场景中原逻辑：

1. text embedding model 输出 `text_embeds`。
2. 若无 image/video，仍分配一个同 shape 的 `inputs_embeds`。
3. `memcpy(text_embeds → inputs_embeds)` 后返回。

对无 image/video 的 benchmark，这个 copy 不改变任何值，也不参与视觉 embedding merge。opt-7 只在 `images.empty() && videos.empty()` 时直接返回 `text_embeds`，保留 image/video 路径原样。曾尝试让 embedding 直接输出 remote tensor，但 continuous-batching 后续路径会访问 `Tensor::data()`，remote tensor 抛 `Not Implemented`，因此只保留 CPU tensor copy-elision。

patch：`benchmark/openvino/qwen_patch/opt7_genai_qwen35_text_embedding_copy_elision.patch`（应用位置是 sibling repo `benchmark/openvino.genai`）。

### 4.8.3 效果（356H，PA，GPU，ic=1）

| 模型 | opt-3+5+6 infer/e2e | **+opt-7 infer/e2e** | opt-7 e2e 增量 | 最终 e2e vs pristine |
|---|---:|---:|---:|---:|
| Qwen3.5-0.8B | 137.3 / 185.7 | **137.7 / 185.1** | −0.3% | **−5.9%** |
| Qwen3.5-2B | 196.9 / 244.8 | **197.3 / 243.9** | −0.3% | **−6.4%** |
| Qwen3.5-4B | 481.2 / 529.5 | **480.8 / 527.7** | −0.3% | **−4.4%** |

这是一个小优化：对 infer-TTFT 基本无影响，但稳定减少 text-only e2e 中的一次大 tensor copy。md5 多重集保持一致：0.8B `cae8d14e×4 + 514faf5a×1 + dc27db97×1`（n=5+warmup），2B `dd753679...×6`，4B `e067b6f4...×6`。

## 4.9 【356H】第四轮 review + fresh vtune：进一步优化空间评估（2026-07-03）

> 应「审视现有 opt-3/5/6/7 是否还有进一步优化 + 是否有其他可优化点」的要求，本轮在当前 opt-3+opt-5+opt-6 栈上重新 vtune，并**逐一动手验证/证伪**了几个尚未在前文覆盖的新角度（不只引用旧结论）。核心结论：**当前栈已接近本机 md5-无损局部优化的实用上限；本轮试的新角度要么负收益、要么 marginal。**

### 4.9.1 fresh vtune 复核（opt-3+5+6，2B PA prefill，n=5，op-sum≈234.8ms/iter）

| kernel | ms/iter | 占比 | occ | 限制因素 | 状态 |
|---|---:|---:|---:|---|---|
| `gemm_kernel` | 116.65 | **49.7%** | 40% | Work Size | oneDNN XMX，不可编辑（§4.6.1 已证伪 OCL 替代 5× 慢） |
| `paged_gated_delta_net_opt` | 48.20 | 20.5% | 100% | — | 串行 SSM 递归，满占用（§6.1） |
| `rms_gpu_bfyx_opt` | 18.39 | 7.8% | 40% | SLM | 访存延迟 bound（§4.6.2 证伪） |
| `dynamic_quantize_gpu_opt` | 12.39 | 5.3% | 100% | — | opt-3 覆盖，满占用 |
| `paged_causal_conv1d_ref` | 10.45 | 4.5% | 100% | — | opt-5 覆盖 |
| `activation_opt` | 6.98 | 3.0% | 100% | — | opt-6 覆盖 |
| `sdpa_micro` | 6.45 | 2.7% | 50.9% | SLM | full-attn 层（6/24 层），本轮新查 |
| `generic_eltwise_ref` | 6.20 | 2.6% | 100% | — | GDN gating，本轮新查 |
| `strided_slice_ref` | 4.87 | 2.1% | 100% | — | SSM slice，本轮新查 |

opt-3/5/6 全部按预期生效（conv1d/activation/DQ 均 100% occ）。gemm 仍占近半，是硬上限。

### 4.9.2 新角度 A（实测）：4-way FC horizontal fusion —— 负收益，已否决

**动机**：OV 已有 `FullyConnectedHorizontalFusion`（`plugin/transformations/fc_horizontal_fusion.cpp`），把共享同一激活的多个 FC 的权重在 N 轴 concat 成一个更大的 gemm + `VariadicSplit` 输出。但它硬编码 `max_num_fcs_to_fuse = 3`。用 IR 分析确认 Qwen3.5-2B：**18 个 SSM 层的 `input_layernorm` 喂 4 个 FC**（in_proj `qkv=6144 / a=16 / b=16 / z=2048`，共享 K=2048），因超过 3 而**未被融合**；6 个 attention 层的 q/k/v（3-way）与 gate/up（2-way）已被融合。

**实测**：把 cap 改成 4、重建。加调试打印确认 **4-way 融合确实触发**（18 层 fused-4FCs + 6 层 fused-3FCs）。但交错热插拔 A/B（cap3 vs cap4，n=8，ic=1，多 reps，md5 全程逐字节一致）：

| 模型 | cap3（当前） | cap4（4-way 融合） | 变化 |
|---|---:|---:|---:|
| Qwen3.5-0.8B | 138.4 ms | 140.3 ms | **+1.4%（回退）** |
| Qwen3.5-2B | 198.6 ms | 200.1 ms | **+0.8%（回退）** |
| Qwen3.5-4B | 481.9 ms | 491.4 ms | **+2.0%（回退）** |

**结论：否决。** 把两个极小的 N=16 SSM gate FC 与大的 qkv(6144)/z(2048) concat 成 N=8192 的单个 gemm，多出的 `VariadicSplit`（8192 宽）开销 + 大 gemm 的 tiling 没有更优，净回退（4B 上 −9.5ms 最明显）。qkv/z 单独已被 oneDNN 高效 tile，融合无增益。**现有 `max_num_fcs_to_fuse=3` 对本模型是恰当的，不要提高。** md5 全程一致（融合数值安全，只是不划算）。

### 4.9.3 新角度 B（分析）：sdpa_micro —— dead-end

`sdpa_micro`（2.7%，6/24 个 full-attention 层）外壳是可编辑 OCL，但**核心两个 GEMM（`ugemm_kq`/`ugemm_vs`）是 oneDNN gemmstone micro-kernel**（与 `gemm_kernel` 同类，经 `selectGEMM`+`generateShim` 生成），不可改。config 表（`choose_config_xe3p`）对 head≤128 无条件返回固定 tile，理论上可调 tile，但**没有 md5-无损的 lever**（改 tile 改累加顺序）。**dead-end。**

### 4.9.4 新角度 C（分析）：generic_eltwise_ref / SSM 尾部搬运 —— marginal

`generic_eltwise_ref`（2.6%）是 GDN 的 gating multiply（`2048×1002`）。与 opt-6 的动机类似：**向量化 eltwise kernel（`vload8`/`blocked_opt`）不支持 dynamic shape，只有 `eltwise_kernel_ref` 声明了 `EnableDynamicShapesSupport`**，所以 dynamic 走标量 ref。理论上可仿 opt-6 给 `vload8` 加 dynamic support。**但关键差异**：`generic_eltwise_ref` 已经是 **100% occupancy**（activation_ref 当初不是），说明它是**访存带宽 bound**（简单逐元素读写），向量化对带宽 bound 的 kernel 收益很小——与 rms（§4.6.2）同一教训。`strided_slice_ref`(2.1%)/`concatenation`(0.6%) 同为 100%-occ 的必要数据搬运。要真正省掉它们需要把 GDN 前后的 slice/concat/eltwise 融进一个有状态 fused op（类似 SDPA 文档 §9 的 conv-sandwich 设想），是**数天级立项**、且大概率破 md5，不属于本轮低风险局部优化。**marginal / 需立项。**

### 4.9.5 rms 二次复核 —— 维持 §4.6.2 结论

重读 `rms_gpu_bfyx_opt.cl`：主实例 `16×16032`（q/k head-norm，16 heads×1002 tok×head_dim 128）用 LWS=16（`get_item_num_and_lws` 因 `itemsNum>8` 停在 128/16=8）。本轮考虑「加 worker 数」的新角度，但机理上它是 **16032 行 × 每行 128 个 f16（≈4MB 读）的访存 bound**，加 worker 不减少总读字节数；与 §4.6.2「提占用率无墙钟收益」一致。**维持：无 md5-无损收益。**

### 4.9.6 本轮总结

| 角度 | 类型 | 结论 | 证据 |
|---|---|---|---|
| 4-way FC 融合（cap 3→4） | 实测 | **否决（负收益 +0.8~2.0%）** | 交错 A/B ×3 尺寸，md5 一致 |
| sdpa_micro tile | 分析 | dead-end | 核心是 oneDNN gemmstone，无 md5-无损 lever |
| eltwise/slice/concat 向量化 | 分析 | marginal（100%-occ 带宽 bound） | 与 rms 同教训；真省需 GDN fused op 立项 |
| rms 加 worker | 分析 | 无收益（访存 bound） | 维持 §4.6.2 |
| gemm / GDN | 复核 | 不可改 / 负期望 | §4.6.1 / §6.1 |

> **诚实结论：opt-3 + opt-5 + opt-6（+ e2e 的 opt-7）已是 356H PA prefill 在「md5 逐字节无损 + 不重写算法 + 不动 oneDNN」约束下的实用上限。** 本轮验证的新角度中，唯一能安全落地的（4-way FC 融合）反而负收益，已否决。真正的剩余大头（gemm 50% + GDN 20%）分别是 oneDNN 硬上限和串行递归，只有「放宽精度验收 + GDN chunkwise 重写」或「改 oneDNN XMX tiling」两条高成本立项路径能再动，均超出局部优化范畴（见 §6.1）。

## 4.10 【356H】第五轮 fresh profiling：当前二进制复核 + eltwise probe（2026-07-09）

> 按「不要被已有结论限制」的要求，本轮重新从当前工作树与二进制出发：重建 GPU plugin + GenAI，使用 `source /home/chaofan/envs/qwen_env/bin/activate`，重新跑 2B PA benchmark 和 vtune gpu-hotspots。原始 profile 已保存为 `repro_356H_20260701/vtune/vt_20260709_opt7_2b.csv`。

### 4.10.1 fresh benchmark + vtune

当前 2B（opt-3+5+6，GenAI opt-7 已安装，PA，GPU，真实 1000-token prompt，`n=5/ic=1`）基准：

| 指标 | 数值 |
|---|---:|
| e2e-TTFT / Prefill | 247.9 ms |
| `first_token_infer` | 198.0 ms |
| e2e | 249.6 ms |
| md5 | `dd753679...` ×6 |

vtune（`n=3`，task time sum≈785ms，约 3 measured + warmup 采样口径）top kernels：

| kernel | total time | 占比 | GWS/LWS | 结论 |
|---|---:|---:|---|---|
| `gemm_kernel`（三大主实例合计） | 0.369s | **47.0%** | `256×8×1 / 32×8×1` | oneDNN XMX GEMM，仍是硬上限 |
| `paged_gated_delta_net_opt` | 0.161s | **20.4%** | `1×16×512 / 1×1×16` | 串行 SSM，100% occ，低风险 dispatch 无空间 |
| `paged_causal_conv1d_ref` | 0.035s | 4.45% | `1×6144×3 / 1×256×1` | opt-5 生效 |
| `rms_gpu_bfyx_opt`（主实例） | 0.032s | 4.03% | `16×16032×1 / 16×1×1` | 维持访存延迟 bound 结论 |
| `dynamic_quantize_gpu_opt`（主实例） | 0.021s | 2.65% | `16×48×1002 / 16×32×1` | opt-3 后剩余必要量化 |
| `activation_opt`（主实例） | 0.017s | 2.19% | `1539072×1×1 / 1024×1×1` | opt-6 生效 |
| `generic_eltwise_ref`（三项合计） | 0.020s | 2.5% | `2048×1×1002 / 1024×1×1` | 本轮 probe 对象 |
| `sdpa_micro` | 0.022s | 2.74% | `128×256×1 / 16×32×1` | 维持 gemmstone dead-end |

与 §4.9 一致：当前低风险 GPU kernel 空间已经是尾部小项，主瓶颈仍是 gemm/GDN。

### 4.10.2 新实测 probe：dynamic `eltwise_simple_vload8` —— 未命中目标，已回退

**动机**：`generic_eltwise_ref` 仍有约 2.5%，而 `activation_ref` 曾通过 opt-6 切到已有向量化 kernel 获益。因此尝试给 `eltwise_simple_vload8` 增加 dynamic-shape support 和 `OPTIONAL_SHAPE_INFO_ARG`，希望 dynamic eltwise 从 `generic_eltwise_ref` 切到 vload8。

**改动 probe**（已回退）：
- `cl_kernels/eltwise_simple_vload8.cl`：kernel 参数加入 `OPTIONAL_SHAPE_INFO_ARG`。
- `kernels/eltwise/eltwise_kernel_vload8.cpp`：`GetSupportedKey()` 增加 `EnableDynamicShapesSupport()`。

**结果**：
- 构建通过。
- 2B benchmark 可运行，md5 多重集与 baseline 一致（`dd753679...×6`）。
- 但 vtune 中目标 kernel **仍然是 `generic_eltwise_ref_*`**，没有出现 `eltwise_simple_vload8`；说明 selector 仍因其他条件（动态 shape 下 layout/size/broadcast/pitch 等约束）未选择 vload8。
- benchmark 上的约 1ms 波动不可归因于该 probe，因为目标 kernel 未切换。

**结论：不落地，已回退。** 若要继续推动 eltwise/slice/concat，需要进一步做图融合或改 dynamic shape selector/JIT 的更深层适配；考虑到它们合计只是尾部、且多为 100% occ 数据搬运，预期收益很小。

### 4.10.3 本轮结论

本轮没有发现新的低风险 patch。当前实用栈仍是：

- GPU plugin：opt-3 + opt-5 + opt-6；不启用 opt-4。
- GenAI text-only e2e：额外叠 opt-7。

想要明显超过当前 356H e2e ~6% 的收益，只剩两类高成本方向：

1. **GDN chunkwise / 并行递归重写**：可能动到 20% 级 hotspot，但会改变浮点累加顺序，几乎必然破 byte-level md5，需要重新定义正确性验收。
2. **oneDNN XMX INT4 GEMM tiling / kernel 级优化**：当前最大 hotspot，但在 oneDNN ngen/gemmstone 层，不是 OpenVINO GPU OCL 插件的局部 patch。

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

# 2) 应用 356H PA 推荐 GPU 优化（opt-3 + opt-5 + opt-6；不要 opt-4）+ 重建 GPU plugin
cd ../benchmark/openvino
git apply qwen_patch/opt3_dynamic_quantize_reuse.patch
git apply qwen_patch/opt5_conv1d_token_tiled.patch
git apply qwen_patch/opt6_dynamic_activation_opt.patch
bash rebuild_gpu_plugin.sh

# 2b) 可选：应用 GenAI text-only VLM e2e 小优化 opt-7，并重建/安装 openvino_genai
cd ../openvino.genai
git apply ../openvino/qwen_patch/opt7_genai_qwen35_text_embedding_copy_elision.patch
cmake --build build --target openvino_genai -j$(nproc)
cp -av build/openvino_genai/libopenvino_genai.so* ../openvino/install/runtime/lib/intel64/

# 3) A/B（n=8，PA，对比 first_token_infer_mean_ms + iterations[].result_md5；prefill ic=1 + decode ic=128）
cd ../../model_bench
./build-local/model_bench vlm -m ../benchmark/models/Qwen3.5-2B-INT4 -d GPU -n 8 -ic 1 \
  -pf ./prompts/qwen3.5_1k.jsonl --no-monitor -lc /tmp/pa.json -r /tmp/x.csv -rj /tmp/out.json
```

---

## 6. 还能继续做什么

> 本节前半保留 358H / opt-4+opt-3 时代的剩余 hotspot 记录，作为为什么不动 GDN/gemm/rms 的依据。对 356H 当前最终栈（opt-3+opt-5+opt-6），`activation_ref` 已由 §4.7 覆盖，剩余低风险空间主要仍是 gemm/GDN/rms 之外的尾部小项；gemm/GDN/rms 的负结论不变。

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
| `activation_ref` | 6.0%，occ 100% | 356H 上由 opt-6 覆盖：简单 dynamic activation 改走 `activation_opt`。358H 尚未补测 opt-6。 |
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
> **历史总结（358H / opt-4+opt-3 阶段）：当时低风险 dispatch 类优化已挖到底，opt-4（conv1d）+ opt-3（DQ）是不重写算法前提下的 PA 实用上限（2B infer-TTFT −6.5% / e2e −4.4%）。356H 当前 GPU kernel 结论已由 §4.7 更新为 opt-3 + opt-5 + opt-6；text-only e2e 另由 §4.8 的 opt-7 小幅更新。**

---

## 附：产物清单

| 文件 | 说明 |
|---|---|
| `src/plugins/intel_gpu/src/graph/impls/ocl_v2/paged_causal_conv1d_ref.{cl,cpp}` | opt-4：token 并行 conv1d（kernel 窗口重建 + GWS 加 token 维） |
| `src/plugins/intel_gpu/src/plugin/transformations/dynamic_quantize_fully_connected.cpp` + `src/graph/dynamic_quantize.cpp` | opt-3：共享激活的 FC 复用 DynamicQuantize（PA 下仍有效） |
| `opt_pa_archive/pa_opt4_conv1d.patch` | opt-4 git patch（基于 pristine HEAD `ede283a88e`，2 文件） |
| `opt_pa_archive/pa_opt3_dq_reuse.patch` | opt-3 git patch（基于 pristine HEAD，2 文件；与 opt-4/opt-5 文件不重叠，可一起 apply） |
| **`opt_pa_archive/pa_opt5_conv1d_tiled.patch`** | **opt-5：token-TILED conv1d（2 文件，与 opt-4 互斥）。分块因子 G 自适应（按 GPU 线程预算 EU×threads/eu + 序列长度：356H→3、大 iGPU 更大、decode→1），`OV_CONV1D_TOKEN_GROUPS` 可强制覆盖、`OV_CONV1D_DEBUG=1` 打印选中的 G** |
| **`benchmark/openvino/qwen_patch/opt6_dynamic_activation_opt.patch`** | **【356H】opt-6：dynamic activation 走 `activation_opt`（2 文件）；叠加 opt-3+opt-5 后，356H infer/e2e TTFT 为 0.8B 137.3/185.7ms、2B 196.9/244.8ms、4B 481.2/529.5ms** |
| **`benchmark/openvino/qwen_patch/opt7_genai_qwen35_text_embedding_copy_elision.patch`** | **【356H】opt-7：GenAI Qwen3.5 text-only VLM 去掉 text embedding → inputs_embeds 的冗余 memcpy；最终 e2e TTFT 为 0.8B 185.1ms、2B 243.9ms、4B 527.7ms** |
| `benchmark/openvino/qwen_patch/README.md` | 统一 patch 目录说明：按 PA/SDPA、358H/356H 平台列出推荐 patch、机制、infer/e2e 数据与正确性口径 |
| `repro_356H_20260701/` | 【356H】全部补测：`SUMMARY_356H.md`、`results.tsv`、`pa_sweep*.tsv`、`opt5_final_AB*.tsv`、`vtune/`（B0/opt4/opt5G3/opt5G4/best2b 逐 kernel CSV）、`so_opt5.so`（opt3+opt5 插件，可热插拔安装） |
| `opt_pa_archive/opt4_conv1d_AB_cleanbase.txt` | opt-4 单独 A/B + md5 |
| `opt_pa_archive/opt4_opt3_combined_AB.txt` | opt-4+opt-3 叠加双口径（infer/e2e TTFT）+ md5 汇总 |
| `opt_pa_archive/vtune_PA_prefill_{B0_pristine,B1_opt4}_2b.csv` | opt-4 前后的 PA prefill vtune（占用率对账） |
| `reeval_PA_20260630/vtune_{PA,SDPA}_2b_kernels.csv` | PA vs SDPA kernel 结构对比（§2 根因） |

---
*基于本地源码构建的 OpenVINO 2026.2.1 / model_bench(C++) / vtune 2026.2，Panther Lake + Arc B390。opt-4 已实测、正确性以 128-token 贪心 md5（含 prefill ic=1）校验逐字节一致；所有数据均在 pristine（不含 SDPA opt-1/2/3）基础上、PA backend、n=8、real prompt(~1000 tok)、GPU 测得。*
