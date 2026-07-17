# 356H 复现数据汇总（2026-07-01，2026-07-02 opt-6，2026-07-03 opt-7 追加）

机器：Intel Core Ultra 7 **356H**（对照文档原始数据为 358H）。
方法：与文档一致 —— `model_bench`(C++) 链接本地源码构建 OpenVINO 2026.2.1（HEAD `ede283a88e`），
GPU，`-n 8`（+1 warmup），真实 1000-tok prompt（`prompts/qwen3.5_1k.jsonl` / `qwen3_1k.jsonl`）。
TTFT = 首 token 延迟（prefill），与 `-ic` 无关，故 TTFT 用 `ic=1` 纯 prefill 测；正确性另用 `ic=128` 校验 decode。

6 个插件构建态（改 6 个 opt 源文件 + 增量重建 GPU 插件 ~6s 每次）：
pristine / +opt1 / +opt1+2 / +opt1+2+3 / +opt4 / +opt4+opt3。
PA 侧因单项收益很小且 opt4 出现回退，全部用「.so 热插拔交错 A/B」（2 reps 均值，rep 间抖动 <3ms）规避热漂移。

---

## 1. PA vs SDPA 基线（stock pip OV，官方 runner，n=10，ic=512，来自 results/*.csv）
infer-TTFT（`1st_infer_latency_avg`）：

| 模型 | PA | SDPA | PA vs SDPA |
|---|---:|---:|---:|
| Qwen3.5-0.8B-INT4 | 151.3 | 305.8 | −50.5% |
| Qwen3.5-2B-INT4 | 215.5 | 351.0 | −38.6% |
| Qwen3.5-4B-INT4 | 510.0 | 791.7 | −35.6% |

e2e-TTFT（`1st_latency_avg`）：0.8B 193.9/348.3 (−44.3%)、2B 263.8/398.5 (−33.8%)、4B 563.7/843.6 (−33.2%)。
（本地 pristine 构建实测与 stock-pip 基线高度一致：PA 2B 214.0 vs 215.5、SDPA 2B 348.9 vs 351.0，验证方法可信。）

---

## 2. SDPA doc（opt-1/2/3，A/B/C/D′，ic=1，n=8，单构建态）
infer-TTFT (ms)：

| 模型 | A 基线 | B(+o1) | C(+o12) | D′(+o123) | o1 | o2 | o3 | A→D′ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B | 292.3 | 289.8 | 255.9 | 251.1 | −2.4 | −33.9 | −4.8 | **−14.1%** |
| Qwen3.5-2B | 348.9 | 344.4 | 315.3 | 309.7 | −4.5 | −29.1 | −5.6 | **−11.3%** |
| Qwen3.5-4B | 800.6 | 788.1 | 731.5 | 720.4 | −12.5 | −56.6 | −11.1 | **−10.0%** |
| Qwen3-1.7B dense | 183.0 | — | — | 180.7 | — | — | — | −1.3%（不触发） |

e2e-TTFT (ms)：0.8B 336.7→295.4 (−12.3%)、2B 393.3→354.4 (−9.9%)、4B 845.0→766.1 (−9.3%)、dense 183.4→181.1 (−1.3%)。

正确性：所有 A/B md5 多重集逐一相同（见 §4）。

---

## 3. PA doc（opt-4 / opt-3，交错热插拔 A/B，ic=1，n=8，2 reps 均值）
infer-TTFT (ms)：

| 模型 | B0 pristine | +opt4 | +opt4+3 | opt3-only(=opt123) | opt4% | opt4+3% | opt3% |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B | 150.8 | 161.4 | 157.1 | 148.5 | **+7.0%** | +4.2% | **−2.1%** |
| Qwen3.5-2B | 214.0 | 224.0 | 216.2 | 207.6 | **+4.7%** | +1.0% | **−3.2%** |
| Qwen3.5-4B | 506.1 | 525.7 | 515.3 | 495.9 | **+3.9%** | +1.8% | **−2.1%** |

e2e-TTFT (ms)：0.8B B0 196.3 / opt4 206.6 / opt4+3 201.9 / opt3-only 193.9；
2B 259.8 / 269.6 / 261.7 / 253.7；4B 552.2 / 571.8 / 561.2 / 542.4。

**关键 356H 发现（与 358H 相反）：opt-4（token-parallel conv1d）在 356H 上是净回退（+4~7%），
而 358H 上是 −1.3~4.4% 的收益。opt-3 在 356H 上仍是有效收益（−2~3.2%），与 358H 一致。**
故 356H 上 PA 的最优实用配置是「仅 opt-3」，opt-4 不应启用。opt-4 数值仍逐字节无损（md5 一致），
只是本机 GPU 上 token 并行的 dispatch 没带来收益反而变慢（推测与 356H iGPU 的 EU 数/占用特性不同有关）。

---

## 4. 正确性（md5 多重集，A/B 同机对比）
356H GPU 存在**可复现的**逐迭代 md5 非单值现象（同一 prompt、permutation OFF 时 8 次迭代产生若干不同 md5，
但跨进程完全可复现、逐位置有序一致）。因此正确性判据用「基线 vs 优化态的 md5 多重集是否逐一相同」（文档 §7.2 口径）。

- SDPA 2B ic=128：pristine 与 opt123(D′) 的 128-token decode md5 多重集 **完全相同**（8 个不同 md5，集合与计数一致）。
- PA 2B ic=128：pristine 与 opt4+3 的 128-token decode md5 多重集 **完全相同**（7 个不同 md5）。
- prefill ic=1：各态多重集在所有尺寸上与其基线一致（2B/4B 为单一 md5 `dd753679`/`e067b6f4`；0.8B 为固定三值多重集）。

结论：opt-1/2/3/4 在 356H 上均**数值无损**；opt-4 的问题纯粹是本机性能回退，非正确性问题。

原始数据文件：`results.tsv`（单构建态全量）、`pa_sweep.tsv`（B0/opt4/opt4+3 交错）、`pa_sweep2.tsv`（B0/opt123 交错）。

---

## 5. vtune 归因 + opt-5（token-TILED conv1d，2026-07-01 新增）

vtune gpu-hotspots（xe 驱动，**32 XVE × 10 threads = 320 HW threads**），2B PA prefill，n=5（/5 为每 iter）。

**opt-4 回退归因（conv1d kernel）：**

| 状态 | GWS | occ | 限制因素 | conv1d ms/iter |
|---|---|---:|---|---:|
| pristine 串行 | 1×6144×1 | 60% | Work Size | 14.13 |
| opt-4 全并行 | 1×6144×1002 | 100% | — | 24.71 |
| opt-5 分块 G=3 | 1×6144×3 | 100% | — | 10.45 |

- 串行占用率 60% = 6144/32=192 threads ÷ 320。opt-4 撑满占用率但每 work-item 冗余读 K=4 窗口 → global 读 ×K → memory-bound → +75%。op-sum +9.3ms/iter，对上墙钟 2B +10ms/iter 回退。
- 根因：opt-4 在 358H 大 iGPU「占用率红利 > 冗余读」净赚；356H 小 iGPU（32 XVE）「冗余读 > 占用率红利」净亏。

**opt-5 = token-TILING**：GWS `{seq,hidden,G}`，每 work-item 串行扫一块（块内滑窗、零冗余读），块起点一次性重建 K-1 历史。G≥2→100% 占用，额外读仅 `hidden×G×(K-1)`（可忽略）。数学 bit 级无损。G 扫描 conv1d 谷底在 **G=3**（10.45ms）。

**opt-5 whole-model（交错 A/B，n=8，ic=1，3 reps 均值，vs opt-3 与 pristine）：**

| 模型 | pristine | opt-3 | opt-3+opt-5 | opt-3+5 vs pristine | opt-5 vs opt-3 |
|---|---:|---:|---:|---:|---:|
| 0.8B | 151.3 | 149.0 | 145.3 | **−4.0%** | −2.5% |
| 2B | 214.3 | 207.3 | 204.8 | **−4.4%** | −1.2% |
| 4B | 505.0 | 494.5 | 496.2 | −1.7% | +0.3%(噪声) |

e2e：0.8B 196.7→190.6(−3.1%)、2B 260.7→250.6(−3.9%)、4B 551.9→542.1(−1.8%)。
正确性：三尺寸 prefill(ic=1)+decode(ic=128) md5 多重集与 opt-3 逐一相同（0.8B 5元、2B decode 7元、4B 3元全一致）。

**356H 阶段性落地建议（opt-6 前）：`pa_opt3_dq_reuse.patch` + `pa_opt5_conv1d_tiled.patch`（不用 opt-4）。** opt-6 追加后最终建议见 §6。

**剩余 hotspot（opt-3 态，2B，op-sum≈248ms/iter）：** gemm 47.3%(occ40,不可改) / GDN 19.5%(occ100,负期望) / **rms_gpu_bfyx_opt 7.4%(occ40 SLM 受限，已识别但跨模型共享、暂缓)** / conv1d 4%(opt-5 已满) / DQ 4.9%(opt-3 已满)。

原始数据：`vtune/vt_{B0,opt4,opt5G3,opt5G4,best2b}.csv`、`opt5_final_AB*.tsv`；opt-5 补丁 `../opt_pa_archive/pa_opt5_conv1d_tiled.patch`；插件 `so_opt5.so`。

---

## 6. opt-6（dynamic activation 走 activation_opt，2026-07-02 追加）

继续分析 opt-3+opt-5 态后，剩余 profile 里仍有大尺寸 `activation_ref`：`1×1×6156288` 和 `1×1×2052096`，合计约 6–7% task time。`activation_opt` 已有 float4 线性向量化 kernel，但此前没有 dynamic-shape support，且 OpenCL 签名没有接收 shape-agnostic dispatch 传入的 `OPTIONAL_SHAPE_INFO_ARG`。

opt-6 改动：
- `activation_kernel_opt.cpp`：给 `ActivationKernelOpt` 开启 dynamic-shape support；dynamic 路径只允许「输入/输出 layout 相同、无 fused ops、无 runtime activation 参数」。
- `activation_opt.cl`：参数列表加入 `OPTIONAL_SHAPE_INFO_ARG`。

**最终 whole-model（356H，PA，GPU，真实 1000-token prompt，ic=1）：**

| 模型 | pristine infer | opt-3+5 infer | **opt-3+5+6 infer** | final vs pristine | opt-6 增量 | pristine e2e | opt-3+5 e2e | **opt-3+5+6 e2e** | final e2e vs pristine |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.8B | 151.3 | 145.3 | **137.3** | **−9.2%** | −5.5% | 196.7 | 190.6 | **185.7** | **−5.6%** |
| 2B | 214.3 | 204.8 | **196.9** | **−8.1%** | −3.8% | 260.7 | 250.6 | **244.8** | **−6.1%** |
| 4B | 505.0 | 496.2 | **481.2** | **−4.7%** | −3.0% | 551.9 | 542.1 | **529.5** | **−4.1%** |

0.8B 最终态使用 n=8 复测（infer/e2e = 137.32/185.68ms）；2B/4B 使用 n=5（196.92/244.76ms、481.18/529.54ms）。md5 多重集保持一致：0.8B 为 `cae8d14e×7 + 514faf5a×1 + dc27db97×1`（含 warmup/8 measured 输出），2B/4B 分别为稳定单值 `dd753679...` / `e067b6f4...`，与此前 PA 基线一致。

opt-6 后尝试重新采 vtune，两次触发 vtune collector 内部断言并生成损坏结果，因此 post-profile 不作为证据；本节收益以 `model_bench` A/B 和 md5 校验为准。opt-6 patch 已统一放在 `../../openvino/qwen_patch/opt6_dynamic_activation_opt.patch`。

**356H 当前 GPU kernel 落地建议：`opt3_dynamic_quantize_reuse.patch` + `opt5_conv1d_token_tiled.patch` + `opt6_dynamic_activation_opt.patch`，不要启用 opt-4。** text-only VLM e2e 可继续叠加 §7 的 GenAI opt-7。

---

## 7. opt-7（GenAI text-only embedding copy-elision，2026-07-03 追加）

重新 profiling opt-3+5+6 后，2B 当前 GPU task 主要是 gemm(~49%) / GDN(~20%) / rms(~7%) / DQ(~5%) / conv1d(~4%) / activation_opt(~3%)；opt-3、opt-5、opt-6 均已按预期生效。e2e 仍比 `first_token_infer` 多约 45ms，继续从 GenAI host/embedding 路径查找小优化。

发现 Qwen3.5 text-only VLM 复用 `InputsEmbedderQwen3VL::get_inputs_embeds()`：无 image/video 时，text embedding model 输出 `text_embeds` 后仍分配同 shape 的 `inputs_embeds` 并整块 `memcpy`。opt-7 只在 `images.empty() && videos.empty()` 时直接返回 `text_embeds`，image/video merge 路径不变。尝试 remote tensor 版本会因 continuous-batching 后续访问 `Tensor::data()` 抛 `Not Implemented`，已放弃。

**最终 whole-model（356H，PA，GPU，真实 1000-token prompt，ic=1）：**

| 模型 | opt-3+5+6 infer/e2e | **+opt-7 infer/e2e** | opt-7 e2e 增量 | 最终 e2e vs pristine |
|---|---:|---:|---:|---:|
| 0.8B | 137.3 / 185.7 | **137.7 / 185.1** | −0.3% | **−5.9%** |
| 2B | 196.9 / 244.8 | **197.3 / 243.9** | −0.3% | **−6.4%** |
| 4B | 481.2 / 529.5 | **480.8 / 527.7** | −0.3% | **−4.4%** |

md5 多重集保持一致：0.8B `cae8d14e×4 + 514faf5a×1 + dc27db97×1`，2B `dd753679...×6`，4B `e067b6f4...×6`。opt-7 patch：`../../openvino/qwen_patch/opt7_genai_qwen35_text_embedding_copy_elision.patch`。

**356H 当前最终建议：GPU 插件用 opt-3 + opt-5 + opt-6；GenAI text-only VLM e2e 再叠 opt-7。**

---

## 8. 2026-07-09 fresh profiling 复核 + eltwise probe

按最新源码/二进制重新构建 GPU plugin 与 GenAI 后，重跑 2B PA（GPU，真实 1000-token prompt，`n=5/ic=1`）：

| 指标 | 数值 |
|---|---:|
| TTFT / Prefill | 247.9 ms |
| `first_token_infer` | 198.0 ms |
| E2E | 249.6 ms |
| md5 | `dd753679...×6` |

fresh vtune CSV：`vtune/vt_20260709_opt7_2b.csv`。

top kernels（task time sum≈785ms）：gemm 主实例合计 47.0%、`paged_gated_delta_net_opt` 20.4%、`paged_causal_conv1d_ref` 4.45%、主 `rms_gpu_bfyx_opt` 4.03%、主 `dynamic_quantize_gpu_opt` 2.65%、主 `activation_opt` 2.19%、三项 `generic_eltwise_ref` 合计约 2.5%、`sdpa_micro` 2.74%。结论与 §6/§7 一致：opt-3/5/6 均已生效，主瓶颈仍是 oneDNN GEMM + GDN。

本轮新 probe：给 `eltwise_simple_vload8` 增加 dynamic-shape support + `OPTIONAL_SHAPE_INFO_ARG`，尝试让 dynamic `generic_eltwise_ref` 切到 vload8。

- 构建通过，2B benchmark 可运行，md5 与 baseline 一致。
- 但 vtune 仍显示目标是 `generic_eltwise_ref_*`，没有切到 `eltwise_simple_vload8`。
- 因此该 probe 未命中目标，已回退，不落地。

**当前结论不变：GPU 插件 opt-3 + opt-5 + opt-6，GenAI text-only e2e opt-7。若要再有大幅提升，需要 GDN chunkwise 或 oneDNN GEMM 级别立项。**
