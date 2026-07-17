# Qwen3 / Qwen3.5 (≤4B) 低 TTFT 加速方案分析报告

> 任务来源：leader 要求调研「qwen3/qwen3.5 2B 以下的 LLM，能否用 llama.cpp 或其他方式跑得更快、主要是 TTFT 更短」；后续已补充 4B 模型和 TPOT/TPS decode 指标。
> 适用硬件：本机 Intel Core Ultra X7 358H（Panther Lake-H）+ Arc B390 iGPU + AI Boost NPU，Ubuntu 24.04。
> 撰写目的：给刚接触此类任务的同学一份「怎么入手、有哪些路、怎么一步步做」的完整分析。
> 结论可信度：本报告的事实性结论大多经过本机源码/数据交叉验证或联网核对，并在文末附【证据与可信度】。凡是"推理而非实测"的地方都会明确标注。

---

## 0. 一句话结论（先看这个）

> **对 ≤4B 的 Qwen3/Qwen3.5，本机已经在用的 OpenVINO GenAI + iGPU(GPU) 路径，就是当前 TTFT 最优解；几乎没有"换个框架就明显更快"的捷径。**
> - **【已实测，2026-06-23/24】llama.cpp Vulkan / 本机源码 SYCL F16 / OpenVINO backend 均不能降低 TTFT**。dense Qwen3 最快的 llama.cpp 路径是 OpenVINO backend GPU，但仍慢于原生 OpenVINO GenAI GPU（0.6B：77ms vs 35.6ms；1.7B：118ms vs 63.1ms；4B：275ms vs 147.6ms）。Qwen3.5 最快的 llama.cpp 路径是本机源码 SYCL F16（0.8B：301ms；2B：399ms；4B：897ms），仍慢于原生 OpenVINO GenAI GPU（纯推理 131/157/335ms）。TPOT/TPS 方面，SYCL F16 decode 可接近 OpenVINO GenAI，Qwen3.5-0.8B 甚至略快，但首 token 延迟仍明显更差。复现见 [llamacpp_TTFT对比复现文档.md](llamacpp_TTFT对比复现文档.md)。
> - **ipex-llm / vLLM-XPU / IPEX-XPU / MLC 基本是死路**（要么项目归档、要么面向吞吐而非 TTFT、要么不支持 Qwen3.5）。
> - **Qwen3.5 在 NPU 上的灾难性 TTFT（~222s）不是硬件极限，而是 OpenVINO 的 NPU 插件还没实现 `GatedDeltaNet` 融合算子**（CPU/GPU 插件已实现）。这是上游要做的事，不是你能在配置层面解决的。对 Qwen3.5，老老实实用 iGPU。

---

## 1. 先把问题想清楚：TTFT 到底由什么决定

这是入手这个任务最重要的一步。**TTFT（Time-To-First-Token，首 token 延迟）= 处理完整个输入 prompt 并吐出第 1 个 token 所花的时间，几乎等同于 prefill（预填充）阶段的耗时。** 它和后续的 decode（每 token 速度，TPOT）是两个完全不同的东西，优化手段也不同。

| 阶段 | 计算特征 | 瓶颈 | 谁影响它 |
|---|---|---|---|
| **Prefill（=TTFT）** | 一次性并行处理 ~940 个 token，是**计算密集（compute-bound）**的大矩阵乘 | **算力**（GPU 矩阵引擎吞吐） | prompt 长度、模型大小、矩阵乘 kernel 质量、能否并行 |
| **Decode（TPOT）** | 每次只算 1 个 token，是**访存密集（memory-bound）** | **内存带宽** | 量化、KV-cache 精度、内存带宽 |

**这对本任务的三个直接推论：**

1. **要降 TTFT，本质是要让那一次大矩阵乘跑得更快**——靠的是矩阵引擎（Intel iGPU 的 XMX）和 kernel 质量，而不是量化能省多少内存。所以很多"加速 LLM"的技巧（INT4、KV-cache 量化）其实主要帮 decode，对 TTFT 帮助有限。
2. **prompt 越短，TTFT 越低**，近似线性。所以"控制/截断输入长度"是任何方案都通用的一招。
3. **设备选择决定上限**：在本机，prefill 的算力排序是 **iGPU ≫ NPU ≫ CPU**（见第 7 节实测）。任何框架，只要最终把 prefill 放在 iGPU 上，TTFT 才有竞争力。

> 给新人的提醒：评估任何"更快的方案"前，先问三件事——它能不能用上 iGPU 的 XMX 矩阵引擎？它的 prefill 是不是并行的？它支不支持你要跑的模型架构？只要有一个否，TTFT 就不可能好。

---

## 2. 你当前的基线（必须先有数，才能谈"更快"）

数据来自 `benchmark/results/benchmark_qwen3_official.csv` 和 `benchmark/results/benchmark_qwen3_5_official.csv`，~940 token 输入 / 512 输出，INT4。**这是所有"更快"的对照基准，先记牢。**

### Qwen3（标准 dense Transformer）TTFT（首 token，ms）

| 模型 | CPU | NPU | **GPU(iGPU)** |
|---|---:|---:|---:|
| Qwen3-0.6B-INT4 | 1125 | 149 | **35** |
| Qwen3-1.7B-INT4 | 2903 | 290 | **63** |
| Qwen3-4B-INT4 | 7291 | 703 | **148** |

### Qwen3.5（hybrid 线性注意力 + 全注意力）TTFT（首 token，ms）

| 模型 | CPU | NPU | **GPU(iGPU)** |
|---|---:|---:|---:|
| Qwen3.5-0.8B-INT4 | 2799 | **~222000 ⚠️** | **171** |
| Qwen3.5-2B-INT4 | 4369 | OOM 被杀 | **198** |
| Qwen3.5-4B-INT4 | ~11000 | — | 335 |

**两个关键观察（决定了整份报告的方向）：**
- **iGPU 已经非常快**（dense 35–148ms，hybrid 纯推理 131–335ms / 端到端首 token 171–378ms）。要"更快"，对手是它，不是 NPU/CPU。
- **Qwen3.5 在 NPU 上 ~222s 是坏掉的**（比 GPU 慢约 1300 倍），原因见第 6 节。dense Qwen3 在 NPU 上正常（149–703ms）。

---

## 3. 关键结论速查表（给 leader 看的决策表）

| 方案 | 能用 iGPU? | 能用 NPU? | Qwen3 dense | Qwen3.5 hybrid | TTFT 实测/判断 vs 当前 GPU | 投入产出 | 建议 |
|---|---|---|---|---|---|---|---|
| **OpenVINO GenAI + GPU（现状）** | ✅ 最佳 | ✅(dense) | ✅成熟 | ✅纯推理 131–335ms | **基准** | — | **保持，作为主力** |
| **OpenVINO 栈内调参** | ✅ | ✅ | ✅ | ✅ | 持平（NPU dense 可省 10–25% padding） | 低 | 值得做小幅优化 |
| **llama.cpp (SYCL/Vulkan, iGPU)** | ✅ | ❌* | ✅成熟，已测到 4B | ✅已测到 4B，可跑且输出非乱码 | **TTFT 明显更慢**（dense SYCL F16 248/374/941ms；Qwen3.5 SYCL F16 301/399/897ms），decode 接近但不能解决 TTFT | 已完成 | ❌ 不作为低 TTFT 路径 |
| **llama.cpp (OpenVINO 后端)** | ✅ | ✅ | ✅GPU 可跑但仍慢 1.9–2.2×；4B 为 275ms vs 147.6ms；NPU/CPU 很慢 | ⚠️GPU/NPU warmup 失败；CPU 可跑但秒级 | **无收益** | 已完成 | ❌ 不值得继续投入 |
| **ipex-llm** | ⚠️旧 | ❌(仅Win) | ⚠️混乱 | ❌ | 更差 | 中高 | ❌ 死路（项目已归档） |
| **ONNX Runtime GenAI + OpenVINO EP** | ✅(=OV) | ✅(=OV) | ✅ | ❌(同OV NPU) | 持平（底层就是OV） | 中 | 仅当需要 ORT 生态时 |
| **MLC-LLM / TVM (Vulkan)** | ✅ | ❌ | ✅ | ❌ | 多半更差 | 中高 | 仅作"独立运行时"佐证 |
| **vLLM XPU** | ⚠️未验证 iGPU | ❌ | ✅ | ⚠️仅CUDA | 更差（面向吞吐） | 高 | ❌ 死路 |
| **PyTorch + IPEX-XPU** | ✅ | ❌ | ✅ | ⚠️ eager 慢 | 更差（eager 开销） | 中高 | ❌ 死路 |

\* llama.cpp 的 SYCL/Vulkan 后端**只能跑 iGPU，不能用 NPU**；唯一能碰 NPU 的是它的 OpenVINO 后端，那本质就是套了层壳的 OpenVINO。

---

## 4. 候选方案逐一分析

### 4.1 llama.cpp —— 已完成实测的备选

**它是什么**：基于 GGUF 的 C/C++ 推理引擎。在本机能通过三条路加速：
- **SYCL 后端**（oneAPI/Level-Zero）：Intel 第一方路径，对 Battlemage/Xe2（B 系列）有专门调优，是 iGPU 上 prefill 通常最快的路。
- **Vulkan 后端**（Mesa ANV）：厂商中立，**编译最省事**（不用装 oneAPI），近年越来越能打。
- **OpenVINO 后端**（`GGML_OPENVINO=ON`）：是 llama.cpp 里**唯一能摸到 NPU** 的路，但本质是「GGUF 皮 + OpenVINO 引擎」。

**模型支持（已核实）**：
- **Qwen3 dense**：✅ 完全成熟。`convert_hf_to_gguf.py` + `llama-quantize` 转 Q4_K_M 即可。
- **Qwen3.5 hybrid**：✅ **架构已支持**（arch `qwen35`/`qwen3next`，PR #16095，2025-11 合入），有专门的 `ggml_gated_delta_net` / `SSM_CONV` / `SSM_SCAN` 算子。这是个积极发现。**但有三个重要坑**：
  1. Intel Arc 上 hybrid 的 SYCL/Vulkan 路径在 2026 年有过崩溃/乱码的回归 bug（#24168 等，多数发生在**独显** Arc Pro，本机 iGPU 不一定复现，且部分已有候选修复合入）。**必须实测确认输出不是乱码。**
  2. **关键**：Gated-DeltaNet 的**分块并行 prefill kernel（chunked prefill）目前只在 CUDA/HIP 上有，SYCL 上还是 `//TODO`**。所以 SYCL 跑 hybrid 的 prefill 走的是"分解后的分块图"（仍并行，但不是单一融合 kernel），效率不如已融合的 OpenVINO GPU。
  3. llama.cpp 的 OpenVINO 后端**根本加载不了 Qwen3.5**（issue #20562，已确认：缺 `SSM_CONV` 算子、跨后端 CPY 失败）。所以「llama.cpp 上 NPU 跑 Qwen3.5」这条路是堵死的。

**NPU 真相（最容易误解的点）**：SYCL/Vulkan 后端**不能用 NPU**。唯一能用 NPU 的 OpenVINO 后端，对 NPU 也只是套壳 OpenVINO（无状态、固定 prefill chunk、Q4_0 为主、仅在 Meteor/Lunar Lake 验证过、本身有 bug），**不会比你现在的原生 OpenVINO NPU 更快**。所以：**别指望 llama.cpp 在 NPU 上有优势。**

**TTFT / TPOT / TPS 实测结果（2026-06-23/24，Q4_K_M GGUF，`llama-bench pp/tg512` 换算）**：
- dense Qwen3：Vulkan 0.6B/1.7B/4B TTFT 为 **313/608/1567ms**；本机源码 SYCL F16 为 **248/374/941ms**；llama.cpp OpenVINO backend GPU 为 **77/118/275ms**。三者都慢于原生 OpenVINO GenAI GPU 的 **35.6/63.1/147.6ms**。decode 上 SYCL F16 为 **164/77/37 tok/s**，接近但未整体超过 OpenVINO GenAI GPU 的 **172/85/43 tok/s**。
- hybrid Qwen3.5：Vulkan 0.8B/2B/4B TTFT 为 **312/594/1472ms**；本机源码 SYCL F16 为 **301/399/897ms**。都慢于原生 OpenVINO GenAI GPU 的纯推理 **131/157/335ms**。decode 上 SYCL F16 为 **114/62/30 tok/s**，其中 0.8B 略高于 OpenVINO GenAI GPU 的 **109 tok/s**，但 TTFT 仍慢 2.3×；llama.cpp OpenVINO backend 的 GPU/NPU 对 Qwen3.5 warmup 失败，CPU 可跑但秒级。

**搭建成本与本机注意点**：
- Vulkan：**低**。当前 `llama.cpp/build` 已由 `-DGGML_VULKAN=ON` 编译得到，可直接跑 `Vulkan0`。
- SYCL：**中**。已在本机源码编译 `llama.cpp/build-sycl-local-f16`，使用 `GGML_SYCL_F16=ON`；默认 `GGML_SYCL_F16=OFF` 构建明显更慢。运行前必须显式 `source /opt/intel/oneapi/2025.3/oneapi-vars.sh --force`，否则复用终端时可能出现静默 `exit=3`。
- OpenVINO 后端：**中**。预编译包 `llama/llama-openvino` 在本机 `Illegal instruction (exit=132)`；已用 `/opt/intel/openvino_2026` 重新编译 `llama.cpp/build-openvino` 完成测试，但对 TTFT 无收益。

**结论**：llama.cpp 的主要价值是 GGUF 生态和可移植性，不是本机低 TTFT。对本任务，Vulkan/SYCL/OpenVINO backend 都没有跑赢原生 OpenVINO GenAI GPU；NPU 这条仍然彻底放弃。

---

### 4.2 ipex-llm —— 死路（项目已归档）

Intel 自己的 LLM 加速库（前身 bigdl-llm）。**已核实：GitHub 仓库 `intel/ipex-llm` 已 ARCHIVED 只读，Intel 声明不再维护/修 bug/发版**，最新二进制停在 2025-07。三个致命问题：
1. **Qwen3.5 完全不支持**，且永远不会加（项目已归档）。
2. **NPU 路径仅 Windows**，且只在 Meteor/Lunar/Arrow Lake 验证过，**不含 Panther Lake**——你是 Ubuntu + PTL，直接没戏。
3. dense Qwen3 只能走它捆绑的、一年前的 llama.cpp SYCL 旧版（oneAPI 2024.2），从未在 PTL/Arc B390 上验证；社区在 PTL 上实测到 `could not create a primitive descriptor for a matmul` 崩溃（#13320）和 154s 的超慢首 token（#13314）。

**结论**：不要投入。它能做的 OpenVINO 都做得更好；它独有的 NPU 路径在本机不可用。

---

### 4.3 ONNX Runtime GenAI + OpenVINO ExecutionProvider —— 底层就是 OpenVINO

ORT-GenAI 在 Linux 上要用 Intel iGPU/NPU，**只能通过 OpenVINO ExecutionProvider**，也就是最终还是调你现在那套 OpenVINO 插件。所以：
- TTFT **最多持平**（多了一层 ORT 抽象，往往略差），没有上行空间。
- Qwen3.5 在 NPU 上**继承同样的坏掉问题**。
- DirectML EP 是 Windows 专属，Linux 不可用。

**结论**：除非 leader 明确需要 ORT-GenAI 的 API/生态，否则没必要。可作为"API 可移植性"的备选，但别期望更快。

---

### 4.4 MLC-LLM / TVM (Vulkan)、vLLM-XPU、PyTorch+IPEX-XPU —— 对本任务都是死路

| 框架 | 为什么不行 |
|---|---|
| **MLC-LLM / TVM (Vulkan)** | 能在 iGPU 上跑 Qwen3 dense，但 Vulkan codegen 在 Intel Xe 上 kernel 效率不如 OpenVINO 的 GPU 插件；**无 NPU 路径**；**不支持 Qwen3.5 的 Gated-DeltaNet**；还有逐模型编译/autotuning 的首次开销。最多作为"独立运行时"的旁证。 |
| **vLLM XPU** | **已联网核实**：官方只支持 Intel **数据中心 GPU + 独显 Arc**，iGPU 未验证（Panther Lake 支持还是一个**未合入的 PR #40295**，否则会在启动时因 UMA 显存核算报错而 abort）。且 vLLM 是为**批量吞吐/连续批处理**设计的，不是单流 TTFT。**无 NPU。** |
| **PyTorch + IPEX-XPU** | 能在 iGPU 上跑，但是 eager 模式、逐算子启动开销大，正好是 TTFT 的反面；**无 NPU**。 |

**共同点**：除 OpenVINO 外，所有 NPU 路径都不存在——**任何"用 NPU"的答案，最终都得回到改进 OpenVINO 的 NPU 插件。**（顺带澄清一个常见担忧：这些框架在本机的 iGPU 用户态驱动其实**齐全**——clinfo/Level-Zero/Mesa Vulkan 都已识别 Arc B390，所以"新硅片驱动不认"不是障碍；真正的障碍是框架层的算子覆盖和成熟度。）

---

### 4.5 OpenVINO 栈内调参 —— 不换框架，先榨干现状

不换框架，只调你已有的 OpenVINO 2026.2，针对 TTFT 真正有用的旋钮（已对本机源码核实）：

**Qwen3 dense 在 NPU 上**（唯一便宜且真实的 TTFT 杠杆）：
- **调小 `MAX_PROMPT_LEN`**：NPU 会把 prompt 右填充到 `MAX_PROMPT_LEN` 再一次性 prefill，所以你为填充的部分白付算力。把它从默认 1024 降到刚好略大于真实 prompt 长度（按 64 对齐），可省 **~10–25%** 的 NPU dense TTFT。**但永远赢不了 GPU。**
- A/B 测 `NPUW_LLM_PREFILL_ATTENTION_HINT`（2026.2 默认 PYRAMID，可选 STATIC/HFA/DYNAMIC）。
- `GENERATE_HINT`（FAST_COMPILE/BEST_PERF）只影响 decode 和编译时间，**对 TTFT 无效**，别动。
- `NPUW_LLM_PREFILL_CHUNK_SIZE` 在 chunk≥max_prompt_len 时自动关闭分块，≤1024 token 场景下基本无差异。

**GPU 上**：dense TTFT 已接近计算下限；`KV_CACHE_PRECISION`、`DYNAMIC_QUANTIZATION_GROUP_SIZE` 等旋钮主要动 decode，对 TTFT 帮助不大。保持 `INFERENCE_PRECISION_HINT=f16` 即可。

**真实部署场景的额外杠杆**：若有**共享/重复前缀**（多轮对话同 system prompt），开 `NPUW_LLM_ENABLE_PREFIX_CACHING` 能大幅降低有效 TTFT——但注意你现在的 benchmark 故意打乱 prompt 来防止前缀缓存命中（为了测真实 prefill），所以这一项在 benchmark 数字里不会体现，但在真实业务里很值钱。

**结论**：栈内能拿的只有"NPU dense 省填充"这一点小幅收益，以及"别把 Qwen3.5 放 NPU"。栈已接近 TTFT 天花板。

---

## 5. 推荐执行路线（分阶段、可落地）

> 给新人的节奏建议：**先把"对照基准"和"测量方法"固化，再去碰新框架**。否则测出来的数没法横向比较，白做。

### 阶段 0：固化方法学（0.5 天）
1. 锁定统一工作负载：**固定 prompt（~940 或你真实业务的长度）、固定 `-ic`、`ITERS≥10`、warm-up 1 次、打乱 prompt 防前缀缓存命中**。
2. 明确**只比 TTFT（首 token / first-infer）**，并记录是否冷编译（首次 kernel JIT/图编译会虚高，要和稳态分开）。
3. 用现成的 `tests/build_prompt_jsonl.py` 生成精确 token 数的 prompt，保证各框架输入一致。

### 阶段 1：榨干现状（0.5 天，低风险必做）
- 对 Qwen3 dense NPU：扫 `MAX_PROMPT_LEN`（按 64 对齐到刚好覆盖真实 prompt）和 `NPUW_LLM_PREFILL_ATTENTION_HINT`，量化能省多少。
- 把 Qwen3.5 的默认设备从含 NPU 改成 **GPU only**（避免重复踩 222s 的坑），明确写进 runner 注释。

### 阶段 2：llama.cpp 实测对比（已完成）
1. **Vulkan**：已用本地 `llama.cpp/build` 测 Qwen3-0.6B/1.7B/4B 和 Qwen3.5-0.8B/2B/4B，TTFT 均慢于 OpenVINO GenAI GPU。
2. **SYCL**：已用本机源码 `build-sycl-local-f16` 测同一批模型，是 llama.cpp 原生 GPU 后端里最快的一路，但 TTFT 仍慢于 OpenVINO GenAI GPU；默认 FP32 构建更慢，预编译包历史结果与 F16 本机构建同量级。
3. **OpenVINO backend**：已用 `/opt/intel/openvino_2026` 重新编译 `build-openvino` 后测试；dense GPU 可跑但仍慢，Qwen3.5 GPU/NPU warmup 失败。
4. `llama-bench` 的 `pp` 数据已按 `TTFT(ms)=prompt_tokens/pp*1000` 换算并写入 [llamacpp_TTFT对比复现文档.md](llamacpp_TTFT对比复现文档.md)。

### 阶段 3：可选的"完整性"补充（按 leader 是否需要而定）
- 若需要"我们也评估了 ORT 生态"：跑一次 ONNX Runtime GenAI + OpenVINO EP（预期持平）。
- 若需要"独立运行时旁证"：MLC-LLM Vulkan 跑 Qwen3 dense（预期略差）。
- vLLM-XPU / IPEX-XPU / ipex-llm：**直接在报告里写明"评估后判定为死路及原因"**，不必实测。

### 阶段 4：给 leader 的结论（0.5 天）
出一张"框架 × 模型 × 设备 × TTFT"对比表 + 一段结论（见第 10 节模板）。

---

## 6. 专题：为什么 Qwen3.5 这么特殊（必须讲清楚）

这是整个任务里技术含量最高、也最容易被 leader 追问的点。

**Qwen3.5 是 hybrid 架构**：24 层里 **18 层是 `linear_attention`（Gated-DeltaNet / Mamba 式状态空间 SSM，带因果 conv1d + 门控）+ 6 层 `full_attention`**（标准 GQA+RoPE，每 4 层一个）。这从 `config.json` 的 `layer_types`、`linear_conv_kernel_dim=4`、`mamba_ssm_dtype` 可证实。

**为什么 SSM 让 prefill 在 NPU 上爆炸**：SSM 的状态更新是**沿 token 轴顺序递归**的（第 t 个 token 依赖第 t-1 个的状态）。要快，必须用**分块并行扫描（chunked parallel scan）**：把序列切成 64 的块，块内用稠密矩阵乘 + 三角求解并行算，块间只传递小状态。这样 prefill 就变成大部分是可并行的 GEMM。

- **iGPU/CPU**：OpenVINO 2026.2 有**融合的 `GatedDeltaNet` 算子**（带分块并行 OpenCL kernel），所以 iGPU 上 Qwen3.5 prefill 只要 171ms。
- **NPU**：**OpenVINO 的 NPU 插件没有实现这个融合算子**，于是导出的 IR 退化成**逐 token 的顺序扫描**——本机模型里精确地是 **18 个 `Loop`（opset5）算子**（对应 18 个 linear_attention 层），每个 Loop 体是 token 级递归。950 token × 18 层串行 → ~222s。

**最关键的纠正（我已亲自在本机源码核实）**：

```
openvino/src/core/src/op/gated_delta_net.cpp                    ← 核心融合算子存在
openvino/src/plugins/intel_gpu/.../gated_delta_net.hpp          ← GPU 插件已实现
openvino/src/plugins/intel_cpu/src/nodes/gated_delta_net.cpp    ← CPU 插件已实现
grep GatedDeltaNet openvino/src/plugins/intel_npu/  →  0 处     ← NPU 插件零实现
```

**所以："Qwen3.5 在 NPU 上慢"不是 NPU 的硬件物理极限，而是 OpenVINO 的 NPU 插件还没把这个融合算子做出来。** 这是上游（Intel/OpenVINO）的待办，不是你在配置或换框架层面能解决的。社区也确认 Intel 工程师在"working on Qwen3.5 enabling / In Progress"。

**对 leader 的话术**：
> "Qwen3.5 是 SSM hybrid 架构。它的线性注意力在 NPU 上目前会退化成顺序扫描（OpenVINO 的 NPU 插件还没实现并行的 GatedDeltaNet 融合算子，而 CPU/GPU 插件已经实现了），所以 NPU 上 222s。这不是硬件极限，是上游软件待办。在我们能控制的范围内，Qwen3.5 用 iGPU（纯推理 131–335ms，端到端首 token 171–378ms）是正确选择；要 NPU 上变好，得等 OpenVINO 后续版本给 NPU 插件补上这个算子。"

---

## 7. 硬件视角：为什么 iGPU 是 TTFT 之王（本机实测支撑）

| 设备 | 是什么 | TTFT 强在/弱在哪 | 本机 dense Qwen3 TTFT |
|---|---|---|---|
| **Arc B390 iGPU** | Xe2/Xe3 级，96 个 Xe 核 @2.5GHz，**带 XMX 脉动矩阵引擎**（clinfo 确认 `cl_intel_subgroup_matrix_multiply_accumulate`），原生 fp16 | **最强**：大矩阵乘正好喂满 XMX；OpenVINO+oneDNN 是最成熟的 kernel；能容忍动态 shape | **35 / 63 / 148 ms（最优）** |
| **AI Boost NPU** | Panther Lake 级（intel_vpu, PCI 8086:B03E, NPU 5 / IP 50XX），几十 INT8 TOPS | 适合**固定 shape 的稳态低功耗推理**；不擅长大动态 prefill GEMM（比 GPU 慢约 4–5 倍）；**完全不擅长递归/扫描**；动态 shape 会触发重编译 | 149 / 290 / 703 ms |
| **16 核 CPU** | 有 AVX2/AVX-VNNI，**无 AVX-512、无 AMX** | 没有矩阵引擎，向量 FMA 远不及 GPU | 1125 / 2903 / 7291 ms（最差） |

**三个要点：**
1. **iGPU、NPU、CPU 共享 ~30GB 系统内存，没有独立显存。** 这意味着内存带宽限制 decode，但**不限制计算密集的 prefill/TTFT**——所以 TTFT 看算力（iGPU 赢）。
2. **NPU 对 dense 模型比 iGPU 慢约 4–5 倍**（且这个比例随模型增大略升，不是恒定），对 hybrid 模型则是灾难。**把 prefill 放 NPU 只会让 TTFT 更差。**
3. **iGPU 已接近计算下限**（实测大模型 XMX 利用率 ~50 TFLOP/s）。但小模型（0.6B）利用率只有 ~32 TFLOP/s，**留有余量**——这也是为什么此前值得实测 llama.cpp；本轮实测已经证明 llama.cpp 没有把这部分空间转化成更低 TTFT。

**关于 HETERO（prefill 在 iGPU、decode 在 NPU）**：对 TTFT **无意义**——TTFT 完全由 prefill 决定，只要 prefill 在 iGPU 上，NPU 帮不上首 token；反而引入设备切换 + KV-cache 跨设备拷贝的开销。仅在"长 decode 想省功耗"且是 dense 模型时才值得考虑，不在本任务目标内。

---

## 8. 如何把新框架接入你现有的 benchmark 框架

你的框架已经很规整（`runners/*.sh` → `benchmark.py` → `results/*.csv|json`，统一 metric schema）。接 llama.cpp 时建议**复用而非另起炉灶**：

1. **指标对齐**：llama.cpp 的 `llama-bench` 输出 `pp`（prompt processing，tok/s）和 `tg`（text generation）。把 **prefill 时间 = prompt_tokens / pp_speed** 换算成毫秒，对齐到你 CSV 的 `ttft_*_ms` / `1st_infer_latency` 口径。decode 用 `tg` 对齐 `tpot`。
2. **新增 framework 维度**：你的 CSV 已有 `framework` 列（现在全是 `ov`）。新增 `llamacpp-sycl` / `llamacpp-vulkan` 取值即可横向对比，下游聚合脚本对未知列按 0 处理（README 已说明）。
3. **新增 runner**：照 `tests/README.md` 的"Adding a new benchmark"流程，写 `tests/runners/run_llamacpp.sh`，source `_common.sh` 复用 `bench_run_with_log` / `bench_make_log_dir` / `bench_write_manifest` / `bench_parse_devices`，把 `llama-bench` 的 JSON 解析进同样的 metric 写入器。
4. **保持工作负载一致**：用同一个 `tests/prompts/*.jsonl`，同样的 prompt 长度和 `-ic`，同样打乱 prompt 防缓存。
5. **环境**：注意本机 Python 环境是 `source /home/user/envs/model_test_env/bin/activate`；llama.cpp 是独立 C++ 构建，转 GGUF 时用这个环境里的 `transformers` 即可。

---

## 9. 风险与注意事项

- **冷编译 vs 稳态**：llama.cpp（SYCL AOT/JIT）和 OpenVINO（图编译）首次运行都有一次性高开销，会让第一次 TTFT 虚高。**测稳态，把 warm-up 排除**（你的框架已默认排除 iter 0）。
- **Qwen3.5 在 llama.cpp 上要验证正确性**：Intel Arc 上 hybrid 有过乱码/崩溃回归。**先确认输出合理（对比 MD5 或人读），再信它的 TTFT 数。** 钉住一个已知良好的 build。
- **多轮对话的 TTFT 陷阱**：llama.cpp 对 hybrid 模型的 prompt 缓存会失效、强制全量重算 prefill（#19794/#22746）——多轮场景 TTFT 会回退。
- **"llama.cpp 打不过 OpenVINO"已经由本轮实测确认**：Vulkan / 本机源码 SYCL F16 / OpenVINO backend 均没有降低 TTFT。后续报告中不要再写成"待实测"或"预期略差"。
- **不要在 NPU 上反复试 Qwen3.5**：~222s + 可能 OOM，浪费时间。

---

## 10. 一页结论模板（给 leader）

> **调研结论：≤4B 的 Qwen3/Qwen3.5，在本机（Core Ultra X7 358H + Arc B390 iGPU + NPU）上，当前的 OpenVINO GenAI + iGPU 路径已是 TTFT 最优解，没有"换框架就明显更快"的捷径。**
>
> - **dense Qwen3**：iGPU 上 35–148ms，已接近硬件计算下限。我们用 llama.cpp Vulkan / 本机源码 SYCL F16 / OpenVINO backend 做了实测对比，最快的 llama.cpp OpenVINO backend GPU 仍慢于原生 OpenVINO GenAI GPU（0.6B：77ms vs 35.6ms；1.7B：118ms vs 63.1ms；4B：275ms vs 147.6ms），SYCL/Vulkan 更慢。decode 指标上 SYCL F16 接近但未整体超过 OpenVINO GenAI。
> - **Qwen3.5 hybrid**：iGPU 上 131–335ms 纯推理 TTFT 是当前正确选择。llama.cpp SYCL F16 decode 可接近 OpenVINO GenAI，但 TTFT 仍慢 2.3–2.7×；llama.cpp OpenVINO backend GPU warmup 失败。NPU 上的 222s 不是硬件极限，而是 OpenVINO 的 NPU 插件尚未实现并行的 GatedDeltaNet 融合算子（CPU/GPU 插件已实现）——属上游软件待办，需等后续 OpenVINO 版本。
> - **其他框架**：ipex-llm（已归档、不支持 Qwen3.5、NPU 仅 Windows）、vLLM-XPU（面向吞吐、iGPU 未验证）、IPEX-XPU（eager 慢）均判定为死路；ONNX Runtime GenAI 底层即 OpenVINO，最多持平。
> - **NPU 是唯一可能的差异化方向，但任何 NPU 改进都必须来自 OpenVINO NPU 插件本身**（如补齐 GatedDeltaNet 融合 / 改善动态 shape），不是换个推理框架能解决的。
>
> **建议**：主力保持 OpenVINO+iGPU；对 dense Qwen3 NPU 做 `MAX_PROMPT_LEN` 微调（省 10–25%）；持续关注 OpenVINO 后续版本对 NPU 上 hybrid 模型的支持。

---

## 11. 附录：证据与可信度

> 标注说明：✅已核实（本机源码/数据或联网）｜⚠️部分成立/有纠偏｜❓推理未实测。完整证据链见研究底稿。

| # | 结论 | 可信度 | 证据要点 |
|---|---|---|---|
| 1 | OpenVINO 2026.2 有 `GatedDeltaNet` 融合算子，CPU/GPU 插件已实现，**NPU 插件零实现** | ✅高（本人本机 grep 核实） | `intel_gpu`/`intel_cpu` 下有 gated_delta_net 实现；`grep GatedDeltaNet intel_npu/` = 0 |
| 2 | Qwen3.5 NPU ~222s，IR 里精确是 18 个 `Loop`(opset5) | ✅高 | `benchmark_qwen3_5_official.csv` first_latency=222338ms；`openvino_language_model.xml` 18×Loop |
| 3 | dense Qwen3 NPU 正常（149/290/703ms），iGPU 最优（35/63/148ms） | ✅高 | `benchmark_qwen3_official.csv` |
| 4 | llama.cpp 支持 Qwen3.5 hybrid（arch qwen35/qwen3next，PR #16095） | ✅高（联网核实） | GitHub PR #16095，2025-11-28 合入 |
| 5 | llama.cpp OpenVINO 后端对 Qwen3.5 **GPU/NPU warmup 失败**，CPU 可跑但秒级 | ✅高（本机实测 + 联网核实） | 本机 `failed to decode prompt batch, res = -3`；issue #20562 提到缺 SSM_CONV/CPY |
| 6 | llama.cpp SYCL 缺融合 chunked GDN prefill kernel（仍 `//TODO`） | ✅高 | `ggml-sycl/gated_delta_net.cpp` TODO；chunked kernel 仅 CUDA/HIP |
| 7 | ipex-llm 已归档、NPU 仅 Windows、不支持 Qwen3.5 | ✅高（联网核实） | repo archived=true；npu_quickstart 仅 MTL/LNL/ARL |
| 8 | vLLM XPU 仅支持数据中心/独显 Arc，iGPU 未验证，面向吞吐 | ✅高（联网核实） | docs.vllm.ai XPU 页；未合入 PR #40295（PTL iGPU 支持） |
| 9 | 本机 iGPU 用户态驱动齐全（非"新硅片不认"），OpenCL/Level-Zero/Vulkan 都识别 Arc B390 | ✅高（本机 probe） | clinfo / libze-intel-gpu1 / mesa-vulkan-drivers 均识别 |
| 10 | NPU 是 Panther Lake 级（IP 50XX），几十 INT8 TOPS，dense 上比 iGPU 慢 ~4–5× | ⚠️高（比例非恒定） | 实测比例 4.17/4.58/4.76，随模型增大略升，非"恒定 4.5×" |
| 11 | "llama.cpp 打不过 OpenVINO GPU" | ✅高（本机实测） | Vulkan/本机源码 SYCL F16/OpenVINO backend 已测到 4B；最快 llama.cpp 路径仍慢于原生 OpenVINO GenAI GPU 的 TTFT，TPOT/TPS 已补充，详见复现文档 |

**研究方法说明**：本报告基于一次多智能体并行调研（6 个维度研究 + 11 条高/中风险结论的对抗式独立核验），辅以本机硬件 probe、源码 grep 和现有 benchmark 数据交叉验证。涉及 2026 年最新框架状态的部分尽量联网核对；联网受限时以本机 2026.2.0 tag 的源码树为准并已注明。

---
*报告生成于 2026-06-22，2026-06-24 补充本机源码 SYCL F16 复测、4B 模型对比和 TPOT/TPS 指标；针对 OpenVINO 2026.2.0 / openvino-genai 2026.2.0.0 / 本机 Panther Lake 平台。框架状态会随上游更新变化，建议每隔 1–2 个月复查 llama.cpp 的 Intel SYCL chunked GDN kernel 与 OpenVINO NPU 插件对 hybrid 的支持进展。*
