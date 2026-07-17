# Pipeline.MX (ovmx) 架构分析 + Qwen3/Qwen3.5 TTFT 实测（356H）

日期：2026-07-06。机器：Intel Core Ultra 7 **356H**（iGPU，`GPU_HW_MATMUL` 但无
CM systolic PagedAttention）。方法与 `OV源码优化降低TTFT_PA.md` 一致：`-n 5`(+1
warmup)、真实 1000-token prompt、逐迭代 permutation 关闭 CB 前缀缓存以测真实 prefill
TTFT、md5 多重集作正确性判据。

测试代码：[`../../ovmx_model_test`](../../ovmx_model_test)（`ovmx_bench` C++，直接链接
release bundle）。参考部署配置：[`../../ovmx_deploy`](../../ovmx_deploy)（OVMS Qwen3-Omni
YAML）。

---

## 0. 结论先行

1. **pipeline_mx（ovmx）是什么**：一个**只有 release、没有源码**的 OpenVINO+GenAI
   精简再发行版，专为 Qwen 系列裁剪。它把「模型图 + pipeline 编排」从 GenAI 的通用栈里
   抽出来，用**YAML 声明的 module DAG** + **手写 C++ 建图器** + **CM PagedAttention 内核**
   重新实现，目标是更小体量、更强的每模块设备编排（GPU/NPU/CPU 混布）与专用算子。

2. **它到底精简了什么**（见 §2）：不是把 OV 变小，而是**绕开 GenAI 通用 pipeline**——
   用 `libpipeline.so`（`ov::pipeline::*`）的 module-DAG 引擎替换 `ov::genai` 的固定
   LLM/VLM pipeline；模型图由**手写建图器**（`create_qwen3_dense_model`、`Qwen3_5Model`、
   `Qwen3_5GatedDeltaNet` …）直接从权重构造，而非走 IR frontend + 通用变换。attention 用
   专用 `pa_cm`（C-for-Metal）内核。

3. **它是不是 drop-in**：是。bundle 的 `<openvino/genai/llm_pipeline.hpp>` 是**影子头**，
   把 `ov::genai::LLMPipeline` 别名到 `ov::pipeline::LLMPipeline`。facade 按模型目录里
   **有没有 `pipeline.yaml`** 决定走 **NATIVE（mx 引擎）** 还是 **BYPASS（回退到 stock
   genai）**。因此同一个二进制、同一份权重、同一 runtime，唯一变量是 `pipeline.yaml`——
   这就是最干净的 A/B。

4. **在 356H 上能否让 TTFT 更优**：**不能，反而更慢**。dense Qwen3 三个尺寸 native mx 的
   TTFT 比 stock genai **高 +19%～+46%**，TPOT/吞吐基本持平，md5 均为合法多重集。根因：
   ovmx 的招牌加速点 `pa_cm`（CM PagedAttention）**要求 systolic-array GPU + immad + CM
   JIT**，**356H iGPU 不支持**，于是 mx 回退到与 stock 相同的 PA 内核，却额外背上了随
   prefill 长度增长的 module-DAG 开销（见 §4 的长度扫描：短 prompt 持平、1000-token 差
   +44ms）。

5. **给决策的建议**（见 §6）：**在 356H 这类无 CM-systolic 的 iGPU 上，从 ovmx 方向出发
   拿不到 TTFT 收益**，因为它的核心加速内核被硬件门槛挡住了。ovmx 的价值应在**支持
   `pa_cm` 的 systolic GPU（如带 XMX/immad 的 Arc/Lunar-Lake/Battlemage 级 iGPU/dGPU）**、
   或**多模态/Omni 混布 + TTS 流式**这类它专门重写了算子和编排的场景上评估。若目标机型是
   356H 且只做纯文本 Qwen3/3.5 的 TTFT，本仓库已有的 `opt-3/5/6` GPU 插件补丁仍是更实的路径。

---

## 1. Release 结构总览

```
pipeline_mx/runtime/
├── include/
│   ├── openvino/            # 标准 OV + GenAI 公共头（含"影子" llm/vlm pipeline 头）
│   └── pipeline/            # ovmx 引擎公共头
│       ├── facade/          # ov::pipeline::{LLMPipeline,VLMPipeline,CB,Omni,...}
│       └── engine/          # module schema / backend / registry
├── lib/intel64/
│   ├── libpipeline.so                 # ★ ovmx 引擎（module DAG + 手写建图器 + pa_cm）
│   ├── libopenvino_genai_mx.so        # ★ mx 版 GenAI（含 modeling 框架）
│   ├── libopenvino_mx.so              # ★ mx 版 OV core
│   ├── libopenvino.so / _genai.so     # 标准 OV/GenAI（BYPASS 路径 + 插件依赖）
│   ├── libopenvino_intel_gpu_plugin.so# 标准 GPU 插件（2026.2.0-21921-9f38e4bea4c）
│   ├── cache.json                     # 预置 GPU kernel cache
│   └── *_frontend.so                  # 全套 IR/PT/TF/ONNX/Paddle/JAX frontend（~66MB）
└── 3rdparty/tbb, cmake/               # 依赖 + find_package(pipeline) 配置
```

**两套并行栈**：标准栈（`libopenvino.so` soname `.2620` / `libopenvino_genai.so`）与
mx 栈（`libopenvino_mx.so` / `libopenvino_genai_mx.so`，各自独立 soname）。
`libpipeline.so` 链接的是**标准** `libopenvino.so`+`libopenvino_genai.so`，并把手写建图
器**静态编进**自身（872 处 `modeling::models` 符号）。GPU 插件绑定标准 `libopenvino.so.2620`。

版本：**Pipeline.MX 0.2.0**，底层 **OpenVINO 2026.2.0-21921**（对照基线 pip OV 是
2026.2.1-21919，几乎同源）。

---

## 2. 精简了什么：ovmx vs 标准 GenAI

关键不是"把 OV 做小"（bundle 总体 560MB，比 pip OV 还大，因为塞了全套 frontend），而是
**换掉推理编排与建图路径**：

| 维度 | 标准 `ov::genai` | Pipeline.MX (`ov::pipeline`) |
|---|---|---|
| pipeline 编排 | 固定的 LLM/VLM/CB C++ 类，内部硬编码数据流 | **YAML 声明的 module DAG**：`ParameterModule → (ChatHistoryPromptBuilder / TextEncoder / TextEmbedding / VisionEncoder / EmbeddingFusion) → LLMInferenceCBModule → ResultModule`；每模块可指定 `device:` 与 `execution_mode: serial/parallel` |
| 模型图来源 | 加载导出的 IR（`openvino_model.xml`）走 frontend + 通用变换 | **两种 backend**（`BackendFactory::resolve_backend_type`）：`IRBackend`（有 IR 时）或 **`ModelingBackend`（手写 C++ 建图器从权重直接构图）**。手写器：`create_qwen3_dense_model`、`create_qwen3_lm_head_model`、`Qwen3_5Model`、`Qwen3_5GatedDeltaNet`、`Qwen3VL*`、`Qwen3Omni*`、`Qwen3ASR*`、`create_qwen3_tts_*` 等一整套 |
| 建图基础设施 | 无（依赖优化器已导出的图） | `modeling::BuilderContext / WeightParameter / VocabEmbedding / FP32LayerNorm / ops::*`——一套在 C++ 里搭 OV 图 + 绑权重的 mini 框架（`libopenvino_genai_mx.so` 内） |
| Attention 内核 | PA / SDPA（`ATTENTION_BACKEND`）走 GPU 插件通用内核 | 额外的 **`pa_cm`**（C-for-Metal PagedAttention），及 `attn_kernel_mode: auto` 回退 |
| 权重来源 | IR `.bin` | IR `.bin` **或** `model.safetensors`（`WeightSource`：`PrefixMappedWeightSource` / `SyntheticWeightSource` …），可从 safetensors 现场切分出 `openvino_language_model` / `text_embeddings` / `merge_embeds` |
| 目标形态 | 库，进程内调用 | 既可库内调用，也是 **OVMS/OVMX 容器**的引擎（见 `ovmx_deploy`：一个 YAML 把 image/video/audio 预处理、vision/audio encoder、LLM(GPU)、TTS(NPU) 全部编排起来，OpenAI 兼容 HTTP 服务） |

一句话：**ovmx = 把 Qwen 的图和编排从 GenAI 通用栈里"抽出来专门重写"**，换取
(a) 每模块设备混布 + 并行调度、(b) 手写建图（可上 safetensors、可做算子级定制）、
(c) 专用 `pa_cm` attention。这正是用户描述的"把 OV+GenAI 里 Qwen 相关的算子和代码专门
抽取出来做的小体量 OV+GenAI"。

### 2.1 drop-in / 路由机制（决定怎么测）

`<openvino/genai/llm_pipeline.hpp>` 影子头：
```cpp
namespace ov { namespace genai {
    using LLMPipeline = ::ov::pipeline::LLMPipeline;   // 别名到 mx facade
}}
```
`ov::pipeline::LLMPipeline(model_dir, device, props)` 构造时：
- 目录含 `pipeline.yaml`/`pipeline_config.yaml` → **NATIVE**（`DirectFacade` 驱动 module DAG）；
- 否则 → **BYPASS**，`genai_compat::*` 把调用转发给 stock `ov::genai` pipeline。

YAML 路径下 `get_generation_config()` 等接口不可用（"is not supported on the YAML
path"），生成长度、attn 内核等由 YAML 里 `LLMInferenceCBModule` 的 params 控制
（`max_new_tokens`、`attn_kernel_mode`、`model_path`…）。

---

## 3. 测试方法（ovmx_model_test）

`ovmx_bench` 显式 include mx facade 头，一个二进制同时能跑两条路：
- **BYPASS**：`-m benchmark/models/<M>`（无 yaml）→ stock genai。
- **NATIVE**：`-m ovmx_model_test/mxcfg/<M>`（软链权重 + 手写 `pipeline.yaml`）→ mx 引擎。

指标口径对齐 `model_bench`：TTFT=generate() 到**首个 streamer 回调**的墙钟；
TPOT=(E2E−TTFT)/(tok−1)；md5=输出文本 MD5（逐迭代 permutation 打乱前缀，故 md5 为多值
多重集，与文档 §7.2 口径一致）。注意 mx YAML 路径**不回填 genai PerfMetrics**（`get_ttft()`
返回 -1），所以 TTFT 必须用 streamer 自测——这也是本 harness 用回调计时的原因。

NATIVE 文本 pipeline.yaml（dense）：`ParameterModule(chat_history,text_streamer) →
ChatHistoryPromptBuilderModule → TextEncoderModule(→input_ids) →
LLMInferenceCBModule(model_path=openvino_model.xml, attn_kernel_mode=auto) →
ResultModule`。

---

## 4. 实测结果（356H，GPU，PA，1000-tok prompt，ic=128，n=5）

### 4.1 Dense Qwen3 —— BYPASS vs NATIVE（同 bundle A/B）

| 模型 | 路径 | TTFT (ms) | TPOT (ms) | E2E (ms) | TPS | TTFT Δ |
|---|---|---:|---:|---:|---:|---:|
| Qwen3-0.6B | BYPASS(genai) | **134.9** | 7.90 | 1110 | 112.1 | — |
| Qwen3-0.6B | NATIVE(mx)    | 196.8 | 8.10 | 1183 | 103.8 | **+45.9%** |
| Qwen3-1.7B | BYPASS(genai) | **201.7** | 13.05 | 1796 | 68.6 | — |
| Qwen3-1.7B | NATIVE(mx)    | 245.3 | 12.83 | 1828 | 68.0 | **+21.6%** |
| Qwen3-4B | BYPASS(genai)   | **436.1** | 26.58 | 3673 | 33.4 | — |
| Qwen3-4B | NATIVE(mx)      | 520.3 | 26.47 | 3745 | 32.8 | **+19.3%** |

**NATIVE mx 三尺寸 TTFT 全面更慢（+19%～+46%）**，TPOT/TPS 在噪声内持平，md5 均为合法
多重集（5 元，与 permutation 对应）。

### 4.2 prefill 长度扫描（Qwen3-1.7B）——差距来自 prefill，不是固定开销

| prompt 长度 | BYPASS TTFT | NATIVE TTFT | Δ |
|---|---:|---:|---:|
| ~2 tok   | 54.9 | 53.5 | **−1.4（持平/略快）** |
| ~256 tok | 93.6 | 98.0 | +4.4 |
| ~1000 tok| 201.7 | 245.3 | **+43.6** |

短 prompt 两条路**完全持平**（甚至 mx 略快），差距**随 prefill 长度线性增长**——说明
不是每次调用的固定 dispatch 开销，而是 **mx 引擎的长 prefill 计算路径效率低于 stock
genai**（native 走 input_ids→TextEmbedding→CB(inputs_embeds) 的分模块 embeds 路径，
stock genai 走直接 token-path prefill），且**没有 `pa_cm` 来补偿**。

### 4.3 Qwen3.5（hybrid SSM/VLM）—— 仅 BYPASS（参考）

| 模型 | 路径 | TTFT (ms) | TPOT (ms) | E2E (ms) | TPS |
|---|---|---:|---:|---:|---:|
| Qwen3.5-0.8B | BYPASS(genai VLM) | 235.4 | 8.58 | 1288 | 96.1 |
| Qwen3.5-2B  | BYPASS(genai VLM) | 308.3 | 14.25 | 2035 | 60.1 |
| Qwen3.5-4B  | BYPASS(genai VLM) | 628.3 | 29.99 | 4107 | 28.5 |

Qwen3.5 的 **NATIVE 文本路走不通**：facade 对 `model_type: qwen3_5` 仍经 stock genai 的
VLM ModelsMap 装载，报 `Model with key 'vision_embeddings_pos' not found in models map`
（genai `utils.cpp:809`），与 YAML 内容无关——即该 release 未提供可用的 Qwen3.5**纯文本**
native 模板（它把 qwen3_5 当作强多模态）。故 Qwen3.5 只测到 BYPASS，其数与本机 stock-pip
基线一致，可作 sanity。

### 4.4 `pa_cm` 硬件门槛（关键根因）

给 `LLMInferenceCBModule` 设 `attn_kernel_mode: pa_cm` 时，构图直接失败：
```
[GPU] ov::hint::attn_kernel_mode=PA_CM requested, but the CM paged attention kernel
is not supported on this GPU architecture. The CM kernel requires a systolic-array
GPU with immad support and CM JIT enabled. Set attn_kernel_mode=AUTO to fall back.
```
356H iGPU 的 `OPTIMIZATION_CAPABILITIES` 含 `GPU_HW_MATMUL`，但 CM PagedAttention 仍拒绝
加载（需要 systolic-array + immad + CM JIT）。因此 native 只能 `auto` 回退到与 stock
**相同**的 PA 内核——**ovmx 的招牌加速在本机被硬件挡掉了**，剩下的只有 module-DAG 开销。

---

## 5. 正确性

所有 dense A/B 的 md5 均为 5 元合法多重集（permutation 每迭代变 prompt → 每迭代不同 md5，
但集合大小与 permutation 数一致），native 与 bypass 各自内部自洽、文本连贯（人工抽查输出
均为正常英文回答）。native 与 bypass 的 md5 集合**不要求逐一相同**——两者是不同实现路径
（甚至可能不同 backend），此处只验证各自数值正常、无崩溃/乱码，而非跨实现逐字节等价。

---

## 6. 结论与建议

**问题「用 pipeline_mx 能否让 TTFT 更优」的答案（356H）：不能。** dense Qwen3 native mx
TTFT +19%～+46%，Qwen3.5 native 文本路不可用。根因是 ovmx 的核心加速内核 `pa_cm`
**需要 systolic-array/immad/CM-JIT 的 GPU**，356H iGPU 不满足，mx 遂回退到与 stock 相同的
PA 内核，却多出随 prefill 增长的编排开销。

**因此，是否"从 ovmx 方向出发"取决于目标硬件与场景：**
- ✅ **值得评估 ovmx 的情形**：目标机是**支持 `pa_cm` 的 systolic GPU**（带 XMX/immad 的
  Arc / Lunar Lake / Battlemage 级）；或需要 **Qwen3-Omni / VLM 的多设备混布 + TTS 流式**
  （ovmx 为此专门重写了 vision/audio/talker/code2wav 算子与 OVMS 编排，见 `ovmx_deploy`）；
  或想用 **safetensors 直接建图**（跳过 optimum 导出 IR）。这些是 ovmx 的真实卖点，本机
  356H 上无法体现。
- ❌ **不值得的情形（=当前 356H 纯文本 Qwen3/3.5 TTFT）**：CM 内核用不了，native 更慢。
  此时本仓库既有的 **GPU 插件补丁 `opt-3 + opt-5 + opt-6`**（见 `OV源码优化降低TTFT_PA.md`，
  356H 上 PA prefill −5%～−9%，md5 逐字节无损）仍是更实、更可控的 TTFT 路径。

**若要继续验证 ovmx 上限**，建议下一步：(1) 在一块支持 `pa_cm` 的 systolic GPU 上重跑本
harness 的同一 A/B（预期 native 才可能反超）；(2) 若坚持 356H，测 ovmx 的
**safetensors→ModelingBackend** 建图路（`create_qwen3_dense_model`）看手写图 + `auto`
attention 是否比 IR 路更省 prefill——但 CM 缺席时上限有限。

---

## 附：复现

```bash
cd ovmx_model_test
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./run_ab.sh GPU 128 5          # 写 results_ab_GPU.tsv
# 单点：
./build/ovmx_bench -m ../benchmark/models/Qwen3-1.7B-INT4          -d GPU -t llm -pf ../model_bench/prompts/qwen3_1k.jsonl -ic 128 -n 5   # BYPASS
./build/ovmx_bench -m mxcfg/Qwen3-1.7B-INT4                        -d GPU -t llm -pf ../model_bench/prompts/qwen3_1k.jsonl -ic 128 -n 5   # NATIVE
```
原始结果：`ovmx_model_test/results_ab_GPU.tsv`。
