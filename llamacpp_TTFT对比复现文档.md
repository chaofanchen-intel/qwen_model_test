# llama.cpp vs OpenVINO TTFT / TPOT / TPS 实测对比复现文档

> 目标：验证在本机 Intel Core Ultra X7 358H + Arc B390 iGPU + AI Boost NPU 上，Qwen3 / Qwen3.5 4B 及以下 LLM 是否能通过 llama.cpp 的 Vulkan / SYCL / OpenVINO backend 获得更低 TTFT，并补充 TPOT / TPS decode 指标。
> 读者：第一次使用 llama.cpp 的同学。本文从 GGUF 准备、环境初始化、benchmark 命令、指标换算到结果解释逐步说明。
> OpenVINO GenAI 基线：来自 `benchmark/results/benchmark_qwen3_official.csv` 与 `benchmark/results/benchmark_qwen3_5_official.csv`。
> llama.cpp 原始结果：`benchmark/results/llamacpp_tg512_20260624.jsonl` 与 `benchmark/results/llamacpp_4b_20260624.jsonl`。
> 实测日期：2026-06-23 至 2026-06-24。

---

## 0. 结论先行

**在本机上，llama.cpp 仍不能降低 Qwen3 / Qwen3.5 4B 及以下模型的 TTFT。当前 TTFT 最优路径仍是原生 OpenVINO GenAI + GPU。**

补充 TPOT / TPS 后结论更完整：

1. **TTFT：OpenVINO GenAI GPU 全部领先。** llama.cpp 的 Vulkan / 本机源码 SYCL F16 / OpenVINO backend 均没有跑赢 OpenVINO GenAI GPU。
2. **TPOT / TPS：SYCL F16 在 decode 上明显强于 Vulkan，部分模型接近 OpenVINO GenAI。** Qwen3.5-0.8B 的 `tg512` generation-only TPS 甚至略高于 OpenVINO CSV 的纯推理 decode TPS，但它的 TTFT 仍是 OpenVINO 的 2.3x，不能解决首 token 延迟目标。
3. **llama.cpp OpenVINO backend 不是 OpenVINO GenAI。** Qwen3 dense 能跑，但 TTFT/TPOT/TPS 都慢于 OpenVINO GenAI；Qwen3.5 hybrid 在 GPU 上 warmup 失败。

### 0.1 完整 GPU 结果总表

口径说明：OpenVINO GenAI 使用官方 CSV 的纯推理字段；llama.cpp TTFT 使用 `pp` 换算，TPOT/TPS 使用独立 `tg512` 测试。

| 模型 | 后端 | TTFT(ms) | TPOT(ms/token) | TPS(tok/s) |
|---|---|---:|---:|---:|
| Qwen3-0.6B | OpenVINO GenAI GPU | **35.6** | **5.81** | **172.3** |
| Qwen3-0.6B | llama.cpp OpenVINO GPU | 77.1 | 11.26 | 88.8 |
| Qwen3-0.6B | llama.cpp SYCL F16 GPU | 247.7 | 6.09 | 164.3 |
| Qwen3-0.6B | llama.cpp Vulkan GPU | 313.0 | 9.39 | 106.5 |
| Qwen3-1.7B | OpenVINO GenAI GPU | **63.1** | **11.69** | **85.5** |
| Qwen3-1.7B | llama.cpp OpenVINO GPU | 117.6 | 18.71 | 53.4 |
| Qwen3-1.7B | llama.cpp SYCL F16 GPU | 373.6 | 12.96 | 77.2 |
| Qwen3-1.7B | llama.cpp Vulkan GPU | 608.0 | 21.77 | 45.9 |
| Qwen3-4B | OpenVINO GenAI GPU | **147.6** | **23.26** | **43.0** |
| Qwen3-4B | llama.cpp OpenVINO GPU | 275.0 | 35.82 | 27.9 |
| Qwen3-4B | llama.cpp SYCL F16 GPU | 940.7 | 27.07 | 36.9 |
| Qwen3-4B | llama.cpp Vulkan GPU | 1567.1 | 41.43 | 24.1 |
| Qwen3.5-0.8B | OpenVINO GenAI GPU | **131.5** | 9.14 | 109.4 |
| Qwen3.5-0.8B | llama.cpp OpenVINO GPU | warmup 失败 | warmup 失败 | warmup 失败 |
| Qwen3.5-0.8B | llama.cpp SYCL F16 GPU | 301.2 | **8.75** | **114.3** |
| Qwen3.5-0.8B | llama.cpp Vulkan GPU | 312.3 | 12.51 | 80.0 |
| Qwen3.5-2B | OpenVINO GenAI GPU | **157.3** | **15.18** | **65.9** |
| Qwen3.5-2B | llama.cpp OpenVINO GPU | warmup 失败 | warmup 失败 | warmup 失败 |
| Qwen3.5-2B | llama.cpp SYCL F16 GPU | 399.4 | 16.01 | 62.5 |
| Qwen3.5-2B | llama.cpp Vulkan GPU | 593.9 | 25.87 | 38.7 |
| Qwen3.5-4B | OpenVINO GenAI GPU | **335.2** | **29.57** | **33.8** |
| Qwen3.5-4B | llama.cpp OpenVINO GPU | warmup 失败 | warmup 失败 | warmup 失败 |
| Qwen3.5-4B | llama.cpp SYCL F16 GPU | 897.2 | 33.18 | 30.1 |
| Qwen3.5-4B | llama.cpp Vulkan GPU | 1472.2 | 48.23 | 20.7 |

### 0.2 一句话给 leader

llama.cpp 的 Vulkan / 本机源码 SYCL F16 / OpenVINO backend 在 4B 及以下 Qwen3/Qwen3.5 上没有降低 TTFT；decode 阶段 SYCL F16 表现接近 OpenVINO GenAI，个别小模型甚至略高，但首 token 延迟仍明显更差。继续追求 TTFT 的主线应留在 OpenVINO GenAI GPU、prompt 长度控制和业务 prefix caching，而不是把默认推理栈切到 llama.cpp。

---

## 1. 指标和换算方法

本文使用三个指标：

| 指标 | 含义 | 越小/越大 |
|---|---|---|
| TTFT | Time To First Token，首 token 延迟，主要反映 prompt prefill | 越小越好 |
| TPOT | Time Per Output Token，首 token 后每个输出 token 的平均 decode 时间 | 越小越好 |
| TPS | decode tokens/s，首 token 后 decode 吞吐 | 越大越好 |

llama.cpp 使用 `llama-bench`：

```text
TTFT(ms) = prompt_tokens / pp_tokens_per_second * 1000
TPS(tok/s) = tg_tokens_per_second
TPOT(ms/token) = 1000 / TPS
```

对应命令：

```bash
# TTFT / prefill：只跑 prompt processing
llama-bench -p 940 -n 0 ...

# TPOT / TPS：只跑 generation，本文固定 512 output tokens
llama-bench -p 0 -n 512 ...
```

注意：不要把 `pp+tg` 的 `avg_ts` 当作 decode TPS。源码里 `avg_ts()` 对 `pp+tg` 使用 `(n_prompt + n_gen) / elapsed_time`，它是混合总吞吐，不是首 token 后 decode 速度。本文 TPOT/TPS 均来自独立 `tg512`。

OpenVINO GenAI CSV 使用这些字段：

| CSV 字段 | 本文指标 | 说明 |
|---|---|---|
| `1st_infer_latency_avg(ms)` | TTFT | 纯推理首 token 延迟，最适合和 `llama-bench pp` 对齐 |
| `1st_latency_avg(ms)` | E2E first-token latency | 包含更多端到端开销，本文只作为补充口径 |
| `2nd_infer_avg_latency_avg(ms)` | TPOT | 纯推理 decode 平均每 token 延迟 |
| `1000 / 2nd_infer_avg_latency_avg(ms)` | TPS | 本文计算的纯推理 decode TPS |
| `2nd_tokens_throughput(tokens/s)` | E2E decode TPS | CSV 自带端到端 decode 参考值 |
| `total_throughput(tokens/s)` | 总吞吐 | 包含首 token 和后续 token 的整体参考 |

本次 prompt 长度对齐：

```text
Qwen3:   OpenVINO input_size 约 937，llama.cpp 使用 -p 940
Qwen3.5: OpenVINO input_size 约 948-950，llama.cpp 使用 -p 950
decode:  OpenVINO infer_count 512，llama.cpp 使用 -n 512
```

---

## 2. 环境和目录

本文按当前 workspace 路径书写：

```bash
cd /home/user/chaofan/model_test
```

当前使用的构建和 runtime：

| 内容 | 路径 |
|---|---|
| Vulkan 源码构建 | `llama.cpp/build/bin/llama-bench` |
| SYCL 本机源码 F16 构建 | `llama.cpp/build-sycl-local-f16/bin/llama-bench` |
| SYCL 本机源码默认构建 | `llama.cpp/build-sycl-local/bin/llama-bench` |
| OpenVINO backend 本机源码构建 | `llama.cpp/build-openvino/bin/llama-bench` |
| 历史 SYCL 预编译包 | `llama/llama-sycl/llama-bench` |
| 历史 OpenVINO 预编译包 | `llama/llama-openvino/llama-bench` |
| GGUF 模型目录 | `gguf/` |
| OpenVINO runtime | `/opt/intel/openvino_2026` |
| OpenVINO GenAI | `/opt/intel/openvino_genai` |
| oneAPI 2025.3 | `/opt/intel/oneapi/2025.3` |

重要说明：

1. `llama/llama-openvino` 预编译包在本机上会 `Illegal instruction (exit=132)`，所以本文使用 `llama.cpp/build-openvino/bin/llama-bench`。
2. SYCL 已按用户要求在本机从源码重新编译。默认 `GGML_SYCL_F16=OFF` 可以运行但 prefill 明显偏慢；本文最终 SYCL 口径采用 `GGML_SYCL_F16=ON` 的本机构建。
3. `python` 命令在本机可能不存在，建议显式使用 `/home/user/envs/model_test_env/bin/python` 或激活项目环境。

---

## 3. 准备 GGUF 模型

llama.cpp 通常使用 **GGUF** 模型文件，不能直接读取 OpenVINO IR，例如 `openvino_model.xml` / `openvino_language_model.xml`。本文使用 Q4_K_M GGUF 与 OpenVINO INT4 做量级对比。二者不是完全相同的量化格式，但都属于 4-bit 权重量化，适合比较推理路径的大致性能差距。

推荐下载命令：

```bash
cd /home/user/chaofan/model_test
mkdir -p gguf
/home/user/envs/model_test_env/bin/python - <<'PY'
from huggingface_hub import hf_hub_download

jobs = [
    ("unsloth/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q4_K_M.gguf"),
    ("unsloth/Qwen3-1.7B-GGUF", "Qwen3-1.7B-Q4_K_M.gguf"),
    ("unsloth/Qwen3-4B-GGUF", "Qwen3-4B-Q4_K_M.gguf"),
    ("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    ("unsloth/Qwen3.5-2B-GGUF", "Qwen3.5-2B-Q4_K_M.gguf"),
    ("unsloth/Qwen3.5-4B-GGUF", "Qwen3.5-4B-Q4_K_M.gguf"),
]

for repo, filename in jobs:
    path = hf_hub_download(repo_id=repo, filename=filename, local_dir="/home/user/chaofan/model_test/gguf")
    print(path)
PY
ls -lh gguf/*Q4_K_M.gguf
```

如果 `hf_hub_download` 续传长时间无输出，也可以用新版 `hf` CLI 单文件下载：

```bash
cd /home/user/chaofan/model_test
/home/user/envs/model_test_env/bin/hf download unsloth/Qwen3-4B-GGUF Qwen3-4B-Q4_K_M.gguf --local-dir /home/user/chaofan/model_test/gguf --max-workers 1 --format human
/home/user/envs/model_test_env/bin/hf download unsloth/Qwen3.5-4B-GGUF Qwen3.5-4B-Q4_K_M.gguf --local-dir /home/user/chaofan/model_test/gguf --max-workers 1 --format human
```

本机实测文件大小：

```text
Qwen3-0.6B-Q4_K_M.gguf       373 MiB
Qwen3-1.7B-Q4_K_M.gguf       1.1 GiB
Qwen3-4B-Q4_K_M.gguf         2.4 GiB
Qwen3.5-0.8B-Q4_K_M.gguf     508 MiB
Qwen3.5-2B-Q4_K_M.gguf       1.2 GiB
Qwen3.5-4B-Q4_K_M.gguf       2.6 GiB
```

排错：`curl` 直连 Hugging Face 在本机可能 SSL 失败，使用 `huggingface_hub` / `hf download` 更稳。下载中断后重新运行同一个下载命令会断点续传。

---

## 4. Vulkan backend 复现

### 4.1 确认设备

```bash
cd /home/user/chaofan/model_test/llama.cpp
export LD_LIBRARY_PATH="$PWD/build/bin:$LD_LIBRARY_PATH"
./build/bin/llama-bench --list-devices
```

期望看到：

```text
Available devices:
  Vulkan0: Intel(R) Graphics (PTL) (... MiB free)
```

### 4.2 Qwen3 dense

TTFT / prefill：

```bash
cd /home/user/chaofan/model_test/llama.cpp
export LD_LIBRARY_PATH="$PWD/build/bin:$LD_LIBRARY_PATH"

./build/bin/llama-bench -m /home/user/chaofan/model_test/gguf/Qwen3-0.6B-Q4_K_M.gguf -p 940 -n 0 -ngl 99 -b 940 --device Vulkan0 -r 3
./build/bin/llama-bench -m /home/user/chaofan/model_test/gguf/Qwen3-1.7B-Q4_K_M.gguf -p 940 -n 0 -ngl 99 -b 940 --device Vulkan0 -r 3
./build/bin/llama-bench -m /home/user/chaofan/model_test/gguf/Qwen3-4B-Q4_K_M.gguf   -p 940 -n 0 -ngl 99 -b 940 --device Vulkan0 -r 3
```

TPOT / TPS：

```bash
cd /home/user/chaofan/model_test/llama.cpp
export LD_LIBRARY_PATH="$PWD/build/bin:$LD_LIBRARY_PATH"

for model in \
  /home/user/chaofan/model_test/gguf/Qwen3-0.6B-Q4_K_M.gguf \
  /home/user/chaofan/model_test/gguf/Qwen3-1.7B-Q4_K_M.gguf \
  /home/user/chaofan/model_test/gguf/Qwen3-4B-Q4_K_M.gguf
do
  ./build/bin/llama-bench -m "$model" -p 0 -n 512 -ngl 99 --device Vulkan0 -r 3 -o jsonl
done
```

实测结果：

| 模型 | `pp` tok/s | TTFT(ms) | `tg512` TPS | TPOT(ms/token) |
|---|---:|---:|---:|---:|
| Qwen3-0.6B | 3003.19 | 313.0 | 106.52 | 9.39 |
| Qwen3-1.7B | 1546.11 | 608.0 | 45.93 | 21.77 |
| Qwen3-4B | 599.84 | 1567.1 | 24.14 | 41.43 |

### 4.3 Qwen3.5 hybrid

TTFT / prefill：

```bash
cd /home/user/chaofan/model_test/llama.cpp
export LD_LIBRARY_PATH="$PWD/build/bin:$LD_LIBRARY_PATH"

./build/bin/llama-bench -m /home/user/chaofan/model_test/gguf/Qwen3.5-0.8B-Q4_K_M.gguf -p 950 -n 0 -ngl 99 -b 950 --device Vulkan0 -r 3
./build/bin/llama-bench -m /home/user/chaofan/model_test/gguf/Qwen3.5-2B-Q4_K_M.gguf   -p 950 -n 0 -ngl 99 -b 950 --device Vulkan0 -r 3
./build/bin/llama-bench -m /home/user/chaofan/model_test/gguf/Qwen3.5-4B-Q4_K_M.gguf   -p 950 -n 0 -ngl 99 -b 950 --device Vulkan0 -r 3
```

TPOT / TPS：

```bash
cd /home/user/chaofan/model_test/llama.cpp
export LD_LIBRARY_PATH="$PWD/build/bin:$LD_LIBRARY_PATH"

for model in \
  /home/user/chaofan/model_test/gguf/Qwen3.5-0.8B-Q4_K_M.gguf \
  /home/user/chaofan/model_test/gguf/Qwen3.5-2B-Q4_K_M.gguf \
  /home/user/chaofan/model_test/gguf/Qwen3.5-4B-Q4_K_M.gguf
do
  ./build/bin/llama-bench -m "$model" -p 0 -n 512 -ngl 99 --device Vulkan0 -r 3 -o jsonl
done
```

实测结果：

| 模型 | `pp` tok/s | TTFT(ms) | `tg512` TPS | TPOT(ms/token) |
|---|---:|---:|---:|---:|
| Qwen3.5-0.8B | 3042.34 | 312.3 | 79.96 | 12.51 |
| Qwen3.5-2B | 1599.62 | 593.9 | 38.65 | 25.87 |
| Qwen3.5-4B | 645.28 | 1472.2 | 20.73 | 48.23 |

---

## 5. SYCL backend 复现

### 5.1 本机源码编译 SYCL F16

SYCL 依赖 oneAPI 2025.x 的 `libsycl.so.8`。稳定写法是显式使用 2025.3 的版本化脚本并加 `--force`。

```bash
cd /home/user/chaofan/model_test/llama.cpp
unset ONEAPI_DEVICE_SELECTOR SYCL_DEVICE_FILTER SYCL_DEVICE_ALLOWLIST ZE_AFFINITY_MASK EnableImplicitScaling GGML_OPENVINO_DEVICE SETVARS_COMPLETED
source /opt/intel/oneapi/2025.3/oneapi-vars.sh --force

cmake -B build-sycl-local-f16 \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_SYCL=ON \
  -DCMAKE_C_COMPILER=icx \
  -DCMAKE_CXX_COMPILER=icpx \
  -DLLAMA_OPENSSL=OFF \
  -DGGML_NATIVE=OFF \
  -DGGML_SYCL_F16=ON

cmake --build build-sycl-local-f16 --config Release -j "$(nproc)"
```

说明：本机没有安装 `ninja`，因此使用 CMake 默认 Makefiles。编译过程中 `ggml-sycl/.../dpct/helper.hpp` 会有 `_abort` / `return-type` 相关 warning，本轮没有编译错误。

### 5.2 确认设备

```bash
cd /home/user/chaofan/model_test/llama.cpp
unset ONEAPI_DEVICE_SELECTOR SYCL_DEVICE_FILTER SYCL_DEVICE_ALLOWLIST ZE_AFFINITY_MASK EnableImplicitScaling GGML_OPENVINO_DEVICE SETVARS_COMPLETED
source /opt/intel/oneapi/2025.3/oneapi-vars.sh --force
export LD_LIBRARY_PATH="$PWD/build-sycl-local-f16/bin:$LD_LIBRARY_PATH"

grep '^GGML_SYCL_F16' build-sycl-local-f16/CMakeCache.txt
ldd build-sycl-local-f16/bin/llama-bench | grep -E 'sycl|OpenCL|ze|mkl|tbb|ggml'
./build-sycl-local-f16/bin/llama-bench --list-devices
```

本机实测关键输出：

```text
GGML_SYCL_F16:BOOL=ON
libggml-sycl.so.0 => /home/user/chaofan/model_test/llama.cpp/build-sycl-local-f16/bin/libggml-sycl.so.0
libsycl.so.8 => /opt/intel/oneapi/2025.3/lib/libsycl.so.8
Available devices:
  SYCL0: Intel(R) Arc(TM) B390 GPU (... MiB free)
```

### 5.3 Qwen3 dense

```bash
cd /home/user/chaofan/model_test/llama.cpp
unset ONEAPI_DEVICE_SELECTOR SYCL_DEVICE_FILTER SYCL_DEVICE_ALLOWLIST ZE_AFFINITY_MASK EnableImplicitScaling GGML_OPENVINO_DEVICE SETVARS_COMPLETED
source /opt/intel/oneapi/2025.3/oneapi-vars.sh --force
export LD_LIBRARY_PATH="$PWD/build-sycl-local-f16/bin:$LD_LIBRARY_PATH"

# TTFT / prefill
./build-sycl-local-f16/bin/llama-bench -m /home/user/chaofan/model_test/gguf/Qwen3-0.6B-Q4_K_M.gguf -p 940 -n 0 -ngl 99 -b 940 --device SYCL0 -r 3
./build-sycl-local-f16/bin/llama-bench -m /home/user/chaofan/model_test/gguf/Qwen3-1.7B-Q4_K_M.gguf -p 940 -n 0 -ngl 99 -b 940 --device SYCL0 -r 3
./build-sycl-local-f16/bin/llama-bench -m /home/user/chaofan/model_test/gguf/Qwen3-4B-Q4_K_M.gguf   -p 940 -n 0 -ngl 99 -b 940 --device SYCL0 -r 3

# TPOT / TPS
./build-sycl-local-f16/bin/llama-bench -m /home/user/chaofan/model_test/gguf/Qwen3-0.6B-Q4_K_M.gguf -p 0 -n 512 -ngl 99 --device SYCL0 -r 3 -o jsonl
./build-sycl-local-f16/bin/llama-bench -m /home/user/chaofan/model_test/gguf/Qwen3-1.7B-Q4_K_M.gguf -p 0 -n 512 -ngl 99 --device SYCL0 -r 3 -o jsonl
./build-sycl-local-f16/bin/llama-bench -m /home/user/chaofan/model_test/gguf/Qwen3-4B-Q4_K_M.gguf   -p 0 -n 512 -ngl 99 --device SYCL0 -r 3 -o jsonl
```

实测结果：

| 模型 | 参数口径 | `pp` tok/s | TTFT(ms) | `tg512` TPS | TPOT(ms/token) |
|---|---|---:|---:|---:|---:|
| Qwen3-0.6B | 默认 `n_ubatch=512` | 3794.16 | 247.7 | 164.28 | 6.09 |
| Qwen3-1.7B | 默认 `n_ubatch=512` | 2516.39 | 373.6 | 77.15 | 12.96 |
| Qwen3-4B | 默认 `n_ubatch=512` | 999.29 | 940.7 | 36.94 | 27.07 |

### 5.4 Qwen3.5 hybrid

```bash
cd /home/user/chaofan/model_test/llama.cpp
unset ONEAPI_DEVICE_SELECTOR SYCL_DEVICE_FILTER SYCL_DEVICE_ALLOWLIST ZE_AFFINITY_MASK EnableImplicitScaling GGML_OPENVINO_DEVICE SETVARS_COMPLETED
source /opt/intel/oneapi/2025.3/oneapi-vars.sh --force
export LD_LIBRARY_PATH="$PWD/build-sycl-local-f16/bin:$LD_LIBRARY_PATH"

# TTFT / prefill
./build-sycl-local-f16/bin/llama-bench -m /home/user/chaofan/model_test/gguf/Qwen3.5-0.8B-Q4_K_M.gguf -p 950 -n 0 -ngl 99 -b 950 --device SYCL0 -r 3
./build-sycl-local-f16/bin/llama-bench -m /home/user/chaofan/model_test/gguf/Qwen3.5-2B-Q4_K_M.gguf   -p 950 -n 0 -ngl 99 -b 950 -ub 950 --device SYCL0 -r 3
./build-sycl-local-f16/bin/llama-bench -m /home/user/chaofan/model_test/gguf/Qwen3.5-4B-Q4_K_M.gguf   -p 950 -n 0 -ngl 99 -b 950 -ub 950 --device SYCL0 -r 3

# TPOT / TPS
./build-sycl-local-f16/bin/llama-bench -m /home/user/chaofan/model_test/gguf/Qwen3.5-0.8B-Q4_K_M.gguf -p 0 -n 512 -ngl 99 --device SYCL0 -r 3 -o jsonl
./build-sycl-local-f16/bin/llama-bench -m /home/user/chaofan/model_test/gguf/Qwen3.5-2B-Q4_K_M.gguf   -p 0 -n 512 -ngl 99 --device SYCL0 -r 3 -o jsonl
./build-sycl-local-f16/bin/llama-bench -m /home/user/chaofan/model_test/gguf/Qwen3.5-4B-Q4_K_M.gguf   -p 0 -n 512 -ngl 99 --device SYCL0 -r 3 -o jsonl
```

实测结果：

| 模型 | 参数口径 | `pp` tok/s | TTFT(ms) | `tg512` TPS | TPOT(ms/token) |
|---|---|---:|---:|---:|---:|
| Qwen3.5-0.8B | 默认 `n_ubatch=512` | 3154.15 | 301.2 | 114.30 | 8.75 |
| Qwen3.5-2B | `-ub 950` | 2378.52 | 399.4 | 62.46 | 16.01 |
| Qwen3.5-4B | `-ub 950` | 1058.89 | 897.2 | 30.14 | 33.18 |

### 5.5 默认 FP32 构建和预编译包对照

本机默认 `GGML_SYCL_F16=OFF` 构建可运行，但比 F16 明显慢，不适合作为最终 SYCL 口径：

| 模型 | 默认 FP32 `pp` | 默认 FP32 TTFT | F16 TTFT |
|---|---:|---:|---:|
| Qwen3-0.6B | 2639.82 tok/s | 356.1 ms | 247.7 ms |
| Qwen3-1.7B | 1371.27 tok/s | 685.5 ms | 373.6 ms |
| Qwen3.5-0.8B | 2330.60 tok/s | 407.6 ms | 301.2 ms |
| Qwen3.5-2B | 1367.49 tok/s | 694.7 ms | 399.4 ms |

历史预编译包 `llama/llama-sycl` 的结果为 Qwen3-0.6B/1.7B **247.6/373.7ms**，Qwen3.5-0.8B/2B **293.9/389.9ms**。本机源码 F16 构建和预编译包在同一量级，结论不变：SYCL 能跑，通常快于 Vulkan，但 TTFT 仍明显慢于原生 OpenVINO GenAI GPU。

---

## 6. llama.cpp OpenVINO backend 复现

### 6.1 预编译包问题

你解压的 OpenVINO 预编译包在：

```text
/home/user/chaofan/model_test/llama/llama-openvino
```

本机实测即使使用 `/opt/intel/openvino_2026` runtime，也会在初始化时崩溃：

```bash
cd /home/user/chaofan/model_test/llama/llama-openvino
source /opt/intel/openvino_2026/setupvars.sh
export LD_LIBRARY_PATH="$PWD:$LD_LIBRARY_PATH"
GGML_OPENVINO_DEVICE=GPU ./llama-bench --list-devices
```

实测输出：

```text
OpenVINO: using device GPU
Illegal instruction (core dumped)
exit=132
```

CPU / GPU / NPU 都是同样 `exit=132`。因此不要直接用这个预编译包下结论。

### 6.2 本机重新编译 OpenVINO backend

```bash
cd /home/user/chaofan/model_test/llama.cpp
cmake -B build-openvino \
  -DGGML_OPENVINO=ON \
  -DGGML_VULKAN=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/opt/intel/openvino_2026/runtime/cmake
cmake --build build-openvino --config Release -j "$(nproc)"
```

确认设备：

```bash
cd /home/user/chaofan/model_test/llama.cpp
source /opt/intel/openvino_2026/setupvars.sh
export LD_LIBRARY_PATH="$PWD/build-openvino/bin:$LD_LIBRARY_PATH"

GGML_OPENVINO_DEVICE=GPU ./build-openvino/bin/llama-bench --list-devices
GGML_OPENVINO_DEVICE=NPU ./build-openvino/bin/llama-bench --list-devices
GGML_OPENVINO_DEVICE=CPU ./build-openvino/bin/llama-bench --list-devices
```

期望看到：

```text
Available devices:
  OPENVINO0: OpenVINO Runtime (... MiB free)
```

### 6.3 Qwen3 dense

GPU TTFT / prefill：

```bash
cd /home/user/chaofan/model_test/llama.cpp
source /opt/intel/openvino_2026/setupvars.sh
export LD_LIBRARY_PATH="$PWD/build-openvino/bin:$LD_LIBRARY_PATH"

GGML_OPENVINO_DEVICE=GPU ./build-openvino/bin/llama-bench -m /home/user/chaofan/model_test/gguf/Qwen3-0.6B-Q4_K_M.gguf -p 940 -n 0 -ngl 99 -b 940 -ub 940 --device OPENVINO0 -r 3
GGML_OPENVINO_DEVICE=GPU ./build-openvino/bin/llama-bench -m /home/user/chaofan/model_test/gguf/Qwen3-1.7B-Q4_K_M.gguf -p 940 -n 0 -ngl 99 -b 940 -ub 940 --device OPENVINO0 -r 3
GGML_OPENVINO_DEVICE=GPU ./build-openvino/bin/llama-bench -m /home/user/chaofan/model_test/gguf/Qwen3-4B-Q4_K_M.gguf   -p 940 -n 0 -ngl 99 -b 940 -ub 940 --device OPENVINO0 -r 3
```

GPU TPOT / TPS：

```bash
cd /home/user/chaofan/model_test/llama.cpp
source /opt/intel/openvino_2026/setupvars.sh
export LD_LIBRARY_PATH="$PWD/build-openvino/bin:$LD_LIBRARY_PATH"

GGML_OPENVINO_DEVICE=GPU ./build-openvino/bin/llama-bench -m /home/user/chaofan/model_test/gguf/Qwen3-0.6B-Q4_K_M.gguf -p 0 -n 512 -ngl 99 --device OPENVINO0 -r 3 -o jsonl
GGML_OPENVINO_DEVICE=GPU ./build-openvino/bin/llama-bench -m /home/user/chaofan/model_test/gguf/Qwen3-1.7B-Q4_K_M.gguf -p 0 -n 512 -ngl 99 --device OPENVINO0 -r 3 -o jsonl
GGML_OPENVINO_DEVICE=GPU ./build-openvino/bin/llama-bench -m /home/user/chaofan/model_test/gguf/Qwen3-4B-Q4_K_M.gguf   -p 0 -n 512 -ngl 99 --device OPENVINO0 -r 3 -o jsonl
```

实测结果：

| 模型 | `pp` tok/s | TTFT(ms) | `tg512` TPS | TPOT(ms/token) | 原生 OpenVINO GenAI GPU TTFT / TPS |
|---|---:|---:|---:|---:|---:|
| Qwen3-0.6B | 12184.41 | 77.1 | 88.83 | 11.26 | **35.6 ms / 172.3** |
| Qwen3-1.7B | 7996.07 | 117.6 | 53.44 | 18.71 | **63.1 ms / 85.5** |
| Qwen3-4B | 3417.94 | 275.0 | 27.92 | 35.82 | **147.6 ms / 43.0** |

NPU / CPU 对 Qwen3 dense 也能跑 TTFT，但很慢，继续作为对照即可：

| 模型 | OpenVINO backend NPU TTFT | OpenVINO backend CPU TTFT |
|---|---:|---:|
| Qwen3-0.6B | 2864.1 ms | 1507.0 ms |
| Qwen3-1.7B | 3213.7 ms | 3266.6 ms |

### 6.4 Qwen3.5 hybrid

Qwen3.5 在 llama.cpp OpenVINO backend 上不适合使用。0.8B / 2B / 4B GPU 均 warmup 失败。

复现命令：

```bash
cd /home/user/chaofan/model_test/llama.cpp
source /opt/intel/openvino_2026/setupvars.sh
export LD_LIBRARY_PATH="$PWD/build-openvino/bin:$LD_LIBRARY_PATH"

GGML_OPENVINO_DEVICE=GPU ./build-openvino/bin/llama-bench -m /home/user/chaofan/model_test/gguf/Qwen3.5-4B-Q4_K_M.gguf -p 950 -n 0 -ngl 99 -b 950 -ub 950 --device OPENVINO0 -r 1 -o jsonl
```

GPU 失败信息：

```text
OpenVINO: using device GPU
test_prompt: failed to decode prompt batch, res = -3
llama_bench: error: failed to run prompt warmup
Qwen3.5-4B OpenVINO backend GPU exit=1
```

此前 0.8B / 2B 的 GPU / NPU 也是同类失败：

```text
test_prompt: failed to decode prompt batch, res = -3
llama_bench: error: failed to run prompt warmup
```

CPU 可以跑但非常慢，不适合作为低 TTFT 方案：

| 模型 | OpenVINO backend CPU `pp` | 换算 TTFT |
|---|---:|---:|
| Qwen3.5-0.8B | 305.10 tok/s | 3113.7 ms |
| Qwen3.5-2B | 208.45 tok/s | 4557.4 ms |

---

## 7. 和 OpenVINO GenAI 基线如何对齐

OpenVINO GenAI 基线 CSV：

```text
benchmark/results/benchmark_qwen3_official.csv
benchmark/results/benchmark_qwen3_5_official.csv
```

本次使用的 GPU INT4 基线如下：

| 模型 | input tokens | output tokens | TTFT `1st_infer_latency_avg` | TPOT `2nd_infer_avg_latency_avg` | TPS `1000/TPOT` | CSV total throughput |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B-INT4 | 937 | 512 | 35.6 ms | 5.81 ms | 172.3 | 170.5 |
| Qwen3-1.7B-INT4 | 937 | 512 | 63.1 ms | 11.69 ms | 85.5 | 84.8 |
| Qwen3-4B-INT4 | 937 | 512 | 147.6 ms | 23.26 ms | 43.0 | 42.6 |
| Qwen3.5-0.8B-INT4 | 950 | 512 | 131.5 ms | 9.14 ms | 109.4 | 102.4 |
| Qwen3.5-2B-INT4 | 950 | 512 | 157.3 ms | 15.18 ms | 65.9 | 62.9 |
| Qwen3.5-4B-INT4 | 948 | 512 | 335.2 ms | 29.57 ms | 33.8 | 32.6 |

如果要换成真实业务 prompt 长度，只要把 `-p`、`-b` 和可选 `-ub` 换成相同 token 数，再重新按公式换算即可。

---

## 8. 分模型分析和结论

### Qwen3-0.6B

OpenVINO GenAI GPU 在 TTFT 和 decode 都是最优：35.6 ms / 172.3 tok/s。llama.cpp OpenVINO backend GPU 的 TTFT 是 77.1 ms，虽然是 llama.cpp 里首 token 最快的路径，但仍慢 2.2x；SYCL F16 的 decode TPS 164.3 接近 OpenVINO，但 TTFT 是 247.7 ms，不适合低 TTFT 目标。

### Qwen3-1.7B

OpenVINO GenAI GPU 仍然整体最优：63.1 ms / 85.5 tok/s。llama.cpp OpenVINO backend GPU TTFT 为 117.6 ms，仍慢 1.9x；SYCL F16 decode 为 77.2 tok/s，接近但低于 OpenVINO，TTFT 373.6 ms 明显偏高。

### Qwen3-4B

4B 下 llama.cpp 的 TTFT 差距继续扩大。OpenVINO GenAI GPU 是 147.6 ms；llama.cpp OpenVINO backend GPU 是 275.0 ms；SYCL F16 是 940.7 ms；Vulkan 是 1567.1 ms。decode 方面 OpenVINO GenAI GPU 43.0 tok/s，SYCL F16 36.9 tok/s，OpenVINO backend 27.9 tok/s，Vulkan 24.1 tok/s。结论是 4B dense 仍应使用 OpenVINO GenAI GPU。

### Qwen3.5-0.8B

OpenVINO GenAI GPU TTFT 最优：131.5 ms。SYCL F16 的 `tg512` decode 为 114.3 tok/s，略高于 OpenVINO 纯推理 decode 的 109.4 tok/s，但 SYCL TTFT 是 301.2 ms，仍慢 2.3x。若业务只关心长输出 decode、且首 token 不敏感，可以把 SYCL 作为实验项；若核心指标是 TTFT，则仍选 OpenVINO GenAI GPU。

### Qwen3.5-2B

OpenVINO GenAI GPU TTFT/TPS 分别为 157.3 ms / 65.9 tok/s。SYCL F16 为 399.4 ms / 62.5 tok/s，decode 接近但 TTFT 明显偏高。Vulkan 593.9 ms / 38.7 tok/s 明显落后。llama.cpp OpenVINO backend GPU/NPU warmup 失败。

### Qwen3.5-4B

OpenVINO GenAI GPU 仍是最优：335.2 ms / 33.8 tok/s。SYCL F16 为 897.2 ms / 30.1 tok/s，decode 接近但 TTFT 慢 2.7x；Vulkan 为 1472.2 ms / 20.7 tok/s；llama.cpp OpenVINO backend GPU warmup 失败。因此 Qwen3.5 4B 不建议切换到 llama.cpp。

---

## 9. 为什么 llama.cpp 没有更低 TTFT

1. **TTFT 主要由 prefill 大矩阵乘决定。** OpenVINO GenAI 在 Intel iGPU 上走厂商优化图编译和 GPU kernel，dense Qwen3 的 prefill 已经非常快。
2. **Vulkan / SYCL 是更通用的 ggml 后端。** 它们能用 iGPU，但本机 prefill kernel 效率不如 OpenVINO GenAI。SYCL F16 通常快于 Vulkan，但仍没有追上 OpenVINO GenAI 的首 token 延迟。
3. **llama.cpp OpenVINO backend 不等于 OpenVINO GenAI。** 它从 GGUF/ggml 图走 backend 路径，性能和算子覆盖不等于原生 OpenVINO GenAI LLMPipeline。dense GPU 能跑但仍慢；Qwen3.5 GPU/NPU 直接 warmup 失败。
4. **Qwen3.5 是 hybrid 架构。** 它依赖 GatedDeltaNet / SSM 类线性注意力。OpenVINO GenAI GPU 当前已有更合适的优化路径；llama.cpp SYCL/Vulkan 虽能跑，但没有降低 TTFT。

---

## 10. llama.cpp 之外还可以尝试什么

如果目标仍然是缩短 TTFT，优先级建议如下。

| 方向 | 是否建议 | 原因 |
|---|---|---|
| OpenVINO GenAI + GPU 继续作为主路径 | 强烈建议 | 当前实测 TTFT 最优，Qwen3/Qwen3.5 到 4B 都成立 |
| 控制 prompt 长度 | 强烈建议 | TTFT 近似随输入 token 数增长，任何框架都受益 |
| 真实业务 prefix caching | 建议 | 多轮或固定 system prompt 场景可降低有效 TTFT，普通 benchmark 不会体现 |
| Qwen3 dense NPU 调小 `MAX_PROMPT_LEN` | 可做小优化 | NPU 会按 max prompt padding，调小可省填充算力，但仍跑不赢 GPU |
| Qwen3.5 固定 GPU only | 强烈建议 | NPU 路径当前异常慢或失败，不应作为默认设备 |
| ONNX Runtime GenAI + OpenVINO EP | 低优先级 | 底层仍是 OpenVINO，通常最多持平，多一层抽象 |
| MLC-LLM / TVM Vulkan | 低优先级 | 无 NPU，Intel iGPU 上预期不如 OpenVINO；Qwen3.5 支持风险更高 |
| vLLM XPU | 不建议 | 更偏服务端 batching/吞吐，不是单流低 TTFT；iGPU 和 Qwen3.5 支持都不是最稳路径 |
| ipex-llm / IPEX-XPU | 不建议 | ipex-llm 已归档；PyTorch eager/XPU 路径通常不适合低 TTFT |

真正可能改变 Qwen3.5 NPU TTFT 的，是 OpenVINO NPU 插件后续补齐 GatedDeltaNet / SSM 的融合实现。这属于上游能力，不是切换到 llama.cpp 能解决的。

---

## 11. 一页复现 checklist

```bash
# 1) 下载 GGUF
cd /home/user/chaofan/model_test
/home/user/envs/model_test_env/bin/python - <<'PY'
from huggingface_hub import hf_hub_download
for repo, filename in [
    ("unsloth/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q4_K_M.gguf"),
    ("unsloth/Qwen3-1.7B-GGUF", "Qwen3-1.7B-Q4_K_M.gguf"),
    ("unsloth/Qwen3-4B-GGUF", "Qwen3-4B-Q4_K_M.gguf"),
    ("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    ("unsloth/Qwen3.5-2B-GGUF", "Qwen3.5-2B-Q4_K_M.gguf"),
    ("unsloth/Qwen3.5-4B-GGUF", "Qwen3.5-4B-Q4_K_M.gguf"),
]:
    print(hf_hub_download(repo_id=repo, filename=filename, local_dir="/home/user/chaofan/model_test/gguf"))
PY

# 2) Vulkan 示例：TTFT + TPOT/TPS
cd /home/user/chaofan/model_test/llama.cpp
export LD_LIBRARY_PATH="$PWD/build/bin:$LD_LIBRARY_PATH"
./build/bin/llama-bench -m /home/user/chaofan/model_test/gguf/Qwen3-4B-Q4_K_M.gguf -p 940 -n 0 -ngl 99 -b 940 --device Vulkan0 -r 3
./build/bin/llama-bench -m /home/user/chaofan/model_test/gguf/Qwen3-4B-Q4_K_M.gguf -p 0 -n 512 -ngl 99 --device Vulkan0 -r 3 -o jsonl

# 3) SYCL F16 示例：TTFT + TPOT/TPS
cd /home/user/chaofan/model_test/llama.cpp
unset ONEAPI_DEVICE_SELECTOR SYCL_DEVICE_FILTER SYCL_DEVICE_ALLOWLIST ZE_AFFINITY_MASK EnableImplicitScaling GGML_OPENVINO_DEVICE SETVARS_COMPLETED
source /opt/intel/oneapi/2025.3/oneapi-vars.sh --force
export LD_LIBRARY_PATH="$PWD/build-sycl-local-f16/bin:$LD_LIBRARY_PATH"
./build-sycl-local-f16/bin/llama-bench -m /home/user/chaofan/model_test/gguf/Qwen3-4B-Q4_K_M.gguf -p 940 -n 0 -ngl 99 -b 940 --device SYCL0 -r 3
./build-sycl-local-f16/bin/llama-bench -m /home/user/chaofan/model_test/gguf/Qwen3-4B-Q4_K_M.gguf -p 0 -n 512 -ngl 99 --device SYCL0 -r 3 -o jsonl

# 4) OpenVINO backend 示例：dense Qwen3 可跑，Qwen3.5 GPU 会 warmup 失败
cd /home/user/chaofan/model_test/llama.cpp
source /opt/intel/openvino_2026/setupvars.sh
export LD_LIBRARY_PATH="$PWD/build-openvino/bin:$LD_LIBRARY_PATH"
GGML_OPENVINO_DEVICE=GPU ./build-openvino/bin/llama-bench -m /home/user/chaofan/model_test/gguf/Qwen3-4B-Q4_K_M.gguf -p 940 -n 0 -ngl 99 -b 940 -ub 940 --device OPENVINO0 -r 3
GGML_OPENVINO_DEVICE=GPU ./build-openvino/bin/llama-bench -m /home/user/chaofan/model_test/gguf/Qwen3-4B-Q4_K_M.gguf -p 0 -n 512 -ngl 99 --device OPENVINO0 -r 3 -o jsonl

# 5) 换算
# TTFT(ms) = prompt_tokens / pp_tokens_per_second * 1000
# TPS(tok/s) = tg_tokens_per_second
# TPOT(ms/token) = 1000 / TPS
```

---

## 12. 最终判断

本轮实测覆盖了 Qwen3-0.6B、Qwen3-1.7B、Qwen3-4B、Qwen3.5-0.8B、Qwen3.5-2B、Qwen3.5-4B，以及 llama.cpp 的 Vulkan、SYCL F16、OpenVINO backend。

```text
OpenVINO GenAI GPU 仍是本机 TTFT 最优路径。
llama.cpp 没有让 TTFT 更低。
llama.cpp SYCL F16 的 decode 能力值得记录，但不能替代低 TTFT 路径。
llama.cpp OpenVINO backend 也不能替代原生 OpenVINO GenAI。
```

后续如果继续探索，建议把精力放在 OpenVINO GenAI 栈内参数、业务 prompt 长度、prefix caching，以及关注 OpenVINO NPU 插件对 Qwen3.5 hybrid 算子的后续优化上。