# 通过优化 OpenVINO 源码降低 Qwen3 / Qwen3.5 TTFT（SDPA backend）—— 复现文档

> **适用范围：`ATTENTION_BACKEND=SDPA`。** 本文记录在 SDPA 路径下落地的三个 GPU 插件源码优化（opt-1/2/3），针对 Qwen3.5（hybrid/SSM）的 prefill。
> ⚠️ **重要：这三个优化是 SDPA 路径专属的。** 在 `ATTENTION_BACKEND=PA`（PagedAttention，OpenVINO 现默认）下它们基本失效（实测总收益从 13–21% 跌到 ~2%），原因见配套的 PA 文档 `OV源码优化降低TTFT_PA.md`。本文保留作为 SDPA 时代的完整方法学与数据记录。
> 代码改动已打成 patch：`benchmark/tests/opt_sdpa_archive/sdpa_opt123.patch`（基于 OpenVINO 2026.2.1 HEAD `ede283a88e`，4 文件）。当前工作树默认 **不含** 这三个优化；需要复现时 `cd benchmark/openvino && git apply ../tests/opt_sdpa_archive/sdpa_opt123.patch`。
> 硬件：Intel Core Ultra X7 358H（Panther Lake）+ Arc B390 iGPU + AI Boost NPU，Ubuntu 24.04。
> 软件：本地源码构建的 OpenVINO 2026.2 + OpenVINO GenAI 2026.2（Release），vtune 2026.2。
> 撰写日期：2026-06-25（opt-3 增补 2026-06-26）。所有 TTFT/op-sum 数值均为 SDPA 口径。
>
> **【356H 补充数据，2026-07-01】** 本文正文数据来自 358H。本次在 **Intel Core Ultra 7 356H**（同 Panther Lake，本机 `python` 环境 `source /home/chaofan/envs/qwen_env/bin/activate`）上按相同方法补测：`model_bench` C++ + 本地源码 OV（同一 HEAD `ede283a88e`）、GPU、n=8、真实 1000-tok prompt；A/+opt1/+opt1+2/+opt1+2+3 四态各自重建 GPU 插件（增量 ~6s）后测量，TTFT 用 `ic=1` 纯 prefill、正确性用 `ic=128` decode 校验。356H 行以「**[356H]**」标注，358H 原数据保留不动。原始测量见 `repro_356H_20260701/`。
> **356H 结论：三优化在 356H 上依然全部有效且数值无损，opt-2 仍是最大贡献项；总降幅 infer-TTFT −10.0~14.1%（vs 358H −12.6~20.8%），量级略小但趋势完全一致（小模型收益更大）。** 356H GPU 存在可复现的逐迭代 md5 非单值现象（见 §7 [356H] 注），故正确性以「A vs D′ 的 md5 多重集逐一相同」判定。

## 0. 结论先行

**思路可行，落地了三个互相独立、可叠加、且数值完全无损（md5 逐字节一致）的 GPU 插件优化，专门加速 Qwen3.5（hybrid/SSM）的 prefill。**

三个优化分别打掉了 prefill 里三类不同的浪费——两类「数据搬运」、一类「重复计算」，都不动 INT4 矩阵乘本体：

| # | 优化 | 针对的 hotspot | 改的文件 | 行数 |
|---|---|---|---|---|
| **opt-1** | depthwise conv 保持 planar 布局，消除 conv 前后的 Reorder | `Reorder`（布局转换） | `layout_optimizer.cpp` | +16 |
| **opt-2** | `permute_ref` 把工作组放到合并访存的 F 轴，修复 LWS 派发 | `Transpose`（permute） | `permute_kernel_ref.cpp` | +31 |
| **opt-3** | 共享同一激活的多个 FC 复用一个 `DynamicQuantize`，消除每层 2–4 次重复量化 | `DynamicQuantize`（激活量化） | `dynamic_quantize_fully_connected.cpp` + `dynamic_quantize.cpp` | +46 / +19 |

**最终实测效果（纯推理 prefill / infer-TTFT，n=8，本地 OV，GPU）：**

| 模型 | A 基线 | +opt-1 | +opt-1+2 (C) | +opt-1+2+3 (D′) | **总降幅** |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B-INT4 | 132.4 ms | 120.2 ms | 108.2 ms | **104.9 ms** | **−20.8%** |
| Qwen3.5-2B-INT4 | 157.7 ms | 146.1 ms | 134.6 ms | **132.3 ms** | **−16.1%** |
| Qwen3.5-4B-INT4 | 333.0 ms | 315.1 ms | 293.8 ms | **290.9 ms** | **−12.6%** |
| Qwen3-1.7B-INT4（dense，对照） | 63.1 ms | 62.3 ms | 62.6 ms | 62.6 ms | −0.7%（不触发，符合预期） |

**[356H]** 同口径（infer-TTFT，n=8，ic=1，本地 OV，GPU）在 356H 上补测：

| 模型 | A 基线 | +opt-1 | +opt-1+2 (C) | +opt-1+2+3 (D′) | **总降幅** |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B-INT4 | 292.3 ms | 289.8 ms | 255.9 ms | **251.1 ms** | **−14.1%** |
| Qwen3.5-2B-INT4 | 348.9 ms | 344.4 ms | 315.3 ms | **309.7 ms** | **−11.3%** |
| Qwen3.5-4B-INT4 | 800.6 ms | 788.1 ms | 731.5 ms | **720.4 ms** | **−10.0%** |
| Qwen3-1.7B-INT4（dense，对照） | 183.0 ms | — | — | 180.7 ms | −1.3%（不触发，符合预期） |

> 356H 绝对值约为 358H 的 2×（iGPU 规格更低），但三优化叠加趋势一致：opt-2 是最大项（356H 上 C 相对 B 降 29~57ms），opt-1/opt-3 各贡献几 ms，dense 基本不动。三个 Qwen3.5 的 A vs D′ md5 多重集逐一相同（见 §7 [356H]）。

- 三个 Qwen3.5 的生成结果 **md5 与基线逐字节一致**（正确性见 §7；opt-3 的 prefill 与 128-token decode md5 都逐字节一致，见 §7.4）。
- dense Qwen3 不受影响（三个优化都只针对 SSM/hybrid 特有的算子模式或图结构）。
- 三个优化是**逐轮 re-profile 找出来的**：opt-1/opt-2 来自基线 profile 的两大搬运项（§2.2），opt-3 来自「opt-1/opt-2 落地后再 profile」时升为第二大算子的 `DynamicQuantize`（§5.1）。每个优化的小幅但叠加，构成总降幅。
- **§6 专门回答**「profiling 里 Reorder+Transpose 占 ~50%，为什么整体只提升 ~10-18%」——这不是 profiling 出错，而是「占比的分母」和「TTFT 的分母」不是同一个量。

一句话给 leader：**改 OV 源码 + C++ profiling 这条路可行。我们在 Qwen3.5 上逐轮 profile，找到并修掉了三处浪费——depthwise conv 的多余 Reorder、转置 kernel 的 GPU 工作组派发缺陷、以及共享激活的多个 FC 重复做激活量化——纯 prefill 降了 13–21%（小模型更明显），结果逐字节一致；dense Qwen3 已在算力下限，无可优化空间。**

---

## 1. 为什么这条思路可行（先把闭环跑通）

「改 OV → 测效果」需要三件事成立，逐一验证：

### 1.1 本地 OV 是完整 Release 构建
```bash
C=/home/user/chaofan/model_test/benchmark/openvino/build/CMakeCache.txt
grep -iE "^CMAKE_BUILD_TYPE|ENABLE_INTEL_GPU:|ENABLE_INTEL_NPU:|CMAKE_INSTALL_PREFIX:" "$C"
```
实测：`CMAKE_BUILD_TYPE=Release`、GPU/NPU/CPU 插件全开、`CMAKE_INSTALL_PREFIX=.../benchmark/openvino/install`。runtime 库、GPU 插件、GenAI、cmake config 都在 `install/` 里齐全。

> ⚠️ CMake 缓存在 `benchmark/openvino/build/`（不是 `benchmark/openvino/`）。构建产物 `.so` 落 `benchmark/openvino/bin/intel64/Release/`，`install/` 是独立拷贝。

### 1.2 让 model_bench 链接「本地 OV」而非 /opt/intel
```bash
cd /home/user/chaofan/model_test/model_bench
source /home/user/chaofan/model_test/benchmark/openvino/install/setupvars.sh   # 导出 OpenVINO[GenAI]_DIR
cmake -S . -B build-local -DCMAKE_BUILD_TYPE=Release
cmake --build build-local -j"$(nproc)"
ldd build-local/model_bench | grep -iE "openvino|genai"    # 全部指向 .../benchmark/openvino/install/...
```
每次运行 `build-local/model_bench` 前：
```bash
source /home/user/chaofan/model_test/benchmark/openvino/install/setupvars.sh >/dev/null 2>&1
export LD_LIBRARY_PATH="/home/user/chaofan/model_test/benchmark/openvino/install/runtime/lib/intel64:${LD_LIBRARY_PATH:-}"
```

### 1.3 改源码后增量重建 GPU 插件只要 ~9 秒（关键）
脚本 `benchmark/openvino/rebuild_gpu_plugin.sh`：
```bash
#!/bin/bash
set -eo pipefail
OV=/home/user/chaofan/model_test/benchmark/openvino
cd "$OV/build"
make openvino_intel_gpu_plugin -j"$(nproc)"
cp -v "$OV/bin/intel64/Release/libopenvino_intel_gpu_plugin.so" \
      "$OV/install/runtime/lib/intel64/libopenvino_intel_gpu_plugin.so"   # 必须刷新 install 里的副本
echo "GPU plugin rebuilt + installed."
```
> 为什么必须 `cp` 到 install：`model_bench` 加载的是 `install/runtime/lib/intel64` 里的 `.so`，而 `make` 只更新 `bin/intel64/Release`。两者不同 inode，不刷新就测的是旧插件——最容易踩的坑。

**三件事都成立，思路可行。**

---

## 2. 建立基线 + 找 hotspot（迭代式）

**本文三个优化不是一次想全的，而是「profile → 改掉最大项 → re-profile → 改掉新的最大项」迭代出来的。** 这一节先讲指标口径与基线 profile（产出 opt-1/opt-2 的目标）；opt-3 的目标是在 opt-1/opt-2 落地后再 profile 才浮现的，见 §5.1。

### 2.1 两个指标的口径（务必分清）

| 指标 | model_bench 字段 | 含义 |
|---|---|---|
| **infer-TTFT** | `first_token_infer_mean_ms` | **纯推理**首 token 延迟（去掉 tokenize 等），最适合衡量 prefill 计算本身，本文优化主要看它 |
| **e2e-TTFT** | `ttft_mean_ms` | 端到端首 token 延迟，含 GenAI pipeline 的 host 开销（embeddings 调用、采样、调度等）；业务真正感受到的 |

两者差异（§6 会用到）：e2e 比 infer 大 ~40ms，主要是 GenAI VLMPipeline 的 host 开销，**不是** tokenization（实测 tokenization 仅 0.6ms）。

### 2.2 找 hotspot：model_bench `--profile`（逐算子）

`--profile` 用 raw `ov::Core` 复现真实 prefill+decode 循环，调 `get_profiling_info()` 导出每个算子的 `node_type/exec_type(kernel)/real_time`，prefill 和 decode 分开。这是 Python `openvino_genai` 拿不到的能力。

```bash
cd /home/user/chaofan/model_test/model_bench
# -pf 提供真实 prompt；--profile 现已自动复用 -pf 的真实长度（见附录），日志打印 using REAL prompt
./build-local/model_bench vlm -m ../benchmark/models/Qwen3.5-2B-INT4 \
  -d GPU -n 1 -ic 8 -pf ./prompts/llama-2-7b-chat_l.jsonl --no-monitor \
  --profile --profile-iters 8 --profile-warmup 3 --profile-topk 200 \
  --profile-csv /tmp/profA_2b.csv -r /tmp/x.csv -rj /tmp/x.json
```

**基线 prefill 逐算子分解（Qwen3.5-2B，938 token，op-time SUM）：**
```
Transpose                 20.98ms  31.9%   <- permute_ref，最大单项      → opt-2
FullyConnectedCompressed  16.60ms  25.2%   <- INT4 矩阵乘（已最优）
Reorder                   13.81ms  21.0%   <- conv 前后布局转换          → opt-1
DynamicQuantize            5.41ms   8.2%   <- 此时只占 8.2%，还不显眼    → 后来升为 opt-3 目标
GatedDeltaNet              2.95ms   4.5%   <- 融合 SSM 算子本身只占 4.5%
```
> ⚠️ 这里的百分比是「**逐算子 real_time 求和**」的占比，分母是 op-sum（≈66ms），**不是** wall-clock TTFT（≈158ms）。两者关系是本文 §6 的核心，先记住这个区别。

**关键观察：Qwen3.5 prefill 里 Transpose+Reorder 合计 ~53%（op-sum 口径）是纯数据搬运**，矩阵乘只占 25%。对照 dense Qwen3-1.7B（94%+ 在矩阵乘+量化，几乎无布局浪费、已在算力下限）。所以 hybrid 模型的可优化点首先在「搬运」——这正是 opt-1（§3）和 opt-2（§4）的目标。

### 2.3 vtune 交叉验证（kernel 级）
```bash
source /opt/intel/oneapi/setvars.sh
vtune -collect gpu-hotspots -r /tmp/vtune_q35 -- model_bench/build-local/model_bench vlm \
  -m benchmark/models/Qwen3.5-2B-INT4 -d GPU -n 3 -ic 16 --input-tokens 950 --no-monitor ...
vtune -report hotspots -group-by computing-task -r /tmp/vtune_q35 -format csv
```
独立确认了同一批 hotspot：`gemm_kernel`、`permute_ref`、`reorder_data`、`dynamic_quantize_gpu_opt` 等（原始数据库见 `vtune-base/results-gpu-qwen3.5-2B/`）。vtune 还给出每个 kernel 的硬件效率：`dynamic_quantize_gpu_opt` 在 prefill 是 40.5% active / 36.4% XMX 的**实打实计算**（不是带宽 bound）——这为 opt-3 的「冗余计算」判断提供了佐证（§5.1）。

---

## 3. 优化 1：消除 depthwise conv 的 Reorder

### 3.1 根因（源码定位 `layout_optimizer.cpp`）
- Qwen3.5 SSM 层的因果 conv1d 在 IR 里是 `GroupConvolution`：input `[?,6144,?]`、weights `[6144,1,1,4]`，即 `groups == in_ch == out_ch == 6144`、kernel=4——**完全 depthwise**（每个输出通道只读 1 个输入通道）。
- 这个 conv 是 **dynamic shape**（seq=-1）。Battlemage 是 immad 设备，走 `set_onednn_dyn_conv_preferred_format()`，该函数对 fp16 dynamic conv **无条件设成 fsv16 blocked 布局**，只有 `channels<=16` 才退回 planar——我们 6144 通道，于是被设成 fsv16。
- conv 邻居都是 planar bfyx，于是 `reorder_inputs` 在 conv 前后各插一个 Reorder（bfyx↔fsv16）。

**为什么是浪费**：depthwise conv 没有跨通道复用，blocked 布局对 conv kernel 毫无收益；kernel=4 的 conv1d 计算量极小。结果是「为一个不需要 blocked 的算子，付了两次全 tensor 的布局搬运」。

### 3.2 改动（+16 行）
在 `set_onednn_dyn_conv_preferred_format()` 拿到 in/out 通道数后、设布局前加 depthwise 短路：
```cpp
// depthwise conv (groups==in==out) gains nothing from blocked fsv16 -> keep planar, no reorder.
if (input_channels > 0 && static_cast<uint32_t>(input_channels) == node.get_groups() &&
    input_channels == output_channels) {
    node.set_preferred_input_fmt(0, format::get_default_format(rank));
    node.set_preferred_output_fmt(0, format::get_default_format(rank));
    return;
}
```
要点：用 dynamic-safe 的 `get_convolution_channel_count()`（**不能用 `is_depthwise(node)`**，后者调 `feature()→get_dims()` 对 dynamic 会抛 `get_dims() is called for dynamic shape`）；dynamic OCL conv impl 原生支持 bfyx，conv 仍正确执行。

### 3.3 效果验证
重建后 re-profile：**36 个 Reorder（18 层 ×2）全部消失**，Reorder op-time 从 13.81ms → 0。墙钟 infer-TTFT 的增量见 §8.1（2B −11.6ms）。

---

## 4. 优化 2：修复 Transpose（permute_ref）的 GPU 工作组派发

### 4.1 关键发现：每层 2 个同形状 transpose，成本差 10×
Qwen3.5 每个 SSM 层有两个 `[1,6144,S]↔[1,S,6144]`（order `[0,2,1]`）transpose，同一个 `permute_ref` kernel，但 op-time 差 10×：
- **pre-conv**（conv 前，`[B,S,6144]→[B,6144,S]`，喂给 Concat→Conv）：~76 µs/层
- **post-conv**（conv 后，`[B,6144,S]→[B,S,6144]`，喂给 VariadicSplit）：~778 µs/层

### 4.2 用调试打印定位真实机制（实测推翻了直觉）
在 `permute_kernel_ref.cpp::SetDefault` 临时加一行受环境变量 `MB_PERMUTE_DBG` 控制的 `fprintf`，打印每次派发的 `gws/lws`，再用独立 micro-benchmark 扫不同 S：

| S（因子） | 选中的 LWS | 单次耗时 |
|---|---|---:|
| 937（质数） | `{1,1,1024}` | **965 µs（快）** |
| 938 = 2·7·67 | `{1,7,128}` | 2123 µs（慢） |
| 1024 = 2¹⁰ | `{1,1024,1}` | 2393 µs（慢） |
| 1023 = 3·11·31 | `{1,1,1024}` | **1146 µs（快）** |

**机制（和「整除性好就快」恰好相反）**：`permute_ref` 的 else 分支 GWS=`{X=1, Y=S, F·B=6144}`，输出写 `OUT_IDX` 沿 **F 轴连续**。`GetOptimalLocalWorkGroupSizes` 用 bfyx 优先级 x→y→f，**贪心地先把 LWS 放在 Y(=S) 轴**：
- S 整除性好 → 大 LWS 落在 Y 轴（这是写散乱的方向）→ 访存不合并 → **慢**
- S 是质数/无好因子 → Y 上只能放 1，整组 1024 落到 **F 轴（写合并方向）→ 快**

所以「慢」恰恰发生在 S 整除性好的时候，这是 LWS 分配启发式的一个缺陷。（之前我先假设「让专用 kernel `permute_f_y_axes` 接管」能加速，micro-bench 实测它反而慢 1.8×，已否决——可优化点必须实测。）

### 4.3 改动（+31 行，纯 LWS 选择，不改数学）
在 else 分支拿到分配器结果后，当 F·B 轴占主导时把 LWS 强制放到 F 轴（取 ≤maxWG 的最大 F 因子），仅当不缩小工作组时才覆盖：
```cpp
const size_t gws_y = dispatchData.gws[1];
const size_t gws_f = dispatchData.gws[2];
const size_t cur_total_lws = dispatchData.lws[0]*dispatchData.lws[1]*dispatchData.lws[2];
const size_t max_wg = params.engineInfo.maxWorkGroupSize ? params.engineInfo.maxWorkGroupSize : 256;
if (gws_f >= gws_y && gws_f >= 256) {
    size_t lws_f = 1;
    for (size_t cand = std::min<size_t>(max_wg, gws_f); cand >= 1; --cand)
        if (gws_f % cand == 0) { lws_f = cand; break; }
    if (lws_f >= cur_total_lws && lws_f > dispatchData.lws[2])
        dispatchData.lws = {1, 1, lws_f};
}
```
因为只改 LWS（工作组怎么切分），不动任何索引/数学，所以数值必然一致。

### 4.4 效果验证
standalone：所有 S 的「悬崖」消失（S=938: 2123→979 µs，S=1024: 2393→1193 µs，全部 ~1ms）。
in-model re-profile：post-conv transpose op-time 18.7ms → 3.97ms（4.7×），Transpose 总 op-time 20.69ms → 5.76ms。墙钟增量见 §8.1（2B −11.5ms）。

> ⚠️ 测量陷阱（这次踩到了）：infer-TTFT 对 warmup 敏感。`-n 5` 数据抖到 147ms，`-n 8`（更充分 warmup）稳定在 134ms。所有 A/B 对比统一用 `-n 8`。

---

## 5. 优化 3：共享激活的多个 FC 复用一个 DynamicQuantize

### 5.1 怎么发现的：opt-1/opt-2 落地后再 profile，DynamicQuantize 升为第二大算子
opt-1/opt-2 把两大搬运项干掉后，重新跑 `--profile`（真实 938-token prompt）看 Qwen3.5-2B prefill 的剩余结构：

```
FullyConnectedCompressed  16.65ms  44.6%   <- INT4 矩阵乘（XMX 最优，不动）
Transpose                  5.74ms  15.4%   <- opt-2 已压过，剩余靠图级消除（§9）
DynamicQuantize            5.35ms  14.3%   <- 175 个节点！新的第二大算子
GatedDeltaNet              2.89ms   7.7%
StridedSlice               1.85ms   5.0%
```

注意：`DynamicQuantize` 的**绝对时间没变**（基线 5.41ms → 现在 5.35ms），但**占比从 8.2% 升到 14.3%**——纯粹是因为分母（op-sum）被 opt-1/opt-2 从 66ms 缩到了 37ms。这正是「迭代 profile」的意义：把大头削掉后，原来第四的算子浮上来成了新目标。**关键线索**：175 个 DQ 节点远多于「~88 个 FC 应有一份」的量级——存在冗余。

### 5.2 根因（IR + 转换 pass 双重定位）
- Qwen3.5 每个 decoder 层里，**4 个 FC（`in_proj_qkv`/`in_proj_a`/`in_proj_b`/`in_proj_z`）共享同一个 `input_layernorm` 输出**作为激活输入；**2 个 FC（`mlp.gate_proj`/`up_proj`）共享同一个 `post_attention_layernorm` 输出**（用 IR 边分析确认，全模型 48 组共享输入）。
- GPU 插件的 `DynamicQuantizeFullyConnected` pass 在**每个** FC 前**各插一份** `DynamicQuantize`（`dynamic_quantize_fully_connected.cpp:108`，`std::make_shared<DynamicQuantize>(m_fc->input_value(0), config)`，无任何去重）。于是同一份 hidden state 被重复量化 **4 次 / 2 次每层**。
- 为什么通用 CSE 不管：OV 的 `ov::pass::SharedOpOptimization` 在 pipeline **更早**就跑完了（`transformations_pipeline.cpp:498`），而 DQ 是在 1685 行才插入的——插入时已无人去重；且 `DynamicQuantize` **没有实现 `visit_attributes`**，通用 CSE 比较属性时拿到空 map，会把配置不同的 DQ 也错误合并，不能直接复用。

### 5.3 改动（两处）
**1. `dynamic_quantize_fully_connected.cpp`（+46 行）——去重**：插 DQ 前，在「同一激活的现有消费者」里找一个**配置逐字段相等**（`group_sizes`/`quantization_type`/`scale_dt`/`zp_dt`/`precomputed_reduction…` 全等，见 `dq_attrs_equal`）的 DQ 复用；找不到才新建。配置不等就不合并 → **零正确性风险**，最坏只是少省一点。
```cpp
auto dyn_quan = find_reusable_dq(m_fc->input_value(0), config);
if (!dyn_quan)
    dyn_quan = std::make_shared<ov::op::internal::DynamicQuantize>(m_fc->input_value(0), config);
```
**2. `dynamic_quantize.cpp`（+19 行）——放宽 skip 断言**：原断言 `users==outputs_count`（一个 DQ 只喂一个 FC、它消费 DQ 的全部输出 = 2/3 个，故 users==outputs）。共享后 `users == outputs_count × 消费者数`。decode（batch=1）阶段 DQ 本就被 skip 成 f16 passthrough，N 个 FC 读同一个 passthrough buffer 与「各自一份被 skip 的 DQ」**数值完全等价**。故放宽为「users 是 outputs 的整数倍且全部为 FC」即可继续 skip——这一步是保住 decode 逐字节一致的关键（见 §5.4 踩坑）。

### 5.4 踩坑记录（一个调试循环的代价）
第一版只做了 §5.3 的第 1 处去重，并在 `mark_runtime_skippable_nodes.cpp` 里**禁止**共享 DQ 被标记 skippable（这样图能跑通，不触发那条断言）。结果：

- prefill 首 token md5 与基线**逐字节一致**（ic=1 验证），但
- **128-token decode md5 变了**。

根因：禁掉 skip 后，decode 改走「量化成 i8 再算 FC」，而基线 decode 是「DQ 被 skip 成 f16 passthrough、FC 直接吃 f16」。两条路输出都合法但数值不同。

修法：**撤掉**对 `mark_runtime_skippable_nodes.cpp` 的改动，改为 §5.3 的第 2 处「放宽 skip 断言」——让共享 DQ 在 decode 仍能 skip 成 f16，N 个 FC 读同一 passthrough，与基线逐字节一致。**教训：DQ 的 skip 是 decode 专属优化，任何碰 skippable 的改法都会改 decode 数值，必须让共享 DQ 仍可 skip。** 正确性复核见 §7.4。

### 5.5 效果验证
re-profile（Qwen3.5-2B prefill）：DynamicQuantize 节点 **175 → 97**（−78，与「24 层 × 每层（4-FC 组省 3 + 2-FC 组省 1）= 24×4 ≈ 96 处冗余」的量级预测一致，差额来自非每层都命中『配置逐字段相等』而被复用），DQ op-time **5.35ms → 3.44ms（−36%）**。墙钟增量（交错 A/B 实测）：0.8B −3.3ms、2B −2.3ms、4B −2.9ms，见 §8.1。

> opt-3 比 opt-1/opt-2 改动「更靠图结构」，所以收益相对小（搬运是 ~13/15ms 的大块，重复量化只有 ~2ms），但它**完全通用**：任何「多个 FC 共享同一激活」的模型（绝大多数 transformer 的 qkv / gate-up）都受益，不限 Qwen3.5。

---

## 6. 为什么 profiling 占 ~50%，整体只提升 ~10-18%？（重点解释，非 profiling 出错）

这是最容易困惑的点（主要针对 opt-1+opt-2：Reorder+Transpose 占 profile ~50%）。**结论：profiling 没出错。问题在于「占比的分母」和「TTFT 的分母」是两个完全不同的量。**

### 6.1 三个不同的「时间」量（Qwen3.5-2B，938 token，基线）

| 量 | 数值 | 是什么 |
|---|---:|---|
| **deep-profile op-sum** | **65.8 ms** | `--profile` 把每个算子的 `real_time` **逐个相加**。`Transpose 31.9% / Reorder 21.0%` 这些百分比的分母就是它 |
| **infer-TTFT** | **157.7 ms** | GenAI pipeline 实测的纯推理首 token 墙钟时间 |
| **e2e-TTFT** | **201.4 ms** | 端到端首 token 墙钟时间（含 host 开销） |

**关键：「Reorder 占 21%、Transpose 占 32%」这个百分比，分母是 op-sum 的 65.8ms，不是 TTFT 的 157.7ms 或 201.4ms。** 所以「占比 ~50%」从一开始就不是「占 TTFT 的 50%」。

### 6.2 为什么 op-sum(66ms) 和 infer-TTFT(158ms) 差这么多？
两者是**两条不同的执行路径**，不能直接相减：
- **op-sum** 来自 `--profile` 的 raw `ov::Core` 循环，它只跑 language-model 图、且把各算子 `real_time` 简单相加。`get_profiling_info` 的 `real_time` 是「每个算子被计时的时长」，**它既不等于墙钟（GPU 上多 kernel 可并行/排队），也不含 GenAI pipeline 的其它阶段**。OV 官方文档明确说：「由于并行执行，per-op real_time 之和可能与 wall-clock 不一致，应作为相对分解、而非可加的时间预算来读。」
- **infer-TTFT** 来自真正的 `VLMPipeline`，除了 LM 图，还含 text-embeddings 模型调用、PagedAttention 的 KV 准备、首 token 的采样/调度等。

所以 op-sum 只能告诉你「**哪个算子相对最重、改完哪个相对变轻了**」，不能用它的百分比直接预测墙钟提升百分比。

### 6.3 用「绝对毫秒」对账（这才是对的口径）

| | op-sum（--profile） | infer-TTFT（墙钟） |
|---|---:|---:|
| 基线 A | 65.8 ms | 157.7 ms |
| +opt-1（去 Reorder） | 52.4 ms（−13.4ms） | 146.1 ms（**−11.6ms**） |
| +opt-2（修 Transpose） | 37.3 ms（−15.1ms） | 134.6 ms（**−11.5ms**） |
| +opt-3（DQ 复用） | 36.5 ms（−0.8ms\*） | 132.3 ms（**−2.3ms**） |
| 合计 A→D′ | −29.3 ms | **−25.4ms** |

\* op-sum 口径下 DQ 节点降了 1.9ms，但 op-sum 合计只降 0.8ms——因为 op-sum 噪声较大、且被去掉的 DQ 与相邻算子有重叠；墙钟（−2.3ms）才是准的。

**对账结论**：opt-1 在 profile 里移除了 13.4ms 的 Reorder op-time，实测墙钟 infer-TTFT 降了 11.6ms——**绝对毫秒高度吻合**（差异来自并行/重叠和路径差异）。opt-2 同理（移除 ~15ms transpose op-time → 墙钟降 11.5ms）；opt-3 量级小（~2ms），但同样可对账。

所以正确的理解是：
- ❌ 错误推算：「Reorder 占 profile 的 21%，所以 TTFT 应降 21%」——拿 op-sum 的占比套到 TTFT 的分母上，分母用错了。
- ✅ 正确推算：「Reorder 是 ~13ms 的绝对开销，去掉后 TTFT 从 158→146ms，即 −11.6ms / −7.3%」。13ms 占 158ms 的 infer-TTFT 约 8%、占 201ms 的 e2e-TTFT 约 6%——所以单个搬运优化 ~6-8%，叠加 opt-3 后总计 ~13-21%，与实测完全一致。

### 6.4 「Reorder 基本消除」与「只提升 7%」并不矛盾
- 「Reorder **基本消除**」指的是：profile 里 36 个 Reorder 算子的 op-time 从 13.8ms 降到 **0**（这是真的，re-profile 验证过）。
- 「只提升 ~7%」指的是：这 13.8ms 的绝对节省，相对于 158ms 的 infer-TTFT（或 201ms 的 e2e）是 ~7-9%。

两句话说的是同一件事的两个口径，没有冲突：**算子被完全干掉了，但它本来也只占总时间的 ~8%（而不是占 profile 的 21% 那个数）。**

### 6.5 那 prefill 的大头在哪？
去掉两处搬运 + 一处重复量化后，prefill 的绝对大头回到了 **FullyConnectedCompressed（INT4 矩阵乘，op-sum 占比从 25% 升到 44%）+ 剩余 DynamicQuantize + GatedDeltaNet 本体**——这些是真正的计算，已走 oneDNN XMX 最优路径，**不是局部源码能再压的**（详见对 dense Qwen3 的同类结论）。所以 ~13-21% 基本就是「能从搬运/冗余里榨出来的全部」，再往下要动图结构（见 §9）。

---

## 7. 正确性：三个优化都不改变模型结果（含 md5 是什么）

### 7.1 概念上为什么不该变
- **opt-1（format）**：blocked fsv16 vs planar bfyx 只是数据在内存里怎么排，两种布局下的 conv kernel 算同一个卷积；reorder 是纯数据拷贝。换成 kernel 本来就支持的布局、从而不需要 reorder，不可能改变算术。
- **opt-2（LWS）**：只改 GPU 工作组怎么切分（每个 work-item 算哪个元素不变），按定义不改结果。
- **opt-3（DQ 复用）**：复用的 DQ 与原本各自的 DQ **输入相同（同一激活）+ 配置逐字段相等**，量化输出必然 bit 级相同；decode 的 skip 路径被保留（§5.3 第 2 处），f16 passthrough 也与基线一致。是三者里唯一动「图结构」的，所以单列 §7.4 做更细的 prefill/decode 双重校验。

### 7.2 md5 是什么、怎么判断没改变结果
**`result_md5` = 对「模型解码后的输出文本」做的 MD5 哈希**，不是权重、不是 logits。代码（`model_bench/src/tasks/vlm.cpp:281,342`、`llm.cpp:264,321`）：
```cpp
std::string output_text = res.texts.empty() ? "" : res.texts[0];   // VLM：生成的文字
m.result_md5 = MD5::hex(output_text);
// LLM：output_text = tokenizer.decode(res.tokens[0]);
```
**为什么 md5 一致就能证明结果没变**：benchmark 用**贪心解码**（`do_sample=false`），给定输入是确定性的——每步取 logits 的 argmax。让它生成 **128 个 token**，任何一步的数值扰动都可能让 argmax 选到不同 token，一旦选错后续整条序列发散、文本必然不同。所以 **128 步生成文本逐字节相同（md5 相同）⇒ 这 128 步每一步的数值都没有发生足以翻转 argmax 的变化** ⇒ 优化数值无损。这是一个「确定性解码 + 长序列累积放大」的端到端行为指纹，专门适合验证布局/派发/图结构类改动。

> 一个口径细节：VLM（Qwen3.5）路径默认**关闭** prompt permutation，所以 8 次迭代用同一个 prompt → 8 个 md5 完全相同（单一稳定 md5）。LLM（dense Qwen3）路径默认**开启** prompt permutation（每次迭代换一个 token 防前缀缓存），所以 8 次迭代会得到一组**不同的** md5——这是 permutation 故意造成的，不是数值不稳。判断「优化是否改变结果」时，要比的是 **基线和优化态产出的 md5 多重集（multiset）是否完全相同**，而不是「单个 md5 是否稳定」。

### 7.3 实测（128-token 贪心，基线 A vs 三优化 D′，n=8，比 md5 多重集）

| 模型 | permutation | 每个状态的 md5 情况 | A vs D′ 多重集 |
|---|---|---|---|
| Qwen3.5-0.8B-INT4 | OFF | 8 迭代全为 `8c2ae831…`（单一） | ✅ 完全相同 |
| Qwen3.5-2B-INT4 | OFF | 8 迭代全为 `0a7720c3…`（单一） | ✅ 完全相同 |
| Qwen3.5-4B-INT4 | OFF | 8 迭代全为 `c7d14131…`（单一） | ✅ 完全相同 |
| Qwen3-1.7B-INT4（dense） | ON | 8 迭代 6 个不同 md5（permutation 所致），如 `1856c535…`×3 等 | ✅ 多重集逐一相同 |

即：3 个 Qwen3.5 优化前后**单一 md5 逐字节一致**；dense Qwen3 优化前后**那一组 6 个非确定性 md5 的多重集（含重复次数）也逐一相同**——两种情况都证明优化没有改变任何一次生成的输出。

> **[356H] 重要差异：356H 上即便 permutation OFF、固定 prompt，8 次迭代的 md5 也不是单一值**（如 SDPA 2B ic=128 出现 8 个不同 md5、0.8B ic=1 出现固定三值多重集 `514faf5a×2|cae8d14e×5|dc27db97×1`）。但这是**可复现的**——跨进程重跑得到逐位置完全一致的 md5 序列（连测两次序列一字不差），属 356H iGPU 在长序列贪心下的可复现数值抖动，非随机。因此 356H 的正确性判据统一用「**基线 vs 优化态的 md5 多重集是否逐一相同**」（与本节 dense 那一行同口径）。实测：
> - **SDPA**：pristine(A) vs opt-1+2+3(D′)，2B ic=128 的 8 元 md5 多重集**逐一相同**；0.8B/2B/4B 各态 ic=1 多重集也与其 A 相同（2B/4B 为单一 `dd753679`/`e067b6f4`，0.8B 为固定三值集）。
> - **PA**（见 PA 文档）：pristine vs opt-4+opt-3，2B ic=128 的 7 元 md5 多重集**逐一相同**。
> 结论：opt-1/2/3（及 PA 的 opt-4）在 356H 上均**数值无损**，只是 356H 的「无损」表现为「多重集一致」而非 358H 的「单一 md5 一致」。

### 7.4 opt-3 专项：prefill 与 decode 都逐字节一致
opt-3 改的是「算子图的结构」（合并节点），且踩过一次坑（§5.4），所以单独把 prefill 和 decode 分开校验：
- **prefill 精确**：实测 ic=1（仅 prefill 出 1 个 token）md5 在 C（opt-1+2）与 D′（opt-1+2+3）下**逐字节一致**（`dd753679…`）——证明去重对 prefill 数值无损。
- **decode 精确**：放宽 skip 断言后（§5.3），共享 DQ 在 decode 仍被 skip 成 f16 passthrough，N 个 FC 读同一 buffer，与基线「各自一份被 skip 的 DQ」等价。
- **端到端**：128-token 贪心 md5，C vs D′ **三个尺寸全部逐字节一致**（与 §7.3 表中同一组 md5）。

| 模型 | prefill md5 (ic=1) | 128-token md5 | C vs D′ |
|---|---|---|---|
| Qwen3.5-0.8B-INT4 | C=D′ `dd753679…` | `8c2ae831…`（单一） | ✅ 逐字节一致 |
| Qwen3.5-2B-INT4 | C=D′ `dd753679…` | `0a7720c3…`（单一） | ✅ 逐字节一致 |
| Qwen3.5-4B-INT4 | — | `c7d14131…`（单一） | ✅ 逐字节一致 |

### 7.5 作用域：会不会影响别的模型？
- **opt-1** 仅命中「dynamic + immad + 完全 depthwise（groups==in==out）」的 conv。实测枚举本机所有模型，只有 Qwen3.5 系列有 conv；连同一个 Qwen3.5 里的 vision-tower patch-embedding conv（3→1024，groups=1，非 depthwise）都**不触发**。对 MobileNet/EfficientNet/ConvNeXt/Conformer 等含 depthwise 的模型，blocked 布局对 depthwise 本就无正确性必需，最坏只是性能权衡；且系统会在格式不匹配处自动重新插 reorder（自我纠正），**最坏结果是性能回退，绝不会算错或编译失败**。
- **opt-2** 仅在 permute_ref 的 else 分支、F·B 轴占主导且当前 LWS 偏小时触发，只放大工作组、从不缩小；纯 LWS 选择，对所有 permute 都数值无损。
- **opt-3** 只命中「同一激活、配置逐字段相等」的 FC（Qwen3.5 的 q/k/v/z 与 gate/up）。配置不等的情形一律不合并（§5.3）；decode skip 行为保留。dense Qwen3 的 attention 用标准 q/k/v 投影同样可能共享输入，若命中也只是同样的无损复用——实测 dense md5 多重集逐一相同。

> 局限性：md5 比的是「这个固定 prompt + 贪心」下的输出，是非常强的回归证据但非形式化证明；要更严格可加多 prompt / 比 logits 余弦相似度。但对布局/派发/图结构类改动，长序列贪心 md5 已足够——这类改动要么完全不影响，要么会让某步 argmax 翻转从而被立刻抓到（opt-3 第一版的 decode 偏差正是这样被抓到的，§5.4）。

---

## 8. 完整测试数据（e2e + infer，每个优化 + 总计）

口径：本地 OV，GPU，`-pf ./prompts/llama-2-7b-chat_l.jsonl`（真实 prompt，input≈950 token），`-n 8`（+1 warmup），`-ic`=16/128。A/B/C 三态用 `git stash` 切换插件分别构建测量；**C→D′ 用「插件 .so 热插拔交错 A/B」**（同一会话内反复 `cp plugin_C.so / plugin_Dprime.so` 到 install 后立即测，规避构建间的热漂移——这次正是靠它纠正了一个把 4B 冷启动 317ms 误当成 opt-3 收益的假信号）。
- A = 基线（无改动）｜B = 仅 opt-1（Reorder）｜C = opt-1 + opt-2（Reorder + Transpose）｜**D′ = opt-1+2+3（＋DynamicQuantize 复用）**

### 8.1 infer-TTFT（纯推理 prefill，`first_token_infer_mean_ms`）

| 模型 | A 基线 | B (+opt-1) | C (+opt-1+2) | D′ (+opt-3) | opt-1 增量 | opt-2 增量 | opt-3 增量 | **A→D′ 总计** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B | 132.4 ms | 120.2 ms | 108.2 ms | **104.9 ms** | −12.2 (−9.2%) | −12.0 (−10.0%) | −3.3 (−3.0%) | **−27.5ms (−20.8%)** |
| Qwen3.5-2B | 157.7 ms | 146.1 ms | 134.6 ms | **132.3 ms** | −11.6 (−7.4%) | −11.5 (−7.9%) | −2.3 (−1.7%) | **−25.4ms (−16.1%)** |
| Qwen3.5-4B | 333.0 ms | 315.1 ms | 293.8 ms | **290.9 ms** | −17.9 (−5.4%) | −21.3 (−6.8%) | −2.9 (−1.0%) | **−42.1ms (−12.6%)** |
| Qwen3-1.7B（dense） | 63.1 ms | 62.3 ms | 62.6 ms | 62.6 ms | — | — | — | −0.5ms / −0.7%（不触发，噪声内） |

> opt-3 增量为「C 与 D′ 插件热插拔交错」实测（同一会话、相邻测量，3 reps σ≤0.1ms）。opt-3 在小模型上收益占比更大（DQ 占其小 prefill 的比例更高）。

**[356H]** infer-TTFT（`first_token_infer_mean_ms`，n=8，ic=1，四态各自重建插件测量）：

| 模型 | A 基线 | B (+opt-1) | C (+opt-1+2) | D′ (+opt-3) | opt-1 增量 | opt-2 增量 | opt-3 增量 | **A→D′ 总计** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B | 292.3 ms | 289.8 ms | 255.9 ms | **251.1 ms** | −2.5 (−0.9%) | −33.9 (−11.7%) | −4.8 (−1.9%) | **−41.2ms (−14.1%)** |
| Qwen3.5-2B | 348.9 ms | 344.4 ms | 315.3 ms | **309.7 ms** | −4.5 (−1.3%) | −29.1 (−8.4%) | −5.6 (−1.8%) | **−39.2ms (−11.2%)** |
| Qwen3.5-4B | 800.6 ms | 788.1 ms | 731.5 ms | **720.4 ms** | −12.5 (−1.6%) | −56.6 (−7.2%) | −11.1 (−1.5%) | **−80.2ms (−10.0%)** |
| Qwen3-1.7B（dense） | 183.0 ms | — | — | 180.7 ms | — | — | — | −2.3ms / −1.3%（不触发，噪声内） |

> 356H 与 358H 定性一致：opt-2 是最大贡献项（356H 占总降幅约 7~8 成），opt-1/opt-3 各贡献 1~2%。dense 在噪声内。注意 356H 上 opt-1 相对占比比 358H 小（−0.9~1.6% vs 358H −5.4~9.2%）——depthwise-conv Reorder 在这块 iGPU 上的绝对开销较小。

### 8.2 e2e-TTFT（端到端，`ttft_mean_ms`）

| 模型 | A 基线 | B (+opt-1) | C (+opt-1+2) | D′ (+opt-3) | opt-3 增量 | **A→D′ 总计** |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B | 175.6 ms | 163.6 ms | 151.0 ms | **149.6 ms** | −1.4ms (−0.9%) | **−26.0ms (−14.8%)** |
| Qwen3.5-2B | 201.4 ms | 189.9 ms | 179.8 ms | **~177.5 ms** | ~−2.3ms | **~−23.9ms (−11.9%)** |
| Qwen3.5-4B | 376.6 ms | 359.5 ms | 338.7 ms | **~335.8 ms** | ~−2.9ms | **~−40.8ms (−10.8%)** |

> e2e 比 infer 大 ~40ms（GenAI pipeline host 开销，非 tokenization）。opt-3 的绝对节省在 prefill 与 e2e 上一致（它只动 prefill 计算）；2B/4B 的 e2e 增量按 prefill 实测平移（ic=128 全量 e2e 交错测量耗时过长，未逐一跑满，标 ~）。

**[356H]** e2e-TTFT（`ttft_mean_ms`，n=8，ic=1，四态各自重建插件实测，无 ~ 估算）：

| 模型 | A 基线 | B (+opt-1) | C (+opt-1+2) | D′ (+opt-3) | opt-3 增量 | **A→D′ 总计** |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B | 336.7 ms | 333.8 ms | 299.9 ms | **295.4 ms** | −4.5ms (−1.5%) | **−41.3ms (−12.3%)** |
| Qwen3.5-2B | 393.3 ms | 389.2 ms | 360.8 ms | **354.4 ms** | −6.4ms (−1.8%) | **−38.9ms (−9.9%)** |
| Qwen3.5-4B | 845.0 ms | 832.6 ms | 776.2 ms | **766.1 ms** | −10.1ms (−1.3%) | **−78.9ms (−9.3%)** |

> 356H 上 e2e host 开销约 44ms（0.8B e2e 336.7 − infer 292.3），与 358H 的 ~40ms 量级一致。dense Qwen3-1.7B e2e：A 183.4 → D′ 181.1（−1.3%，不触发）。

### 8.3 deep-profile prefill op-time 分解（Qwen3.5-2B，op-sum 口径，用于看相对结构变化）

| 算子 | A 基线 | B (+opt-1) | C (+opt-1+2) | D′ (+opt-3) |
|---|---:|---:|---:|---:|
| op-sum 合计 | 65.8 ms | 52.4 ms | 37.3 ms | 36.5 ms |
| Transpose | 20.98ms (31.9%) | 20.69ms (39.5%) | 5.74ms (15.4%) | ~5.8ms |
| Reorder | 13.81ms (21.0%) | **0 (0%)** | 0 (0%) | 0 (0%) |
| FullyConnectedCompressed | 16.60ms (25.2%) | 16.58ms (31.6%) | 16.65ms (44.6%) | ~17.0ms |
| **DynamicQuantize** | 5.41ms (175 节点) | 5.39ms | 5.35ms (175 节点) | **3.44ms (97 节点)** |
| GatedDeltaNet | 2.95ms | ~2.9ms | ~2.9ms | ~2.9ms |

（再次强调 §6：这张表的百分比分母是 op-sum，不是 TTFT；用它看「结构怎么变」，用 8.1/8.2 看「TTFT 降多少」。opt-3 把 DQ 从 175→97 节点、op-time −1.9ms，墙钟 prefill −2~3ms，对账吻合。）

---

## 9. 还能继续做什么（按性价比）

> 2026-06-29 第二轮 hotspot 复核（详见 `OV当前hotspot分析与后续优化方向.md`）后修订：在 md5 逐字节无损约束下，**3-opt 态已无可安全落地的新局部优化点**；下表把已实测排除的方向标注清楚，避免重复踩。

| 方向 | 预期 | 风险/工作量 | 建议 |
|---|---|---|---|
| 把 opt-1/opt-2/opt-3 整理成上游 PR + 补单测 | 已落地，巩固 | 低 | 做（opt-3 尤其通用：任何「多 FC 共享激活」模型都受益，不限 Qwen3.5） |
| **conv-sandwich 融成一个有状态 layout-aware fused op**：把 `Transpose(pre)+Concat(state)+GroupConv+Slice_2+Transpose_1` 整段收进一个 kernel | 可省 `Transpose_1`(~3.57ms)+`Slice_2`(~1.65ms)+concat | 高（新 internal op+pass+kernel，且要同时保 prefill/decode md5） | **唯一值得立项的性能方向**，先 POC 验证 decode md5 不破 |
| ~~kernel 级修 post-conv transpose~~ | — | — | **已排除**：permute_ref 已被 opt-2 调过；tiled SLM kernel 要 F-旋到内层，我们是 F↔Y 两轴 swap，模式不匹配。无 md5-无损局部修法 |
| ~~GDN `gated_delta_net_ref` 调 V_BLOCK_SIZE~~ | — | — | **已实测排除**：VBS=4 已最优；VBS=8 的「加速」是 kernel float4 写死导致漏算一半 V 列的假象，正确实现下 VBS=8 净回退 |
| ~~进一步系统化 DynamicQuantize 去重~~ | — | — | **已到 floor**：opt-3 已消掉 dense 全部 84 处共享冗余（198 MatMul→113 DQ 恰好对上），剩余 DQ 不共享输入。补 `visit_attributes`+CSE 只是整洁度收益，无额外性能 |
| FullyConnectedCompressed（GEMM）| 已在 XMX 算力下限 | — | 不做 |
| dense Qwen3 prefill | 已在算力下限（94% 在 GEMM+量化） | — | 不做 |
| greedy-only GPU ArgMax 输出 token id | 降 e2e 的 logits copy+CPU scan | 中高 | dense 小项，见 hotspot 文档 §10 |
| Qwen3.5 NPU 的 GatedDeltaNet 融合算子（~222s 问题） | 巨大，属上游 NPU 插件待办 | 极高 | 跟进 OpenVINO 上游 |

---

## 10. 一页复现 checklist

```bash
# 0) 环境
cd /home/user/chaofan/model_test
source benchmark/openvino/install/setupvars.sh
export LD_LIBRARY_PATH="$PWD/benchmark/openvino/install/runtime/lib/intel64:$LD_LIBRARY_PATH"

# 1) 构建链接本地 OV 的 model_bench（一次性）
cd model_bench && cmake -S . -B build-local -DCMAKE_BUILD_TYPE=Release && cmake --build build-local -j$(nproc)
ldd build-local/model_bench | grep openvino    # 确认指向 benchmark/openvino/install

# 2) profiling 找 hotspot（--profile 会自动用 -pf 的真实长度）；落地一个优化后 RE-PROFILE，hotspot 会变
./build-local/model_bench vlm -m ../benchmark/models/Qwen3.5-2B-INT4 -d GPU -n 1 -ic 8 \
  -pf ./prompts/llama-2-7b-chat_l.jsonl --no-monitor \
  --profile --profile-iters 8 --profile-warmup 3 --profile-csv /tmp/prof.csv -r /tmp/x.csv -rj /tmp/x.json

# 3) 三个优化已在源码里（layout_optimizer.cpp + permute_kernel_ref.cpp
#    + dynamic_quantize_fully_connected.cpp + dynamic_quantize.cpp）；改完重建（~9s）
cd ../benchmark/openvino && bash rebuild_gpu_plugin.sh

# 4) A/B + 正确性（n=8！，对比 first_token_infer_mean_ms / ttft_mean_ms / iterations[].result_md5）
#    opt-3 这类「改图结构」的优化，务必同时验 prefill(ic=1) 和 decode(ic=128) 两个 md5
cd ../../model_bench
./build-local/model_bench vlm -m ../benchmark/models/Qwen3.5-2B-INT4 -d GPU -n 8 -ic 128 \
  -pf ./prompts/llama-2-7b-chat_l.jsonl --no-monitor -r /tmp/x.csv -rj /tmp/out.json
```

---

## 附：deep-profile 复用 `-pf` 真实 prompt（model_bench 改动）

**背景**：旧版 `--profile` 只用 `--input-tokens`（默认 64）合成 prompt，**忽略 `-pf`**。导致只传 `-pf` 时，计时 benchmark 用真实长度、但 deep-profile 偷偷按 64 token 跑，逐算子占比严重失真（64-token 时 Transpose+Reorder 仅 ~17%，真实 ~950-token 是 ~53%）。

**改动（4 处，已落地）**：
- `deep_profile.hpp`：`DeepProfileConfig` 新增 `std::optional<std::string> prompt_text`。
- `deep_profile.cpp`：新增 `resolve_prefill_ids()`——`prompt_text` 非空用真实 prompt 自然长度，否则回退合成；打印 `using REAL prompt: N tokens` / `using synthetic prompt`。
- `llm.cpp`/`vlm.cpp`：profile 块里，未传 `--input-tokens` 时把 benchmark 第一条 prompt（按相同 chat-template 处理）填入 `dp.prompt_text`。`--input-tokens` 仍走显式合成（二者互斥）。

**验证**：`-pf` 跑 `--profile`，日志 `using REAL prompt: 938 tokens`，与 benchmark 的 `raw_text_tokens=938` 一致。

---

## 附：产物清单

| 文件 | 说明 |
|---|---|
| `benchmark/openvino/src/plugins/intel_gpu/src/graph/layout_optimizer.cpp` | opt-1（depthwise conv 保持 bfyx，+16 行） |
| `benchmark/openvino/src/plugins/intel_gpu/src/kernel_selector/kernels/permute/permute_kernel_ref.cpp` | opt-2（permute_ref LWS 放 F 轴，+31 行） |
| `benchmark/openvino/src/plugins/intel_gpu/src/plugin/transformations/dynamic_quantize_fully_connected.cpp` | opt-3a（共享激活的 FC 复用 DynamicQuantize，+46 行） |
| `benchmark/openvino/src/plugins/intel_gpu/src/graph/dynamic_quantize.cpp` | opt-3b（放宽 skip 断言，让共享 DQ 仍可在 decode skip，+19 行） |
| `benchmark/openvino/rebuild_gpu_plugin.sh` | 增量重建 GPU 插件 + 刷新 install（~9s） |
| `model_bench/build-local/model_bench` | 链接本地 OV 的 C++ benchmark |
| `model_bench/transpose_microbench.cpp`、`transpose_dir_microbench.cpp` | Transpose kernel 对比 / 方向不对称 micro-benchmark |
| `model_bench/results/`、`/tmp/finaldata/`、`/tmp/plugin_C.so`、`/tmp/plugin_Dprime.so` | A/B/C/D′ 状态原始 JSON（TTFT + md5）、逐算子 profile CSV、交错 A/B 用的两份插件 |
| `vtune-base/results-gpu-qwen3.5-2B/` | 原始 vtune gpu-hotspots 数据库（hotspot 交叉验证 + opt-3 复盘起点） |

---
*基于本地源码构建的 OpenVINO 2026.2 / OpenVINO GenAI 2026.2 / model_bench(C++) / vtune 2026.2，Panther Lake + Arc B390。三个优化均已实测、正确性以 128-token 贪心 md5 校验（opt-3 同时校验 prefill ic=1 与 decode 128-token 均逐字节一致）。所有 TTFT 数据为 n=8、real prompt(~950 tok)、本地 OV、GPU；opt-3 增量用插件热插拔交错 A/B 实测。*
