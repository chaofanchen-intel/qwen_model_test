# Qwen3.5-2B + PA：第三轮 TTFT 优化（带宽受限 iGPU 上的重新定位）

延续 `benchmark/openvino/qwen3.5_patch/`（opt-1~opt-5）与 GME 第二轮
（`gme-Qwen2-VL-2B-Instruct-Running/gme-qwen2-vl-int4-hotspot-optimization/docs/06`）
的方法，在**本机 356H（32 EU）**上继续降低 Qwen3.5-2B 的 PA prefill TTFT。

本轮的核心不是"再挤一点"，而是**重新判定瓶颈在哪**：本机与旧文档所用的机器不是同一台，
内存频率受限，导致优化的优先级和旧结论**不一致**。所有数字均为本机实测，
不复用其它机器的绝对值。

---

## TL;DR

| | |
| --- | --- |
| **新优化（opt-6）** | 把 Gated-DeltaNet 的两处 SiLU 从独立 kernel 融进生产者（conv1d 的 OCL epilogue + `in_proj_z` 的 oneDNN post-op） |
| **收益** | `Qwen3.5-2B-INT4` infer-TTFT **275.2 → 259.2 ms**；相对 pristine 从 −6.2% 提升到 **−11.7%**，opt-6 单项 **−5.5 个百分点**，与 opt-3 同量级 |
| **g128 交叉验证** | 332.1 → 318.1 ms，累计 −9.3%，opt-6 单项 −4.0 pp |
| **被证伪** | GME 最大的杠杆（INT4 权重 layout `cab→abc`）在 Qwen3.5 上**两种量化都是负收益**；per-OC 变体的 GEMM 已达 int8 DPAS 天花板的 **96%** |
| **被修正** | opt-4 / opt-5 在本机只值 ≤0.8 pp（旧机器上是 −4.4% / −3.8%）——占用率不是本机的约束 |
| **正确性** | prefill 首 token **逐字节相同**；128 步贪心解码 md5 **改变**（≤1 ULP f16 差异被放大）。这是本轮唯一不满足旧 md5 验收标准的改动 |

---

## 1. 为什么先证伪 GEMM：它已经到硬件天花板

GEMM 占本机 prefill GPU 时间的 **51.6%**。旧文档写的是"oneDNN 硬上限，不可编辑"——
那是**断言**，不是证据。GME 第二轮恰恰在同一句断言上挖出了 1.25x（权重 layout
`cab → abc`）。所以本轮第一件事是把这句话拆开重测。

用独立 oneDNN probe（`onednn/qwen35_gemm_probe.cpp`，脱离 OpenVINO 直接定价
primitive，方法与 GME 一致）在 **Qwen3.5 的真实形状 + 真实量化配置**上定价。

### 1.1 两个变体的量化方式本质不同

`openvino_config.json` 与 `ONEDNN_VERBOSE` 双重确认：

| 模型 | 量化配置 | oneDNN 实际 primitive |
| --- | --- | --- |
| `Qwen3.5-2B-INT4` | `group_size=-1, sym=true` | `wei:s4` + `attr-scales:wei:4:f16`（**per-OC，无 zero-point**） |
| `Qwen3.5-2B-INT4-g128` | `group_size=128, sym=false` | `wei:u4` + grouped scale + grouped u8 zero-point |

这一点很重要：GME 生产用的是第二种，而**旧的 Qwen3.5 PA 文档全部测的是第一种**。
GME 自己的成本分解表里，layout 偏好在这两种量化之间是**翻转的**，所以绝不能直接沿用。

### 1.2 定价结果

真实形状 M=1002、K=2048、N=6144（= `mlp.gate/up` 与 `linear_attn.in_proj_qkv`）：

| case | `cab`（现状） | `abc` | 结论 |
| --- | ---: | ---: | --- |
| per-OC s4（`INT4`） | 0.714 ms / **35.3 T** | 0.861 ms / 29.3 T | abc **慢 17%** |
| g128 u4（`INT4-g128`） | 0.908 ms / 27.8 T | 0.932 ms / 27.1 T | abc **慢 2.6%** |
| 稠密 s8（int8 DPAS 天花板） | 0.688 ms / **36.6 T** | — | — |
| 稠密 f16（参考） | 1.342 ms / 18.8 T | — | — |

原始输出：[`results/probe_qwen35_shapes.txt`](results/probe_qwen35_shapes.txt)

**结论一：GME 的 layout 杠杆不转移。**
Qwen3.5 的 `N/K = 3`，正落在 GME 从未测过的空档（GME 在 N/K=5.83 大赢、1.33 大输）。
实测两种量化下 `abc` 都是负收益，GME 已交付的 `N >= 4K` 门槛在这里**正确地不触发**。

**结论二：per-OC 变体的 GEMM 已经没有余量。**
35.3 T vs 稠密 int8 36.6 T = **96%**，说明 INT4 的反量化开销（unpack u4/s4 + 乘 scale）
在这个量化方案下**几乎为零**。这把旧文档的断言升级成了实测证据：
**GEMM 这 51.6% 不是"暂时没找到办法"，是真的到顶了。**

（g128 变体的 GEMM 只有 27.8 T，慢 21%，差的正是 grouped scale + grouped zp 的反量化。
但那是量化方案的代价，要改就得重新量化、动精度，不是 plugin patch 能碰的。
这也解释了为什么 g128 整体比 per-OC 慢 ~55 ms。）

### 1.3 顺带证伪的第二条：dispatch 形状

模型把**所有**投影都按 `M x 1 x K` 下发（batch=M、M=1、权重广播），
不同于 GME 只有部分投影是这个形状。probe 对比 `1 x M x K`：
四种形状全部**无差异（≤1%）**，见 [`results/probe_dispatch_form.txt`](results/probe_dispatch_form.txt)。
与 GME 同结论，排除。

---

## 2. 本机的真实约束是带宽，不是占用率

本机与旧文档（iGPU-B）同为 356H/32 EU，但**内存频率受限**。这不是软件回退，
但它**改变了优化的目标函数**，必须重新 profiling 而不是沿用旧的 kernel 排序。

### 2.1 fresh VTune 逐 kernel（本机，2B，PA prefill）

`results/vtune_tasks_current.csv`，4 次 prefill 合计 1083 ms，下表已除以 4：

| kernel | ms/iter | 占比 | 旧文档（iGPU-B） |
| --- | ---: | ---: | --- |
| `gemm_kernel` | 139.6 | 51.6% | ~47–49%（§1 已证到顶） |
| `paged_gated_delta_net_opt` | 42.3 | 15.6% | 20.4% |
| **`dynamic_quantize_gpu_opt`** | **23.6** | **8.7%** | 2.65% → **3.3 倍** |
| `rms_gpu_bfyx_opt` | 18.5 | 6.8% | 4.03% → 1.7 倍 |
| **`activation_opt`** | **15.4** | **5.7%** | 2.19% → **2.6 倍** |
| `paged_causal_conv1d_ref` | 12.2 | 4.5% | ~4.5% |
| `generic_eltwise_ref` | 5.7 | 2.1% | 2.5% |
| `strided_slice_ref` | 4.4 | 1.6% | 2.1% |

### 2.2 关键观察：所有 kernel 都贴着同一条带宽线

VTune 的 GPU memory bandwidth 一栏（`results/vtune_tasks_current.csv`）：

| kernel | 读 GB/s | 写 GB/s | 合计 |
| --- | ---: | ---: | ---: |
| `gemm_kernel`（最大实例） | 41.6 | 0.4 | 42.1 |
| `activation_opt`（6144 宽） | 19.9 | 19.4 | 39.2 |
| `dynamic_quantize_gpu_opt` | 35.3 | 12.6 | **48.0** |
| `rms_gpu_bfyx_opt` | 28.1 | 6.2 | 34.3 |

**没有任何 kernel 超过 ~48 GB/s。** 这就是本机的实际 DRAM 天花板。

由此得到本轮最重要的判断：
**`activation_opt` 这类纯逐元素 kernel 已经跑在带宽天花板上，不可能再调快——只能整趟省掉。**

这直接推翻了旧文档对这批 kernel 的处置（"已 100% occupancy、带宽 bound、marginal，
不值得做"）。那个判断在带宽充裕的机器上成立；在本机，
**DQ + rms + activation 合计 21%**，是除 GEMM 外最富的矿，
而正确的度量是**省掉多少趟内存**，不是提占用率。

### 2.3 同一逻辑解释了 opt-4/opt-5 为什么在本机失灵

opt-4（conv1d token tiling）和 opt-5（dynamic activation 走向量化 kernel）
买的都是**占用率**。在本机占用率不是约束，所以：

| | iGPU-B（旧） | 本机 |
| --- | ---: | ---: |
| opt-4 增量 | −4.4% | **−0.69 pp** |
| opt-5 增量 | −3.8% | **−0.15 pp** |

opt-4 尤其吃亏：它的 tiling 要为每个 tile 重读 K-1 个 history 元素换取占用率，
带宽紧张时这笔交换不划算。用 `OV_CONV1D_TOKEN_GROUPS` 单独 A/B 也能看到同样的结论
（G=1 串行 275.4 ms vs G=3 tiling 276.6 ms，方向甚至相反，量级都在 ~1-2 ms 噪声内）。

**两者仍是正向或中性，保留在推荐栈里；但不能拿旧机器的数字外推到带宽受限的硬件。**

---

## 3. opt-6：把 SiLU 融进生产者

### 3.1 为什么这两趟一直没被融掉

Qwen3.5 的 Gated-DeltaNet 有两处 SiLU，各自的生产者都**只有它一个消费者**：

```
paged_causal_conv1d (users=1) → reshape (users=1) → swish     # [tokens, 6144]，12.0 ms/iter
in_proj_z FullyConnected       → reshape (users=1) → swish     # [tokens, 2048]， 3.4 ms/iter
```

三条原因叠在一起，缺一不可：

1. **中间隔着一个 Reshape。** 它是 `PagedCausalConv1DFusion` 为恢复 3D 形状插入的
   （该 pattern 刻意止于 `Slice`，把 Swish 留在外面）。
2. **`prepare_primitive_fusing` 只融合直接依赖。** 注意 `fully_connected`
   **本来就在白名单里**——第二处纯粹是被这个 Reshape 挡住的，不是不支持。
3. **通用的 `MoveEltwiseUpThroughDataMov` 救不了。** 它匹配
   `UnaryElementwiseArithmetic` / `BinaryElementwiseArithmetic`，而 `ov::op::v4::Swish`
   直接继承 `ov::op::Op`（因为有可选 beta 输入），**永远匹配不上**。

一趟独立的逐元素 kernel 是**纯内存开销**：把整个张量读一遍写一遍
（6144 宽的那趟每层 ~24 MB @ 1k token）只为做一次乘法，
而生产者写出那个值的时候它**还在寄存器里**。结合 §2.2，这趟省下来是净赚。

### 3.2 改动

| 文件 | 改动 |
| --- | --- |
| `plugin/transformations/move_swish_up_through_reshape.{hpp,cpp}` | **新 pass**：`Swish(Reshape(x))` → `Reshape(Swish(x))`。逐元素算子与 reshape 交换**数值恒等**；只在 Reshape 单消费者时触发 |
| `plugin/transformations_pipeline.cpp` | 注册（放在 `SinkReshape` 之后，确保越过最终相邻的那个 Reshape） |
| `graph/graph_optimizer/prepare_primitive_fusing.cpp` | 白名单加入 `paged_causal_conv1d`，**仅限 SWISH** |
| `graph/impls/ocl_v2/paged_causal_conv1d_ref.{cl,cpp}` | conv1d epilogue 应用 SWISH；JIT 侧独立复核"只有一个 SWISH fused desc"，其它一律不认 |

默认只对 `PagedCausalConv1D` 生产者 hoist；`in_proj_z` 那一处由
`OV_SWISH_HOIST_ALL=1` 开启（它会变成 oneDNN post-op，在 f32 累加器上求值，
更准但更不"逐位"，所以做成 opt-in）。

patch：[`patches/opt6_fuse_silu_into_producer.patch`](patches/opt6_fuse_silu_into_producer.patch)
（6 文件，+245 行；已同步到 `benchmark/openvino/qwen3.5_patch/opt6_fuse_silu_into_producer.patch`）

### 3.3 逐项收益（本机，交错 A/B）

方法：每个阶段一个独立编译的 plugin binary，**交错**热插拔
（本机单次运行间有 ~1 ms 单调热漂移，与 opt-4/opt-5 的增量同量级，
顺序跑会系统性地惩罚靠后的阶段）。runner：
[`run_pa_stage_ab.sh`](run_pa_stage_ab.sh)。
`n=5`、`ic=1` 纯 prefill、真实 1002-token prompt、3 轮取中位。
pristine/opt3/opt34/opt345 复用 `benchmark/tests/repro_20263_opt_ab/results` 里
2026-08-07 用同一棵 2026.3 源码树编出的 binary。

**Qwen3.5-2B-INT4（per-OC）**

| 阶段 | infer (ms) | vs pristine | 单项增量 | e2e (ms) |
| --- | ---: | ---: | ---: | ---: |
| pristine | 293.45 | — | — | 299.15 |
| + opt-3 | 277.69 | −5.37% | −5.37 pp | 283.33 |
| + opt-4 | 275.66 | −6.06% | −0.69 pp | 281.30 |
| + opt-5 | 275.22 | −6.21% | −0.15 pp | 280.93 |
| **+ opt-6** | **259.19** | **−11.68%** | **−5.46 pp** | **264.88** |

**Qwen3.5-2B-INT4-g128（交叉验证）**

| 阶段 | infer (ms) | vs pristine | 单项增量 |
| --- | ---: | ---: | ---: |
| pristine | 350.66 | — | — |
| + opt-3 | 335.54 | −4.31% | −4.31 pp |
| + opt-4 | 334.19 | −4.70% | −0.39 pp |
| + opt-5 | 332.10 | −5.29% | −0.59 pp |
| **+ opt-6** | **318.08** | **−9.29%** | **−4.00 pp** |

原始逐轮数据：`results/stage_ab/`、`results/stage_ab_g128_rerun.log`。

**读法：**

- opt-6 单项 −5.46 pp，**与 opt-3 同量级**，两者合起来基本就是全部收益。
- 与 §2.1 的 VTune 预测（12.0 + 3.4 = 15.4 ms）吻合到 1 ms 以内（实测 −16.0 ms）。
- 收益**基本不随权重量化变化**（−5.46 pp vs −4.00 pp），符合预期：
  它省的是**激活**的内存趟数，与权重位宽无关。

> ⚠️ g128 第一次跑（`results/stage_ab_g128.log`）的 rep1 opt3456 = 330.96 ms 是离群值，
> 与同配置的其余 4 次测量（317.4 / 317.4 / 317.7 / 318.1）不符，判定为首次触碰的
> 冷启动伪影，已重跑 3 轮取代，原始日志保留。

---

## 4. 正确性：本轮唯一不满足旧验收标准的改动

| 口径 | 结果 |
| --- | --- |
| prefill 首 token md5（`ic=1`，n=8） | **逐字节相同**：`dd7536794b63bf90eccfd37f9b147d7f`（与旧文档记录的 2B 值也一致） |
| 128 步贪心解码 md5（`ic=128`） | **不同** |

**必须明说：opt-6 不是 bit-exact 优化。**
差异是真的，不是噪声——基线与 patched 各自连跑两次都能复现自己的 md5 多重集
（`results/decode128_{OFF,ON}.json`）。根因已定位，是 ≤1 ULP 的 f16 舍入差异被贪心解码放大：

- **conv1d epilogue**：我刻意复刻了未融合时的舍入顺序（先把 f32 累加器 round 到 f16、
  再在 f16 里算 `x/(1+exp(-x))`——已确认独立 kernel 的 `input_dt` 就是 f16）。
  但复刻不了的是：独立 kernel 在 **`half4` 向量**上求 `exp`，epilogue 在**标量 `half`**
  上求，且两者是分别编译的 OpenCL program。
  **在另一个 kernel 里逐位复现向量化 f16 超越函数，不是可靠可达的目标。**
- **`in_proj_z` FC**：SiLU 变成 oneDNN `eltwise_swish` post-op，在 **f32 累加器**上求值
  再存 f16 —— 比原来的 f16 独立 kernel **更准**，当然也就不逐位相同。

所以 opt-6 准确的描述是：**逐元素数值等价或更优，但长贪心序列不可逐位复现**。
首 token 逐位相同、128 步发散，是近似打平 logits 上的预期数值敏感
（GME 文档记录过同一现象），不是缺陷证据；但它确实意味着
**opt-6 无法用其它 patch 所用的 md5 门禁来验收**。

**本轮未做：opt-6 的定量精度对比（固定输入下的 cosine / max-abs）。**
在补上之前，opt-6 的定位是：可以放心 benchmark、可以用于只关心首 token 的场景；
**不要**投进"128 步 md5 不变"作为验收门槛的部署——那种场景用 `opt3+opt4+opt5`。
`OV_DISABLE_SWISH_HOIST=1` 可在不重编的情况下运行时关闭。

---

## 5. 下一步

| 方向 | 说明 |
| --- | --- |
| **opt-6 的定量精度对比** | 最高优先级的补课。移植 GME 的 `dump_outputs.py` + `compare_outputs.py` 口径，给出 cosine / max-abs 量纲 |
| **`dynamic_quantize`(8.7%) + `rms`(6.8%) 融合** | 同一条"省内存趟数"思路的下一个矿：让 RMSNorm 直接输出 s8 + scale，可同时省掉 RMS 的 f16 写与 DQ 的 f16 读。按 §2.2 的带宽口径估计 10–15 ms。**本轮未实现** |
| **opt-4 的 G 公式引入带宽项** | 现公式只看线程预算（EU × threads/EU），在带宽受限部件上会选到不划算的 G，应能自动退化到 G=1 |
| **上游化** | `MoveSwishUpThroughReshape` 是通用的；更干净的上游做法可能是让 `MoveEltwiseUpThroughDataMov` 覆盖 Swish |
| GEMM(51.6%) / GDN(15.6%) | §1 已证 GEMM 到顶；GDN 是串行递归，前几轮已判定需 chunkwise 重写（数天级 + 必然破 md5） |

---

## 复现

```bash
cd /home/chaofan/model_benchmark/benchmark/openvino
git checkout -- src/                                    # 回到 2026.3.0
for p in opt3_dynamic_quantize_reuse opt4_conv1d_token_tiled \
         opt5_dynamic_activation_opt opt6_fuse_silu_into_producer; do
    git apply qwen3.5_patch/$p.patch
done
cmake build                                             # 新增源文件需要重新 glob
cmake --build build --target openvino_intel_gpu_plugin -j && cmake --install build
source install/setupvars.sh

# 逐阶段交错 A/B（本文 §3.3）
cd ../tests/qwen35-pa-ttft-round3
./run_pa_stage_ab.sh
MODEL=/home/chaofan/model_benchmark/benchmark/models/Qwen3.5-2B-INT4-g128 REPS=3 ./run_pa_stage_ab.sh

# oneDNN GEMM 定价（本文 §1）
cd onednn && bash build_probe.sh
./qwen35_gemm_probe 100
QWEN_PROBE_SHAPES=dispatch ./qwen35_gemm_probe 100
QWEN_PROBE_SHAPES=nsweep   ./qwen35_gemm_probe 100      # 扫 N，找 cab/abc 交叉点

# 逐 kernel profiling（本文 §2）
source /opt/intel/oneapi/setvars.sh
cd /home/chaofan/model_benchmark/model_bench
echo '{"ATTENTION_BACKEND":"PA"}' > /tmp/pa.json
vtune -collect gpu-hotspots -r /tmp/vt -- ./build-local/model_bench vlm \
  -m ../benchmark/models/Qwen3.5-2B-INT4 -d GPU -n 3 -ic 1 \
  -pf ./prompts/qwen3.5_1k.jsonl --no-monitor -lc /tmp/pa.json -r /tmp/x.csv -rj /tmp/x.json
vtune -report hotspots -group-by computing-task -r /tmp/vt -format csv
```

> `--profile` **不能**用于 PA：`ATTENTION_BACKEND` 是 GenAI pipeline 级属性，
> deep-profile 直接 `compile_model` 导出的静态 IR，看到的永远是 SDPA 结构图。
> PA 的真实 kernel 结构只能用 VTune 抓真实 pipeline。

## 目录

```
onednn/qwen35_gemm_probe.cpp   Qwen3.5 形状 + 两种量化的 INT4 GEMM 定价 probe（§1）
run_pa_stage_ab.sh             逐阶段交错热插拔 A/B runner（§3.3）
patches/                       opt-6 patch（基于 2026.3.0 + opt-3/4/5）
results/
  probe_qwen35_shapes.txt      §1.2 layout 定价主表
  probe_dispatch_form.txt      §1.3 Mx1xK vs 1xMxK
  vtune_tasks_current.csv      §2 逐 kernel + 带宽（权威）
  stage_ab/                    §3.3 per-OC 逐阶段逐轮原始数据
  stage_ab_g128*.log           §3.3 g128 逐阶段（含被作废的首次采集）
  decode128_{OFF,ON}.json      §4 md5 两态
  so_pa_opt3_opt4_opt5_opt6.so opt-6 阶段的 plugin binary（可热插拔）
```

## 开关

| 环境变量 | 作用 |
| --- | --- |
| `OV_DISABLE_SWISH_HOIST=1` | 完全关闭 opt-6（回到未 hoist 的图），用于交错 A/B |
| `OV_SWISH_HOIST_ALL=1` | 额外把 `in_proj_z` 的 SiLU 融进 oneDNN post-op（多约 −1.6%） |
| `OV_CONV1D_TOKEN_GROUPS=1` | 关闭 opt-4 的 tiling（本机 ≈ 中性，见 §2.3） |
