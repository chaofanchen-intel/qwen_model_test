# Qwen3.5-2B GPU 继续优化：356H / Xe3 专项探索日志

日期：2026-07-13 起。基线栈 = 已落地的 opt-3 + opt-5 + opt-6（GPU plugin `de95bee0`）。
目标：在当前栈之上，针对本机 **Core Ultra 7 356H（Xe3 架构，iGPU `0xb0a0`，32 XVE）** 找新的优化点，允许**非普适、Xe3 专用、甚至激进（可破 bit-exact）**的尝试。本文件完整记录分析→尝试→结论→下一步，便于后续接续。

> 测量口径：`model_bench vlm -m Qwen3.5-2B-INT4 -d GPU -n5 -ic1 -lc gpu_pa.json`（PA、isolated prefill）用于 kernel 归因；`-ic 512` 用于业务 TTFT/TPOT/TPS。所有 A/B 均**逐次温度门控冷启动**（跑前把 `x86_pkg_temp` 冷却回 ~50°C 空载地板），避免热漂移污染（见统一版报告 §6）。

---

## 1. 基线 hotspot 全景（fresh vtune，2026-07-13，2B PA ic=1 n=5）

原始数据：`results-gpu-qwen3.5-2B/`（vtune gpu-hotspots，含完整 XVE 占用/停顿/L3/内存指标）。GPU task 合计 ≈1177 ms（n=5+warmup 采样和）。按 kernel 家族（时间加权指标）：

| kernel 家族 | ms | 占比 | occ | XVE active | stalled | XMX act | ALU0 act | L3 R GB/s | mem R GB/s | 主停顿 | 判读 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `gemm_kernel` | 584.8 | **49.7%** | 76 | 53 | 45 | **56** | 20 | 134 | 36 | SBID 26% | oneDNN XMX INT4 GEMM；XMX 只占 56%、45% 停顿在 SBID（等权重/反量化数据）→ 部分**数据供给 bound** |
| `paged_gated_delta_net_opt` | 242.0 | **20.6%** | 78 | **84** | 15 | **0** | **65** | 21 | 17 | SBID 16% | 串行 SSM 递归；**XMX 完全闲置**、纯 ALU 忙。最大**可编辑**热点 |
| `rms_gpu_bfyx_opt` | 91.9 | 7.8% | 70 | 69 | 30 | 23 | 46 | 73 | 27 | SBID 22% | 访存延迟 bound（前序已证伪提占用率无收益） |
| `dynamic_quantize_gpu_opt` | 62.0 | 5.3% | 74 | 55 | 45 | 46 | 23 | 113 | 34 | SBID 29% | opt-3 已去重，剩余为必要量化 |
| `paged_causal_conv1d_ref` | 52.7 | 4.5% | 82 | 63 | 35 | 50 | 25 | 122 | 26 | SBID 21% | opt-5 已打满占用 |
| `activation_opt` | 34.6 | 2.9% | 80 | 65 | 33 | 33 | 27 | 88 | 27 | SBID 27% | opt-6 已向量化 |
| `sdpa_micro__prefill` | 32.4 | 2.8% | 64 | 37 | **59** | 22 | 6 | 76 | 15 | **SBID 50%** | full-attn 层(6/24)，重停顿；核心是 oneDNN gemmstone |
| `generic_eltwise_ref` | 31.1 | 2.6% | 73 | 54 | 42 | 46 | 17 | 137 | 17 | SBID 26% | GDN gating，100%-occ 带宽 bound |
| `strided_slice_ref` | 24.6 | 2.1% | 57 | 44 | 45 | 2 | 8 | 21 | 20(W78) | SBID 47% | SSM 数据搬运，写带宽高 |

**两大项占 ~70%**：`gemm`（oneDNN，不可编辑）与 `GDN`（可编辑 OCL，但串行）。

### 1.1 关键新观察（相比前序文档）
1. **gemm 的 XMX 只忙 56%、SBID 停顿 26%**：说明 XMX systolic 阵列有近一半时间在等数据（INT4 权重解包 + 动态量化激活的供给）。纯算力并非唯一瓶颈，**数据供给/解包**也占一部分——这给"减少权重/激活内存往返"类优化留了一线空间（虽然 kernel 本体不可改）。
2. **GDN 的 XMX 利用率 = 0%**：整段 20% 的 prefill 时间里矩阵引擎完全闲置，全靠 ALU0（65%）做逐 token 的点积/FMA。**这是本机最大的"架构性浪费"**——若能把 chunk 内计算改写成矩阵乘走 XMX，理论收益最大。
3. `sdpa_micro` 59% 停顿（SBID 50%）是停顿占比最高的核，但只有 2.8%，且核心是 oneDNN gemmstone，改不动。

---

## 2. 优化候选清单（按 期望收益 × 可行性 排序）

标注【新】= 前序文档未尝试；【已证伪】= 前序已实测否决，不重复。

| # | 候选 | 目标核 | 类型 | 期望 | 可行性 | 备注 |
|---|---|---|---|---|---|---|
| A | **GDN chunkwise + XMX** | GDN 20.6% | 激进/算法级 | 高（理论 GDN −30~50%）| 低（数天，破 md5）| 本机最大可编辑杠杆；§4 详细计划 |
| B | 【新】GDN `SUBGROUP_SIZE` 16→32（Xe3 SIMD32） | GDN | Xe3 专用 | 中/未知 | 高（改 1 函数）| 会改归约顺序→md5 变；§3.2 |
| C | 【新】GDN `V_BLOCK_SIZE` 在 356H 上重扫 | GDN | 调参 | 低/中 | 高 | md5-无损；§3.1 |
| D | 【新】GDN 延迟衰减/减少 subgroup 归约次数 | GDN | 微优化 | 低 | 中 | 每 token 有 2×VBS 次 subgroup_reduce |
| E | 【新】gemm 数据供给：减少 DQ↔GEMM 往返 / 权重预取 | gemm | 激进 | 中/未知 | 低 | gemm 本体不可改，只能改其上下游 |
| F | 【已证伪】rms SLM / 占用率 | rms | — | — | — | 访存延迟 bound，提占用率无墙钟收益 |
| G | 【已证伪】4-way FC 融合 / eltwise vload8 / sdpa tile | 多 | — | — | — | 见前序文档 §4.9/§4.10 |

**本轮执行顺序**：先做低风险快验证 C（重扫 VBS，确认基线在 Xe3 上是否最优）→ 再做 Xe3 专用 B（SIMD32）→ 记录 → A（chunkwise）作为主攻方向详细规划（本轮不一定跑完）。

---

## 3. 实验记录

（下面各节在实验执行后填入 A/B 数据与结论）

**方法**：给 GDN host 代码（`paged_gated_delta_net.cpp` 的 `get_v_block_size` / `get_subgroup_size`）加环境变量旋钮（`OV_GDN_VBLOCK` / `OV_GDN_SG`），重建一次插件（~10s），用 `benchmark/tests/gdn_ab.sh` 做逐次温度门控冷启动 A/B（2B PA ic=1 n=5，取 `first_token_infer_mean_ms`）。旋钮改动已存为 `benchmark/openvino/qwen_patch/dev_gdn_tuning_knobs.patch`；实验后源码已还原、安装二进制已还原为 canonical `de95bee0`。原始数据：`benchmark/tests/gdn_ab_results.tsv`、`gdn_ab_runs/`。

### 3.0–3.2 结果汇总（first_token_infer，越小越好）

| 变体 | 参数 | first_token_infer ms | ±std | 相对 base | md5 | 结论 |
|---|---|---:|---:|---:|---|---|
| **base** | VBLOCK=4, SG=16（当前） | **197.37** | 0.48 | — | dd753679 | 基准 |
| base_recheck | 同上 | 197.86 | 0.36 | +0.25% | dd753679 | 梯度对照通过（Δ0.5ms）|
| vblock2 | VBLOCK=2 | 217.77 | 0.20 | **+10.3%** | dd753679 | 回退 |
| vblock8 | VBLOCK=8 | 224.72 | 0.47 | **+13.9%** | dd753679 | 回退 |
| sg32 | SUBGROUP_SIZE=32（Xe3 SIMD32）| 246.77 | 8.88 | **+25.0%** | dd753679 | 明显回退 |

**结论（均为负结果，但有价值）**：

1. **V_BLOCK_SIZE=4 在 356H 上确是最优**。VBLOCK=2/8 都更慢；且 vblock2 那一跑起始温度反而比 base 更低（52℃ vs 62℃）却仍慢 +20ms，排除热因素，回退是真实的。硬编码的 4 无需改。
2. **Xe3 的 SIMD32 对 GDN 是明显回退（+25%）**。原因符合机理：GDN 的瓶颈是每个 subgroup ~1002 token 的**串行递归延迟**，不是 SIMD 宽度；SG=32 只是把 K 维归约摊到更多 lane、加深归约树，并未缩短串行链，反而增加每步开销。故"Xe3 更宽 SIMD"这一朴素方向对 GDN 无效。
3. md5 全部保持 `dd753679`（含 SG=32——归约恰好 bit 相同），即这些旋钮都是数值安全的，纯性能维度。
4. **含义**：GDN 的 dispatch/调参空间已被穷尽（本轮 + 前序 VBS 扫描）。要继续压 GDN，只剩 **算法级 chunkwise + XMX 重写**（§4）。gemm 是 oneDNN 不可改。**当前 opt-3/5/6 栈已是"局部 dispatch 优化"的实用上限。**

---

## 4. 主攻方向：GDN chunkwise + XMX（详细计划，供后续接续）

**为什么是它**：GDN 占 prefill 20.6%，是**唯一可编辑**的大热点；当前 kernel `XMX 利用率 = 0%`（全靠 ALU0 65% 做逐 token 点积/FMA）。矩阵引擎整段闲置 = 本机最大的架构性浪费。dispatch 调参已穷尽（§3），只剩算法级重写。

### 4.1 当前 kernel 的串行本质（`paged_gated_delta_net_opt.cl`）
每个 `(head, v_block)` 由一个 subgroup 持有 `128×V_BLOCK` 的 state 于寄存器，**串行**扫 ~1002 token，每 token：
```
b_g = exp(gate); state *= b_g;               // 衰减
h_k = <state, k>;                             // subgroup 归约
update = (v - h_k) * beta;
state += k ⊗ update;                          // rank-1 更新（delta rule）
out = <state, q>;                             // subgroup 归约
```
token t 依赖 t-1（state 递归）→ 无 token 并行；512 个 subgroup 各跑一条 1002 步的串行链（active 84%）。

### 4.2 chunkwise 并行形（flash-linear-attention 的 gated delta net）
把 T 个 token 切成大小 C 的块（如 C=64）。块内用矩阵运算（可上 XMX），块间只串行传递 state S（K×V），串行步数从 T 降到 T/C（1002→~16）。核心（每块，省略跨块衰减细节）：

1. 块内衰减累积：`γ_i = Πexp(gate)`（块内前缀积）。
2. `K̃ = diag(β)·K`（C×d）；构造严格下三角 `A = tril(K̃ Kᵀ, -1)`（C×C），解 `T = (I − A)⁻¹`（单位下三角求逆，块内 forward-substitution）。
3. `W = T·(diag(β)·V)` 等中间量（C×V 矩阵乘）。
4. 块内输出 `O_intra = tril(Q K̃ᵀ)·W`（C×C · C×V），块间 `O_inter = Q·S_in`（C×d · d×V）。
5. 更新块末 state：`S_out = decay·S_in + K̃ᵀ·(…)`（d×C · C×V）。

其中 **QKᵀ(C×C)、T·V(C×V)、Kᵀ·(…)(d×V)、Q·S(C×V)** 都是可走 XMX 的 f16 矩阵乘（C=64、d=128、V=128 对 XMX tile 友好）。唯一必须串行/非 XMX 的是 (I−A)⁻¹ 三角求逆（C×C，每块一次）。

### 4.3 XMX-in-OCL 可行性（已确认）
OpenVINO OCL 内可用 **`DPAS`/`MMAD_8`/`MMAD_16` 宏**（`cl_kernels/` 里 conv/FC-INT8 及 sdpa systolic 路径已在用）表达 systolic 矩阵乘；f16 走 dpas builtin。小矩阵（如 LoRA）用 subgroup-FMA（`lora_opt.cl`），大矩阵用 gemmstone。**结论：C×C / C×d 中等块矩阵乘在手写 OCL 里用 DPAS 宏有先例、可行**，但需仔细安排 tile/布局（state 常驻 SLM 或寄存器，Q/K/V 块从 global 经 2D block load）。

### 4.4 实施步骤（建议顺序）
1. **先在 Python/numpy 写 chunkwise 参考实现**，对拍现有串行 kernel 输出（用 `paged_gated_delta_net_ref` 路径 dump 一层的 q/k/v/gate/beta/state），确认公式与 gating/decay/`use_qk_l2norm`/scale/cache 边界(`cache_interval`) 完全对齐。**这步最关键，占一半风险。**
2. 写**不带 XMX 的 chunkwise OCL**（块内矩阵乘先用 subgroup-FMA），跑通并验证数值接近串行版（cos-sim / 最大逐元素误差）。此时块间串行步数 ↓ 已可能有收益。
3. 再把块内 4 个大矩阵乘换成 **DPAS/XMX**，量 kernel 时间与 vtune `XMX active%`（目标从 0% 拉到显著非零）。
4. 调 chunk size C（32/64/128）、SLM vs 寄存器放 state、2D block load。

### 4.5 正确性与风险
- **几乎必然破 byte-level md5**（矩阵乘累加顺序 ≠ 串行 rank-1）。用户已同意放宽 bit-exact。验收改为：**贪心解码 token 序列与参考一致 / 或 logits cos-sim > 阈值 + 人工可读性**。需建立这套"非 bit-exact 验收"流程。
- 工作量：**数天级**，是本项目最大一块。上限（前序 VBS U 曲线）：GDN 105→~85ms、对总 prefill ~ −3~4%；若 XMX 真能吃下块内矩阵乘，理论可更好（前序估算未用 XMX，偏保守）。**实际上限需步骤 3 后才能定。**
- 风险：cache/paging 边界（`block_indices`/`cache_interval`/prefix caching）在 chunkwise 下比串行版复杂；decode（单 token）保留原串行路径或退化处理。

---

## 5. 下一步 / 待办

- [ ] **（主攻）GDN chunkwise 参考实现 + 对拍**（§4.4 步骤 1）——先 numpy，不碰 kernel。
- [ ] gemm 数据供给角度（候选 E）：vtune 显示 gemm XMX 仅 56% 忙、SBID 停顿 26%，即 XMX 半闲等数据。调研能否减少 DQ↔GEMM 中间往返（如 DQ 输出常驻 L3 / 融合 scale），属 oneDNN 上下游、非 kernel 本体，低确定性。
- [ ] 若 chunkwise 暂不做：**当前 opt-3/5/6 栈可定为本机局部优化交付上限**，后续收益需算法级或 oneDNN 级立项。
- [ ] 复现旋钮：`git apply benchmark/openvino/qwen_patch/dev_gdn_tuning_knobs.patch` + `bash benchmark/openvino/rebuild_gpu_plugin.sh`，再 `bash benchmark/tests/gdn_ab.sh <label> OV_GDN_VBLOCK=.. OV_GDN_SG=..`。

### 附：本轮环境与产物
- fresh vtune：`results-gpu-qwen3.5-2B/`（本轮分析源）。
- A/B 数据：`benchmark/tests/gdn_ab_results.tsv`、`gdn_ab_runs/`；harness `gdn_ab.sh`。
- dev 旋钮 patch：`benchmark/openvino/qwen_patch/dev_gdn_tuning_knobs.patch`。
- 安装二进制已还原 canonical `de95bee0`（opt-3+5+6）；openvino 源码树已还原干净。
