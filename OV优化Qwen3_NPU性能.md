# 优化 Qwen3（dense）在 NPU 上的 TTFT / decode 性能 —— 复现文档

> **适用范围：Qwen3 dense（0.6B / 1.7B / 4B INT4），设备 = NPU（`-d NPU`）。**
> 硬件：Intel Core Ultra 7 356H（Panther Lake）的 AI Boost NPU（`intel_vpu`，PCI `8086:B03E`），Ubuntu 24.04。
> 软件：本地源码构建的 OpenVINO 2026.2.1 + OpenVINO GenAI 2026.2（HEAD `ede283a88e`），NPU level-zero 驱动 `libze_intel_npu`。
> 测量工具：`model_bench`（C++，链接本地 OV），LLMPipeline；NPU 参数通过 `-lc <file>` 传 JSON。
> 撰写日期：2026-07-01。所有数值为 NPU 口径、真实 1000-tok prompt、n=8（+1 warmup）、ic=64。

---

## 0. 结论先行

**NPU 上的 Qwen3 优化与 GPU 完全不同：NPU 没有可编辑的 OCL kernel，整模型由 NPU 编译器（经 NPUW LLM 管线）编译成 blob。优化面是「NPUW / NPU 编译期配置项」，不是逐 kernel 改源码。** 用不动源码、纯配置扫参 + 实测，找到两个有效开关：

| 开关 | 作用 | 效果 | 适用 |
|---|---|---|---|
| **`NPU_TURBO=YES`** | 拉高 NPU 时钟 | TTFT −5~9%，decode 中性~略升 | **全部尺寸通用（安全）** |
| **`NPUW_LLM_GENERATE_HINT=BEST_PERF`** | generate 子图 `NPUW_ONLINE_PIPELINE=NONE`，去掉逐 token 的切分开销 | decode **+17~22%**（小/中模型） | **0.6B / 1.7B**；**4B 上反而 −5%，不要用** |

**最终实测（Qwen3 dense，NPU，n=8，ic=64，真实 1000-tok prompt，md5 逐字节无损）：**

| 模型 | 推荐配置 | TTFT | decode 吞吐 |
|---|---|---:|---:|
| Qwen3-0.6B-INT4 | BEST_PERF + turbo | 152.2 → **138.7 ms（−8.8%）** | 92.80 → **113.65 tok/s（+22.5%）** |
| Qwen3-1.7B-INT4 | BEST_PERF + turbo | 295.6 → **269.1 ms（−9.0%）** | 56.92 → **66.90 tok/s（+17.5%）** |
| Qwen3-4B-INT4 | **turbo only**（不加 BEST_PERF） | 692.3 → **659.4 ms（−4.7%）** | 32.25 → **32.32 tok/s（+0.2%）** |

- **正确性**：所有配置、所有尺寸的 128-token 贪心 md5 与 baseline **逐字节一致**（turbo 只改时钟、BEST_PERF 只改子图切分，均不动数值）。
- **decode 是 NPU 上的大头**：对 ≥128 token 的生成，decode 占端到端 >85%（见 §2），所以 decode 吞吐是最高价值目标——BEST_PERF 正好打这里。

一句话给 leader：**Qwen3 在 NPU 上不能像 GPU 那样改 kernel，只能调 NPUW 编译配置。实测两个开关：`NPU_TURBO` 全模型安全提速（TTFT −5~9%）；`GENERATE_HINT=BEST_PERF` 给小/中模型的 decode 带来 +17~22% 的大幅提升（但 4B 上会 −5%，需按尺寸区分）。全部数值无损。**

### 0.1 跨机器补充（358H）

本页主结论来自 **356H** 机器。2026-07-02 在另一台 **Core Ultra X7 358H** 上按同样 workload 复测，发现结论**大体一致但不是逐项完全可迁移**：

- `1.7B`：与本页一致，`NPU_TURBO=YES` 降 TTFT，`BEST_PERF + turbo` 提升 decode，且 md5 不变。
- `4B`：`BEST_PERF only` 仍然回退，和本页一致；但 `BEST_PERF + turbo` 在 358H 上不再回退，而是 decode 近乎持平到微升。
- `0.6B`：`BEST_PERF`（含 `BEST_PERF + turbo`）在 358H 上会改变输出文本，不再满足本页的“逐字节无损”结论。逐 token 对比显示，它不是整段重写，而是在很早的位置稳定地多出一个 `" you"`，并在固定 `64`-token 上限下挤掉结尾句号。

358H 的原始结果与逐 token 对比见 [npu_qwen3_358h_20260702/README.md](npu_qwen3_358h_20260702/README.md)。

---

## 1. 为什么 NPU 的方法和 GPU 不同

- GPU 插件（intel_gpu）用 OCL kernel，可逐 kernel 改源码 + vtune 逐 kernel profiling（PA/SDPA 文档就是这么做的）。
- **NPU 插件（intel_npu）不同**：LLM 走 **NPUW LLM 管线**（`src/plugins/intel_npu/src/plugin/npuw/llm_compiled_model.cpp`），把 prefill 与 generate 拆成两个子模型，各自交给 NPU 编译器编译成 blob。没有运行时可换的 OCL kernel。
- 因此 `model_bench --profile`（GPU 那套逐算子 `get_profiling_info`）在 NPU 上不适用：它直接 `compile_model` 已导出的 IR 并传 `MAX_PROMPT_LEN` 等 NPUW 级属性，raw NPU 插件会报 `Option 'MAX_PROMPT_LEN' is not supported`（这些是 GenAI/NPUW 管线级属性，不是 plugin 级）。
- **可行的优化面 = NPUW/NPU 编译期配置项**（`NPUW_LLM_*`、`NPU_*`），通过 `-lc` JSON 传入。方法学 = 逐项配置扫参 + 交错 A/B 实测 TTFT 与 decode。

### 1.1 NPUW LLM 的关键配置项（源码定位）
`llm_compiled_model.cpp` 里 hint → 配置的映射：
- `NPUW_LLM_GENERATE_HINT`：默认 **FAST_COMPILE**（dense 模型），映射 `NPUW_UNFOLD_IREQS=YES`（编译快、但 decode 每 token 有额外开销）；`BEST_PERF` 映射 `NPUW_ONLINE_PIPELINE=NONE`（generate 子图不做在线切分 → decode 更快）。
- `NPUW_LLM_PREFILL_HINT`：默认 **DYNAMIC**（配合 chunk size 走 chunked prefill）。
- `NPUW_LLM_PREFILL_ATTENTION_HINT` / `GENERATE_ATTENTION_HINT`：默认 STATIC，可选 HFA（fused flash attention）。
- `NPU_TURBO`：NPU 时钟 turbo（plugin 级 `ov::intel_npu::turbo`）。

---

## 2. 建立基线 + 确认 hotspot（decode 主导）

基线（Qwen3-1.7B，runner 配置 `MAX_PROMPT_LEN=1024, MIN_RESPONSE_LEN=512, NPUW_LLM_PREFILL_CHUNK_SIZE=2048`，n=8）：
- TTFT ≈ 296 ms、decode ≈ 57 tok/s（TPOT ≈ 17.4 ms）。与官方 runner CSV 基线（TTFT ~290ms、~57 tok/s）一致。

**prefill vs decode 占比（TTFT 296ms、TPOT 17.4ms）：**

| 生成长度 | 端到端 | prefill 占比 | decode 占比 |
|---|---:|---:|---:|
| 32 token | 0.84 s | 36% | 64% |
| 128 token | 2.51 s | 12% | **88%** |
| 512 token | 9.19 s | 3% | **97%** |

**结论：对任何 ≥128 token 的真实生成，decode 是绝对大头。** 所以最高价值目标是 decode 吞吐（TPOT），其次才是 TTFT。这决定了后面优先看 `GENERATE_HINT`（直接打 decode）。

---

## 3. 配置扫参（逐项，Qwen3-1.7B）

每个配置独立 `-lc` 传入、重新编译（NPU 编译 ~10–18s）、n=5 实测。相对 baseline（decode 56.8 tok/s、TTFT 298ms）：

| 配置 | TTFT ms | decode tok/s | 结论 |
|---|---:|---:|---|
| baseline（runner 配置） | 298.2 | 56.82 | — |
| `GENERATE_HINT=BEST_PERF` | 299.1 | 58.34 | decode 小升，TTFT 中性 |
| `PREFILL_ATTENTION_HINT=HFA` | 297.3 | 57.02 | 无效（prefill attention 走 HFA 在本 NPU 无收益） |
| `NPU_TURBO=YES` | 272.5 | 57.98 | TTFT 明显降，decode 略升 |
| **`BEST_PERF` + `NPU_TURBO`** | **277.5** | **60.13** | **两者叠加最佳** |
| `BEST_PERF` + DQ(`NPU_COMPILER_DYNAMIC_QUANTIZATION=YES`) | 297.3 | 59.40 | DQ 无额外收益（默认已启用 DQ），且丢了 turbo 的 TTFT 收益 |

**已排除项：**
- **`GENERATE_ATTENTION_HINT=HFA` / generate 侧 HFA**：编译**病态地慢**（单配置 >12 分钟仍未完成，CPU 满载、1.1GB RSS），实测不可用，已放弃。类似 SDPA 文档提到的 NPU 融合算子编译爆炸问题。
- **DQ 显式开启**：无收益（NPUW 默认已对 generate/prefill 应用 DQ，见 `get_default_generate_config` / `get_default_prefill_config`）。

---

## 4. 跨尺寸验证（n=8）+ 关键发现：BEST_PERF 对 4B 反而回退

把「BEST_PERF + turbo」放到三个尺寸上（n=8，ic=64），并对 4B 做 turbo/BEST_PERF 单项拆分：

| 模型 | baseline decode | turbo-only decode | BEST_PERF-only decode | BEST_PERF+turbo decode | 最优选择 |
|---|---:|---:|---:|---:|---|
| 0.6B | 92.80 | — | — | **113.65（+22.5%）** | BEST_PERF+turbo |
| 1.7B | 56.92 | 57.27 | 58.34 | **66.90（+17.5%）** | BEST_PERF+turbo |
| 4B | 32.25 | **32.32（中性）** | 30.67（**−4.9% 回退**） | 30.01（−7%） | **turbo only** |

**发现：`GENERATE_HINT=BEST_PERF`（即 `NPUW_ONLINE_PIPELINE=NONE`）在小/中模型上是 decode 大赢（+17~22%），但在 4B 上反而回退 ~5%。** 机理推测：4B 的 generate 子图更大，关闭在线切分（ONLINE_PIPELINE=NONE）后子图无法被有效拆分调度，反而降低 NPU 利用率；小模型子图小、切分开销占比高，去掉切分净赚。`NPU_TURBO` 则在所有尺寸上安全（提速或中性），是通用开关。

- 1.7B 拆分佐证：turbo-only decode 57.27 → 加 BEST_PERF 到 66.90，**BEST_PERF 独立贡献 +16.8% decode**。
- 4B 拆分佐证：turbo-only 32.32（中性）、BEST_PERF-only 30.67（回退），叠加后 30.01（回退），证明是 BEST_PERF 拖累 4B。

---

## 5. 正确性

- 贪心解码（`do_sample=false`），关闭 prompt permutation（`--disable-prompt-permutation`）跑固定 1000-tok prompt，比 baseline 与优化态的 `result_md5` 多重集。
- **在本页对应的 356H 机器上**，三尺寸、各自最优配置 vs baseline 均 **单一 md5 逐字节一致**（如 1.7B/0.6B/4B 在该 prompt 下均为 `b1be2553ef3b…`）。
- **这个“无损”结论不是跨机器绝对成立。** 358H 上，`0.6B` 只要启用 `BEST_PERF`，输出文本就会稳定地发生一次早期 token 插入，并因 `64`-token 上限丢掉结尾句号；`turbo only` 则保持 baseline md5。
- 概念上无损：`NPU_TURBO` 只改时钟频率、不改数值；`GENERATE_HINT` 只改 generate 子图的切分/调度方式、不改算术。

---

## 6. 落地建议

**按模型尺寸选择 NPU 配置（在 runner 已有的 `MAX_PROMPT_LEN/MIN_RESPONSE_LEN/PREFILL_CHUNK_SIZE` 之上追加）：**

```jsonc
// Qwen3-0.6B / 1.7B（NPU）：
{ "NPU_TURBO": "YES", "NPUW_LLM_GENERATE_HINT": "BEST_PERF" }

// Qwen3-4B（NPU）：只加 turbo，不要 BEST_PERF
{ "NPU_TURBO": "YES" }
```

- 收益：0.6B/1.7B decode +17~22%、TTFT −9%；4B TTFT −5%、decode 持平。
- 全部数值无损、纯配置项（无源码改动、无重编译 OV 插件）。
- **注意**：`GENERATE_HINT=BEST_PERF` 是尺寸相关的开关——小/中模型开、4B（及更大）关。上线前对目标模型跑一次 A/B 确认。
- **再补一条**：上面的“0.6B / 1.7B 开 `BEST_PERF`”是 **356H 上的经验结论**。如果目标机器不是同一台 356H，尤其是像 358H 这种不同 stepping / host 组合，`0.6B` 必须单独做正确性复验；在 358H 上，`0.6B` 的 `turbo only` 保持 baseline md5，而 `BEST_PERF` 会改变输出文本。

---

## 7. 一页复现 checklist

```bash
# 0) 环境
cd <repo-root>
source benchmark/openvino/install/setupvars.sh
export LD_LIBRARY_PATH="$PWD/benchmark/openvino/install/runtime/lib/intel64:$LD_LIBRARY_PATH"

# 1) baseline（runner 配置）
echo '{"MAX_PROMPT_LEN":1024,"MIN_RESPONSE_LEN":512,"NPUW_LLM_PREFILL_CHUNK_SIZE":2048}' > /tmp/npu_base.json
# 2) 优化（小/中模型）
echo '{"MAX_PROMPT_LEN":1024,"MIN_RESPONSE_LEN":512,"NPUW_LLM_PREFILL_CHUNK_SIZE":2048,"NPU_TURBO":"YES","NPUW_LLM_GENERATE_HINT":"BEST_PERF"}' > /tmp/npu_opt.json

# 3) A/B（n=8，对比 ttft_mean_ms + decode = 1000/tpot_mean_ms；正确性用 --disable-prompt-permutation 比 result_md5）
cd model_bench
./build-local/model_bench llm -m ../benchmark/models/Qwen3-1.7B-INT4 -d NPU -n 8 -ic 64 \
  -pf ../benchmark/tests/prompts/qwen3_1k.jsonl --no-monitor -lc /tmp/npu_opt.json -r /tmp/x.csv -rj /tmp/out.json
```

原始数据：`npu_qwen3_356h_20260701/`（各配置 A/B 的 JSON + md5 校验 JSON）。
