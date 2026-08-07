# Qwen3.5-2B：OpenVINO 2026.3 本机 opt-1~opt-5 逐项验证

## 1. 测试目的

验证 `qwen3.5_patch` 中的五个 GPU plugin 优化，在本机 OpenVINO 2026.3 base 上是否仍然有效。测试采用累计启用方式：每个状态都从同一个 2026.3 pristine plugin 出发，只增加当前状态需要的 patch。

本报告只记录本机实测结果，不直接复用另一台机器上的绝对性能数据。

## 2. 测试环境和口径

- OpenVINO base：2026.3，源码工作树当前分支 `qwen-ttft-optimizations`
- Runtime：`benchmark/openvino/install` 下的 2026.3 build
- GenAI：2026.3 build
- 设备：本机 Intel iGPU，`GPU`
- 模型：`Qwen3.5-2B-INT4`
- 输入：`model_bench/prompts/qwen3.5_1k.jsonl`，实际约 1002 input tokens
- 输出：512 tokens
- 测量：5 次 measured iterations + 1 次 warm-up
- Prompt permutation：开启
- 性能配置：`PERFORMANCE_HINT=LATENCY`
- 结果目录：[results](results)
- Runner：[run_qwen35_opt_ab.sh](run_qwen35_opt_ab.sh)

每个状态都重新编译 `openvino_intel_gpu_plugin`，保存对应 plugin binary，然后运行相同的 `model_bench vlm` 命令。测试结束后 runner 自动恢复测试前的源码文件和安装 plugin。

## 3. SDPA 逐项结果

SDPA 只测试 opt-1、opt-2、opt-3，累计状态为 pristine → opt1 → opt1+opt2 → opt1+opt2+opt3。

| 状态 | TTFT ms | 相对 pristine | TPOT ms | Decode TPS |
|---|---:|---:|---:|---:|
| pristine | 481.90 | 0.00% | 38.39 | 26.08 |
| opt1 | 470.23 | **-2.42%** | 39.00 | 25.64 |
| opt1 + opt2 | 450.27 | **-6.56%** | 39.07 | 25.61 |
| opt1 + opt2 + opt3 | 428.69 | **-11.04%** | 37.37 | 26.77 |

### SDPA 分项判断

- opt1 在本机有效，TTFT 下降约 2.4%。
- opt2 在 opt1 基础上继续带来约 4.2 个百分点的绝对增益，累计 TTFT 下降约 6.6%。
- opt3 在 opt1+opt2 基础上继续带来约 4.8 个百分点的绝对增益，最终累计 TTFT 下降约 11.0%。
- TPOT/TPS 主要是小幅波动，优化目标是 prefill/TTFT，不能用 decode 指标判断 opt1/2/3 是否生效。

## 4. PA 逐项结果

PA 测试累计状态为 pristine → opt3 → opt3+opt4 → opt3+opt4+opt5。

| 状态 | TTFT ms | 相对 pristine | TPOT ms | Decode TPS |
|---|---:|---:|---:|---:|
| pristine | 310.07 | 0.00% | 32.61 | 30.66 |
| opt3 | 291.22 | **-6.08%** | 32.53 | 30.74 |
| opt3 + opt4 | 290.17 | **-6.42%** | 32.35 | 30.91 |
| opt3 + opt4 + opt5 | 288.76 | **-6.87%** | 32.59 | 30.68 |

### PA 分项判断

- opt3 在本机有效，TTFT 下降约 6.1%，是 PA 状态的主要收益来源。
- opt4 在 opt3 基础上继续下降约 1.05 ms，累计收益从 6.08% 增加到 6.42%；本机收益很小，但方向为正。
- opt5 在 opt3+opt4 基础上继续下降约 1.41 ms，最终累计 TTFT 下降约 6.87%。
- opt4/opt5 的增量收益受本机模型、GPU 和 memory behavior 影响较大；本次结果证明它们在本机 2026.3 build 上仍有正向效果，但收益小于 SDPA opt2/opt3。

## 5. 结论

五个优化点在本机 OpenVINO 2026.3 base 上都没有失效：

- SDPA：opt1、opt2、opt3 全部有效，累计 TTFT `481.90 → 428.69 ms`，下降 **11.04%**。
- PA：opt3、opt4、opt5 全部有效，累计 TTFT `310.07 → 288.76 ms`，下降 **6.87%**。

本机实际收益排序：

1. SDPA opt3 / opt2：主要收益点；
2. PA opt3：主要收益点；
3. SDPA opt1：有稳定但较小的收益；
4. PA opt4 / opt5：本机增量较小，但仍为正收益。

这次验证只覆盖 `Qwen3.5-2B-INT4` 和本机 Intel iGPU，且每个状态 n=5；它证明 patch 在 2026.3 上可以编译、运行并带来本机 TTFT 收益，不代表所有 Qwen3.5 尺寸或其他 GPU 的绝对收益相同。

## 6. 原始产物

每个状态均保存了：

- `*.log`：完整 model_bench 输出
- `*.json`：结构化性能结果
- `*.csv`：迭代明细
- `libopenvino_intel_gpu_plugin_*.so`：对应状态的 2026.3 GPU plugin

所有产物位于本目录的 [results](results)。
