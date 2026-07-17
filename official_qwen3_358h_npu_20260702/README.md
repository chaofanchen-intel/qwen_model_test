# Official llm_bench Qwen3 NPU check on 358H (2026-07-02)

Workload for this re-check:

- tool: `benchmark/openvino.genai/tools/llm_bench/benchmark.py`
- device: `NPU`
- prompt: `benchmark/tests/prompts/qwen3_1k.jsonl`
- prompt index: `0`
- `-ic 512`
- `-n 10` measured iterations (`+1` warm-up)
- `-bs 1`
- `--disable_prompt_permutation`
- `--resource_monitor 0`
- `-mc 0`

Configs reused from `benchmark/tests/npu_qwen3_356h_20260701/`.

## Main results

| Model | Config | Compile s | TTFT ms | decode tok/s | total tok/s | md5 |
|---|---|---:|---:|---:|---:|---|
| Qwen3-0.6B-INT4 | baseline | 6.53 | 151.86 | 96.14 | 93.66 | `2720e56e1b6d9f555ca4438496aeaf77` |
| Qwen3-0.6B-INT4 | turbo only | 6.50 | 140.04 | 96.44 | 94.15 | `2720e56e1b6d9f555ca4438496aeaf77` |
| Qwen3-0.6B-INT4 | BEST_PERF only | 12.42 | 150.00 | 101.86 | 99.09 | `b6cb5167c4428eabe6cd2c09b5193a1d` |
| Qwen3-0.6B-INT4 | BEST_PERF + turbo | 12.60 | 139.29 | 112.86 | 109.70 | `b6cb5167c4428eabe6cd2c09b5193a1d` |
| Qwen3-1.7B-INT4 | baseline | 9.88 | 298.65 | 58.25 | 56.44 | `cd00de93ec286ab253b5c979a6658545` |
| Qwen3-1.7B-INT4 | turbo only | 9.93 | 276.69 | 57.83 | 56.18 | `cd00de93ec286ab253b5c979a6658545` |
| Qwen3-1.7B-INT4 | BEST_PERF only | 17.05 | 298.61 | 58.04 | 56.26 | `cd00de93ec286ab253b5c979a6658545` |
| Qwen3-1.7B-INT4 | BEST_PERF + turbo | 16.78 | 271.34 | 65.82 | 63.73 | `cd00de93ec286ab253b5c979a6658545` |
| Qwen3-4B-INT4 | baseline | 15.03 | 719.36 | 31.87 | 30.56 | `7b561c08e09e67dec579f8a525fe6897` |
| Qwen3-4B-INT4 | turbo only | 14.82 | 650.98 | 32.19 | 30.98 | `7b561c08e09e67dec579f8a525fe6897` |
| Qwen3-4B-INT4 | BEST_PERF only | 28.74 | 697.59 | 30.51 | 29.35 | `7b561c08e09e67dec579f8a525fe6897` |
| Qwen3-4B-INT4 | BEST_PERF + turbo | 28.65 | 660.62 | 30.36 | 29.27 | `7b561c08e09e67dec579f8a525fe6897` |

## Conclusions for official benchmark.py on this 358H host

- `0.6B`: the official path reproduces the same correctness split seen in `model_bench`. `turbo only` preserves the baseline md5, while any config that enables `BEST_PERF` switches to a different stable md5. `BEST_PERF + turbo` gives the largest speedup (`TTFT -8.3%`, `decode +17.4%`), but it is not byte-identical to baseline.
- `1.7B`: the main recommendation still holds on the official path. `BEST_PERF + turbo` improves both TTFT and decode throughput with unchanged md5. However, `BEST_PERF only` is nearly neutral here, so the safest recommendation is the combo, not `BEST_PERF` alone.
- `4B`: the official path matches the original 356H warning, not the `model_bench` 358H exception. `BEST_PERF only` regresses decode (`31.87 -> 30.51 tok/s`), and `BEST_PERF + turbo` also regresses decode (`31.87 -> 30.36 tok/s`). On the official path, `4B` should still use `turbo only`.

## 0.6B text check

- `baseline` and `turbo only` export identical output text.
- `BEST_PERF only` and `BEST_PERF + turbo` export identical output text.
- The opening change is the same as in the `model_bench` re-check: `"before. You have ..."` becomes `"before you. You have ..."`.

## Artifact map

- Per-run JSON: `*_iters.json`
- Per-run CSV: `*_iters.csv`
- Raw logs: `*.log`
- Output text dumps: subdirectories such as `0p6b_base/`, `0p6b_bestperf/`, `1p7b_opt/`
- Cross-run summary: `benchmark_summary.csv`, `benchmark_summary.json`