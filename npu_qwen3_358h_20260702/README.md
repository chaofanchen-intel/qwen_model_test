# Qwen3 dense NPU check on 358H (2026-07-02)

Workload matches the 356H study as closely as possible:

- device: `NPU`
- prompt: `benchmark/tests/prompts/qwen3_1k.jsonl`
- `-n 8` measured iterations (`+1` warm-up)
- `-ic 64`
- `--disable-prompt-permutation`
- `--no-monitor`
- tool: `model_bench/build-local/model_bench llm`

Configs reused from `benchmark/tests/npu_qwen3_356h_20260701/`.

## Main results

| Model | Config | TTFT ms | decode tok/s | md5 |
|---|---|---:|---:|---|
| Qwen3-0.6B-INT4 | baseline | 154.50 | 96.68 | `c8a27c900e6d63b0d8c2257fc1b08851` |
| Qwen3-0.6B-INT4 | turbo only | 138.94 | 96.91 | `c8a27c900e6d63b0d8c2257fc1b08851` |
| Qwen3-0.6B-INT4 | BEST_PERF only | 153.49 | 94.83 | `75355c2b11e6d26a291fd6b29c444f67` |
| Qwen3-0.6B-INT4 | BEST_PERF + turbo | 137.65 | 111.91 | `75355c2b11e6d26a291fd6b29c444f67` |
| Qwen3-1.7B-INT4 | baseline | 292.05 | 58.13 | `2f29a2a294b4953abf84a03366a68f37` |
| Qwen3-1.7B-INT4 | turbo only | 272.07 | 58.48 | `2f29a2a294b4953abf84a03366a68f37` |
| Qwen3-1.7B-INT4 | BEST_PERF + turbo | 272.71 | 67.00 | `2f29a2a294b4953abf84a03366a68f37` |
| Qwen3-4B-INT4 | baseline | 716.02 | 32.53 | `2f29a2a294b4953abf84a03366a68f37` |
| Qwen3-4B-INT4 | turbo only | 660.88 | 32.37 | `2f29a2a294b4953abf84a03366a68f37` |
| Qwen3-4B-INT4 | BEST_PERF only | 712.70 | 30.37 | `2f29a2a294b4953abf84a03366a68f37` |
| Qwen3-4B-INT4 | BEST_PERF + turbo | 658.27 | 32.70 | `2f29a2a294b4953abf84a03366a68f37` |

## Conclusions on this 358H host

- `1.7B`: same qualitative result as 356H. `NPU_TURBO=YES` cuts TTFT, and adding `NPUW_LLM_GENERATE_HINT=BEST_PERF` gives a large decode gain with unchanged md5.
- `4B`: `BEST_PERF` alone still regresses decode (`32.53 -> 30.37 tok/s`), matching the 356H warning. But `BEST_PERF + turbo` no longer regresses on this host; it is roughly neutral to slightly positive on decode (`32.70 tok/s`).
- `0.6B`: performance direction is similar only for the full combo, but md5 changes as soon as `BEST_PERF` is enabled. On this host, `turbo only` keeps the baseline md5, while `BEST_PERF only` and `BEST_PERF + turbo` switch to a different stable md5.

## 0.6B token-level diff

- `BEST_PERF only` and `BEST_PERF + turbo` produce identical output text and identical token IDs on this host. The md5 change is therefore caused by `NPUW_LLM_GENERATE_HINT=BEST_PERF`, not by `NPU_TURBO`.
- Relative to baseline, the difference is a small early shift rather than a full rewrite: the optimized output inserts one token, `" you"`, at token index `4`, then drops the baseline's final `"."` to stay within the fixed `64`-token output budget.
- Under sequence alignment, the outputs share `63/64` token IDs (`4` equal tokens before the insertion, then `59` equal tokens after re-alignment). The minimal edit script is `insert " you"` plus `delete trailing "."`.
- The opening text changes from `"pew by yourself before. You have ..."` to `"pew by yourself before you. You have ..."`, and the ending changes from `"... hands clasped together."` to `"... hands clasped together"`.

## Artifact map

- `0p6b_base.*`, `0p6b_turbo.*`, `0p6b_bestperf.*`, `0p6b_opt.*`
- `1p7b_base.*`, `1p7b_turbo.*`, `1p7b_opt.*`
- `4b_base.*`, `4b_turbo.*`, `4b_bestperf.*`, `4b_opt.*`
- `0p6b_base_output.txt`, `0p6b_bestperf_output.txt`, `0p6b_opt_output.txt`
- `0p6b_base_tokens.json`, `0p6b_bestperf_tokens.json`, `0p6b_opt_tokens.json`
- `0p6b_base_vs_bestperf_compare.json`, `0p6b_base_vs_opt_compare.json`

Each `*.json` file contains the aggregate metrics and per-iteration `result_md5` values. Each `*.log` file contains the raw console output from `model_bench`.