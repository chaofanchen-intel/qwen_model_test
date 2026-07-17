# OpenVINO Benchmark Suite

Performance benchmarks for **Qwen3 LLMs**, **Qwen3.5 VLMs**, **LLMLingua-2** prompt-compression models, **Qwen3-Embedding** text-embedding models, and **Qwen3-Reranker** text-reranking models on CPU / GPU / NPU via OpenVINO and OpenVINO GenAI.

This copy lives under `/home/user/chaofan/model_test/benchmark/` and uses the freshly exported models in `benchmark/models/` plus the local source trees in `benchmark/openvino/` and `benchmark/openvino.genai/`. For tasks supported by official OpenVINO GenAI `llm_bench` (**Qwen3**, **Qwen3.5**, **embedding**, **reranker**), use the `run_official_*.sh` wrappers first. The direct Python implementations are retained for debugging, cross-checking, and LLMLingua-2.

---

## Directory layout

```
benchmark/
├── models/                    ← exported models used by this benchmark suite
├── openvino/                  ← local OpenVINO source tree
├── openvino.genai/            ← local OpenVINO GenAI source tree, including tools/llm_bench
└── tests/
    ├── README.md              ← this file
    ├── lib/                   ← shared library (monitors, metrics, reports, ...)
    ├── benchmark_qwen3.py     Qwen3 text-only LLM direct benchmark
    ├── benchmark_qwen3_5.py   Qwen3.5 VLM direct benchmark
    ├── benchmark_lingua.py    LLMLingua-2 direct benchmark
    ├── benchmark_embedding.py Qwen3-Embedding direct benchmark
    ├── benchmark_reranker.py  Qwen3-Reranker direct benchmark
    ├── build_prompt_jsonl.py  Helper: build a JSONL prompt with a target token count
    ├── runners/               convenience shell wrappers
    ├── prompts/               pre-built JSONL prompts
    ├── results/               direct-script default CSV/JSON output dir
    └── logs/                  timestamped per-run logs
```

---

## Quick start

From the **benchmark root** (`/home/user/chaofan/model_test/benchmark/`):

```bash
# Official-first suite over CPU/GPU/NPU. Qwen3/Qwen3.5 default to llama-2-7b-chat_l.jsonl
# prompt index 0 as the approximate 1k input, with -ic 512 output tokens.
./tests/runners/run_all.sh

# Official Qwen3, CPU/GPU/NPU, auto-discovers models/Qwen3-* and models/qwen3-*, approximate 1k input / 512 output
./tests/runners/run_official_qwen3.sh

# Official Qwen3.5 VLM, auto-discovers models/Qwen3.5-*, text-only by default, same 1k/512 workload
./tests/runners/run_official_qwen3_5.sh

# LLMLingua-2 direct path, because official llm_bench does not cover prompt compression
./tests/runners/run_lingua.sh

# Official Qwen3-Embedding / Qwen3-Reranker wrappers
./tests/runners/run_official_embedding.sh
./tests/runners/run_official_reranker.sh

# Direct retained implementations for cross-checks
./tests/runners/run_qwen3.sh
./tests/runners/run_qwen3_5.sh
./tests/runners/run_embedding.sh
./tests/runners/run_reranker.sh
```

Each runner writes:
- `tests/logs/<bench>_<timestamp>/` — per-run log files + a `run_info.txt` manifest
- `tests/logs/<bench>_<timestamp>/results_*.csv` — append-mode CSV
- `tests/logs/<bench>_<timestamp>/results_*.json` — append-mode JSON

---

## Hardware and memory monitoring

Direct Python benchmark scripts default to official-like no resource monitor unless you pass `--monitor` / `--memory-monitor`. Runner scripts default to resource collection so batch runs produce CPU/GPU/NPU and memory columns in CSV/JSON.

The custom runner monitors track:
- **CPU per-core utilization** — `/proc/stat` poller (no privileges needed)
- **GPU utilization** — shells out to `xpu-smi stats -d 1 -j` (needs sudo for engine_util)
- **NPU utilization** — reads `/sys/bus/pci/drivers/intel_vpu/<dev>/npu_busy_time_us` and PMT GUID
- **Process/System memory** — background sampler for process RSS and host system memory, reported in MiB

Because xpu-smi/PMT need root, the benchmarks call `sudo -v` once before running so the password prompt does not interleave with output. Use `--no-monitor` to skip CPU/GPU/NPU utilization monitoring.

Memory monitoring is separate from hardware utilization monitoring and does not require sudo. It is disabled by default to match official `llm_bench` (`-mc 0`) and to avoid perturbing latency numbers. Enable it with `--memory-monitor`; tune the sampling period with `--memory-monitor-interval <ms>`.

Memory fields follow the useful subset of official `llm_bench -mc 1/2`:
- `initial_*_mem_mib`: process/system memory before model compilation or loading
- `compile_max_*_mem_mib`: max memory observed while constructing/compiling the pipeline/model
- `compile_max_*_mem_increase_mib`: increase relative to the compile-stage start
- `runtime_*`: measured-iteration summaries; per-iteration memory is also stored in JSON under `iterations[]`

Like the official tool, memory sampling can slightly perturb performance timings. For strict performance runs, leave memory monitoring off and run a separate memory-focused pass when needed.

### Official benchmark wrappers

The official [openvino.genai/tools/llm_bench/benchmark.py](../openvino.genai/tools/llm_bench/benchmark.py) has a compact resource monitor enabled by default via `--resource_monitor 1`. It adds these fields to detailed CSV/JSON reports without changing the core official metric path:

```
2nd_tokens_throughput(tokens/s), total_throughput(tokens/s),
resource_cpu_avg_util(%), resource_cpu_cpu0_avg_util(%), resource_cpu_cpu1_avg_util(%), ...,
resource_gpu_avg_util(%), resource_gpu_memory_avg_util(%), resource_npu_avg_util(%),
resource_avg_rss_mem(MiB), resource_peak_rss_mem(MiB),
resource_avg_sys_mem(MiB), resource_peak_sys_mem(MiB), resource_memory_samples
```

The official runner scripts wrap common commands and behave closer to the custom runners:

```bash
./tests/runners/run_official_qwen3.sh
./tests/runners/run_official_qwen3_5.sh
```

They write a descriptive log directory such as `tests/logs/official_qwen3_Qwen3-0.6B-INT4_GPU_<ts>/`, containing:
- `run_info.txt`: runner manifest and software versions
- `<model>_<device>_ic<N>_n<M>_iters.csv/json`: detailed official report with every iteration plus official avg/min/median rows

They also append avg-only benchmark summaries to stable files by default:
- `results/benchmark_qwen3_official.csv/json`
- `results/benchmark_qwen3_5_official.csv/json`

Set `BENCHMARK_CSV` / `BENCHMARK_JSON` when you want a different shared summary file. Set `BENCHMARK_DIR` to move both defaults together.

Useful overrides:

```bash
MODEL=/home/user/chaofan/model_test/benchmark/models/Qwen3-1.7B-INT4 ITERS=10 INFER_COUNT=512 ./tests/runners/run_official_qwen3.sh
MODEL=/home/user/chaofan/model_test/benchmark/models/Qwen3.5-4B-INT4 ITERS=10 INFER_COUNT=512 ./tests/runners/run_official_qwen3_5.sh
MODELS="0.6B 1.7B" ./tests/runners/run_official_qwen3.sh GPU
MODELS="0.8B 2B" QUANTS="INT4" ./tests/runners/run_official_qwen3_5.sh GPU
NPU_MAX_PROMPT_LEN=1024 ./tests/runners/run_official_qwen3.sh NPU
RESOURCE_MONITOR=0 ./tests/runners/run_official_qwen3.sh
MEMORY_CONSUMPTION=2 ./tests/runners/run_official_qwen3.sh
BENCHMARK_CSV=results/qwen3_benchmark.csv BENCHMARK_JSON=results/qwen3_benchmark.json ./tests/runners/run_official_qwen3.sh GPU
```

The wrappers run `sudo -v` before benchmarking when `RESOURCE_MONITOR=1` and `SUDO_REFRESH=1` (default). The monitor itself uses `sudo -n` for `xpu-smi` and NPU sysfs reads, so it will never block in a background sampling thread. If you call `benchmark.py` directly and want GPU/NPU telemetry, run `sudo -v` first:

```bash
cd openvino.genai/tools/llm_bench
sudo -v
python benchmark.py -m /home/user/chaofan/model_test/benchmark/models/Qwen3-0.6B-INT4 \
    -d GPU -n 10 -ic 512 -pf prompts/llama-2-7b-chat_l.jsonl \
    --task text_gen -bs 1 \
    -r results/qwen3_iters.csv -rj results/qwen3_iters.json \
    --benchmark_csv results/qwen3_benchmark.csv \
    --benchmark_json results/qwen3_benchmark.json
```

`resource_gpu_avg_util(%)` is filled only when `xpu-smi` exposes a compute/GPU utilization metric. On the current PTL image, `xpu-smi stats` exposes memory utilization and frequency but not compute utilization, so `resource_gpu_avg_util(%)` may be blank; this avoids reporting memory utilization as GPU utilization. `resource_gpu_memory_avg_util(%)` records the memory-utilization metric separately when available.

Official `llm_bench` does not implement the LLMLingua-2 token-classification compression workflow used by `tests/benchmark_lingua.py`. The LLMLingua BERT model can be forced through official `--task text_rerank` because its `model_type` is `bert`, but that measures `TextRerankPipeline` query/document scoring, not prompt compression quality or compression throughput. `--task text_embed` is not suitable for these token-classification exports.

---

## Common CLI surface

`benchmark_qwen3.py` and `benchmark_qwen3_5.py` share the same option set, modeled on the official `llm_bench/benchmark.py`:

| Flag | Default | Meaning |
|---|---|---|
| `-m / --model-dir` *(req)* | — | One or more OV model directories |
| `-d / --device` | `CPU` | One or more devices (`CPU GPU NPU AUTO …`) |
| `-p / --prompt` | — | Inline prompt (mutually exclusive with `-pf`) |
| `-pf / --prompt-file` | — | One or more JSONL files (`{"prompt": "..."}` per line) |
| `-pi / --prompt-index` | — | Subset of indices to run (post-concatenation, deduplicated) |
| `--input-tokens` | — | Build one inline prompt near this final input-token count; mutually exclusive with `-p/-pf` |
| `--input-base-text` | — | Optional `.txt` source for `--input-tokens` prompt construction |
| `-ic / --infer-count` | `512` | Requested maximum output tokens (`max_new_tokens`); actual output may be shorter |
| `--exact-output-tokens` | off | Best-effort fixed output: sets `min_new_tokens=max_new_tokens=--infer-count` when the backend supports it |
| `-n / --num-iters` | `5` | Measured iterations per prompt (warm-up iter 0 always extra) |
| `-bs / --batch-size` | `1` | Informational; pipeline still invoked once per prompt |
| `--num-beams` | `1` | Beam-search width |
| `--apply-chat-template` | off | Wrap prompt in tokenizer's chat template |
| `--disable-prompt-permutation` | off | **See "Prefix-cache & permutation" below** |
| `-s / --seed` | `42` | RNG seed |
| `-lc / --load-config / --load_config` | — | JSON file with extra OV plugin properties (`PERFORMANCE_HINT`, etc.) |
| `--cache-dir` | `""` | OpenVINO compiled-model `CACHE_DIR`; empty string matches official `llm_bench` |
| `--num-streams` | — | `NUM_STREAMS` plugin property |
| `-r / --csv-out` | `results/<bench>.csv` | CSV report path (append mode if exists) |
| `-rj / --json-out` | `results/<bench>.json` | JSON report path (append mode if exists) |
| `--monitor` | off | Enable HW utilization monitoring (CPU/GPU/NPU); direct scripts default off to match official `llm_bench` |
| `--no-monitor` | on | Keep HW utilization monitoring disabled |
| `--memory-monitor` | off | Enable process/system memory monitoring |
| `--no-memory-monitor` | off | Explicitly disable process/system memory monitoring |
| `--memory-monitor-interval` | `100` | Memory sampling interval in milliseconds |
| `--cooldown-between-runs` | `5` | Seconds slept between (model, device) pairs |

`benchmark_qwen3_5.py` adds:
- `--image <path>` — activates multimodal mode
- `--optimum-fallback` — explicitly use the legacy optimum.intel GPU fallback instead of `VLMPipeline`
- VLM prompt permutation is **ON by default** (via the shared `--disable-prompt-permutation`, default off), matching the official `visual_text_gen` path. Pass `--disable-prompt-permutation` for a cache-warm scenario.

`benchmark_lingua.py` differs because it benchmarks an encoder-only token-classification model:
| Flag | Default | Meaning |
|---|---|---|
| `--input-lengths` | `64 128 256 512` | List of token counts (each ≤ 512) |
| `--compression-rates` | `0.3 0.5 0.7` | Target keep-ratios (0.5 = keep 50%) |
| `--warmup` / `--iters` | `5 / 50` | Warm-up vs measured iterations |

---

## Qwen3-Embedding & Qwen3-Reranker

These two families are **single forward-pass encoders** (Qwen3-Embedding and Qwen3-Reranker are decoder models used as encoders). There is **no autoregressive decode loop**, so the generation metrics (TTFT / TPOT / decode TPS) do not apply. Instead we measure end-to-end **latency** and **throughput** (documents/s and tokens/s), mirroring the official `llm_bench` `text_embed` / `text_rerank` GenAI paths.

Key API facts (verified against `openvino_genai` 2026.2):
- `TextEmbeddingPipeline.embed_documents(texts)` / `.embed_query(text)` and `TextRerankPipeline.rerank(query, texts)` return plain Python lists.
- Neither pipeline exposes a `perf_metrics` object or `get_tokenizer()`. So we time the call directly with `time.perf_counter()` (exactly what the official tool does) and load a standalone `ov_genai.Tokenizer` from the model dir for token accounting. Tokenization is kept out of the pure-latency window but folded into the reported `e2e` latency.

### `benchmark_embedding.py`

Defaults follow the Qwen3-Embedding model card: **LAST_TOKEN pooling**, **left padding**, **L2 normalize on**, and an instruction-prefixed query.

| Flag | Default | Meaning |
|---|---|---|
| `-m / --model-dir` *(req)* | — | One or more OV embedding model dirs |
| `-d / --device` | `CPU` | One or more devices |
| `--mode` | `documents` | `documents` → `embed_documents(batch)`; `query` → `embed_query(single)`; `both` |
| `-bs / --batch-size` | `1` | Texts encoded per `embed_documents()` call (documents mode) |
| `--pooling-type` | `last_token` | `cls` / `mean` / `last_token` |
| `--normalize` / `--no-normalize` | on | L2-normalize embeddings |
| `--padding-side` | `None` | `left` / `right`; Qwen3 last-token pooling expects `left` |
| `--max-length` | pipeline default | Pad/truncate length (Qwen3 default is large) |
| `--pad-to-max-length` | off | Fix input shape to `--max-length` (useful for NPU) |
| `--query-instruction` / `--embed-instruction` | `None` | Instruction prefixes (see model card) |
| `-p / -pf / -pi` | — | Inline text / JSONL file(s) (`{"prompt": ...}` or `{"text": ...}`) / index subset |
| `--input-tokens` | — | Build one inline text near this token count |
| `-n / --num-iters` | `5` | Measured iterations (+1 warm-up) |

Example — sweep batch size on CPU and GPU:
```bash
python tests/benchmark_embedding.py \
    -m models/Qwen3-Embedding-0.6B-int8-ov \
    -d CPU GPU --mode both -bs 16 -n 10
```

### `benchmark_reranker.py`

`TextRerankPipeline.rerank(query, documents)` scores every (query, document) pair in one call and returns the top-N `(index, score)` tuples.

| Flag | Default | Meaning |
|---|---|---|
| `-m / --model-dir` *(req)* | — | One or more OV reranker model dirs |
| `-d / --device` | `CPU` | One or more devices |
| `-p / --prompt / --query` | built-in | Inline query |
| `-pf / -pi` | — | JSONL query file(s) / index subset |
| `--texts` | built-in set | Candidate documents (inline) |
| `--texts-file` | — | JSONL candidate docs (`{"text": ...}` per line) |
| `--num-documents` | as-is | Replicate/trim candidates to exactly N (throughput sweep) |
| `--top-n` | `3` | Results returned (`TextRerankPipeline.Config.top_n`) |
| `--max-length` | pipeline default | Pad/truncate length (Qwen3 default 8192) |
| `-n / --num-iters` | `5` | Measured iterations (+1 warm-up) |

Example — candidate-set-size sweep:
```bash
for N in 4 16 64; do
  python tests/benchmark_reranker.py \
      -m models/Qwen3-Reranker-0.6B-int8-ov -d CPU \
            -pf tests/prompts/reranker/queries.jsonl \
            --texts-file tests/prompts/reranker/texts_64.jsonl \
            --num-documents $N --top-n 10 --max-length 512 -n 10
done
```

Both record the same memory / CPU / GPU / NPU columns as the generation benchmarks, plus per-iteration history in JSON. The reranker JSON additionally records `top1_index` / `top1_score` so you can confirm scoring determinism across iterations.

> **NPU note:** the Qwen3 encoder IRs are dynamic-shape; the NPU compiler may reject them unless you pass `--pad-to-max-length --max-length <N>` to fix the shape. CPU and GPU work out of the box.

### Reranker Benchmark 推荐测法

Reranker 的输入应该拆成两个文件：`-pf/--prompt-file` 是 query 列表，每行 `{"prompt": ...}`；`--texts-file` / official `--texts_file` 是候选文档列表，每行 `{"text": ...}`。不要用 `prompts/llama-2-7b-chat_l.jsonl` 作为默认 query，它是长聊天/总结 prompt，不代表检索请求。

The repo now ships a small retrieval-shaped input set:
- `tests/prompts/reranker/queries.jsonl` — 10 concise search queries.
- `tests/prompts/reranker/texts_16.jsonl` — 16 mixed relevant and distractor passages for quick runs.
- `tests/prompts/reranker/texts_64.jsonl` — 64 passages for candidate-count scaling.

Recommended sweeps:
- **Candidate count:** keep query and max length fixed, test `texts_16.jsonl`, `texts_64.jsonl`, or local `NUM_DOCUMENTS=4 16 64` against `texts_64.jsonl`.
- **Max length:** keep query/docs fixed and sweep `RERANKING_MAX_LENGTH=256 512 1024 2048` for official runs or `MAX_LENGTH=256 512 1024 2048` for local runs.
- **Device/precision:** keep the workload fixed and vary CPU/GPU/NPU plus model precision (`int8`, `int4`, `fp16`) so latency/resource differences are attributable to deployment choice.
- **Quality pass:** performance CSV/JSON does not validate ranking quality; use labeled data and metrics such as MRR, nDCG, Hit@k, or Recall@k when quality matters.

Official runner examples:
```bash
# Default: queries.jsonl + texts_16.jsonl + RERANKING_MAX_LENGTH=512
./tests/runners/run_official_reranker.sh GPU

# Larger candidate set
TEXTS_FILE=$PWD/tests/prompts/reranker/texts_64.jsonl \
RERANKING_MAX_LENGTH=512 \
./tests/runners/run_official_reranker.sh GPU

# Sequence-length sweep
for L in 256 512 1024 2048; do
    RERANKING_MAX_LENGTH=$L \
    TEXTS_FILE=$PWD/tests/prompts/reranker/texts_64.jsonl \
    ./tests/runners/run_official_reranker.sh GPU
done
```

Local runner examples:
```bash
# Default: queries.jsonl + texts_16.jsonl + MAX_LENGTH=512
./tests/runners/run_reranker.sh CPU

# Candidate-count sweep using one 64-document file
for N in 4 16 64; do
    NUM_DOCUMENTS=$N \
    TEXTS_FILE=tests/prompts/reranker/texts_64.jsonl \
    MAX_LENGTH=512 \
    ./tests/runners/run_reranker.sh CPU
done
```

Interpret the core dimensions as `query length`, `candidate document length`, `candidate documents per query`, and `max_length`. In official GenAI reranker logs, `Input token size` is not as clean as text generation's prompt token count; use the manifest fields (`texts_file`, `reranking_max_len`) and candidate file size as the workload definition.

### Official `llm_bench` parity

The official upstream tool fully supports these via `--task text_embed` / `--task text_rerank`. The wrappers `run_official_embedding.sh` / `run_official_reranker.sh` drive `benchmark.py` with the same resource-monitor extensions as the other official wrappers and write `results/benchmark_{embedding,reranker}_official.{csv,json}` summaries:

```bash
MODEL=models/Qwen3-Embedding-0.6B-int8-ov ./tests/runners/run_official_embedding.sh CPU
MODEL=Qwen3-Reranker-0.6B-int8-ov        ./tests/runners/run_official_reranker.sh CPU
```

`MODEL=` accepts absolute paths, project-relative paths such as `models/<dir>`, or a bare directory name under `models/`. Auto-discovery resolves absolute paths before the official wrapper `cd`s into `llm_bench`.

Official embedding defaults now use `tests/prompts/embedding/texts.jsonl`, `EMBEDDING_MAX_LENGTH=512`, and `PAD_TO_MAX_LENGTH=1` so the run is a fixed-shape embedding workload instead of a Llama chat prompt. Set `PROMPT_FILE=`, `EMBEDDING_MAX_LENGTH=`, or `PAD_TO_MAX_LENGTH=0` when you intentionally want the upstream default text, pipeline max length, or natural unpadded lengths.

Official reranker defaults now use `tests/prompts/reranker/queries.jsonl`, `tests/prompts/reranker/texts_16.jsonl`, and `RERANKING_MAX_LENGTH=512`. Set `TEXTS_FILE=` to fall back to benchmark.py's tiny built-in candidate list, or set `TEXTS_FILE=tests/prompts/reranker/texts_64.jsonl` for a more useful candidate-count workload.

For NPU, the official embedding/reranker wrappers pass an automatic `NPU_LOAD_CONFIG` only to NPU runs. It currently sets an OpenVINO compiled-model `CACHE_DIR` under `cache_npu/embedding` or `cache_npu/reranker`. Set `NPU_LOAD_CONFIG=` to disable it, set `NPU_LOAD_CONFIG='{"CACHE_DIR":"..."}'` to override it, or set `LOAD_CONFIG=...` when you want one config to apply to every device.

A small upstream-compatibility fix was applied to `llm_bench_utils/output_json.py:estimate_throughput`: for `text_embed`/`text_rerank` there is no 2nd-token latency, so the averaged value arrives as `""` and the original `latency > 0` comparison raised `TypeError`, blocking the `--benchmark_csv` summary. It now coerces defensively (`float(latency)` inside a `try`). The authoritative per-iteration `-r` CSV already recorded correct embed/rerank `generation_time`/`latency`; only the optional summary file was affected.

---

## Prefix-cache & prompt permutation

OpenVINO GenAI keeps a **KV prefix cache** between iterations of a single pipeline. If you call `pipe.generate("same prompt")` twice in a row, the second call can re-use the cached prefill; TTFT may collapse from realistic prefill latency to a tiny cache-hit number.

To get *honest* TTFT numbers in repeated iterations, the official `llm_bench` mutates the prompt to a per-iteration value before each call, forcing the prefix cache to miss. For LLMs it mutates an early input token (preserving BOS when the tokenizer has one); for VLMs it mutates the prompt string. We do the same:

| Pipeline | How permutation works |
|---|---|
| `LLMPipeline` (qwen3) | Pre-tokenize, mutate token 0 if there is no BOS or token 1 if BOS exists, then pass `TokenizedInputs` to `generate()`. This matches upstream. |
| `VLMPipeline` (qwen3.5) | `VLMPipeline.generate()` takes a string prompt (no `TokenizedInputs`), so permutation prepends the per-iteration index to the prompt. This changes the first token and misses the KV prefix cache from position 0. Both our direct Qwen3.5 script and the official `visual_text_gen` path now do this. |

> **Patched upstream:** the official `visual_text_gen` path (`openvino.genai/tools/llm_bench/task/visual_language_generation.py`) originally ignored `--disable_prompt_permutation` entirely and always sent the same prompt string — so its TTFT was a cache-warm optimistic value. We patched it to permute by default (string-prefix mutation) and to honor `--disable_prompt_permutation`, matching the `text_gen` behavior. The prompt grows by one leading token per iteration.

**Default:** Both Qwen3 text and Qwen3.5 visual default to permutation ON, matching the (now-patched) official text_gen / visual_text_gen paths. Set `PERMUTE=0` / `--disable-prompt-permutation` (direct) or `--disable_prompt_permutation` (official) for a cache-warm scenario with stable, identical-prompt iterations.

This is separate from `--cache-dir`: `--cache-dir` controls OpenVINO's compiled-model cache and affects pipeline construction time, not TTFT inside measured `generate()` iterations. For generation latency, the important protection is prompt permutation. If you need a completely cold compile every process, pass `--cache-dir ""` or remove the cache directory before running.

For official comparability, the direct scripts mirror upstream OpenVINO config defaults: Qwen3 text uses `ATTENTION_BACKEND=PA`, Qwen3.5 visual/text uses `ATTENTION_BACKEND=SDPA`, and `CACHE_DIR=""`. Use `-lc/--load-config` to override these plugin properties intentionally.

The runner scripts now keep the same model-facing defaults as the direct scripts/official benchmark. Their extra default is resource collection: they pass `--monitor` for CPU/GPU/NPU utilization unless `HW_MONITOR=0` is set, and pass `--memory-monitor` unless `MEMORY_MONITOR=0` is set.

Empirical example (`Qwen3-0.6B-INT4`, 256-token prompt, 16-output, CPU):

| permutation | iter 1 TTFT | iter 2 TTFT |
|---|---|---|
| OFF (`--disable-prompt-permutation`) | 9.4 ms | 9.5 ms  ← prefix-cache hit |
| ON (default) | ~250 ms | ~250 ms ← honest prefill |

The VLM path is even more sensitive because GPU TTFT is small. Example (`Qwen3.5-2B-INT4`, 1000-token prompt, GPU, `ATTENTION_BACKEND=PA`, `visual_text_gen`):

| permutation | iter 1 TTFT | iter 2+ TTFT | effect on the mean |
|---|---|---|---|
| OFF (`--disable-prompt-permutation`) | ~191 ms (cold) | ~72 ms (cache-warm) | one cold outlier; `n=5` mean ≈ 95 ms vs `n=10` mean ≈ 84 ms — small-`n` swings |
| ON (default) | ~191 ms | ~191 ms (every iter cold) | honest cold prefill on every iteration; stable across `n` |

With permutation OFF, only the first counted iteration is cold (the single warm-up does not pre-warm the PA prefix cache), so its weight — and the reported mean — depends heavily on `-n`. Permutation ON removes that artifact: TTFT is consistent and `n=5` ≈ `n=10`.

---

## How to test input/output token pairs

You do **not** need a prompt file for common input/output token sweeps. Use `--input-tokens` for the input length and `-ic/--infer-count` for the requested output length. The input count is measured with the same tokenizer used by the benchmark and includes `--apply-chat-template` when that flag is enabled. For Qwen3.5, the default permutation sentinel is also included so the target matches the actual request sent to `generate()`.

`-ic` maps to `max_new_tokens`, so it is a maximum/requested output length, not a mathematical guarantee. With GenAI paths we set `ignore_eos=True`, so CPU/GPU text generation often reaches the requested count, but a backend can still stop early because of model/runtime limits. If you need a fixed-length best effort, add `--exact-output-tokens`; this also sets `min_new_tokens` to the same value when supported. The per-iteration JSON records actual generated tokens in both `output_tokens` and `infer_count`.

Example: 512-input / 256-output on Qwen3:

```bash
python tests/benchmark_qwen3.py \
    -m models/Qwen3-1.7B-INT4 -d CPU \
    --input-tokens 512 \
    -ic 256 -n 5 \
    --apply-chat-template \
    --exact-output-tokens
```

The runners expose the same control through env vars:

```bash
PROMPT_FILE= INPUT_TOKENS=512 INFER_COUNT=256 EXACT_OUTPUT_TOKENS=1 ./tests/runners/run_qwen3.sh CPU
PROMPT_FILE= INPUT_TOKENS=512 INFER_COUNT=256 EXACT_OUTPUT_TOKENS=1 MODE="text-only" ./tests/runners/run_qwen3_5.sh CPU
```

Use `--input-base-text path/to/source.txt` or `INPUT_BASE_TEXT=...` if you want the generated prompt content to come from your own corpus. The builder binary-searches a slice/repetition of that text until the tokenizer count lands near the target.

### Optional JSONL prompt files

The migrated runners default to `llama-2-7b-chat_l.jsonl` prompt index 0 as an approximate 1k-input text-generation workload. If you need a prompt targeted to an exact token count, build one with the same tokenizer the benchmark will use. `tests/build_prompt_jsonl.py` does that:

```bash
# Build a 1024-token prompt for any qwen3 / qwen3.5 model (uses ov_genai.Tokenizer).
python tests/build_prompt_jsonl.py \
    --model models/Qwen3-1.7B-INT4 \
    --tokens 1024 \
    --apply-chat-template \
    --out tests/prompts/qwen3_1k.jsonl
```

Pass `--apply-chat-template` here if the benchmark will also use `--apply-chat-template`; otherwise the JSONL file is targeted to the raw prompt length and the final benchmark `input_tokens` will be slightly larger after templating.

Then either:

```bash
# Option A: our wrapper (with HW monitoring + append CSV)
python tests/benchmark_qwen3.py \
    -m models/Qwen3-1.7B-INT4 -d CPU \
    --prompt-file tests/prompts/qwen3_1k.jsonl --prompt-index 0 \
    -ic 512 -n 5 \
    --apply-chat-template
# CSV  : tests/results/qwen3.csv (append mode)
# JSON : tests/results/qwen3.json (append mode)
```

```bash
# Option B: the official benchmark.py
cd openvino.genai/tools/llm_bench
python benchmark.py \
    -m /home/user/chaofan/model_test/benchmark/models/Qwen3-1.7B-INT4 \
    -d CPU --task text_gen \
    -pf /home/user/chaofan/model_test/benchmark/tests/prompts/qwen3_1k.jsonl \
    -pi 0 \
    -ic 512 \
    -n 5 \
    -r results.csv -rj results.json
```

Flag mapping (official ↔ ours):

| Official | Ours | Meaning |
|---|---|---|
| `-pf` | `--prompt-file` | JSONL with `{"prompt": "..."}` lines |
| `-pi 0` | `--prompt-index 0` | Run only prompt index 0 |
| `-ic 512` | `--infer-count 512` | `max_new_tokens` |
| `-n 5` | `--num-iters 5` | 5 measured iterations + 1 warm-up |
| `--task text_gen` | (implicit) | Our wrappers are task-specific |
| `--disable_prompt_permutation` | `--disable-prompt-permutation` | Same semantics |

---

## Runner cheat-sheet

All runners accept devices as positional args, override defaults via env vars, and support `--help`.

### `run_qwen3.sh`

```bash
./tests/runners/run_qwen3.sh [DEVICES...]

# Common overrides:
MODEL=models/Qwen3-1.7B-INT4 ./tests/runners/run_qwen3.sh CPU
MODELS="0.6B 1.7B" QUANTS="INT4" ITERS=10 ./tests/runners/run_qwen3.sh CPU GPU
PROMPT_FILE=tests/prompts/my_2k.jsonl INFER_COUNT=128 ./tests/runners/run_qwen3.sh CPU
PROMPT_FILE= INPUT_TOKENS=512 INFER_COUNT=256 ./tests/runners/run_qwen3.sh CPU
PROMPT_FILE= INPUT_TOKENS=512 INFER_COUNT=256 EXACT_OUTPUT_TOKENS=1 ./tests/runners/run_qwen3.sh CPU
LOAD_CONFIG=configs/gpu_perf.json ./tests/runners/run_qwen3.sh GPU
MEMORY_MONITOR=0 ./tests/runners/run_qwen3.sh CPU      # disable RSS/System memory sampling
HW_MONITOR=0 ./tests/runners/run_qwen3.sh CPU          # disable CPU/GPU/NPU utilization sampling
MEMORY_INTERVAL_MS=500 ./tests/runners/run_qwen3.sh CPU
PERMUTE=0 ./tests/runners/run_qwen3.sh CPU         # cache-warm scenario

# Defaults: DEVICE="CPU GPU NPU"  MODEL unset  MODELS unset => auto-discover models/Qwen3-* and models/qwen3-*
#           ITERS=10  INFER_COUNT=512  NPU_MAX_PROMPT_LEN=1024
#           EXACT_OUTPUT_TOKENS=0  APPLY_CHAT_TEMPLATE=0  PERMUTE=1  INTERVAL=0
#           PROMPT_FILE=tests/prompts/llama-2-7b-chat_l.jsonl  PROMPT_INDEX=0
#           Set PROMPT_FILE= to use INPUT_TOKENS instead.
#           CACHE_DIR=""  HW_MONITOR=1  MEMORY_MONITOR=1  MEMORY_INTERVAL_MS=100
#           LOAD_CONFIG unset; NPU_LOAD_CONFIG auto for NPU
```

### `run_qwen3_5.sh`

```bash
./tests/runners/run_qwen3_5.sh [DEVICES...]

# Common overrides:
MODEL=models/Qwen3.5-2B-INT4 ./tests/runners/run_qwen3_5.sh CPU
MODELS="0.8B 2B" QUANTS="INT4" MODE="text-only multimodal" IMAGE_PATH=path/to/photo.jpg ./tests/runners/run_qwen3_5.sh CPU
MODE="text-only" ./tests/runners/run_qwen3_5.sh GPU                # GenAI VLMPipeline on GPU
IMAGE_PATH=path/to/photo.jpg ./tests/runners/run_qwen3_5.sh CPU
PROMPT_FILE= INPUT_TOKENS=512 INFER_COUNT=256 MODE="text-only" ./tests/runners/run_qwen3_5.sh CPU
PROMPT_FILE= INPUT_TOKENS=512 INFER_COUNT=256 EXACT_OUTPUT_TOKENS=1 MODE="text-only" ./tests/runners/run_qwen3_5.sh CPU
OPTIMUM_FALLBACK=1 MODE="text-only" ./tests/runners/run_qwen3_5.sh GPU  # legacy fallback path
LOAD_CONFIG=configs/gpu_perf.json MODE="text-only" ./tests/runners/run_qwen3_5.sh GPU
PERMUTE=0 MODE="text-only" ./tests/runners/run_qwen3_5.sh GPU     # cache-warm scenario (permute is ON by default)
MEMORY_MONITOR=0 MODE="text-only" ./tests/runners/run_qwen3_5.sh CPU
HW_MONITOR=0 MODE="text-only" ./tests/runners/run_qwen3_5.sh CPU

# Defaults: DEVICE="CPU GPU NPU"  MODEL unset  MODELS unset => auto-discover models/Qwen3.5-*
#           MODE="text-only"  IMAGE_PATH=test_image.jpg
#           ITERS=10  INFER_COUNT=512  EXACT_OUTPUT_TOKENS=0  NPU_MAX_PROMPT_LEN=1024
#           PROMPT_FILE=tests/prompts/llama-2-7b-chat_l.jsonl  PROMPT_INDEX=0
#           Set PROMPT_FILE= to use INPUT_TOKENS instead.
#           APPLY_CHAT_TEMPLATE=0  PERMUTE=1  OPTIMUM_FALLBACK=0
#           CACHE_DIR=""  HW_MONITOR=1  MEMORY_MONITOR=1  MEMORY_INTERVAL_MS=100
#           LOAD_CONFIG unset; NPU_LOAD_CONFIG auto for NPU
```

### `run_lingua.sh`

```bash
./tests/runners/run_lingua.sh [DEVICES...]

# Common overrides:
MODELS="bert" QUANTS="INT4 INT8" ./tests/runners/run_lingua.sh CPU GPU
LOAD_CONFIG=configs/cpu_perf.json ./tests/runners/run_lingua.sh CPU
MEMORY_INTERVAL_MS=500 ./tests/runners/run_lingua.sh CPU

# Defaults: DEVICE="CPU GPU NPU"  MODELS="bert xlm-roberta"  QUANTS="INT4"
#           WARMUP=5  ITERS=500  INTERVAL=0  MEMORY_MONITOR=0  MEMORY_INTERVAL_MS=100
```

### `run_embedding.sh`

```bash
./tests/runners/run_embedding.sh [DEVICES...]

# Common overrides:
MODELS="models/Qwen3-Embedding-0.6B-int8-ov" MODE=both BATCH_SIZE=16 ./tests/runners/run_embedding.sh CPU GPU
INPUT_TOKENS=512 ./tests/runners/run_embedding.sh CPU
PROMPT_FILE=tests/prompts/embedding/texts.jsonl ./tests/runners/run_embedding.sh CPU
POOLING=mean NORMALIZE=0 ./tests/runners/run_embedding.sh CPU
MAX_LENGTH=512 PAD_TO_MAX_LENGTH=1 ./tests/runners/run_embedding.sh NPU   # fixed shape for NPU
NPU_LOAD_CONFIG='{"CACHE_DIR":"/tmp/ov_embedding_cache"}' ./tests/runners/run_embedding.sh NPU

# Defaults: MODELS auto-discovered (models/Qwen3-Embedding-*)  MODE=both  BATCH_SIZE=8
#           POOLING=last_token  NORMALIZE=1  PADDING_SIDE=left  MAX_LENGTH unset
#           ITERS=10  CACHE_DIR=""  NPU_LOAD_CONFIG auto for NPU
#           HW_MONITOR=1  MEMORY_MONITOR=1  MEMORY_INTERVAL_MS=100
```

### `run_reranker.sh`

```bash
./tests/runners/run_reranker.sh [DEVICES...]

# Common overrides:
MODELS="models/Qwen3-Reranker-0.6B-int8-ov" TOP_N=10 ./tests/runners/run_reranker.sh CPU GPU
NUM_DOCUMENTS=64 ./tests/runners/run_reranker.sh CPU             # candidate-set-size sweep
TEXTS_FILE=tests/prompts/reranker/texts_64.jsonl MAX_LENGTH=1024 ./tests/runners/run_reranker.sh CPU
NPU_LOAD_CONFIG='{"CACHE_DIR":"/tmp/ov_reranker_cache"}' ./tests/runners/run_reranker.sh NPU
PROMPT_FILE= QUERY="best laptop CPU" TEXTS_FILE=tests/prompts/reranker/texts_16.jsonl ./tests/runners/run_reranker.sh CPU

# Defaults: MODELS auto-discovered (models/Qwen3-Reranker-*)  TOP_N=3  NUM_DOCUMENTS unset
#           PROMPT_FILE=tests/prompts/reranker/queries.jsonl
#           TEXTS_FILE=tests/prompts/reranker/texts_16.jsonl  MAX_LENGTH=512
#           ITERS=10  CACHE_DIR=""  NPU_LOAD_CONFIG auto for NPU
#           HW_MONITOR=1  MEMORY_MONITOR=1  MEMORY_INTERVAL_MS=100
```

---

## CSV / JSON output

All writers are **append-mode**: re-running a benchmark that points at the same output file appends a new row (CSV) or appends new entries to the top-level array (JSON). The header is written exactly once. To start fresh, delete the file first.

### Generation CSV (qwen3 + qwen3.5) columns

```
model,framework,pipeline,device,precision,mode,
num_beams,batch_size,apply_chat_template,disable_prompt_permutation,
prompt_idx,input_tokens,infer_count_arg,exact_output_tokens,num_iters,measured_iters,compile_time_s,
output_tokens_{mean,std,p50,p90,p95,min,max},
generation_time_{mean,std,p50,p90,p95,min,max}_s,
ttft_{mean,std,p50,p90,p95,min,max}_ms,
tpot_{mean,std,p50,p90,p95,min,max}_ms,
first_token_infer_{mean,std,p50,p90,p95,min,max}_ms,
other_tokens_infer_{mean,std,p50,p90,p95,min,max}_ms,
tps_{mean,std,p50,p90,p95,min,max},
decode_tps_{mean,std,p50,p90,p95,min,max},
prefill_{mean,std,p50,p90,p95,min,max}_ms,
decode_{mean,std,p50,p90,p95,min,max}_ms,
e2e_{mean,std,p50,p90,p95,min,max}_ms,
tokenization_{mean,std,p50,p90,p95,min,max}_ms,
detokenization_{mean,std,p50,p90,p95,min,max}_ms,
initial_{rss,sys}_mem_mib, initial_{rss,sys}_mem_share,
compile_max_{rss,sys}_mem_mib, compile_max_{rss,sys}_mem_increase_mib,
compile_max_{rss,sys}_mem_share,
runtime_{mean,max}_{rss,sys}_mem_mib,
runtime_{mean,max}_{rss,sys}_mem_increase_mib,
runtime_max_{rss,sys}_mem_share,memory_samples,
cpu_avg_util,gpu_avg_util,npu_avg_util,
image_path,
openvino_version,openvino_genai_version,run_started_at
```

The same fields plus per-iteration history (`iterations[]`) are persisted in JSON.

### LLMLingua CSV columns

```
model,framework,device,precision,
input_tokens,compression_rate,iters,compile_time_s,
initial_{rss,sys}_mem_mib, initial_{rss,sys}_mem_share,
compile_max_{rss,sys}_mem_mib, compile_max_{rss,sys}_mem_increase_mib,
compile_max_{rss,sys}_mem_share,
runtime_{mean,max}_{rss,sys}_mem_mib,
runtime_{mean,max}_{rss,sys}_mem_increase_mib,
runtime_max_{rss,sys}_mem_share,memory_samples,
latency_{mean,std,p50,p90,p95,min,max}_ms,
throughput_samples_{mean,std,p50,p90,p95,min,max},
throughput_tokens_{mean,std,p50,p90,p95,min,max},
avg_compressed_tokens,avg_compression_ratio,
cpu_avg_util,gpu_avg_util,npu_avg_util,
openvino_version,run_started_at,
cpu_cpu0_util, cpu_cpu1_util, …            ← per-core CPU utilization
```

### Embedding CSV columns

```
model,framework,pipeline,device,precision,mode,
batch_size,pooling_type,normalize,max_length,padding_side,pad_to_max_length,
query_instruction,embed_instruction,
prompt_idx,input_tokens,total_input_tokens,embedding_dim,num_iters,measured_iters,compile_time_s,
latency_{mean,std,p50,p90,p95,min,max}_ms,
e2e_{mean,std,p50,p90,p95,min,max}_ms,
tokenization_{mean,std,p50,p90,p95,min,max}_ms,
throughput_docs_{mean,std,p50,p90,p95,min,max},
throughput_tokens_{mean,std,p50,p90,p95,min,max},
initial_/compile_max_/runtime_ memory fields (same shape as generation), memory_samples,
cpu_avg_util,gpu_avg_util,npu_avg_util,
cpu_cpu0_util, cpu_cpu1_util, …            ← per-core CPU utilization
openvino_version,openvino_genai_version,run_started_at
```

### Reranker CSV columns

```
model,framework,pipeline,device,precision,
num_documents,top_n,max_length,padding_side,pad_to_max_length,
prompt_idx,input_tokens,total_input_tokens,num_iters,measured_iters,compile_time_s,
latency_{mean,std,p50,p90,p95,min,max}_ms,
e2e_{mean,std,p50,p90,p95,min,max}_ms,
tokenization_{mean,std,p50,p90,p95,min,max}_ms,
throughput_docs_{mean,std,p50,p90,p95,min,max},
throughput_tokens_{mean,std,p50,p90,p95,min,max},
initial_/compile_max_/runtime_ memory fields (same shape as generation), memory_samples,
cpu_avg_util,gpu_avg_util,npu_avg_util,
cpu_cpu0_util, cpu_cpu1_util, …            ← per-core CPU utilization
openvino_version,openvino_genai_version,run_started_at
```

For embedding, `input_tokens` is the per-text token count and `total_input_tokens` is the batch sum; `throughput_docs` is texts/s and `throughput_tokens` is tokens/s. For reranking, `input_tokens` is the max (query+document) pair length and `total_input_tokens` sums all pairs; `throughput_docs` is candidate-documents scored per second. Per-iteration history (`iterations[]`, including `top1_index`/`top1_score` for rerank) is persisted in JSON.

---

## Metric definitions

The generation scripts report both official-aligned latency metrics and local resource/summary extensions. All summary statistics are computed over measured iterations only; warm-up iteration 0 is stored in JSON but excluded from aggregates.

| Metric | Meaning | Official alignment |
|---|---|---|
| **input_tokens** | Prompt token count used for the measured call. Qwen3 text is the encoded `TokenizedInputs` length after optional chat template and permutation. Qwen3.5 VLM uses `perf_metrics.get_num_input_tokens()` when available, because VLMPipeline may add internal template/control tokens. | Matches official text_gen / visual_text_gen input-size reporting. |
| **infer_count_arg** | The requested `-ic/--infer-count` value passed as `max_new_tokens`. | Matches official `-ic`; it is a maximum/requested value, not proof that exactly that many tokens were emitted. |
| **output_tokens** | Actual generated-token count from GenAI perf metrics or output token tensors/streamer fallback. | Corresponds to official `Output size`. Use this to confirm whether `-ic` was reached. |
| **TTFT** | Time to first token. Qwen3 text uses `perf_metrics.get_ttft().mean`, matching official text_gen. Qwen3.5 VLM uses official visual_text_gen's adjustment: `get_ttft().mean - tokenization_duration`. | Official-aligned; equivalent to official `First token latency` / `1st token latency`. |
| **TPOT** | Average decode cadence after the first token. Qwen3 text uses `perf_metrics.get_tpot().mean`. Qwen3.5 VLM uses deltas of `raw_metrics.m_new_token_times`, matching official visual_text_gen. | Official-aligned; equivalent to official `other tokens latency` / `2nd token latency`. |
| **Decode TPS** | `1000 / TPOT(ms)`. | Official-aligned with `2nd tokens throughput`. |
| **Throughput** | `output_tokens / E2E_seconds`, total generated-token throughput over the whole `generate()` call. | Local extension; useful but not the same as official `2nd tokens throughput`. |
| **Gen time / generation_time** | Wall-clock duration of one `generate()` call. It excludes prompt-file reading, outer loop overhead, and pipeline construction. | Same timing window as official `Generation Time`. |
| **Prefill** | Reported as TTFT. It is the observable prompt-processing + first-token latency proxy when no separate prefill-only metric is exposed. | Name is local; value aligns with TTFT/first-token latency. |
| **Decode** | `E2E - TTFT`, the wall-clock part after first token until generation returns. | Local convenience summary. |
| **E2E** | End-to-end wall-clock of the `generate()` call. In practice this is nearly the same window as Gen time, while keeping millisecond units and the identity `E2E = Prefill + Decode`. | Local convenience summary; official prints `Generation Time` and per-token latency instead. |
| **Tokenization** | Encode/tokenization time. Qwen3 text measures the explicit host tokenizer encode before `generate()`. Qwen3.5 VLM uses GenAI raw tokenization duration from perf metrics. | Qwen3.5 VLM matches official raw perf metric; Qwen3 text is local wrapper timing and should be read as host preprocessing cost. |
| **Detokenization** | Decode/detokenization time after generation. Qwen3 text measures explicit host detokenizer decode. Qwen3.5 VLM uses GenAI raw detokenization duration when exposed. | Qwen3.5 VLM matches official raw perf metric; Qwen3 text is local wrapper timing. |
| **1st infer** | Backend inference duration for the first generated token from `raw_metrics.token_infer_durations`. | Official-aligned with `First infer latency`. |
| **2nd infer / other infer** | Average backend inference duration for generated tokens after the first. This is lower-level runtime inference time and may differ from TPOT because TPOT includes scheduling/runtime overhead around token emission. | Official-aligned with `other infers latency`. |
| **compile_time_s** | Pipeline construction time. | Local extension; official logs pipeline initialization time but does not aggregate it in the same CSV schema. |
| **RSS memory** | Resident Set Size of the benchmark process, including child processes when `psutil` is available. | Local resource extension; official has memory collection only when `-mc` is enabled and does not provide identical CSV fields. |
| **System memory** | Host memory used: total physical memory minus available memory. This can be affected by other processes. | Local resource extension. |
| **cpu_avg_util / gpu_avg_util / npu_avg_util** | Average utilization sampled by the runner-side monitors during the benchmark process. Official extensions use `resource_cpu_avg_util(%)`, per-core `resource_cpu_cpuN_avg_util(%)`, `resource_gpu_avg_util(%)`, `resource_gpu_memory_avg_util(%)`, and `resource_npu_avg_util(%)`. GPU/NPU availability depends on local tooling such as `xpu-smi` and sudo-readable sysfs. | Local resource extension; this workspace's official `benchmark.py` has the same compact resource extension under `resource_*` fields. |
| **result_md5** | MD5 of generated text stored per iteration in JSON. Helpful for checking determinism when permutation is disabled. | Official also logs generated result MD5. |

Statistics summary fields are mean / std (population) / p50 / p90 / p95 / min / max, computed over **measured iterations only**. This matches the official average policy: warm-up iter 0 is excluded.

### Embedding & reranker metrics

These encoders have no decode loop, so the table above does not apply. They report:

| Metric | Meaning |
|---|---|
| **latency** | Pure pipeline call time (`embed_documents` / `embed_query` / `rerank`), timed with `time.perf_counter()`. This is the headline number — no `perf_metrics` is exposed by these pipelines. |
| **e2e** | `latency + tokenization`. Host token accounting is kept out of the latency window but added here so e2e reflects a realistic end-to-end cost. |
| **tokenization** | Host-side `ov_genai.Tokenizer` encode time, used only for the tokens/s denominator. |
| **throughput (docs)** | Embedding: texts encoded per second (`batch_size / latency`). Reranking: candidate documents scored per second (`num_documents / latency`). |
| **throughput (tokens)** | Total input tokens processed per second over the measured call. |
| **embedding_dim** | Output vector dimension (embedding only). |
| **top1_index / top1_score** | Reranker's highest-scoring document index and score, stored per iteration for determinism checks. |

Like the generation benchmarks, all aggregates exclude warm-up iteration 0, and memory / CPU / GPU / NPU utilization are recorded identically.

---

## Models & directory naming

We auto-discover models from `models/` based on these conventions:

| Family | Pattern | Example |
|---|---|---|
| Qwen3 LLM | `Qwen3-<size>-<QUANT>` and legacy `qwen3-*` | `Qwen3-0.6B-INT4`, `qwen3-0.6b-int4-ov` |
| Qwen3.5 VLM | `Qwen3.5-<size>-<QUANT>` | `Qwen3.5-0.8B-INT4`, `Qwen3.5-2B-INT4`, `Qwen3.5-4B-INT4` |
| LLMLingua-2 | `llmlingua-2-<arch>-meetingbank-<QUANT>` | `llmlingua-2-bert-base-multilingual-cased-meetingbank-INT4` |
| Qwen3-Embedding | `Qwen3-Embedding-*` | `Qwen3-Embedding-0.6B-int8-ov`, `Qwen3-Embedding-0.6B-int4-cw-ov` |
| Qwen3-Reranker | `Qwen3-Reranker-*` | `Qwen3-Reranker-0.6B-int8-ov`, `Qwen3-Reranker-0.6B-fp16-ov` |

The Qwen3/Qwen3.5/LLMLingua/embedding/reranker runners auto-discover matching directories under `models/` by default. Qwen3/Qwen3.5 still accept `MODELS`/`QUANTS` for size-based filtering, and every runner accepts explicit model paths via `MODEL="models/<dir> ..."` where supported.

Each directory must contain at minimum:
- `openvino_model.xml` + `openvino_model.bin` (or `openvino_language_model.{xml,bin}` for Qwen3.5 VLM)
- `openvino_tokenizer.xml` / `openvino_detokenizer.xml`

Use `model_download.py` (at the project root) to fetch / convert new models.

---

## Known issues & workarounds

### Qwen3.5 VLM on Panther Lake iGPU

On the current PTL stack in this workspace, `ov_genai.VLMPipeline.generate()` works on GPU for Qwen3.5. The Qwen3.5 benchmark therefore uses GenAI `VLMPipeline` by default on GPU, matching CPU/NPU behavior and avoiding the older automatic optimum.intel fallback path.

The old fallback is still available for debugging regressions:

```bash
python tests/benchmark_qwen3_5.py \
    -m models/Qwen3.5-2B-INT4 -d GPU \
    --input-tokens 512 -ic 256 -n 5 \
    --optimum-fallback
```

For runner usage, set `OPTIMUM_FALLBACK=1`.

### LLMLingua-2 on NPU

The LLMLingua-2 OV models are exported with dynamic input shapes (`[?,?]`), which the NPU compiler rejects. `benchmark_lingua.py` now handles NPU by **reshaping the model to a static `[1, input_len+2]` and (re)compiling per input length** inside the sweep (CPU/GPU keep the single dynamic compile). Inputs are right-padded to that length and the `attention_mask` drives the real-token compression math, so NPU compression results match CPU/GPU exactly. `DEVICE=NPU QUANTS=INT4 ITERS=500 ./tests/runners/run_lingua.sh` works.

Measured on Core Ultra X7 358H (INT4):
- **BERT-base** (`llmlingua-2-bert-...`): works on NPU at all sweep lengths (64/128/256/512). Compile ≤2.7s per shape; inference ~3–17 ms.
- **XLM-RoBERTa-large** (`llmlingua-2-xlm-roberta-large-...`): works at input lengths ≤ 258, but the `[1,512]` shape **hangs the NPU device** (`ZE_RESULT_ERROR_DEVICE_LOST`). The script catches this, records the working configs, and skips the remaining rates for that length. If you only need NPU numbers for XLM, restrict the sweep, e.g. `--input-lengths 64 128 256` (or set it in the runner). The 512-length XLM-large NPU hang appears to be an NPU-plugin/driver limitation for this larger model.

### XLM-RoBERTa missing token_type_ids

XLM-RoBERTa does not have a `token_type_ids` input port (BERT does). `benchmark_lingua.py` introspects the compiled model's input names and only feeds `token_type_ids` when present — should "just work" for both architectures.

### Qwen3-Embedding / Qwen3-Reranker on NPU

The Qwen3 encoder IRs are dynamic-shape. NPU requires static shapes, so a plain `--device NPU` run may fail to compile. Fix the shape with `--pad-to-max-length --max-length <N>` (e.g. `--max-length 512`). CPU and GPU work without any extra flags. The reranker default `max_length` is large (8192 for Qwen3), so pin a smaller `--max-length` on NPU to keep compile time and memory reasonable. The embedding/reranker runners also provide NPU-only `NPU_LOAD_CONFIG` defaults that set `CACHE_DIR`; this controls compiled-model caching and does not replace the fixed-shape settings.

#### Official reranker on NPU: static reshape (`NPU_STATIC_RESHAPE=1`)

The official GenAI `TextRerankPipeline` reads the dynamic IR directly and the NPU compiler rejects it (`Got negative shape dim bound: '-1'`). The reranker scores **all candidate documents for a query in one forward pass**, so the model's batch dim equals the candidate count — a static model is valid only for that exact `(num_candidates, max_length)` pair.

`run_official_reranker.sh` supports `NPU_STATIC_RESHAPE=1`: it computes the candidate count (line count of `TEXTS_FILE`, or 2 for the built-in default texts), calls [`reshape_reranker_static.py`](reshape_reranker_static.py) to write a static `[num_candidates, max_length]` copy under `models/_reranker_npu_static/`, and runs it. `RERANKING_MAX_LENGTH` becomes the static sequence length and is required. On NPU, `ov_utils.create_genai_text_reranker_model` forces `pad_to_max_length=True` so every inference feeds exactly `[num_candidates, max_length]` (CPU/GPU keep dynamic behavior). The static model is cached and reused across runs with the same shape.

```bash
# Built-in default 2 candidates, NPU, static reshape at max_length 512 (the practical NPU tier)
NPU_STATIC_RESHAPE=1 RERANKING_MAX_LENGTH=512 TEXTS_FILE="" PROMPT_FILE="" PROMPT_INDEX="" \
  MODEL=models/Qwen3-Reranker-0.6B-fp16-ov DEVICE=NPU ITERS=200 ./tests/runners/run_official_reranker.sh NPU
```

**NPU is a poor fit for this workload — measured on Core Ultra X7 358H (Qwen3-Reranker-0.6B-fp16):**

| shape `[candidates × max_length]` | NPU compile | per-query latency | viable? |
|---|---|---|---|
| `[2 × 256]` | 64 s | 306 ms | yes |
| `[2 × 512]` | 132 s | 208 ms | yes |
| `[1 × 2048]` | 462 s | 928 ms | slow |
| `[2 × 2048]` | did not finish in 15 min | — | no |
| `[15 × 512]` (texts_16) | did not finish in ~16 min, ~26 GB RAM | — | no |
| `[63 × 512]` (texts_64) | OOM-killed | — | no |

So `RERANKING_MAX_LENGTH=2048` and the `texts_16`/`texts_64` candidate-count workloads are **not feasible on NPU** (compile-time/RAM explosion). For those, use GPU (≈1 ms init, sub-ms scoring) or CPU. Only small candidate counts (a few) with `RERANKING_MAX_LENGTH<=512` are practical on NPU, and even then per-query latency (~200–300 ms) is far worse than GPU.

Note: `Qwen3-Reranker-0.6B-INT4` is exported as a `seq-cls` head (output `[?,2]`, inputs lack `position_ids`) and fails GenAI's Qwen3 postprocessing after static reshape (`is_axis_valid` error). Use the `fp16-ov` / `int8-ov` exports (output `[?,?,vocab]`) for the NPU static path.

### LLMLingua-2 runner picked the wrong Python (`No module named 'numpy'`)

`run_lingua.sh` (and the direct `run_qwen3.sh` / `run_qwen3_5.sh`) used to fall back to whatever `python`/`python3` is on `PATH` when `PYTHON` was unset, which on this box is `/usr/bin/python3` (no numpy/openvino). They now default `PYTHON=/home/user/envs/model_test_env/bin/python` (matching the `run_official_*` runners) when it exists, so `DEVICE=GPU QUANTS=INT4 ITERS=500 ./tests/runners/run_lingua.sh` works out of the box. Override with `PYTHON=/path/to/python` if you use a different env. LLMLingua-2 on NPU remains unsupported (dynamic-shape IR — see below).

### Official `--benchmark_csv` for text_embed / text_rerank

The custom benchmark-summary code (`llm_bench_utils/output_json.py`) was written for generation tasks and assumed a numeric 2nd-token latency. Embedding/reranking have none, so the averaged value is `""` and the summary writer raised `TypeError: '>' not supported between instances of 'str' and 'int'`. `estimate_throughput` now coerces with `float(...)` inside a `try`, so `--benchmark_csv`/`--benchmark_json` summaries write cleanly. The per-iteration `-r`/`-rj` reports were never affected.

---

## Adding a new benchmark

1. Add per-iteration & per-run dataclasses to [`lib/metrics.py`](lib/metrics.py) (or reuse `GenIterationMetrics` / `GenBenchmarkResult` for generation, or `EmbedIterationMetrics` / `RerankIterationMetrics` for single-pass encoders).
2. Add CSV/JSON writers and a console summary printer to [`lib/reports.py`](lib/reports.py). Encoder-style results can reuse `_write_csv_append`, `_stat_cells`, `_memory_cells`, `_util_cells`, and `fill_encoder_stats`.
3. Write `tests/benchmark_<name>.py` consuming `lib.cli`, `lib.monitors.MonitorBundle`, `lib.prompts.load_prompts`, and either `lib.genai_runner.run_llm_iteration` / `run_vlm_iteration` (LLM/VLM), `lib.embed_runner.run_embed_iteration` / `run_rerank_iteration` (encoders), or your own per-iter helper.
4. Add a runner shell at `tests/runners/run_<name>.sh` that sources `_common.sh` for `bench_run_with_log`, `bench_make_log_dir`, `bench_write_manifest`, `bench_parse_devices`.
5. Document the new benchmark in this README.

---

## Tips

- **Re-running** a benchmark with the same `--csv-out`/`--json-out` path appends rows. Delete the file to start clean.
- **Different machines**: the per-core CPU columns in `lingua.csv` will differ in count between machines; downstream tooling should treat unknown columns as `0`.
- **Cache directory**: direct and runner scripts default `--cache-dir ""` to match official `llm_bench`. Set `CACHE_DIR=tests/cache_qwen3` or another path only when you intentionally want OpenVINO compiled-model cache. This is not the KV prefix cache that affects TTFT; prompt permutation handles that.
- **Python executable**: runners honor `PYTHON=/path/to/python`; otherwise they try `python` and then `python3`.
- **Reproducibility**: every runner writes a `run_info.txt` manifest with python / openvino / openvino_genai versions and all parameter values.
