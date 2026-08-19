# Qwen3.5-2B + PagedAttention — TTFT optimization, round 4

Continuation of the round-3 work. Starting point: the `opt-3 + opt-4 + opt-5 + opt-6` plugin
stack from `benchmark/openvino/qwen3.5_patch/`. Goal: lower TTFT further.

Outcome: **one new optimization (opt-7), −11.3 ms infer TTFT (−4.4%)**, shipped as
`benchmark/openvino/qwen3.5_patch/opt7_fuse_rms_into_dynamic_quantize.patch`.

| Build | infer TTFT p50 | pipeline TTFT p50 | e2e TTFT p50 |
|---|---:|---:|---:|
| opt-3+4+5+6 (paired base) | 260.31 ms | 265.89 ms | 268.13 ms |
| **+ opt-7** | **248.97 ms** | **254.60 ms** | **256.77 ms** |
| **opt-7 increment** | **−11.34 ms (−4.36%)** | **−11.29 ms (−4.24%)** | **−11.36 ms (−4.24%)** |

All numbers here were measured on **this** machine (Core Ultra 7 356H, 32 EU,
memory-frequency-limited: nothing exceeds ~48 GB/s measured DRAM bandwidth). Numbers in
older round-1..3 documents came from different parts and are not comparable — nothing in
this round was tuned against them.

## Setup

```
benchmark/openvino @ qwen-ttft-optimizations, clean src + opt3 + opt4 + opt5 + opt6
cmake --build build --target openvino_intel_gpu_plugin -j $(nproc)
model_bench vlm -m benchmark/models/Qwen3.5-2B-INT4 -d GPU -pf model_bench/prompts/qwen3.5_1k.jsonl
  -lc '{"PERFORMANCE_HINT":"LATENCY","ATTENTION_BACKEND":"PA"}'   # 1002-token prompt
env OV_SWISH_HOIST_ALL=1                                          # opt-6's opt-in FC-side fusion
```

`ATTENTION_BACKEND` is a **GenAI pipeline** property. A raw `ov::Core::compile_model`
rejects it, which is why `model_bench --profile` (the deep-profile path) cannot be used to
study the PA graph — it compiles its own model directly. Everything PA-specific here was
measured through the normal pipeline path or with VTune.

The round-start paired-base measurement was **259.31 ms infer / 265.03 ms pipeline TTFT**
(`results/base_r4.json`), consistent with the 259.19 ms recorded for this stack previously.
The fresh opt-7 A/B rerun below independently re-measured its paired base at 260.31 ms
infer / 265.89 ms pipeline TTFT / 268.13 ms e2e TTFT.

For the fresh per-stage rerun below, `infer TTFT` means `first_token_infer_p50_ms`,
`pipeline TTFT` means `ttft_p50_ms`, and `e2e TTFT` means `e2e_p50_ms` from the
structured `model_bench` result. The stage run used `n=5`, `ic=1`, three interleaved
rounds; the opt-7 paired run used four interleaved rounds. Each cell is first the p50 of
the five measured iterations in one process, then the median across interleaved rounds.

## Where the time actually goes

Round 3 ended with the easy structural wins taken: no single kernel is badly dispatched
any more. So round 4 started by asking whether the remaining time is *compute* or *memory*.

### 1. Price every real GEMM shape standalone (`onednn/`)

`onednn/qwen35_gemm_inventory.cpp` enumerates every GEMM the decoder actually issues at a
1002-token prefill (shape, weight quantization, post-ops), and
`onednn/qwen35_gemm_replay.cpp` replays them back-to-back through oneDNN with nothing else
running — a pure compute floor for the exact work the model does.

```
--- count-weighted decoder total (one prefill pass, M=1002) ---
case               predicted ms          GFLOP  eff TFLOP/s
perOC_s4_cab               77.9         2750.9         35.3
```

Ablations (`results/replay_ablation.txt`) confirm the floor is insensitive to the graph-level
details: sharing quantized activations across the q/k/v/z group and dropping post-ops move it
only 78.2 → 81.0 ms.

**In-model, the same GEMMs take ~129.6 ms.** So roughly **52 ms — 40% of decoder GEMM time —
is not compute.**

### 2. Account for the gap: it is activation traffic

At a 1002-token prefill the decoder moves ~2.11 GB of activation tensors. At the ~48 GB/s
this part actually sustains that is **~44.0 ms**, which covers essentially the whole gap.
The GEMMs are not slow; they are waiting on activation bytes that the weight-streaming does
not hide.

That fixes the optimization metric for the rest of the round:

> **The unit of progress is a memory pass eliminated, not a kernel made faster.**

Anything already running at the DRAM ceiling cannot be sped up — only removed. (This is the
same conclusion opt-6 reached for the SiLU passes, arrived at independently.)

### 3. Per-kernel profile → find the removable passes (`results/vtune_opt3456.csv`)

VTune GPU Compute/Media Hotspots on the opt-3+4+5+6 build:

| Kernel | share of GPU busy | what it does |
|---|---:|---|
| `dynamic_quantize_gpu_opt` | 9.2% | read f16 activation → write s8 + scales |
| `rms_gpu_bfyx_opt` | 7.3% | read f16 activation → write f16 activation |

16.5% of GPU time in two kernels that do almost no arithmetic. And they are **adjacent**:
in Qwen3.5 every `input_layernorm` / `post_attention_layernorm` feeds directly into the
DynamicQuantize in front of the following FullyConnected group.

## opt-7: fold RMSNorm into DynamicQuantize

Per `[T, H]` f16 activation the pair moves:

```
rms:  read  T*H*2 (x)           + write T*H*2 (normalized f16)
dq:   read  T*H*2 (normalized)  + write T*H   (s8) + scales
```

The middle f16 tensor is written once and read straight back — it exists purely to hand a
value between two kernels. Folding the normalization into the quantizer removes both
passes: `2 * T*H*2` bytes per site.

24 layers × 2 layernorms = 48 candidate sites; **47 fold**. The one that does not is layer
0's `input_layernorm`, whose normalized tensor has a second consumer, so its `rms` has to
stay. At `1002 × 2048` f16 that is **~385 MB removed** ≈ 8 ms at 48 GB/s.

### Implementation

- `prepare_primitive_fusing::fuse_rms_into_dynamic_quantize` — new pass, runs last in
  `prepare_primitive_fusing`. Sets `fused_rms` / `rms_eps` on the `dynamic_quantize`
  primitive and rewires its inputs to `{x, gamma}`.
- `dynamic_quantize_gpu_opt.cl` — new `FUSE_RMS` path inside `MODE_LARGE_GS`: block-read x,
  reduce, normalize, apply gamma, round to f16, quantize, block-write s8.
- `DynamicQuantizeKernelOpt::Validate` gains a `fused_rms` precondition block; the **ref**
  and **kv-cache** kernels explicitly *refuse* a folded node.
- `dynamic_quantize_inst::should_skip_execution` and `primitive_inst::update_shape` refuse
  to optimize away a folded node.

### Bit-exactness was designed in, not hoped for

A different summation order changes the last f16 bit and would defeat the md5 gate, so the
fused kernel reproduces `rms_gpu_bfyx_opt`'s reduction *structure* element-for-element:

| `rms_gpu_bfyx_opt` | fused kernel |
|---|---|
| `intel_sub_group_block_read_us8` → lane `l` of sub-group `s` owns `{128s + 16j + l}` | identical block read, identical ownership |
| f32 accumulate via `native_powr(x, 2)` | same |
| `sub_group_reduce_add`, then SLM halving tree over `rms_slm[local_id]` driven by `local_id*SIMD + sglid` | same slots, same tree |
| `native_powr(sqrt(mean_sq + eps), -1)` | same |
| writes f16 → dq reads that f16 | rounds to f16 **before** quantizing — same value dq used to read |

Verified: prefill md5 is identical between the two builds.

### The one place it is not bit-exact: decode

`dynamic_quantize` normally turns itself into an f16 pass-through at batch 1
(`should_skip_execution`). A folded node cannot: its input is the *un-normalized*
activation, so aliasing output to input would silently drop the normalization. Serving both
forms from one node would need the output dtype to flip i8↔f16 at runtime, which one
shape-agnostic compiled kernel cannot do.

Decision: **folded nodes never skip.** The 47 folded sites therefore quantize at batch 1
during decode where they previously stayed f16. This is the *same* quantization scheme the
model already applies to every prefill token (per-token symmetric i8, 128-element groups),
and the batch-1 skip exists as a performance heuristic, not an accuracy guarantee.
Measured cost: **none** — TPOT p50 improved 32.601 → 32.389 ms.

### Alternatives considered and rejected

| Idea | Why not |
|---|---|
| Make either kernel faster | Both already at the DRAM ceiling. No headroom to find. |
| Fuse RMSNorm into the FullyConnected instead | The FC reads the **s8** tensor, which is the one tensor in the chain that must exist anyway. Nothing removed. |
| Keep a second RMS-only kernel variant so decode can still skip | Needs a per-shape impl whose output dtype flips i8↔f16, plus a second CL variant, for a decode path that measures *faster* without it. Complexity with negative expected value. |
| Keep `rms` in the graph as a third DQ input for the skip path | `rms` would then always execute, which is exactly the pass we set out to remove. |

### A bug this surfaced: fusing must mirror the kernel's preconditions

The folded form is implemented by exactly one kernel, and the fallbacks refuse it by
design — so an over-eager pass turns into a **compile failure**, not wrong numerics. That
fired during development:

```
Could not find a suitable kernel for dynamicquantize:DynamicQuantize_274568
  params raw string: F16_BFYX_v1_..._v2048_..._v988_...;F16_BFYX_...;INT8_BFYX_...
```

Cause: the plugin defaults `DYNAMIC_QUANTIZATION_GROUP_SIZE` to per-token
(`UINT64_MAX`) on systolic-capable hardware when no property is set — as happens on the raw
`compile_model` path that `--profile` uses. The kernel's fused branch only implements
`MODE_LARGE_GS` with a 128-element group, so `Validate` rejected it and nothing else would
take it.

Fix: the pass now checks, before fusing, everything the kernel checks that cannot change
afterwards — symmetric i8, last-axis group size exactly 128 (all other axes 1), identity
scale order, no precomputed reduction, static reduction extent that is a multiple of 128
and ≤ 32 groups, f16 in/out, gamma covering the axis exactly. Re-verified: the exact
`--profile` invocation that failed now compiles and runs (`rc=0`, no exception), on both
this build and the baseline.

## Measurements

### Interleaved A/B (`run_opt7_ab.sh`, `results/opt7_ab_rerun_20260819/`)

Two plugin binaries from the same tree, hot-swapped alternately within one session. This
part drifts thermally by about 1 ms per run, which is a sizeable fraction of the increment,
so one-pass-per-stage ordering would bias whichever stage runs later. The fresh rerun was
performed on 2026-08-19 with `n=5`, `ic=1`, four reps; each cell is that run's p50 over five
measured iterations.
The paired opt-7 binaries report OpenVINO plugin version
`2026.3.0-22453-32f59cd8191-qwen-ttft-optimizations`; the GenAI version is the same as in
the stage run.

| Rep | base infer | opt-7 infer | Δ | base pipeline TTFT | opt-7 pipeline TTFT | Δ | base e2e TTFT | opt-7 e2e TTFT | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 258.16 | 248.34 | −9.82 | 263.86 | 254.02 | −9.84 | 266.17 | 256.34 | −9.83 |
| 2 | 260.29 | 249.22 | −11.07 | 265.84 | 254.78 | −11.05 | 268.10 | 256.97 | −11.13 |
| 3 | 260.65 | 249.21 | −11.44 | 266.41 | 254.91 | −11.51 | 268.62 | 257.05 | −11.57 |
| 4 | 260.34 | 248.73 | −11.61 | 265.93 | 254.42 | −11.52 | 268.15 | 256.57 | −11.58 |
| **median** | **260.31** | **248.97** | **−11.34 (−4.36%)** | **265.89** | **254.60** | **−11.29 (−4.24%)** | **268.13** | **256.77** | **−11.36 (−4.24%)** |

Every paired rep lands in roughly −9.8…−11.6 ms; the effect is far larger than the run-to-run
spread. **Predicted ~8 ms from bytes alone, measured 11.3 ms** — removing a kernel also
removes its dispatch and its cache pollution, so the byte-count estimate is a floor.

### Correctness (`results/correctness/`)

| Check | opt3456 | opt3456+opt7 |
|---|---|---|
| `ic=1` (prefill only), 4 iterations | `dd7536794b63bf90eccfd37f9b147d7f` ×4 | **same** ×4 |
| `ic=128` greedy decode, 4 iterations | 4 md5s | **all differ** |
| TPOT p50 @ `ic=128` | 32.601 ms | 32.389 ms |

Prefill is bit-identical, as designed. Decode differs for the understood reason above, and
does not cost time. (The four md5s differ from each other *within* a build too — prompt
permutation is on, so each iteration sees a permuted prompt; the comparison is per-position
across builds.)

## Files

```
run_opt7_ab.sh                    interleaved A/B driver
onednn/                           GEMM inventory + replay probe (the compute-floor experiment)
results/base_r4.{json,csv}        round-start baseline
results/probe_model_inventory.txt per-shape GEMM prices, count-weighted decoder total
results/replay_ablation.txt       share-activations / no-post-ops ablations
results/vtune_opt3456.csv         VTune GPU Compute Hotspots on the round-3 stack
results/so_pa_opt3456.so          baseline plugin binary (A)
results/so_pa_opt3456_opt7.so     opt-7 plugin binary (B)
results/opt7_ab/                  Previous opt-7 A/B raw results
../qwen35-pa-ttft-round3/results/stage_ab/
                                 Fresh opt-3…opt-6 stage results used by the companion table
results/correctness/              md5 / TPOT runs
```

The shipped patch and its self-contained write-up live in
`benchmark/openvino/qwen3.5_patch/`.

## Where the remaining time is

After opt-7, the ~44 ms of unhidden activation traffic is down by roughly a quarter. What
is left, in rough order of size:

- The **s8 activation tensors themselves** and the f16 GEMM outputs — these are real
  hand-offs into oneDNN and cannot be removed without fusing quantization into the GEMM
  prologue, which the oneDNN path does not currently expose.
- The **Gated-DeltaNet state tensors** — large, and touched by several kernels in sequence.
- `gemm_kernel` at 77.9 ms is the genuine compute floor and is 35.3 TFLOP/s effective,
  i.e. already close to what this part can do.

The same method applies to the next round: find a tensor that is written once and read once
purely to cross a kernel boundary, and remove the boundary.
