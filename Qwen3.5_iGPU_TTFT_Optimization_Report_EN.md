# Reducing Qwen3.5 Prefill TTFT on Intel iGPU by Optimizing OpenVINO Source Code

**Backends covered: SDPA and PagedAttention (PA).**
**Platforms: Intel Core Ultra X7 358H (Arc B390 iGPU) and Intel Core Ultra 7 356H (32-XVE iGPU), both Panther Lake, Ubuntu 24.04.**
**Software: locally source-built OpenVINO 2026.2 + OpenVINO GenAI 2026.2 (Release), VTune 2026.2.**
**Benchmark workload: real 1000-token input prompt, 512 output tokens (`in=1002, out=512`), greedy decoding.**

> **Scope of this report.** All optimizations here are in the **OpenVINO GPU plugin** source code. We focus on the *general, portable* optimizations that are safe to ship. One optimization we tried, `opt-4` (fully token-parallel conv1d), speeds up the large iGPU (358H) but slows down the small iGPU (356H), so it is **not** a recommended optimization — we only mention it briefly to explain why its safe replacement, `opt-5`, was designed. Every optimization we do recommend is **numerically lossless — the generated text is byte-for-byte identical to the baseline.**

---

## 1. Executive Summary

### 1.1 What we did

Our goal was to reduce the Time-To-First-Token (TTFT), i.e. the prefill latency, of the Qwen3.5 hybrid/SSM models running INT4 on Intel integrated GPUs. We profiled the real workload, found where GPU time is wasted, and patched the OpenVINO GPU plugin source to remove that waste. We did **not** touch the INT4 matrix multiply itself — it is already optimal on the XMX hardware.

OpenVINO has two attention backends. **SDPA** was the older path; **PagedAttention (PA)** is now the default. We optimized both, because the two paths have very different hotspots.

### 1.2 The headline results

**PA is already much faster than SDPA — this is the single biggest win.** On the same model and hardware, just switching from SDPA to PA reduces the first-token latency by 34% to 51%. So the first recommendation is simple: use PA (which is already the default).

On top of PA, our GPU-plugin optimizations give the following additional reduction on the 356H iGPU (our primary test machine, real 1000-token input / 512 output, PA):

| Model | Pure-inference TTFT | End-to-end TTFT | Decode throughput (TPS) |
|---|---:|---:|---:|
| Qwen3.5-0.8B-INT4 | **−10.0%** (153.1 → 137.8 ms) | **−7.6%** (204.5 → 189.0 ms) | unchanged (~110 tok/s) |
| Qwen3.5-2B-INT4 | **−7.9%** (219.8 → 202.4 ms) | **−6.7%** (271.2 → 253.1 ms) | unchanged (~69 tok/s) |
| Qwen3.5-4B-INT4 | **−4.8%** (520.7 → 495.8 ms) | **−4.5%** (575.8 → 549.8 ms) | unchanged (~35 tok/s) |

Under the older SDPA backend, the same style of work produced even larger TTFT reductions (because SDPA had more waste to remove): **−10% to −21% on pure-inference TTFT**.

> **Important scope note on throughput.** All of these optimizations target the **prefill (TTFT)**. They do **not** change the decode speed. So the per-token decode time (TPOT) and the tokens-per-second throughput (TPS) stay the same before and after — as expected. We improve *how fast the first token appears*, not the streaming speed after that.

### 1.3 The general optimizations (all bit-exact, all in the OpenVINO GPU plugin)

| # | Optimization | Backend | What it removes |
|---|---|---|---|
| **opt-1** | Keep depthwise conv in planar layout | SDPA | Redundant layout conversions (Reorder) around the conv |
| **opt-2** | Fix the transpose kernel's GPU work-group dispatch | SDPA | A dispatch defect that made the transpose up to 10× slower |
| **opt-3** | Reuse one DynamicQuantize across FCs that share an input | SDPA + PA | Repeated activation quantization (2–4 times per layer) |
| **opt-5** | Token-tiled conv1d | PA | Low GPU occupancy on the causal conv kernel |
| **opt-6** | Route dynamic-shape activation to the vectorized kernel | PA | Slow scalar element-wise activation |

opt-3 is the one that spans both backends, and it is completely general — any transformer where several fully-connected layers share the same input (almost every QKV or gate/up block) benefits from it.

### 1.4 Correctness

Every recommended optimization is verified numerically lossless. We use greedy (deterministic) decoding and hash the generated text with MD5. When the model generates 128 tokens greedily and the hash is byte-for-byte identical to the baseline, it proves that no step's numerics changed enough to flip a single token choice. All recommended optimizations pass this test on all three model sizes, for both prefill and decode.

---

## 2. Test Platform and Methodology

### 2.1 Hardware and software

| Item | 358H | 356H |
|---|---|---|
| CPU | Intel Core Ultra X7 358H (Panther Lake) | Intel Core Ultra 7 356H (Panther Lake) |
| iGPU | Arc B390 (larger) | 32 XVE = 320 HW threads (smaller, ~2× slower) |
| OS | Ubuntu 24.04 | Ubuntu 24.04 |
| OpenVINO / GenAI | 2026.2 source build (Release) | same HEAD |
| Profiler | VTune 2026.2 | VTune 2026.2 |

The two machines are both Panther Lake, but the 356H iGPU is smaller, so its absolute latencies are roughly twice the 358H. This difference is important: an optimization that helps a large iGPU does not automatically help a small one, which is exactly what happened with opt-4.

### 2.2 Benchmark configuration

Unless stated otherwise, every benchmark uses:
- **Real 1000-token input prompt** (`in=1002` after the chat template), **512 output tokens** (`out=512`).
- The C++ tool `model_bench`, linked against the locally built OpenVINO (not the system `/opt/intel` copy).
- Greedy decoding (`do_sample=false`) for deterministic, verifiable output.

Two measurement styles appear in this report:
- **Full 1k/512 runs** (`ic=512`, n=5): report TTFT plus decode throughput (TPS, TPOT). This is the user-facing benchmark.
- **Pure-prefill runs** (`ic=1`, n=8): generate only the first token, to isolate the prefill and measure each single optimization's increment cleanly without thermal drift from a long decode.

The edit → rebuild → measure loop is fast: rebuilding only the GPU plugin takes about 9 seconds. The one trap: after rebuilding, the new `.so` must be copied into the `install/` directory, because that is the copy the benchmark actually loads.

### 2.3 The metrics — keep them separate

| Metric | Field | Meaning |
|---|---|---|
| **infer-TTFT** | `first_token_infer_mean_ms` | Pure inference first-token latency. Best for measuring the prefill compute itself. Our main metric. |
| **e2e-TTFT** | `ttft_mean_ms` (a.k.a. `Prefill`) | End-to-end first-token latency. Includes GenAI pipeline host overhead. What the user actually feels. |
| **TPOT** | `tpot_mean_ms` | Time per output token (decode step latency). |
| **Throughput / Decode TPS** | `throughput_mean_tps` / `decode_throughput_mean_tps` | Tokens per second (overall / decode only). |

e2e-TTFT is about 40–50 ms larger than infer-TTFT. That extra time is GenAI pipeline host overhead, **not** tokenization (tokenization measured at only ~0.7 ms). Because that host cost is fixed, the percentage reduction on e2e is always smaller than on infer.

### 2.4 How we find hotspots

Two independent tools that cross-check each other:

1. **Deep operator profiling** (`model_bench --profile`): replays the real prefill+decode loop and reports each operator's `real_time`, separately for prefill and decode. Works for the SDPA graph.
2. **VTune GPU hotspots**: captures the real pipeline at the GPU-kernel level, including each kernel's occupancy and what limits it (work size, SLM, barriers). This is the only correct way to see the PA path, because PA is a pipeline-level setting and the static deep-profile always sees the SDPA graph.

The whole project is iterative: profile, remove the biggest editable item, profile again, remove the next one. Every optimization was found this way.

### 2.5 How we verify correctness

`result_md5` is the MD5 hash of the model's decoded output text. With greedy decoding the output is deterministic: every step takes the argmax of the logits. If we let the model generate 128 tokens and the hash matches the baseline exactly, then every one of those 128 steps had no numeric change large enough to flip any token choice. This is a strong end-to-end fingerprint and the right test for layout, dispatch, and graph-structure changes.

One detail on the 356H: even with a fixed prompt, the 356H iGPU can produce a small set of different hashes across iterations (a *reproducible* numeric jitter, not randomness — re-running gives the identical sequence). So on 356H we compare the *multiset* of hashes and require it to be identical. All recommended optimizations pass.

### 2.6 One important measurement caveat

In the profiler, data-movement operators can look like ~50% of the time, yet the wall-clock TTFT only improves ~10%. This is **not** a profiling error — the percentage's denominator (the operator-time sum) is a different, smaller number than the wall-clock TTFT. Section 5 explains this in detail, because it is the most common question this kind of work raises.

---

## 3. SDPA Backend Optimizations

### 3.1 Baseline hotspot analysis

We built a clean baseline and profiled Qwen3.5-2B prefill on a real ~938-token prompt.

**Deep operator profile (prefill only, operator-time sum ≈ 66 ms):**

| Operator | Time | Share | Target |
|---|---:|---:|---|
| Transpose (permute_ref) | 20.98 ms | 31.9% | → opt-2 |
| FullyConnectedCompressed (INT4 matmul) | 16.60 ms | 25.2% | already optimal |
| Reorder (layout conversion) | 13.81 ms | 21.0% | → opt-1 |
| DynamicQuantize | 5.41 ms | 8.2% | → opt-3 (later) |
| GatedDeltaNet | 2.95 ms | 4.5% | the fused SSM operator |

The key observation: in the Qwen3.5 hybrid model, **Transpose plus Reorder together are about 53% pure data movement**, while the matrix multiply is only 25%. For comparison, the dense Qwen3-1.7B spends 94%+ in matmul and quantization — almost no layout waste, already at the compute floor. So the opportunity in the hybrid model is in the *data movement* first.

**VTune cross-check (358H, whole-run kernel view, operator-time sum ≈ 913 ms):**

| Kernel | Time | Share | Occupancy | Note |
|---|---:|---:|---:|---|
| `gemm_kernel` | 474.5 ms | 51.9% | 73% | INT4 matmul (runs every decode step) |
| `gated_delta_net_ref` | 86.9 ms | 9.5% | 53% | SSM |
| `permute_ref` | 69.1 ms | 7.6% | 99% | **→ opt-2** |
| `strided_slice_ref` | 58.2 ms | 6.4% | 94% | SSM slice |
| `conv_reorder` | 45.8 ms | 5.0% | 98% | **→ opt-1** |
| `sdpa_micro` | 45.3 ms | 5.0% | 36% | full-attn layers |
| `gen_conv` (GroupConv) | 20.6 ms | 2.3% | 97% | **→ opt-1** |

> Note the two profilers look different: the deep profile is **prefill only**, so the data-movement operators dominate; VTune captures the **whole 512-token generation**, so `gemm_kernel` (which runs on every decode step) dominates. Both are correct, and both point at the same editable targets: `permute_ref` (opt-2) and the conv `Reorder`/`GroupConv` (opt-1).

### 3.2 opt-1: Eliminate the depthwise conv Reorder

**Root cause.** The Qwen3.5 SSM layer has a causal conv1d that is fully depthwise (groups = input channels = output channels = 6144, kernel size 4). Because it has a dynamic shape, the plugin unconditionally assigned it a blocked memory layout (fsv16). But its neighbors use the planar layout, so the plugin inserted one Reorder before and one after the conv to convert layouts.

**Why it is waste.** A depthwise conv has no cross-channel reuse, so the blocked layout gives it zero benefit. We paid for two full-tensor layout copies for an operator that does not need them.

**The fix (+16 lines).** In the layout-decision function, add a short-circuit: if the conv is fully depthwise, keep it planar. The dynamic OpenCL conv natively supports planar, so the conv still runs correctly and both Reorders disappear.

**Result.** All 36 Reorders (18 layers × 2) disappeared; Reorder operator time went from 13.81 ms to 0.

### 3.3 opt-2: Fix the transpose (permute_ref) GPU work-group dispatch

**The surprising finding.** Each SSM layer has two transposes of the same shape, but their cost differed by 10×. A micro-benchmark showed the cause was a GPU work-group (LWS) assignment defect, behaving opposite to intuition:

| Sequence length | Chosen LWS | Time |
|---|---|---:|
| 937 (prime) | {1,1,1024} | 965 µs (fast) |
| 938 (= 2·7·67) | {1,7,128} | 2123 µs (slow) |
| 1024 (= 2¹⁰) | {1,1024,1} | 2393 µs (slow) |

The dispatcher greedily puts the large work-group on the Y (sequence) axis. When the sequence length has good divisors, the large group lands on the Y axis — the *scattered-write* direction — so memory access is not coalesced and it is slow. When the length is prime, the group is forced onto the F axis — the *coalesced-write* direction — so it is fast. In other words, the slowdown happens exactly when the length divides nicely, a defect in the dispatch heuristic.

**The fix (+31 lines, pure LWS selection, no math change).** When the F axis dominates, force the large work-group onto the F axis, but only when this does not shrink the work-group. Because we only change how the work is split, not what each work-item computes, the result is numerically identical by definition.

**Result.** The slow cases were fixed (sequence 938: 2123 → 979 µs). In the model, total Transpose operator time went from about 20.7 ms to 5.8 ms.

### 3.4 opt-3: Reuse one DynamicQuantize across FCs that share an input

**How we found it.** After opt-1 and opt-2 removed the two big data-movement items, we profiled again. DynamicQuantize was now the second-largest operator. Its absolute time was unchanged (~5.4 ms), but its *share* jumped from 8% to 14% because the denominator shrank. The key clue: 175 DynamicQuantize nodes, far more than the number of FC layers — clear redundancy.

**Root cause.** In each Qwen3.5 decoder layer, four FCs share the same input-layernorm output, and two more share the same post-attention-layernorm output. But the plugin inserted a *separate* DynamicQuantize in front of *every* FC — so the same hidden state was quantized 4× or 2× per layer.

**The fix (two files).** Before inserting a DynamicQuantize, look for an existing one on the same input whose configuration matches field-by-field, and reuse it. If the configs do not match, do not merge — zero correctness risk. We also relaxed a skip-assertion so the shared DynamicQuantize can still be skipped to an f16 passthrough during decode, which keeps decode byte-identical.

**Result.** DynamicQuantize nodes went from 175 to 97; its operator time dropped about 36%. This optimization is completely general — any model with multiple FCs sharing one activation benefits.

### 3.5 SDPA results

Pure-inference TTFT (`first_token_infer_mean_ms`, n=8, real ~1000-token prompt, GPU).

**358H:**

| Model | Baseline | +opt-1 | +opt-1+2 | +opt-1+2+3 | opt-1 | opt-2 | opt-3 | **Total** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B | 132.4 | 120.2 | 108.2 | **104.9** | −9.2% | −10.0% | −3.0% | **−20.8%** |
| Qwen3.5-2B | 157.7 | 146.1 | 134.6 | **132.3** | −7.4% | −7.9% | −1.7% | **−16.1%** |
| Qwen3.5-4B | 333.0 | 315.1 | 293.8 | **290.9** | −5.4% | −6.8% | −1.0% | **−12.6%** |
| Qwen3-1.7B (dense, control) | 63.1 | — | — | 62.6 | — | — | — | −0.7% (not triggered) |

**356H:**

| Model | Baseline | +opt-1 | +opt-1+2 | +opt-1+2+3 | opt-1 | opt-2 | opt-3 | **Total** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B | 292.3 | 289.8 | 255.9 | **251.1** | −0.9% | −11.7% | −1.9% | **−14.1%** |
| Qwen3.5-2B | 348.9 | 344.4 | 315.3 | **309.7** | −1.3% | −8.4% | −1.8% | **−11.2%** |
| Qwen3.5-4B | 800.6 | 788.1 | 731.5 | **720.4** | −1.6% | −7.2% | −1.5% | **−10.0%** |
| Qwen3-1.7B (dense, control) | 183.0 | — | — | 180.7 | — | — | — | −1.3% (not triggered) |

The tables above are pure-inference TTFT (ic=1). These optimizations are prefill-only, so **decode metrics are unchanged**: the SDPA backend baseline decode (356H, official runner, ic=512) is TPOT ~11.6 / 16.2 / 32.2 ms and decode TPS ~86 / 62 / 31 tok/s for 0.8B / 2B / 4B, and the optimizations leave those values untouched (see the PA-vs-SDPA full-metric table in §4.1 for the same numbers alongside PA).

**Reading the tables:**
- The trend is identical on both machines: **opt-2 (the transpose fix) is the biggest single contributor.**
- opt-1 (the Reorder fix) contributes much less on 356H (−0.9% to −1.6%) than on 358H (−5.4% to −9.2%), because the absolute cost of that Reorder is smaller on the smaller iGPU.
- Smaller models benefit more; the dense control model is unaffected, exactly as expected.

### 3.6 Correctness (SDPA)

All three Qwen3.5 sizes produce byte-identical output before and after all three optimizations (128-token greedy). For opt-3 specifically, we verified both prefill (1-token) and 128-token decode are byte-identical.

---

## 4. PA (PagedAttention) Backend Optimizations

### 4.1 PA is already much faster than SDPA

PA is now the OpenVINO default, and it is a large win by itself. Official runner, stock OpenVINO, **real 1000-token prompt, 512 output tokens (ic=512), n=10**, first-inference latency:

**358H:**

| Model | PA | SDPA | PA vs SDPA |
|---|---:|---:|---:|
| Qwen3.5-0.8B | 70.5 ms | 143.8 ms | **−51.0%** |
| Qwen3.5-2B | 99.7 ms | 165.8 ms | **−39.9%** |
| Qwen3.5-4B | 229.6 ms | 350.3 ms | **−34.5%** |

**356H:**

| Model | PA | SDPA | PA vs SDPA |
|---|---:|---:|---:|
| Qwen3.5-0.8B | 151.3 ms | 305.8 ms | **−50.5%** |
| Qwen3.5-2B | 215.5 ms | 351.0 ms | **−38.6%** |
| Qwen3.5-4B | 510.0 ms | 791.7 ms | **−35.6%** |

The speedup ratio is almost identical on both machines. A useful reference: our best SDPA-plus-all-three-optimizations 2B result was 132 ms, but plain PA alone is 99.7 ms — faster by 33 ms. That is the direct evidence that "PA replaces the SDPA-era optimizations."

**PA is faster on decode too, not just prefill.** The two tables above are the first-token latency. Looking at the full generation (356H, official runner, ic=512, n=10) shows PA also wins on the per-token decode latency (TPOT) and the decode throughput (TPS):

| Model | Metric | SDPA | PA | PA vs SDPA |
|---|---|---:|---:|---:|
| **0.8B** | e2e-TTFT | 348.3 ms | 193.9 ms | **−44.3%** |
| | TPOT (decode) | 11.63 ms | 8.45 ms | **−27.3%** |
| | Decode TPS | 86.0 tok/s | 118.4 tok/s | **+37.7%** |
| **2B** | e2e-TTFT | 398.5 ms | 263.8 ms | **−33.8%** |
| | TPOT (decode) | 16.16 ms | 13.78 ms | **−14.7%** |
| | Decode TPS | 61.9 tok/s | 72.6 tok/s | **+17.3%** |
| **4B** | e2e-TTFT | 843.6 ms | 563.7 ms | **−33.2%** |
| | TPOT (decode) | 32.21 ms | 26.53 ms | **−17.6%** |
| | Decode TPS | 31.0 tok/s | 37.7 tok/s | **+21.6%** |

> This is an important distinction. **Switching backend (SDPA → PA) improves both prefill and decode.** Our own source-level optimizations (§4.3–4.6) improve only the prefill (TTFT) — they leave decode unchanged. So the backend choice and our optimizations are complementary: PA gives the big cross-the-board win, and our patches shave the remaining prefill on top.

### 4.2 Which SDPA optimizations survive under PA

PA uses a completely new set of fused SSM/attention kernels, which fuse away or bypass the standalone operators that opt-1 and opt-2 targeted. VTune (358H) confirms it — the PA baseline kernel view no longer has `permute_ref`, `conv_reorder`, or `gen_conv`; instead the conv is a single fused `paged_causal_conv1d_ref` and the transpose is fused into the SSM kernel:

| Optimization | Under SDPA | Under PA | Still useful? |
|---|---|---|---|
| **opt-1** (remove conv Reorder) | targets `conv_reorder` + `gen_conv` | PA uses a fused conv, no Reorder | ❌ nothing to remove |
| **opt-2** (fix permute LWS) | targets `permute_ref` | PA fuses the transpose into the SSM kernel | ❌ not triggered |
| **opt-3** (reuse DynamicQuantize) | targets `dynamic_quantize_gpu_opt` | still present in PA prefill (6–7%) | ✅ **still effective** |

So under PA, opt-1 and opt-2 become zero, but **opt-3 is re-validated as still effective**. This is why opt-3 is our most general, most valuable optimization.

### 4.3 PA hotspot analysis (VTune, 2B prefill)

**358H, pristine (operator-time sum ≈ 572 ms):**

| Kernel | Time | Share | Occupancy | Editable? / Target |
|---|---:|---:|---:|---|
| `gemm_kernel` (INT4 GEMM) | 262.5 ms | 45.8% | 46% | No — oneDNN XMX (compute floor) |
| `paged_gated_delta_net_opt` | 104.1 ms | 18.2% | 53% | Serial SSM recurrence (high risk) |
| `paged_causal_conv1d_ref` | 66.0 ms | 11.5% | **20%** | **→ conv1d target** |
| `dynamic_quantize_gpu_opt` | 36.4 ms | 6.4% | 100% | **→ opt-3** |
| `activation_ref` | 32.6 ms | 5.7% | 100% | **→ opt-6** |
| `rms_gpu_bfyx_opt` | 29.8 ms | 5.2% | 68% | tested, no wall-clock gain |

**356H, pristine (operator-time sum ≈ 1283 units):**

| Kernel | Time | Share | Occupancy | Editable? / Target |
|---|---:|---:|---:|---|
| `gemm_kernel` | 587.8 | 45.8% | 43% | No — oneDNN XMX |
| `paged_gated_delta_net_opt` | 244.1 | 19.0% | 100% | Serial SSM recurrence |
| `dynamic_quantize_gpu_opt` | 94.3 | 7.3% | 100% | **→ opt-3** |
| `rms_gpu_bfyx_opt` | 91.6 | 7.1% | 68% | tested, no wall-clock gain |
| `activation_ref` | 79.8 | 6.2% | 100% | **→ opt-6** |
| `paged_causal_conv1d_ref` | 70.6 | 5.5% | **60%** | **→ opt-5** |
| `sdpa_micro` | 37.6 | 2.9% | 51% | oneDNN gemmstone (dead-end) |
| `generic_eltwise_ref` | 31.4 | 2.4% | 100% | bandwidth-bound |

The conv1d kernel stood out on both machines because of its low occupancy — that was the editable opportunity.

### 4.4 conv1d optimization: opt-5 (token-tiled) — and why opt-4 is excluded

**The problem.** The causal conv1d dispatched only `hidden_size = 6144` threads, and each thread walked serially along the time axis over ~1000 tokens. The token dimension had zero parallelism, so occupancy was only 20% (358H) / 60% (356H).

**First attempt (opt-4), and why we do not recommend it.** The obvious fix is to give every output token its own work-item — fully token-parallel. VTune shows exactly why this is platform-dependent:

| conv1d dispatch (2B PA prefill) | GWS | Occupancy | conv1d kernel time |
|---|---|---:|---:|
| **358H** pristine (serial) | 1×6144×1 | 20% | 66.0 ms |
| **358H** opt-4 (full token-parallel) | 1×6144×1002 | 100% | **45.1 ms — faster ✅** |
| **356H** pristine (serial) | 1×6144×1 | 60% | 14.1 ms/iter |
| **356H** opt-4 (full token-parallel) | 1×6144×1002 | 100% | **24.7 ms/iter — slower ❌** |
| **356H** opt-5 (token-tiled, G=3) | 1×6144×3 | 100% | **10.5 ms/iter — faster ✅** |

On the large 358H iGPU, opt-4 helps: the occupancy benefit outweighs the extra memory reads. But on the small 356H iGPU, opt-4 *hurts*: every work-item re-reads its own K-element window from global memory, which multiplies global reads by ~K and makes the kernel memory-bound. The output was still bit-exact — it was purely a performance regression. **So opt-4 is not a general optimization, and we exclude it.**

**The general fix: opt-5, token-tiled conv1d.** Instead of one work-item per token, split each sequence into a small number G of contiguous blocks (G=3 by default). Each work-item processes one block serially with a sliding window, so there is no redundant reading inside a block. This gets the best of both worlds:

- Occupancy reaches 100% (with G≥2 there are >12,000 work-items, well above the 320 HW threads).
- No memory blow-up — the extra reads are only at block boundaries (negligible).
- The per-element math order is identical to the serial version → bit-exact.

**Result on 356H:** conv1d kernel 14.1 → 10.5 ms/iter (−26%), occupancy 60% → 100%. Unlike opt-4, it does not regress on the small iGPU.

### 4.5 opt-6: Route dynamic-shape activation to the vectorized kernel

**The problem.** After opt-3 and opt-5, the largest remaining *editable* item on the 356H was `activation_ref` (80 units, ~6% of prefill). These are plain element-wise activations. OpenVINO already has a float4-vectorized activation kernel (`activation_opt`), but it did not support dynamic shapes, so dynamic activations fell back to the slow scalar `activation_ref`.

**The fix (two files).** Enable dynamic-shape support on the vectorized kernel, but only for the safe subset (same input/output layout, no fused ops, no runtime activation parameters), and add the shape-info argument the shape-agnostic dispatch requires. This does not change the math — only which kernel is selected.

**Result on 356H (VTune):** the activation dropped from `activation_ref` at ~80 units to `activation_opt` at ~23 units — roughly a **−71% kernel time**. In wall-clock terms it adds another −3.0% to −3.8% on infer-TTFT (see §4.6).

### 4.6 Final PA stack results (356H, real 1000-token input / 512 output)

The recommended 356H PA stack is **opt-3 + opt-5 + opt-6** (all in the OpenVINO GPU plugin). Measured on the definitive full benchmark: 356H, PA, GPU, `in=1002, out=512`, n=5.

**Latency and throughput, all states:**

| Model | State | infer-TTFT (ms) | e2e-TTFT (ms) | TPOT (ms) | Throughput (tok/s) | Decode TPS (tok/s) |
|---|---|---:|---:|---:|---:|---:|
| **0.8B** | pristine | 153.1 | 204.5 | 8.67 | 110.5 | 115.4 |
| | +opt-3 | 147.2 | 197.8 | 8.68 | 110.5 | 115.2 |
| | +opt-3+5 | 144.7 | 195.2 | 8.70 | 110.3 | 115.0 |
| | **+opt-3+5+6** | **137.8** | **189.0** | 8.67 | 110.8 | 115.4 |
| **2B** | pristine | 219.8 | 271.2 | 14.02 | 68.8 | 71.3 |
| | +opt-3 | 212.9 | 264.2 | 14.07 | 68.7 | 71.1 |
| | +opt-3+5 | 209.3 | 260.8 | 14.07 | 68.7 | 71.1 |
| | **+opt-3+5+6** | **202.4** | **253.1** | 14.11 | 68.6 | 70.9 |
| **4B** | pristine | 520.7 | 575.8 | 27.11 | 35.5 | 36.9 |
| | +opt-3 | 510.0 | 565.4 | 27.07 | 35.6 | 37.0 |
| | +opt-3+5 | 511.3 | 565.9 | 27.11 | 35.5 | 36.9 |
| | **+opt-3+5+6** | **495.8** | **549.8** | 27.12 | 35.5 | 36.9 |

**Each optimization's standalone contribution (increment added on top of the previous state):**

| Model | opt-3 (infer) | opt-5 (infer) | opt-6 (infer) | **infer total** | opt-3 (e2e) | opt-5 (e2e) | opt-6 (e2e) | **e2e total** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.8B | −3.8% (−5.9 ms) | −1.7% (−2.5 ms) | −4.8% (−6.9 ms) | **−10.0%** | −3.3% (−6.8 ms) | −1.3% (−2.5 ms) | −3.2% (−6.2 ms) | **−7.6%** |
| 2B | −3.2% (−7.0 ms) | −1.7% (−3.6 ms) | −3.3% (−6.9 ms) | **−7.9%** | −2.6% (−7.1 ms) | −1.3% (−3.4 ms) | −2.9% (−7.6 ms) | **−6.7%** |
| 4B | −2.0% (−10.6 ms) | +0.3% (noise) | −3.0% (−15.6 ms) | **−4.8%** | −1.8% (−10.4 ms) | +0.1% (noise) | −2.9% (−16.2 ms) | **−4.5%** |

> opt-6 (activation) and opt-3 (DynamicQuantize reuse) are the two biggest PA contributors. opt-5 (conv1d) is smaller and shrinks with model size — on the 4B it is within noise, because the conv1d is a smaller fraction of the larger model's prefill.

**Read from the metrics table:** the TPOT, Throughput, and Decode TPS columns are flat across all states. These optimizations improve the prefill (TTFT) only; decode speed is unchanged, as expected. Decode stays at ~115 / ~71 / ~37 tok/s for 0.8B / 2B / 4B.

**Kernel-level accounting (356H VTune, 2B prefill) — this is where each optimization's effect shows up:**

| Kernel | pristine | opt-4 (rejected) | +opt-3+5 | +opt-3+5+6 | which opt |
|---|---:|---:|---:|---:|---|
| `paged_causal_conv1d` | 70.6 (occ 60%) | 123.6 (occ 100%, memory-bound) | 52.2 (occ 100%) | 34.9 (occ 100%) | **opt-5** |
| `dynamic_quantize_gpu_opt` | 94.3 | 94.3 | 61.8 | 41.3 | **opt-3** |
| activation | 79.8 `_ref` | 79.8 `_ref` | 79.1 `_ref` | 23.1 `_opt` | **opt-6** |
| `gemm_kernel` | 587.8 | 587.7 | 580.3 | (unchanged) | not editable |
| `paged_gated_delta_net_opt` | 244.1 | 243.6 | 240.1 | (unchanged) | not editable |

> Note on 358H: on the large iGPU the historical PA stack was opt-3 + opt-4 (infer-TTFT −6.7% / −6.5% / −3.9%), because opt-4 works there. Since opt-4 regresses on the small iGPU, the *portable* PA optimization that holds on both machines is opt-3, with opt-5/opt-6 as the 356H-validated replacements/additions. We did not extrapolate the 356H opt-5/opt-6 numbers to 358H.

### 4.7 Correctness (PA)

Every recommended PA optimization is verified bit-exact. For all three model sizes, the 1-token prefill hash and the 128-token decode hash multisets are identical to the pristine baseline. (opt-4, which we exclude, was also bit-exact — its problem was performance, not correctness.)

---

## 5. Why does profiling show ~50% but TTFT improves only ~10%?

This is the most common and most important question, so it deserves its own section. **The profiler is not wrong. The confusion is that the percentage's denominator and the TTFT's denominator are two different numbers.**

For Qwen3.5-2B at the SDPA baseline there are three different "times":

| Quantity | Value | What it is |
|---|---:|---|
| Deep-profile operator-time sum | 65.8 ms | The sum of each operator's `real_time`. The "Transpose 32% / Reorder 21%" percentages are computed against *this*. |
| infer-TTFT | 157.7 ms | The real pure-inference first-token wall-clock time. |
| e2e-TTFT | 201.4 ms | The end-to-end first-token wall-clock time (includes host overhead). |

So "Reorder is 21%" means 21% of the 65.8 ms operator sum — **not** 21% of the 158 ms TTFT. The operator sum is smaller than the wall clock because on the GPU multiple kernels overlap and queue (per-operator times are not additive), and because the wall-clock TTFT also includes pipeline stages the operator sum does not.

**The correct way to reconcile them is in absolute milliseconds:**

| | operator sum | infer-TTFT (wall clock) |
|---|---:|---:|
| Baseline | 65.8 ms | 157.7 ms |
| +opt-1 (remove Reorder) | 52.4 ms (−13.4) | 146.1 ms (**−11.6**) |
| +opt-2 (fix Transpose) | 37.3 ms (−15.1) | 134.6 ms (**−11.5**) |
| +opt-3 (reuse DQ) | 36.5 ms (−0.8) | 132.3 ms (**−2.3**) |

opt-1 removed 13.4 ms of Reorder operator time, and the wall clock dropped 11.6 ms — a very close match. So the right statement is not "Reorder is 21%, so TTFT should drop 21%." It is "Reorder is about 13 ms of absolute cost; removing it drops TTFT from 158 to 146 ms, which is about 7–8% of infer-TTFT." Each data-movement optimization is worth about 6–8%, and stacking them gives the 13–21% total we measured.

In one sentence: **the operators really were fully removed, but each one was only about 8% of the true total time — not the 21% you see against the profiler's smaller denominator.**

---

## 6. What's Next — the ceiling

After our optimizations, the remaining PA prefill time is dominated by two things that are hard limits, not local source fixes:

| Item | Share | Why we do not touch it |
|---|---:|---|
| `gemm_kernel` (INT4 GEMM) | ~47–50% | oneDNN's JIT XMX micro-kernel, not an editable OpenCL kernel. Forcing the editable OpenCL FC path is **5× slower**. Its 40% occupancy is normal for XMX. This is the compute floor. |
| `paged_gated_delta_net_opt` (SSM) | ~19–20% | A true serial state recurrence — output tokens are not independent, so it cannot be token-parallelized like the conv. Already at 100% occupancy. A chunkwise rewrite is estimated at only ~3–4% of total prefill, would take days, and would almost certainly break bit-exactness. |

We also tested and rejected several other candidates and recorded them so nobody repeats the work:
- **Larger FC horizontal fusion** (fuse 4 shared-input FCs into one GEMM): measured **negative** (+0.8% to +2.0%) — the extra split op outweighs the fusion.
- **RMS-norm SLM tuning**: the occupancy can be raised, but the kernel is memory-latency-bound, so wall-clock is unchanged.
- **Element-wise / slice vectorization**: already 100% occupancy (bandwidth-bound), so vectorizing gives little.

The honest conclusion: **under the constraints of "bit-exact, no algorithm rewrite, no touching oneDNN," the current stack is at the practical ceiling on this iGPU.** Going further would require relaxing the bit-exact criterion (for a GDN chunkwise rewrite) or working inside oneDNN's XMX GEMM — both are large, separate projects. The one clearly worthwhile follow-up is to upstream opt-3 and the conv1d work as pull requests with unit tests, since they are general.

---

## 7. Conclusion

- **Use PA.** It is the default and it is 34–51% faster than SDPA on both iGPUs. This is the biggest single win.
- **On top of PA, our OpenVINO GPU-plugin optimizations add up to −10% infer-TTFT and −7.6% e2e-TTFT** on the 356H (1000-token input / 512 output), driven by opt-3 (DynamicQuantize reuse), opt-5 (token-tiled conv1d), and opt-6 (vectorized dynamic activation).
- **Under the older SDPA backend, the same approach gave −10% to −21%**, driven mainly by opt-2 (the transpose dispatch fix).
- **These are prefill (TTFT) optimizations** — decode throughput (TPS, TPOT) is unchanged, as expected.
- **opt-3 is the most portable win** — it helps both backends and any model where FCs share an activation.
- **Every recommended optimization is numerically lossless** — the generated text is byte-for-byte identical to the baseline, verified with 128-token greedy MD5.
- **We deliberately exclude opt-4** (full token-parallel conv1d): it helps the large iGPU but regresses the small one. opt-5 is its safe, portable replacement.
- The remaining time is the INT4 GEMM and the serial SSM recurrence — hard limits, not local optimization targets.

*All data measured on locally source-built OpenVINO 2026.2 / OpenVINO GenAI 2026.2 / model_bench (C++) / VTune 2026.2, on Panther Lake 358H (Arc B390) and 356H (32 XVE). Workload: real 1000-token input, 512 output tokens. Correctness verified by 128-token greedy MD5.*
