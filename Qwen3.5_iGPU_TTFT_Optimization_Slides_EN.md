# Qwen3.5 iGPU TTFT Optimization — Presentation Deck (English)

> **How to use this file.** Each `## Slide N` block is one PowerPoint slide. **"On the slide"** is the bullet content. **"Script"** is what to read aloud — short, simple sentences you can read almost word-for-word.
> **Benchmark workload for all numbers: real 1000-token input, 512 output tokens, greedy decoding.**

---

## Slide 1 — Overview

**On the slide:**
- **Topic:** Reducing Qwen3.5 first-token latency (TTFT) on Intel iGPU, by patching the **OpenVINO GPU plugin** source
- **Method:** profile the real workload → find wasted GPU time → remove it. We do NOT touch the INT4 matrix multiply (already optimal on XMX)
- **Backends:** SDPA (older) and PagedAttention (now default) · **Machines:** 358H (larger Arc B390) and 356H (smaller, 32 XVE, ~2× slower)
- **Workload:** real 1000-token input, 512 output tokens, greedy decoding
- **Metrics:** infer-TTFT (pure inference) · e2e-TTFT (+~45 ms host) · TPOT / TPS (decode) — profiled with deep operator profile + VTune (occupancy)
- **Correctness:** greedy → MD5 of the 128-token output → identical hash = **bit-exact** (all recommended opts pass)
- **Agenda:** SDPA → PA → the "50% vs 10%" question → ceiling → summary

**Script:**
"Hello everyone. Today I will share our work on making Qwen3.5 faster on Intel integrated GPUs. Our target is the first-token latency, also called TTFT. The method is simple: we profile the real workload, find where GPU time is wasted, and patch the OpenVINO GPU plugin to remove that waste. We do not touch the INT4 matrix multiply, because it is already optimal on the XMX hardware.

OpenVINO has two attention backends: SDPA, the older one, and PagedAttention, which is now the default. We tested on two machines — the 358H with a larger iGPU, and the 356H with a smaller one, about two times slower. I highlight this size difference now, because later we will see an optimization that helps the large iGPU but hurts the small one. All numbers use a real one-thousand-token input and five hundred twelve output tokens.

We track four metrics: the pure inference first-token time, the end-to-end first-token time which adds about forty-five milliseconds of host overhead, the per-token decode time, and the throughput. And one key point up front: every optimization I recommend is numerically lossless. We prove this with greedy decoding and an MD5 hash of the output — if the hash matches the baseline exactly, no numeric change flipped any token. The model output stays exactly the same, byte for byte. I will go through the two backends, one common question about profiling, and the limits we reached."

---

## Slide 2 — SDPA: Baseline Hotspots

**On the slide:**
Qwen3.5-2B prefill, deep operator profile (prefill only):

```
Transpose                 31.9%   <- data movement  -> opt-2
FullyConnectedCompressed  25.2%   <- INT4 matmul (optimal)
Reorder                   21.0%   <- data movement  -> opt-1
DynamicQuantize            8.2%   <- later          -> opt-3
GatedDeltaNet              4.5%
```
- Transpose + Reorder ≈ 53% is **pure data movement**; matmul only 25%
- VTune (whole run) confirms the same targets: `permute_ref` 7.6%, `conv_reorder` 5.0%, `gen_conv` 2.3%
- Dense Qwen3 has almost no such waste (already at compute floor)

**Script:**
"Let us start with the SDPA backend. Here is the prefill profile for the 2B model. Look at the top items. Transpose is thirty-two percent, and Reorder is twenty-one percent. Together, about fifty-three percent, and they are pure data movement — just moving data around in memory. The actual matrix multiply is only twenty-five percent. VTune confirms the same targets. For comparison, the dense Qwen3 model has almost no data-movement waste; it is already at the compute floor. So in this hybrid model, the opportunity is clearly the data movement. That gives us our first two optimizations."

---

## Slide 3 — SDPA: The Three Fixes

**On the slide:**
- **opt-1 — remove the depthwise conv Reorder (+16 lines)**
  - Depthwise conv was forced into a blocked layout it does not need → keep it planar → Reorder time 13.8 ms → 0
- **opt-2 — fix the transpose GPU work-group dispatch (+31 lines)** — the biggest contributor
  - Two same-shape transposes differed 10×; the large work-group landed on the scattered-write axis when the length divided nicely
  - Force it onto the coalesced axis → Transpose time ~20.7 ms → 5.8 ms
- **opt-3 — reuse one DynamicQuantize across FCs that share an input (2 files)**
  - 4 FCs shared one input but each got its own quantize → 175 quantize nodes → reuse → 97 nodes
  - Config mismatch → no merge → zero correctness risk; completely general
- All three are lossless (layout / work-split / same-input-same-config reuse)

**Script:**
"Here are the three SDPA fixes. opt-1 removes the Reorder. The depthwise convolution was forced into a blocked memory layout it does not need, so the plugin inserted two layout conversions around it. We keep it in the plain layout, and both conversions disappear — from about fourteen milliseconds to zero.

opt-2 is the biggest one. We found two transposes of the same shape that differed by ten times in cost, because of a defect in the GPU work-group assignment. When the sequence length divides nicely, the large work-group lands on the wrong axis, where memory writes are scattered. We force it onto the correct axis. Transpose time drops from about twenty-one milliseconds to under six.

opt-3 removes redundant quantization. Four fully-connected layers share the same input, but each had its own quantize step. There were 175 quantize nodes. We reuse one quantize when the input and configuration match, bringing it to 97 nodes. If they do not match, we do not merge, so there is zero risk. And this one is completely general. All three are lossless."

---

## Slide 4 — SDPA: Results

**On the slide:**
TTFT reduction, baseline → +opt-1+2+3 (both infer-TTFT and e2e-TTFT):

**358H**

| Model | infer-TTFT | e2e-TTFT |
|---|---:|---:|
| Qwen3.5-0.8B | **−20.8%** (132 → 105 ms) | −14.8% (176 → 150 ms) |
| Qwen3.5-2B | **−16.1%** (158 → 132 ms) | ~−11.9% (201 → 178 ms) |
| Qwen3.5-4B | **−12.6%** (333 → 291 ms) | ~−10.8% (377 → 336 ms) |

**356H**

| Model | infer-TTFT | e2e-TTFT |
|---|---:|---:|
| Qwen3.5-0.8B | **−14.1%** (292 → 251 ms) | −12.3% (337 → 295 ms) |
| Qwen3.5-2B | **−11.2%** (349 → 310 ms) | −9.9% (393 → 354 ms) |
| Qwen3.5-4B | **−10.0%** (801 → 720 ms) | −9.3% (845 → 766 ms) |

**Each optimization's standalone contribution (infer-TTFT, added on top of the previous, 0.8B / 2B / 4B):**

| Optimization | 358H | 356H |
|---|---|---|
| opt-1 — Reorder | −9.2 / −7.4 / −5.4% | −0.9 / −1.3 / −1.6% |
| **opt-2 — Transpose** | **−10.0 / −7.9 / −6.8%** | **−11.7 / −8.4 / −7.2%** |
| opt-3 — DQ reuse | −3.0 / −1.7 / −1.0% | −1.9 / −1.8 / −1.5% |

- **opt-2 (transpose fix) is the biggest contributor on both machines**
- opt-1 is large on 358H but small on 356H (that Reorder is cheaper on the smaller iGPU)
- e2e reduction is a bit smaller than infer (fixed ~40–45 ms host overhead)
- Decode metrics unchanged (prefill-only opts): 356H SDPA baseline TPOT ~11.6 / 16.2 / 32.2 ms, decode TPS ~86 / 62 / 31 tok/s
- Smaller models benefit more; dense Qwen3-1.7B control unaffected (~−1%)

**Script:**
"Here are the SDPA results, showing both the pure-inference and the end-to-end latency. On the 358H, the pure-inference first-token latency drops twenty-one percent for the small model, sixteen for the medium, thirteen for the large. End-to-end drops a little less — fifteen, twelve, eleven percent — because it includes a fixed host overhead of around forty-five milliseconds. On the 356H the reductions are ten to fourteen percent on inference, nine to twelve end-to-end.

The third table shows each optimization on its own. The transpose fix, opt-2, is clearly the biggest — around eight to twelve percent on both machines. opt-1, the Reorder fix, is large on the 358H but small on the 356H, because that Reorder is cheaper on the smaller iGPU. opt-3 adds one to three percent. The decode metrics are unchanged, since these only touch the prefill. And the dense control model at the bottom is unaffected, exactly as expected."

---

## Slide 5 — PagedAttention is the New Default — Faster on Prefill AND Decode

**On the slide:**
infer-TTFT reduction, PA vs SDPA (stock OpenVINO, 1000-token input / 512 output):

| Model | 358H | 356H |
|---|---:|---:|
| Qwen3.5-0.8B | **−51.0%** | **−50.5%** |
| Qwen3.5-2B | **−39.9%** | **−38.6%** |
| Qwen3.5-4B | **−34.5%** | **−35.6%** |

Full metrics (356H, official runner, n=10) — PA wins every metric:

| Model | Backend | e2e-TTFT | TPOT (decode) | Decode TPS |
|---|---|---:|---:|---:|
| 0.8B | SDPA | 348 ms | 11.6 ms | 86 tok/s |
| | **PA** | **194 ms** | **8.5 ms** | **118 tok/s** |
| 2B | SDPA | 399 ms | 16.2 ms | 62 tok/s |
| | **PA** | **264 ms** | **13.8 ms** | **73 tok/s** |
| 4B | SDPA | 844 ms | 32.2 ms | 31 tok/s |
| | **PA** | **564 ms** | **26.5 ms** | **38 tok/s** |

- PA vs SDPA: e2e-TTFT −33 to −44%, TPOT −15 to −27%, **decode throughput +17 to +38%**
- Unlike our source optimizations (prefill-only), **PA improves both prefill and decode**
- PA alone (2B) beats SDPA + all 3 opts; under PA, opt-1/opt-2 become zero, **opt-3 still works**
- **First recommendation: use PA (already the default)**

**Script:**
"Now the second backend, and here is the single biggest win. Switching from SDPA to PA reduces the first-token latency by thirty-four to fifty-one percent, on both machines. But it is even better than that — look at the second table. PA is faster on every metric. The end-to-end first-token time drops thirty-three to forty-four percent. The per-token decode time drops fifteen to twenty-seven percent. And the decode throughput, the tokens per second, goes up by seventeen to thirty-eight percent. So unlike our own optimizations, which only help the prefill, PA improves both the prefill and the decode. To put it in perspective, plain PA on the 2B model already beats our best SDPA result with all three optimizations. And under PA, opt-1 and opt-2 become zero, but opt-3 still works. The first and most important recommendation is simply: use PA."

---

## Slide 6 — PA: Hotspots (VTune)

**On the slide:**
Qwen3.5-2B, PA prefill, VTune (356H):

```
gemm_kernel                 45.8%  occupancy 43%    <- INT4 matmul (not editable)
paged_gated_delta_net_opt   19.0%  occupancy 100%   <- serial SSM (high risk)
dynamic_quantize_gpu_opt     7.3%  occupancy 100%   <- opt-3
activation_ref               6.2%  occupancy 100%   <- opt-6
paged_causal_conv1d_ref      5.5%  occupancy 60%    <- LOW occupancy -> opt-5
```

- Conv1d had low occupancy (20% on 358H, 60% on 356H) → editable opportunity
- gemm and the SSM recurrence are the hard limits (see last slide)

**Script:**
"Here is the PA profile for the 2B model. The matrix multiply is the biggest, but that is oneDNN's kernel and we cannot edit it. The SSM core is next, but it is a serial recurrence and high risk. The three editable items are the quantize, the activation, and the convolution. The convolution stands out because of its low occupancy — twenty percent on the large machine, sixty percent on the small one. Low occupancy means the GPU is not fully used. That is our clear opportunity."

---

## Slide 7 — PA: conv1d Optimization (opt-5) — and the Lesson of opt-4

**On the slide:**
- Problem: conv1d used only 6144 threads; each walked ~1000 tokens serially → low occupancy
- **First attempt (opt-4, fully token-parallel): helped 358H but REGRESSED on 356H**

| conv1d dispatch (2B) | Occupancy | conv1d time |
|---|---:|---:|
| 358H serial | 20% | 66.0 ms |
| 358H opt-4 full-parallel | 100% | 45.1 ms ✅ |
| 356H serial | 60% | 14.1 ms/iter |
| 356H opt-4 full-parallel | 100% | 24.7 ms/iter ❌ (memory-bound) |
| **356H opt-5 tiled (adaptive G=3)** | 100% | **10.5 ms/iter ✅** |

- opt-4: every work-item re-reads its window → memory-bound on the small iGPU → **excluded**
- opt-5: split each sequence into a few blocks → 100% occupancy AND no extra memory → **−26%, bit-exact**
- **G is chosen adaptively** from the GPU thread budget (EU × threads/EU) + sequence length: **3 on the 356H, ~8 on the 358H, 1 (serial) for decode** → one portable patch, expected to replace opt-4 on the 358H too

**Script:**
"Now the convolution, and this is a good engineering story. The problem: the kernel used only about six thousand threads, each walking serially over a thousand tokens.

Our first attempt, opt-4, was the obvious one — give every token its own work-item, fully parallel. On the large 358H iGPU this worked: occupancy went to one hundred percent and the kernel got faster. But on the small 356H it got slower — from fourteen to twenty-five milliseconds. The reason is that every work-item re-reads its own window from memory, which multiplies memory traffic and makes it memory-bound on the small iGPU. It was still correct, just slower. So we exclude that version.

The real fix is opt-5. Instead of one work-item per token, we split each sequence into just a few blocks. Each work-item processes one block with a sliding window, so there is no redundant reading. This gives both full occupancy and no extra memory. And importantly, we do not hardcode the number of blocks — it is chosen automatically from the size of the GPU and the sequence length. On the small 356H it picks three; on the larger 358H it would pick about eight; for decode it falls back to one. So a single patch fits every iGPU size, and it should replace the opt-4 version on the 358H as well. The convolution drops twenty-six percent and stays bit-exact. The lesson: the same idea, tuned automatically to the hardware size."

---

## Slide 8 — PA: opt-6 (vectorized dynamic activation)

**On the slide:**
- After opt-3 and opt-5, the largest editable item left was the element-wise activation (`activation_ref`, ~6%)
- OpenVINO had a fast float4-vectorized kernel (`activation_opt`), but not for dynamic shapes
- Fix: enable the fast kernel for the safe subset + add the shape-info argument
- Changes kernel *selection* only, not the math
- VTune: activation ~80 → ~23 units (**−71% kernel time**); wall-clock adds −3.0% to −3.8% on infer-TTFT

**Script:**
"The last GPU optimization is opt-6. After the previous fixes, the biggest editable item left was the element-wise activation, about six percent. OpenVINO already had a fast vectorized activation kernel, but it did not support dynamic shapes, so these activations fell back to a slow scalar kernel. We enabled the fast kernel for the safe cases. This only changes which kernel is selected, not the math. In VTune the activation kernel time dropped by seventy-one percent, and in wall-clock it adds another three to four percent on the first-token latency."

---

## Slide 9 — PA: Final Results (356H, 1000-token input / 512 output)

**On the slide:**
Recommended 356H PA stack: **opt-3 + opt-5 + opt-6** (all in the OpenVINO GPU plugin)

| Model | infer-TTFT | e2e-TTFT | TPOT (decode) | Decode TPS |
|---|---:|---:|---:|---:|
| Qwen3.5-0.8B | **−10.0%** (153 → 138 ms) | **−7.6%** (205 → 189 ms) | 8.7 ms (unchanged) | ~115 tok/s (unchanged) |
| Qwen3.5-2B | **−7.9%** (220 → 202 ms) | **−6.7%** (271 → 253 ms) | 14.1 ms (unchanged) | ~71 tok/s (unchanged) |
| Qwen3.5-4B | **−4.8%** (521 → 496 ms) | **−4.5%** (576 → 550 ms) | 27.1 ms (unchanged) | ~37 tok/s (unchanged) |

**Each optimization's standalone contribution (infer-TTFT, added on top of the previous):**

| Optimization | 0.8B | 2B | 4B |
|---|---:|---:|---:|
| opt-3 (DQ reuse) | −3.8% (−5.9 ms) | −3.2% (−7.0 ms) | −2.0% (−10.6 ms) |
| opt-5 (conv1d tiled) | −1.7% (−2.5 ms) | −1.7% (−3.6 ms) | +0.3% (noise) |
| opt-6 (activation) | −4.8% (−6.9 ms) | −3.3% (−6.9 ms) | −3.0% (−15.6 ms) |

- opt-6 and opt-3 are the biggest; opt-5 shrinks with model size (conv1d is a smaller share on 4B)
- **These are prefill optimizations → TPOT and decode throughput (TPS) are unchanged, as expected**

**Script:**
"Here are the final PA results on the 356H, with the real one-thousand-token input and five hundred twelve output tokens. The recommended stack is opt-3, opt-5, and opt-6. The first-token latency drops ten percent for the small model, eight for the medium, five for the large. End-to-end drops five to eight percent. This is on top of the big PA-versus-SDPA win.

The second table shows each optimization on its own. opt-6, the activation fix, and opt-3, the quantize reuse, give the most — each about three to five percent. opt-5, the convolution fix, gives about one to two percent, and it shrinks on the large model because the convolution is a smaller share there.

And one honest point: look at the last two columns of the first table. The per-token decode time and the throughput are unchanged. That is expected, because all of these target the prefill, the first token. We make the first token appear faster; the tokens-per-second after that stays the same."

---

## Slide 10 — The "50% vs 10%" Question

**On the slide:**
- Profiler showed data movement ≈ 50%, but TTFT improved ~10% — why?
- The percentage's denominator ≠ the TTFT's denominator

| Quantity | 2B value |
|---|---:|
| operator-time sum (profiler denominator) | 65.8 ms |
| infer-TTFT (wall clock) | 157.7 ms |

- Reconcile in milliseconds: opt-1 removed 13.4 ms op-time → 11.6 ms wall-clock (close match)
- Reorder is ~13 ms *absolute* → 13/158 ≈ 8% of TTFT

**Script:**
"I want to answer one question in advance, because people always ask it. In the profiler, the data-movement operators looked like fifty percent, but the actual TTFT only improved around ten percent. Why? Because the two percentages have different denominators. The profiler percentages are against the operator-time sum, about sixty-six milliseconds for the 2B. But the real TTFT is one hundred fifty-eight. So 'Reorder is twenty-one percent' means twenty-one percent of sixty-six, not of one hundred fifty-eight. The correct way is to think in absolute milliseconds. Reorder is about thirteen milliseconds. Remove it, and the wall clock drops about twelve — a close match. Thirteen out of one hundred fifty-eight is about eight percent. The profiler is not wrong; we just read it in milliseconds, not in its own percentages."

---

## Slide 11 — The Ceiling and What's Next

**On the slide:**

| Remaining item | Share | Why we stop here |
|---|---:|---|
| INT4 GEMM (`gemm_kernel`) | ~47–50% | oneDNN XMX micro-kernel, not editable; OpenCL path is 5× slower |
| SSM recurrence (`paged_gated_delta_net_opt`) | ~19–20% | true serial recurrence; already 100% occupancy; rewrite risks bit-exactness |

- Also tested and rejected: larger FC fusion (negative), RMS SLM tuning (memory-bound), eltwise vectorization (bandwidth-bound)
- Near the practical ceiling under "bit-exact + no algorithm rewrite + no oneDNN edits"
- Worthwhile follow-up: upstream opt-3 and the conv1d work as PRs

**Script:**
"So where is the ceiling? After our work, the remaining time is dominated by two things we cannot easily touch. First, the INT4 matrix multiply, about half the time. That is oneDNN's XMX kernel; we tested the editable OpenCL path and it was five times slower, so this is the compute floor. Second, the SSM recurrence, about twenty percent. That is a true serial recurrence, already fully utilized; rewriting it would risk our bit-exact guarantee. We also tried and rejected several smaller ideas and recorded them. Honestly, we are near the practical ceiling under our constraints of staying bit-exact and not rewriting algorithms. The clear next step is to upstream opt-3 and the conv1d work as pull requests, since they are general."

---

## Slide 12 — Summary

**On the slide:**
1. **Use PA** — 34–51% faster than SDPA on both iGPUs (the biggest win)
2. On top of PA: **−10% infer / −7.6% e2e** on 356H (opt-3 + opt-5 + opt-6, 1k input / 512 output)
3. Under SDPA: **−10% to −21%** infer (opt-2 the biggest contributor)
4. These are **prefill (TTFT) optimizations** — decode throughput (TPS) unchanged
5. **opt-3 is the most general** — both backends, any shared-activation model
6. **Everything is bit-exact** — verified with 128-token greedy MD5
7. Lesson: same idea, different hardware — opt-4 helped the big iGPU but hurt the small one → opt-5

**Script:**
"To summarize. Number one, and most important: use PagedAttention. It is thirty-four to fifty-one percent faster than SDPA on both machines. Number two: on top of PA, our GPU-plugin optimizations add another ten percent on inference and about eight percent end-to-end on the 356H. Number three: under the older SDPA backend, the same approach gave ten to twenty-one percent. Number four: these are prefill optimizations, so the decode throughput is unchanged. Number five: the most general optimization is opt-3, which works everywhere. Number six: everything is bit-exact — the output text never changes. And finally, the key lesson: the same optimization must fit the hardware. The fully parallel convolution helped the big iGPU but hurt the small one, so we designed the tiled version that helps both. Thank you. I am happy to take questions."

---

## Appendix — Quick Number Reference (for Q&A)

**Workload:** real 1000-token input, 512 output tokens, greedy decoding, GPU.

**PA vs SDPA (infer-TTFT):** 358H −51 / −40 / −34.5%; 356H −50.5 / −38.6 / −35.6% (0.8B / 2B / 4B)

**PA vs SDPA full metrics (356H, official runner, n=10):**
- e2e-TTFT: 348→194 / 399→264 / 844→564 ms (−44 / −34 / −33%)
- TPOT: 11.6→8.5 / 16.2→13.8 / 32.2→26.5 ms (−27 / −15 / −18%)
- Decode TPS: 86→118 / 62→73 / 31→38 tok/s (+38 / +17 / +22%)

**SDPA total (infer-TTFT):** 358H −20.8 / −16.1 / −12.6%; 356H −14.1 / −11.2 / −10.0%
- e2e-TTFT 356H: −12.3 / −9.9 / −9.3%; SDPA baseline decode TPOT ~11.6 / 16.2 / 32.2 ms, TPS ~86 / 62 / 31 tok/s (unchanged by opts)

**PA stack (356H, opt-3+opt-5+opt-6):**
- infer-TTFT: 153→138 (−10.0%) / 220→202 (−7.9%) / 521→496 (−4.8%)
- e2e-TTFT: 205→189 (−7.6%) / 271→253 (−6.7%) / 576→550 (−4.5%)
- TPOT ~8.7 / 14.1 / 27.1 ms, Decode TPS ~115 / 71 / 37 tok/s — unchanged by the optimizations

**conv1d kernel (per iter):** 356H serial 14.1 ms (60% occ) → opt-4 24.7 ms (100%, memory-bound, rejected) → opt-5 10.5 ms (100%); 358H serial 66 ms (20%) → opt-4 45 ms (100%, helps there)

**opt-6 kernel effect (356H VTune):** activation ~80 → ~23 units (−71%)

**Per-optimization SDPA contribution (358H):** opt-1 −5 to −9%, opt-2 −7 to −10% (biggest), opt-3 −1 to −3%

**Fixed host overhead (e2e − infer):** ~45 ms (not tokenization; tokenization is ~0.7 ms)

**Ceiling:** gemm_kernel ~47–50% (oneDNN XMX, not editable), SSM recurrence ~19–20% (serial, 100% occ)
