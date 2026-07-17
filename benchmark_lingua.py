#!/usr/bin/env python3
"""
OpenVINO benchmark for LLMLingua-2 (token classification, BERT/XLM-RoBERTa).

Measures: inference latency, throughput (samples/s and input tokens/s),
compression ratio, compressed token count — across input lengths and target
keep-rates, on CPU/GPU/NPU.

Tokenization happens OUTSIDE the timing window — we measure inference latency
only, matching the pattern of the official llm_bench tools.
"""

from __future__ import annotations

import argparse
import datetime
import gc
import statistics
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np

try:
    import openvino as ov
except ImportError:
    sys.exit("ERROR: openvino not installed. pip install openvino")

from transformers import AutoTokenizer

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from lib.cli import (
    add_io_args, add_monitor_args, detect_precision, ensure_sudo,
    load_extra_ov_config,
)
from lib.metrics import (
    BertBenchmarkResult, BertIterationMetrics, attach_memory_metrics,
    fill_bert_stats, fill_memory_stats,
)
from lib.monitors import MemoryUsageMonitor, MonitorBundle
from lib.reports import print_bert_summary, write_bert_csv, write_json
from lib.stats import compute_stats


# =============================================================================
# Sample text bank for length-targeted prompt construction
# =============================================================================

SAMPLE_TEXTS = [
    "The quarterly earnings report shows a significant increase in revenue compared to the previous fiscal year. "
    "Our cloud computing division has grown by thirty-seven percent, driven primarily by enterprise adoption of "
    "our AI and machine learning services. The board recommends increasing the dividend payout ratio given the "
    "strong cash flow generation and reduced capital expenditure requirements for the upcoming quarter.",
    "In today's meeting we discussed the product roadmap for Q3 and Q4. The engineering team presented their "
    "progress on the new authentication system which uses biometric verification combined with hardware tokens. "
    "Marketing shared the results of the A/B testing campaign showing a twelve percent improvement in conversion "
    "rates when using the redesigned landing page with the new call-to-action button placement.",
    "The research paper presents a novel approach to distributed consensus in Byzantine fault-tolerant systems. "
    "Unlike previous protocols that require three rounds of communication, our method achieves agreement in just "
    "two rounds under normal conditions while maintaining safety guarantees even when up to one-third of nodes "
    "exhibit arbitrary failures. We prove correctness formally using the TLA+ specification language.",
]


def build_input_text(tokenizer, target_tokens: int) -> str:
    base_text = " ".join(SAMPLE_TEXTS)
    ids = tokenizer(base_text, add_special_tokens=False).input_ids
    if not ids:
        sys.exit("ERROR: sample text tokenized to zero tokens")
    if len(ids) < target_tokens:
        repeat = (target_tokens + len(ids) - 1) // len(ids)
        ids = (ids * repeat)[:target_tokens]
    else:
        ids = ids[:target_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True,
                            clean_up_tokenization_spaces=False)


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)


def _model_input_names(compiled_model) -> set[str]:
    return {inp.any_name for inp in compiled_model.inputs}


def _input_len(value: str) -> int:
    v = int(value)
    if not 1 <= v <= 512:
        raise argparse.ArgumentTypeError(
            f"input length must be in [1, 512] (BERT/XLM-RoBERTa max), got {v}"
        )
    return v


# =============================================================================
# Single inference
# =============================================================================

def run_single_inference(compiled_model, tokenizer, text: str,
                         compression_rate: float,
                         model_input_names: set[str],
                         static_seq: int | None = None) -> tuple[float, int, int]:
    # Tokenization is outside the timing window — we measure inference only.
    # static_seq (NPU): pad/truncate to a fixed sequence length so the input matches the
    # statically-reshaped model. real_tokens (from attention_mask) drives the compression
    # calc so right-padding does not pollute the kept-token selection or the reported metrics.
    if static_seq is not None:
        encoded = tokenizer(text, return_tensors="np", truncation=True,
                            max_length=static_seq, padding="max_length")
    else:
        encoded = tokenizer(text, return_tensors="np", truncation=True, max_length=512)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    real_tokens = int(attention_mask.sum()) if static_seq is not None else int(input_ids.shape[1])
    input_tokens = real_tokens

    feed_dict: dict[str, np.ndarray] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    # BERT has token_type_ids; XLM-RoBERTa does not. Inspect the compiled model.
    if "token_type_ids" in model_input_names:
        feed_dict["token_type_ids"] = encoded.get("token_type_ids", np.zeros_like(input_ids))

    t0 = time.perf_counter()
    result = compiled_model(feed_dict)
    t1 = time.perf_counter()
    latency_ms = (t1 - t0) * 1000.0

    logits = result[compiled_model.output(0)]
    keep_probs = _softmax(logits[0], axis=-1)[:, 1]
    content_probs = keep_probs[1:real_tokens - 1]     # exclude [CLS]+[SEP] and any right-padding
    target_keep = max(1, int(len(content_probs) * compression_rate))
    sorted_idx = np.argsort(content_probs)[::-1]
    threshold = content_probs[sorted_idx[min(target_keep - 1, len(content_probs) - 1)]]
    compressed_tokens = int(np.sum(content_probs >= threshold)) + 2  # [CLS]+[SEP]

    return latency_ms, input_tokens, compressed_tokens


# =============================================================================
# Per-config benchmark
# =============================================================================

def run_one_config(compiled_model, tokenizer, text: str, compression_rate: float,
                   warmup: int, iters: int,
                   model_input_names: set[str],
                   memory_monitor: MemoryUsageMonitor,
                   static_seq: int | None = None) -> list[BertIterationMetrics]:
    for _ in range(warmup):
        run_single_inference(compiled_model, tokenizer, text, compression_rate, model_input_names, static_seq)

    out: list[BertIterationMetrics] = []
    for i in range(iters):
        memory_monitor.reset()
        latency_ms, in_tokens, compressed = run_single_inference(
            compiled_model, tokenizer, text, compression_rate, model_input_names, static_seq
        )
        iteration_memory = memory_monitor.summary()
        metric = BertIterationMetrics(
            iteration=i + 1, input_tokens=in_tokens, latency_ms=latency_ms,
            throughput_samples_per_s=1000.0 / latency_ms if latency_ms > 0 else 0.0,
            throughput_tokens_per_s=in_tokens * 1000.0 / latency_ms if latency_ms > 0 else 0.0,
            compressed_tokens=compressed,
            compression_ratio=compressed / in_tokens if in_tokens else 1.0,
        )
        attach_memory_metrics(metric, iteration_memory)
        out.append(metric)
    return out


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="OpenVINO benchmark for LLMLingua-2 token classification models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-m", "--model-dir", required=True, nargs="+",
                   help="Path(s) to OV model directory (must contain openvino_model.xml).")
    p.add_argument("-d", "--device", default=["CPU"], nargs="+",
                   help="Inference device(s).")
    p.add_argument("--input-lengths", type=_input_len, nargs="+",
                   default=[64, 128, 256, 512],
                   help="List of input token lengths to test (each in [1, 512]).")
    p.add_argument("--compression-rates", type=float, nargs="+",
                   default=[0.3, 0.5, 0.7],
                   help="Target keep ratios (e.g. 0.5 = keep 50%% of tokens).")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("-lc", "--load-config", "--load_config", dest="load_config", default=None,
                   help="Path to JSON file with extra OV plugin properties.")
    p.add_argument("--cache-dir", default="",
                   help="OV CACHE_DIR; '' disables (default).")
    p.add_argument("--num-streams", default=None,
                   help="OV NUM_STREAMS property.")
    p.add_argument("--cooldown-between-runs", type=int, default=5,
                   help="Seconds between (model, device) pairs.")
    add_io_args(p, default_csv="results/lingua.csv", default_json="results/lingua.json")
    add_monitor_args(p)
    return p.parse_args()


# =============================================================================
# Per (model, device) runner
# =============================================================================

def run_for_model_device(model_dir: Path, device: str, args, tokenizer,
                         monitors: MonitorBundle,
                         memory_monitor: MemoryUsageMonitor) -> list[BertBenchmarkResult]:
    print(f"\n{'#' * 80}")
    print(f"  Model: {model_dir.name} | Device: {device.upper()}")
    print(f"{'#' * 80}")

    config: dict[str, str] = {}
    if args.cache_dir is not None:
        config["CACHE_DIR"] = args.cache_dir
    if args.num_streams is not None:
        config["NUM_STREAMS"] = str(args.num_streams)
    if args.load_config:
        config.update(load_extra_ov_config(args.load_config))
    print(f"  Plugin config: {config or '(empty)'}")

    initial_memory = memory_monitor.current()
    memory_monitor.reset()

    # NPU rejects the dynamic-shape ([?,?]) LLMLingua IR. On NPU we reshape the model to a static
    # [1, static_seq] per input length and (re)compile inside the length loop. On CPU/GPU we compile
    # the dynamic model once up front (original behavior). static_seq = input_len + 2 ([CLS]+[SEP]),
    # capped at 512; inputs are right-padded to it and the attention_mask drives the real-token math.
    is_npu = device.upper() == "NPU"
    core = ov.Core()
    try:
        model = core.read_model(str(model_dir / "openvino_model.xml"))
    except Exception as e:
        print(f"  ERROR: failed to read model: {e}")
        return []

    compiled = None
    compile_time = 0.0
    compile_memory = memory_monitor.summary()
    input_names = {inp.get_any_name() for inp in model.inputs}
    if not is_npu:
        print(f"  Compiling model on {device}...")
        t0 = time.perf_counter()
        try:
            compiled = core.compile_model(model, device, config)
        except Exception as e:
            print(f"  ERROR: failed to compile model on {device}: {e}")
            return []
        compile_time = time.perf_counter() - t0
        compile_memory = memory_monitor.summary()
        input_names = _model_input_names(compiled)
        print(f"  Compiled in {compile_time:.2f}s")
    else:
        print(f"  NPU: will reshape+compile a static model per input length.")

    precision = detect_precision(model_dir)
    ov_ver = ov.get_version() if hasattr(ov, "get_version") else ""
    run_started = datetime.datetime.now().isoformat()

    results: list[BertBenchmarkResult] = []
    for input_len in args.input_lengths:
        clamped = min(input_len, 510)   # leave room for [CLS]+[SEP]
        text = build_input_text(tokenizer, clamped)
        actual_tokens = len(tokenizer(text, add_special_tokens=True).input_ids)
        print(f"\n  Input length target: {input_len} tokens (actual: {actual_tokens})")

        static_seq = None
        if is_npu:
            static_seq = min(clamped + 2, 512)   # [CLS]+[SEP]; cap at BERT/XLM max 512
            print(f"    NPU: compiling static [1,{static_seq}] ... ", end="", flush=True)
            t0 = time.perf_counter()
            try:
                static_model = core.read_model(str(model_dir / "openvino_model.xml"))
                static_model.reshape({inp.get_any_name(): ov.PartialShape([1, static_seq])
                                      for inp in static_model.inputs})
                compiled = core.compile_model(static_model, device, config)
            except Exception as e:
                print(f"COMPILE FAILED: {str(e)[:200]}")
                continue
            compile_time = time.perf_counter() - t0
            compile_memory = memory_monitor.summary()
            input_names = _model_input_names(compiled)
            print(f"{compile_time:.1f}s")

        for comp_rate in args.compression_rates:
            print(f"    Compression rate: {comp_rate:.0%} keep ... ", end="", flush=True)
            monitors.reset()

            try:
                iterations = run_one_config(
                    compiled, tokenizer, text, comp_rate,
                    args.warmup, args.iters, input_names, memory_monitor,
                    static_seq=static_seq,
                )
            except Exception as e:
                msg = str(e)
                print(f"ERROR: {msg}")
                # An NPU device-lost/hung error leaves the device in a bad state; retrying the
                # remaining compression rates for this shape will just fail again. Skip them.
                if "DEVICE_LOST" in msg or "device hung" in msg or "ZE_RESULT_ERROR" in msg:
                    print("    (NPU device error — skipping remaining compression rates for this length)")
                    break
                continue
            if not iterations:
                print("FAILED")
                continue

            summary = monitors.summary()

            lat_stats = compute_stats([m.latency_ms for m in iterations])
            thr_s = compute_stats([m.throughput_samples_per_s for m in iterations])
            thr_t = compute_stats([m.throughput_tokens_per_s for m in iterations])

            result = BertBenchmarkResult(
                model_dir=str(model_dir), model_name=model_dir.name,
                framework="ov", device=device.upper(), precision=precision,
                input_length=actual_tokens, compression_rate=comp_rate,
                warmup_iters=args.warmup, measure_iters=len(iterations),
                compile_time_s=compile_time,
                iterations=[asdict(m) for m in iterations],
                cpu_util=summary["cpu"], gpu_util=summary["gpu"], npu_util=summary["npu"],
                openvino_version=ov_ver, run_started_at=run_started,
            )
            fill_bert_stats(
                result, lat_stats, thr_s, thr_t,
                avg_input=statistics.fmean([m.input_tokens for m in iterations]),
                avg_compressed=statistics.fmean([m.compressed_tokens for m in iterations]),
                avg_ratio=statistics.fmean([m.compression_ratio for m in iterations]),
            )
            fill_memory_stats(result, initial_memory, compile_memory, iterations)
            results.append(result)
            print(f"Lat={result.latency_mean_ms:.2f}ms  "
                  f"Thr={result.throughput_tokens_mean:.0f}tok/s  "
                  f"Compressed={result.avg_compressed_tokens:.0f}/{result.avg_input_tokens:.0f}"
                  f" ({result.avg_compression_ratio:.0%})")

    for r in results:
        print_bert_summary(r)
    del compiled
    del model
    gc.collect()
    return results


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    model_dirs = [Path(m) for m in args.model_dir]
    for md in model_dirs:
        if not md.exists():
            sys.exit(f"ERROR: model directory not found: {md}")
        if not (md / "openvino_model.xml").exists():
            sys.exit(f"ERROR: openvino_model.xml not found in: {md}")

    if not args.no_monitor:
        ensure_sudo()

    print("  Loading tokenizer (from first model dir)...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dirs[0]))

    print("=" * 80)
    print("  LLMLingua-2 OpenVINO Benchmark")
    print("=" * 80)
    print(f"  Models:         {[md.name for md in model_dirs]}")
    print(f"  Devices:        {args.device}")
    print(f"  Input lengths:  {args.input_lengths}")
    print(f"  Keep ratios:    {args.compression_rates}")
    print(f"  Warmup/Iters:   {args.warmup}/{args.iters}")
    print(f"  Monitor:        {not args.no_monitor}")
    print(f"  Memory mon:     {args.memory_monitor} ({args.memory_monitor_interval} ms)")
    print(f"  CSV out:        {args.csv_out}")
    print(f"  JSON out:       {args.json_out}")

    monitors = MonitorBundle(
        enabled=not args.no_monitor,
        cpu_interval_ms=args.cpu_monitor_interval,
        gpu_interval_ms=args.gpu_monitor_interval,
        npu_interval_ms=args.npu_monitor_interval,
        gpu_command=args.gpu_command,
    )
    monitors.start()
    memory_monitor = MemoryUsageMonitor(
        enabled=args.memory_monitor,
        interval_ms=args.memory_monitor_interval,
    )
    memory_monitor.start()

    all_results: list[BertBenchmarkResult] = []
    first_run = True
    for md in model_dirs:
        for dev in args.device:
            if not first_run and args.cooldown_between_runs > 0:
                print(f"\n  [cooldown] sleeping {args.cooldown_between_runs}s before next run...")
                time.sleep(args.cooldown_between_runs)
            first_run = False
            try:
                all_results.extend(run_for_model_device(md, dev, args, tokenizer, monitors, memory_monitor))
            except Exception as e:
                print(f"  ERROR running {md.name} on {dev}: {e}")
                traceback.print_exc()

    monitors.stop()
    memory_monitor.stop()

    if not all_results:
        print("\nNo results collected.")
        return

    if args.csv_out:
        write_bert_csv(Path(args.csv_out), all_results)
    if args.json_out:
        write_json(Path(args.json_out), all_results)

    print(f"\n  Total runs: {len(all_results)}")
    print("  Done.")


if __name__ == "__main__":
    main()
