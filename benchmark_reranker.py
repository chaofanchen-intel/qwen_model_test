#!/usr/bin/env python3
"""
OpenVINO GenAI text-reranking benchmark for Qwen3-Reranker models.

Uses openvino_genai.TextRerankPipeline. Distilled from
openvino.genai/tools/llm_bench/task/text_reranker.py (the GenAI rerank path),
plus the shared monitor / report / CLI infrastructure under tests/lib/.

TextRerankPipeline.rerank(query, documents) scores every (query, document)
pair in one call and returns the top_n (index, score) tuples. There is no
decode loop, so we measure end-to-end latency and throughput (documents/s and
tokens/s) rather than TTFT/TPOT.

Inputs:  query   via --prompt / --prompt-file [--prompt-index]
         documents via --texts / --texts-file  (defaults to a small built-in set)
Outputs: append-mode CSV + JSON (paths default to results/reranker.{csv,json}).
"""

from __future__ import annotations

import argparse
import datetime
import gc
import json
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np

try:
    import openvino as ov
    import openvino_genai as ov_genai
except ImportError:
    sys.exit("ERROR: openvino_genai not installed. pip install openvino-genai")

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from lib.cli import (
    add_encoder_common_args, add_io_args, add_monitor_args, add_reranker_args,
    detect_precision, ensure_sudo, load_extra_ov_config,
)
from lib.metrics import (
    RerankBenchmarkResult, attach_memory_metrics, fill_encoder_stats,
    fill_memory_stats,
)
from lib.monitors import MemoryUsageMonitor, MonitorBundle
from lib.prompts import load_prompts, load_texts_jsonl
from lib.reports import print_rerank_summary, write_json, write_rerank_csv
from lib.embed_runner import aggregate_encoder, count_tokens, run_rerank_iteration


# Default candidate documents — mirror the official llm_bench text_reranker defaults.
_DEFAULT_DOCUMENTS = [
    "Intel Core Ultra processors incorporate an AI-optimized architecture that supports "
    "new user experiences and the next wave of commercial applications.",
    "Intel Core Ultra processors are designed to provide enhanced performance and "
    "efficiency for a wide range of computing tasks.",
    "The Eiffel Tower is one of the most recognizable landmarks in Paris, France.",
    "Photosynthesis is the process by which plants convert light energy into chemical energy.",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="OpenVINO GenAI text-reranking benchmark for Qwen3-Reranker models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_encoder_common_args(p)
    add_reranker_args(p)
    add_io_args(p, default_csv="results/reranker.csv", default_json="results/reranker.json")
    add_monitor_args(p)
    return p.parse_args()


def build_config(args) -> "ov_genai.TextRerankPipeline.Config":
    cfg = ov_genai.TextRerankPipeline.Config()
    cfg.top_n = args.top_n
    if args.max_length is not None:
        cfg.max_length = args.max_length
    if args.padding_side:
        cfg.padding_side = args.padding_side
    if args.pad_to_max_length:
        cfg.pad_to_max_length = True
    return cfg


def resolve_documents(args) -> list[str]:
    if args.texts and args.texts_file:
        sys.exit("ERROR: --texts and --texts-file are mutually exclusive")
    if args.texts_file:
        docs = load_texts_jsonl(args.texts_file)
    elif args.texts:
        docs = list(args.texts)
    else:
        docs = list(_DEFAULT_DOCUMENTS)
    if not docs:
        sys.exit("ERROR: no candidate documents resolved")

    if args.num_documents is not None and args.num_documents > 0:
        # Replicate/trim to exactly num_documents for a candidate-set-size sweep.
        if len(docs) >= args.num_documents:
            docs = docs[:args.num_documents]
        else:
            reps = (args.num_documents + len(docs) - 1) // len(docs)
            docs = (docs * reps)[:args.num_documents]
    return docs


def resolve_queries(args) -> list[tuple[int, str]]:
    return load_prompts(args.prompt, args.prompt_file, args.prompt_index)


def build_pipe_config(args) -> dict[str, str]:
    cfg: dict[str, str] = {}
    if args.cache_dir is not None:
        cfg["CACHE_DIR"] = args.cache_dir
    if args.num_streams is not None:
        cfg["NUM_STREAMS"] = str(args.num_streams)
    if args.load_config:
        cfg.update(load_extra_ov_config(args.load_config))
    return cfg


def run_for_model_device(model_dir: Path, device: str, args, documents: list[str],
                         monitors: MonitorBundle,
                         memory_monitor: MemoryUsageMonitor) -> list[RerankBenchmarkResult]:
    print(f"\n{'#' * 80}")
    print(f"  Model: {model_dir.name} | Device: {device.upper()}")
    print(f"  Backend: openvino_genai.TextRerankPipeline")
    print(f"{'#' * 80}")

    pipe_config = build_pipe_config(args)
    print(f"  Plugin config: {pipe_config or '(empty)'}")
    cfg = build_config(args)

    try:
        tokenizer = ov_genai.Tokenizer(str(model_dir))
    except Exception as e:
        print(f"  [warn] could not load tokenizer for token accounting: {e}")
        tokenizer = None

    initial_memory = memory_monitor.current()
    memory_monitor.reset()

    print(f"  Compiling pipeline...")
    t0 = time.perf_counter()
    try:
        pipe = ov_genai.TextRerankPipeline(str(model_dir), device.upper(), cfg, **pipe_config)
    except Exception as e:
        print(f"  ERROR: failed to construct TextRerankPipeline on {device}: {e}")
        return []
    compile_time = time.perf_counter() - t0
    compile_memory = memory_monitor.summary()
    print(f"  Compiled in {compile_time:.2f}s")

    queries = resolve_queries(args)
    effective_top_n = min(args.top_n, len(documents))
    print(f"  Queries: {len(queries)}  Documents: {len(documents)}  top_n: {effective_top_n}")

    precision = detect_precision(model_dir)
    ov_ver = ov.get_version() if hasattr(ov, "get_version") else ""
    ov_genai_ver = ov_genai.__version__ if hasattr(ov_genai, "__version__") else ""
    run_started = datetime.datetime.now().isoformat()

    results: list[RerankBenchmarkResult] = []
    for prompt_idx, query in queries:
        q_tokens = count_tokens(tokenizer, query)
        print(f"\n  [P{prompt_idx}] query_tokens={q_tokens} docs={len(documents)}  "
              f"num_iters={args.num_iters} (+1 warmup)")

        monitors.reset()
        all_iters = []
        for num in range(args.num_iters + 1):
            memory_monitor.reset()
            it = run_rerank_iteration(
                pipe=pipe, query=query, documents=documents, top_n=effective_top_n,
                iter_idx=num, is_warmup=(num == 0), prompt_idx=prompt_idx,
                tokenizer=tokenizer,
            )
            iteration_memory = memory_monitor.summary()
            if it is not None:
                attach_memory_metrics(it, iteration_memory)
                all_iters.append(it)

        if not all_iters:
            print("  [warn] no iterations collected; skipping")
            continue

        summary = monitors.summary()
        measured = [it for it in all_iters if not it.is_warmup] or all_iters
        rep = measured[0]
        result = RerankBenchmarkResult(
            model_dir=str(model_dir), model_name=model_dir.name,
            framework="ov", pipeline="TextRerankPipeline", device=device.upper(),
            precision=precision,
            num_documents=len(documents), top_n=effective_top_n,
            max_length=args.max_length or 0,
            padding_side=args.padding_side or "",
            pad_to_max_length=args.pad_to_max_length,
            prompt_idx=prompt_idx, input_tokens=rep.input_tokens,
            total_input_tokens=rep.total_input_tokens,
            num_iters=args.num_iters,
            measured_iters=sum(1 for it in all_iters if not it.is_warmup),
            compile_time_s=compile_time,
            iterations=[asdict(it) for it in all_iters],
            cpu_util=summary["cpu"], gpu_util=summary["gpu"], npu_util=summary["npu"],
            openvino_version=ov_ver, openvino_genai_version=ov_genai_ver,
            run_started_at=run_started,
        )
        fill_encoder_stats(result, aggregate_encoder(all_iters))
        fill_memory_stats(result, initial_memory, compile_memory, all_iters)
        print_rerank_summary(result)
        results.append(result)

    del pipe
    gc.collect()
    return results


def main():
    args = parse_args()
    np.random.seed(args.seed)

    if args.prompt is None and not args.prompt_file:
        print("  [info] no --prompt / --prompt-file query; using a built-in default query")
        args.prompt = "What is the capital of France?"

    model_dirs = [Path(m) for m in args.model_dir]
    for md in model_dirs:
        if not md.exists():
            sys.exit(f"ERROR: model directory not found: {md}")

    documents = resolve_documents(args)

    if not args.no_monitor:
        ensure_sudo()

    print("=" * 80)
    print("  Qwen3-Reranker OpenVINO GenAI Text-Reranking Benchmark")
    print("=" * 80)
    print(f"  Models:        {[md.name for md in model_dirs]}")
    print(f"  Devices:       {args.device}")
    print(f"  Documents:     {len(documents)}")
    print(f"  top_n:         {args.top_n}")
    print(f"  Max length:    {args.max_length if args.max_length is not None else '(pipeline default)'}")
    print(f"  num_iters:     {args.num_iters} (+1 warm-up)")
    print(f"  Monitor:       {not args.no_monitor}")
    print(f"  Memory mon:    {args.memory_monitor} ({args.memory_monitor_interval} ms)")
    print(f"  CSV out:       {args.csv_out}")
    print(f"  JSON out:      {args.json_out}")

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

    all_results: list[RerankBenchmarkResult] = []
    first_run = True
    for md in model_dirs:
        for dev in args.device:
            if not first_run and args.cooldown_between_runs > 0:
                print(f"\n  [cooldown] sleeping {args.cooldown_between_runs}s before next run...")
                time.sleep(args.cooldown_between_runs)
            first_run = False
            try:
                all_results.extend(run_for_model_device(md, dev, args, documents, monitors, memory_monitor))
            except Exception as e:
                print(f"  ERROR running {md.name} on {dev}: {e}")
                traceback.print_exc()

    monitors.stop()
    memory_monitor.stop()

    if not all_results:
        print("\nNo results collected.")
        return

    if args.csv_out:
        write_rerank_csv(Path(args.csv_out), all_results)
    if args.json_out:
        write_json(Path(args.json_out), all_results)

    print(f"\n  Total runs: {len(all_results)}")
    print("  Done.")


if __name__ == "__main__":
    main()
