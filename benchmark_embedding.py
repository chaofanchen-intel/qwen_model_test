#!/usr/bin/env python3
"""
OpenVINO GenAI text-embedding benchmark for Qwen3-Embedding models.

Uses openvino_genai.TextEmbeddingPipeline. Distilled from
openvino.genai/tools/llm_bench/task/text_embeddings.py (the GenAI embed path),
plus the shared monitor / report / CLI infrastructure under tests/lib/.

Qwen3-Embedding is a *decoder* used as an encoder: the model card recommends
LAST_TOKEN pooling with left padding, an instruction-prefixed query, and L2
normalization. Those are the defaults here; override with --pooling-type /
--padding-side / --query-instruction / --no-normalize.

Per-iteration metrics: latency, e2e (incl. host tokenization accounting),
throughput in documents/s and tokens/s. There is no TTFT/TPOT — embedding is a
single forward pass with no decode loop.

Inputs:  --prompt <text>  OR  --prompt-file <jsonl ...> [--prompt-index ...]
         OR  --input-tokens <N>
Modes:   --mode documents (batch) | query (single) | both
Outputs: append-mode CSV + JSON (paths default to results/embedding.{csv,json}).
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
    add_embedding_args, add_encoder_common_args, add_io_args, add_monitor_args,
    detect_precision, ensure_sudo, load_extra_ov_config,
)
from lib.metrics import (
    EmbedBenchmarkResult, attach_memory_metrics, fill_encoder_stats,
    fill_memory_stats,
)
from lib.monitors import MemoryUsageMonitor, MonitorBundle
from lib.prompts import build_prompt_from_token_target, load_prompts
from lib.reports import print_embed_summary, write_embed_csv, write_json
from lib.embed_runner import aggregate_encoder, count_tokens, run_embed_iteration


_POOLING = {
    "cls": "CLS",
    "mean": "MEAN",
    "last_token": "LAST_TOKEN",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="OpenVINO GenAI text-embedding benchmark for Qwen3-Embedding models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_encoder_common_args(p)
    add_embedding_args(p)
    add_io_args(p, default_csv="results/embedding.csv", default_json="results/embedding.json")
    add_monitor_args(p)
    return p.parse_args()


def build_config(args) -> "ov_genai.TextEmbeddingPipeline.Config":
    cfg = ov_genai.TextEmbeddingPipeline.Config()
    cfg.pooling_type = getattr(ov_genai.TextEmbeddingPipeline.PoolingType, _POOLING[args.pooling_type])
    cfg.normalize = args.normalize
    if args.max_length is not None:
        cfg.max_length = args.max_length
    if args.padding_side:
        cfg.padding_side = args.padding_side
    if args.pad_to_max_length:
        cfg.pad_to_max_length = True
    if args.query_instruction is not None:
        cfg.query_instruction = args.query_instruction
    if args.embed_instruction is not None:
        cfg.embed_instruction = args.embed_instruction
    return cfg


def resolve_texts(args, tokenizer) -> list[tuple[int, str]]:
    if args.input_tokens is not None:
        if args.prompt is not None or args.prompt_file:
            sys.exit("ERROR: --input-tokens is mutually exclusive with --prompt / --prompt-file")
        if args.prompt_index:
            sys.exit("ERROR: --prompt-index only applies to --prompt-file")
        prompt, actual = build_prompt_from_token_target(
            args.input_tokens, lambda t: count_tokens(tokenizer, t), args.input_base_text
        )
        print(f"  Generated text from --input-tokens: target={args.input_tokens}, actual={actual}")
        return [(0, prompt)]
    return load_prompts(args.prompt, args.prompt_file, args.prompt_index)


def build_pipe_config(args, device: str) -> dict[str, str]:
    cfg: dict[str, str] = {}
    if args.cache_dir is not None:
        cfg["CACHE_DIR"] = args.cache_dir
    if args.num_streams is not None:
        cfg["NUM_STREAMS"] = str(args.num_streams)
    if args.load_config:
        cfg.update(load_extra_ov_config(args.load_config))
    return cfg


def run_for_model_device(model_dir: Path, device: str, args,
                         monitors: MonitorBundle,
                         memory_monitor: MemoryUsageMonitor) -> list[EmbedBenchmarkResult]:
    print(f"\n{'#' * 80}")
    print(f"  Model: {model_dir.name} | Device: {device.upper()}")
    print(f"  Backend: openvino_genai.TextEmbeddingPipeline")
    print(f"{'#' * 80}")

    pipe_config = build_pipe_config(args, device)
    print(f"  Plugin config: {pipe_config or '(empty)'}")
    cfg = build_config(args)

    # Standalone tokenizer for token accounting (pipeline exposes no get_tokenizer).
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
        pipe = ov_genai.TextEmbeddingPipeline(str(model_dir), device.upper(), cfg, **pipe_config)
    except Exception as e:
        print(f"  ERROR: failed to construct TextEmbeddingPipeline on {device}: {e}")
        return []
    compile_time = time.perf_counter() - t0
    compile_memory = memory_monitor.summary()
    print(f"  Compiled in {compile_time:.2f}s")

    texts = resolve_texts(args, tokenizer)
    print(f"  Texts loaded: {len(texts)} (indices: {[i for i, _ in texts]})")

    precision = detect_precision(model_dir)
    ov_ver = ov.get_version() if hasattr(ov, "get_version") else ""
    ov_genai_ver = ov_genai.__version__ if hasattr(ov_genai, "__version__") else ""
    run_started = datetime.datetime.now().isoformat()

    modes = ["documents", "query"] if args.mode == "both" else [args.mode]

    results: list[EmbedBenchmarkResult] = []
    for prompt_idx, raw_text in texts:
        for mode in modes:
            # In documents mode we replicate the text up to batch_size to measure
            # batched throughput; query mode is always a single text.
            if mode == "documents":
                batch_texts = [raw_text] * max(1, args.batch_size)
            else:
                batch_texts = [raw_text]

            per_text_tokens = count_tokens(tokenizer, raw_text)
            print(f"\n  [P{prompt_idx}] mode={mode} batch={len(batch_texts)} "
                  f"per_text_tokens={per_text_tokens}  num_iters={args.num_iters} (+1 warmup)")

            monitors.reset()
            all_iters = []
            for num in range(args.num_iters + 1):
                memory_monitor.reset()
                it = run_embed_iteration(
                    pipe=pipe, texts=batch_texts, mode=mode, iter_idx=num,
                    is_warmup=(num == 0), prompt_idx=prompt_idx, tokenizer=tokenizer,
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
            result = EmbedBenchmarkResult(
                model_dir=str(model_dir), model_name=model_dir.name,
                framework="ov", pipeline="TextEmbeddingPipeline", device=device.upper(),
                precision=precision, mode=mode,
                batch_size=len(batch_texts), pooling_type=args.pooling_type,
                normalize=args.normalize,
                max_length=args.max_length or 0,
                padding_side=args.padding_side or "",
                pad_to_max_length=args.pad_to_max_length,
                query_instruction=args.query_instruction or "",
                embed_instruction=args.embed_instruction or "",
                prompt_idx=prompt_idx, input_tokens=rep.input_tokens,
                total_input_tokens=rep.total_input_tokens,
                embedding_dim=rep.embedding_dim,
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
            print_embed_summary(result)
            results.append(result)

    del pipe
    gc.collect()
    return results


def main():
    args = parse_args()
    np.random.seed(args.seed)

    if args.prompt is None and not args.prompt_file and args.input_tokens is None:
        print("  [info] no --prompt / --prompt-file / --input-tokens; using a built-in default text")
        args.prompt = (
            "Given a web search query, retrieve relevant passages that answer the query. "
            "Query: What is the capital of France and what is it famous for?"
        )

    model_dirs = [Path(m) for m in args.model_dir]
    for md in model_dirs:
        if not md.exists():
            sys.exit(f"ERROR: model directory not found: {md}")

    if not args.no_monitor:
        ensure_sudo()

    print("=" * 80)
    print("  Qwen3-Embedding OpenVINO GenAI Text-Embedding Benchmark")
    print("=" * 80)
    print(f"  Models:        {[md.name for md in model_dirs]}")
    print(f"  Devices:       {args.device}")
    print(f"  Mode:          {args.mode}")
    print(f"  Pooling:       {args.pooling_type}")
    print(f"  Normalize:     {args.normalize}")
    print(f"  Padding side:  {args.padding_side or '(default)'}")
    print(f"  Max length:    {args.max_length if args.max_length is not None else '(pipeline default)'}")
    print(f"  Batch size:    {args.batch_size}")
    print(f"  num_iters:     {args.num_iters} (+1 warm-up)")
    if args.input_tokens is not None:
        print(f"  Input tokens:  target {args.input_tokens}")
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

    all_results: list[EmbedBenchmarkResult] = []
    first_run = True
    for md in model_dirs:
        for dev in args.device:
            if not first_run and args.cooldown_between_runs > 0:
                print(f"\n  [cooldown] sleeping {args.cooldown_between_runs}s before next run...")
                time.sleep(args.cooldown_between_runs)
            first_run = False
            try:
                all_results.extend(run_for_model_device(md, dev, args, monitors, memory_monitor))
            except Exception as e:
                print(f"  ERROR running {md.name} on {dev}: {e}")
                traceback.print_exc()

    monitors.stop()
    memory_monitor.stop()

    if not all_results:
        print("\nNo results collected.")
        return

    if args.csv_out:
        write_embed_csv(Path(args.csv_out), all_results)
    if args.json_out:
        write_json(Path(args.json_out), all_results)

    print(f"\n  Total runs: {len(all_results)}")
    print("  Done.")


if __name__ == "__main__":
    main()
