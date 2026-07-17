#!/usr/bin/env python3
"""
OpenVINO GenAI text-generation benchmark for Qwen3 LLMs (text-only).

Uses openvino_genai.LLMPipeline. Distilled from
openvino.genai/tools/llm_bench/task/text_generation.py (the GenAI text_gen path),
plus the shared monitor / report / CLI infrastructure under tests/lib/.

Per-iteration metrics: TTFT, TPOT, throughput, prefill / decode / E2E latency.
Inputs:  --prompt <text>  OR  --prompt-file <jsonl ...> [--prompt-index ...].
Outputs: append-mode CSV + JSON (paths default to results/qwen3.{csv,json}).
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

# Allow `python tests/benchmark_qwen3.py` from anywhere.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from lib.cli import (
    add_common_gen_args, add_io_args, add_monitor_args, detect_precision,
    ensure_sudo, load_extra_ov_config,
)
from lib.metrics import GenBenchmarkResult, attach_memory_metrics, fill_gen_stats, fill_memory_stats
from lib.monitors import MemoryUsageMonitor, MonitorBundle
from lib.prompts import build_prompt_from_token_target, load_prompts
from lib.reports import print_gen_summary, write_gen_csv, write_json
from lib.genai_runner import (
    apply_chat_template_safe, aggregate, build_generation_config,
    run_llm_iteration,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="OpenVINO GenAI text-generation benchmark for Qwen3 LLMs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_gen_args(p)
    add_io_args(p, default_csv="results/qwen3.csv", default_json="results/qwen3.json")
    add_monitor_args(p)
    return p.parse_args()


def resolve_prompts(args, tokenizer) -> list[tuple[int, str]]:
    # Explicit --prompt / --prompt-file always win over the default --input-tokens.
    # (input_tokens now defaults to 1000, so we only treat it as the source when the
    # user did NOT pass an explicit prompt; --input-tokens 0 / <=0 also falls back.)
    if args.prompt is not None or args.prompt_file:
        return load_prompts(args.prompt, args.prompt_file, args.prompt_index)
    if not args.input_tokens or args.input_tokens <= 0:
        return load_prompts(args.prompt, args.prompt_file, args.prompt_index)

    if args.prompt_index:
        sys.exit("ERROR: --prompt-index only applies to --prompt-file")

    def count_final_tokens(raw_prompt: str) -> int:
        prompt_text = apply_chat_template_safe(
            tokenizer, raw_prompt, args.apply_chat_template
        )
        return int(tokenizer.encode(prompt_text).input_ids.shape[-1])

    prompt, actual = build_prompt_from_token_target(
        args.input_tokens, count_final_tokens, args.input_base_text
    )
    print(f"  Generated prompt from --input-tokens: target={args.input_tokens}, actual={actual}")
    return [(0, prompt)]


def run_for_model_device(model_dir: Path, device: str, args,
                         monitors: MonitorBundle,
                         memory_monitor: MemoryUsageMonitor) -> list[GenBenchmarkResult]:
    print(f"\n{'#' * 80}")
    print(f"  Model: {model_dir.name} | Device: {device.upper()}")
    print(f"  Backend: openvino_genai.LLMPipeline")
    print(f"{'#' * 80}")

    pipe_config: dict[str, str] = {}
    if args.cache_dir is not None:
        pipe_config["CACHE_DIR"] = args.cache_dir
    if args.num_streams is not None:
        pipe_config["NUM_STREAMS"] = str(args.num_streams)
    if args.load_config:
        try:
            pipe_config.update(load_extra_ov_config(args.load_config))
        except (OSError, json.JSONDecodeError) as e:
            print(f"  ERROR: --load-config: {e}")
            return []
    # GPU defaults: LATENCY hint + PA attention backend. Each is only injected when the
    # user did not already set it via -lc/--load-config, so they stay overridable.
    if device.upper() != "NPU":
        pipe_config.setdefault("PERFORMANCE_HINT", "LATENCY")
        pipe_config.setdefault("ATTENTION_BACKEND", "PA")
    print(f"  Plugin config: {pipe_config or '(empty)'}")

    initial_memory = memory_monitor.current()
    memory_monitor.reset()

    print(f"  Compiling pipeline...")
    t0 = time.perf_counter()
    try:
        pipe = ov_genai.LLMPipeline(str(model_dir), device.upper(), **pipe_config)
    except Exception as e:
        print(f"  ERROR: failed to construct LLMPipeline on {device}: {e}")
        return []
    compile_time = time.perf_counter() - t0
    compile_memory = memory_monitor.summary()
    print(f"  Compiled in {compile_time:.2f}s")

    tokenizer = pipe.get_tokenizer()
    base_gen_config = pipe.get_generation_config() if hasattr(pipe, "get_generation_config") else None

    prompts = resolve_prompts(args, tokenizer)
    print(f"  Prompts loaded: {len(prompts)} (indices: {[i for i, _ in prompts]})")

    precision = detect_precision(model_dir)
    ov_ver = ov.get_version() if hasattr(ov, "get_version") else ""
    ov_genai_ver = ov_genai.__version__ if hasattr(ov_genai, "__version__") else ""
    run_started = datetime.datetime.now().isoformat()
    permute = not args.disable_prompt_permutation

    results: list[GenBenchmarkResult] = []

    for prompt_idx, raw_prompt in prompts:
        prompt_text = apply_chat_template_safe(tokenizer, raw_prompt, args.apply_chat_template)
        in_tokens = int(tokenizer.encode(prompt_text).input_ids.shape[-1])
        print(f"\n  [P{prompt_idx}] input_tokens={in_tokens}  "
              f"infer_count(max_new)={args.infer_count}  num_iters={args.num_iters} (+1 warmup)  "
              f"permute={permute}")

        monitors.reset()
        gen_config = build_generation_config(
            infer_count=args.infer_count, num_beams=args.num_beams,
            seed=args.seed, base_config=base_gen_config,
            exact_output_tokens=args.exact_output_tokens,
        )

        all_iters = []
        for num in range(args.num_iters + 1):
            memory_monitor.reset()
            it = run_llm_iteration(
                pipe=pipe, prompt_text=prompt_text, gen_config=gen_config,
                iter_idx=num, is_warmup=(num == 0), prompt_idx=prompt_idx,
                permute=permute,
            )
            iteration_memory = memory_monitor.summary()
            if it is not None:
                attach_memory_metrics(it, iteration_memory)
                all_iters.append(it)

        summary = monitors.summary()
        result = GenBenchmarkResult(
            model_dir=str(model_dir), model_name=model_dir.name,
            framework="ov", pipeline="LLMPipeline", device=device.upper(),
            precision=precision, mode="text-only",
            num_beams=args.num_beams, batch_size=args.batch_size,
            apply_chat_template=args.apply_chat_template,
            disable_prompt_permutation=args.disable_prompt_permutation,
            prompt_idx=prompt_idx, input_tokens=in_tokens,
            infer_count_arg=args.infer_count,
            exact_output_tokens=args.exact_output_tokens,
            num_iters=args.num_iters,
            measured_iters=sum(1 for it in all_iters if not it.is_warmup),
            compile_time_s=compile_time,
            iterations=[asdict(it) for it in all_iters],
            cpu_util=summary["cpu"], gpu_util=summary["gpu"], npu_util=summary["npu"],
            openvino_version=ov_ver, openvino_genai_version=ov_genai_ver,
            run_started_at=run_started,
        )
        fill_gen_stats(result, aggregate(all_iters))
        fill_memory_stats(result, initial_memory, compile_memory, all_iters)
        print_gen_summary(result)
        results.append(result)

    del pipe
    gc.collect()
    return results


def main():
    args = parse_args()
    np.random.seed(args.seed)

    if args.prompt is None and not args.prompt_file and (not args.input_tokens or args.input_tokens <= 0):
        print("  [info] no --prompt / --prompt-file and --input-tokens<=0; using a built-in default prompt")
        args.prompt = (
            "Explain the key concepts of machine learning including supervised "
            "learning, unsupervised learning, and reinforcement learning."
        )

    model_dirs = [Path(m) for m in args.model_dir]
    for md in model_dirs:
        if not md.exists():
            sys.exit(f"ERROR: model directory not found: {md}")

    if not args.no_monitor:
        ensure_sudo()

    print("=" * 80)
    print("  Qwen3 OpenVINO GenAI Text-Generation Benchmark")
    print("=" * 80)
    print(f"  Models:        {[md.name for md in model_dirs]}")
    print(f"  Devices:       {args.device}")
    print(f"  infer_count:   {args.infer_count}")
    print(f"  exact_output:  {args.exact_output_tokens}")
    print(f"  num_iters:     {args.num_iters} (+1 warm-up)")
    print(f"  num_beams:     {args.num_beams}")
    print(f"  chat_template: {args.apply_chat_template}")
    print(f"  permutation:   {'OFF' if args.disable_prompt_permutation else 'ON'}")
    if args.input_tokens is not None:
        print(f"  Input tokens:  target {args.input_tokens}")
    if args.prompt:
        print(f"  Prompt:        inline ({len(args.prompt)} chars)")
    if args.prompt_file:
        print(f"  Prompt files:  {args.prompt_file}")
        if args.prompt_index:
            print(f"  Indices:       {args.prompt_index}")
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

    all_results: list[GenBenchmarkResult] = []
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
        write_gen_csv(Path(args.csv_out), all_results)
    if args.json_out:
        write_json(Path(args.json_out), all_results)

    print(f"\n  Total runs: {len(all_results)}")
    print("  Done.")


if __name__ == "__main__":
    main()
