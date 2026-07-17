#!/usr/bin/env python3
"""
OpenVINO GenAI vision-language benchmark for Qwen3.5 VLM models.

Uses ov_genai.VLMPipeline by default on CPU/GPU/NPU. An explicit
--optimum-fallback flag keeps the old optimum.intel OVModelForVisualCausalLM
path available for debugging regressions.

Same CLI shape as benchmark_qwen3.py — adds --image for multimodal runs.

Inputs:  --prompt <text>  OR  --prompt-file <jsonl ...> [--prompt-index ...]
         [--image <path>]   (multimodal mode if set)
Outputs: append-mode CSV + JSON (paths default to results/qwen3_5.{csv,json}).
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
from typing import Optional

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
    add_common_gen_args, add_io_args, add_monitor_args, detect_precision,
    ensure_sudo, load_extra_ov_config,
)
from lib.metrics import GenBenchmarkResult, attach_memory_metrics, fill_gen_stats, fill_memory_stats
from lib.monitors import MemoryUsageMonitor, MonitorBundle
from lib.prompts import build_prompt_from_token_target, load_prompts
from lib.reports import print_gen_summary, write_gen_csv, write_json
from lib.genai_runner import (
    apply_chat_template_safe, aggregate, build_generation_config,
    permute_vlm_prompt, run_optimum_vlm_iteration, run_vlm_iteration,
)


# =============================================================================
# Image loaders
# =============================================================================

def load_image_tensor(image_path: str) -> "ov.Tensor":
    """For VLMPipeline (ov.Tensor with HWC uint8)."""
    from PIL import Image as PILImage
    img = PILImage.open(image_path).convert("RGB")
    return ov.Tensor(np.array(img))


def load_image_pil(image_path: str):
    """For optimum.intel OVModelForVisualCausalLM (PIL.Image)."""
    from PIL import Image as PILImage
    return PILImage.open(image_path).convert("RGB")


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="OpenVINO GenAI vision-language benchmark for Qwen3.5 VLMs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_gen_args(p)
    # Permutation defaults ON (via add_common_gen_args' --disable-prompt-permutation,
    # default False), matching the official visual_text_gen path. The VLM permute is a
    # prompt-prefix mutation (see lib/genai_runner.permute_vlm_prompt); pass
    # --disable-prompt-permutation for a cache-warm scenario.
    p.add_argument("--image", default=None,
                   help="Path to image file. Activates multimodal mode.")
    p.add_argument("--optimum-fallback", action="store_true",
                   help="Use optimum.intel OVModelForVisualCausalLM on GPU instead of GenAI VLMPipeline (legacy workaround).")
    p.add_argument("--strict-device", action="store_true", help=argparse.SUPPRESS)
    add_io_args(p, default_csv="results/qwen3_5.csv", default_json="results/qwen3_5.json")
    add_monitor_args(p)
    return p.parse_args()


# =============================================================================
# Per (model, device) runners
# =============================================================================

def _common_pipe_config(args) -> dict[str, str]:
    cfg: dict[str, str] = {}
    if args.cache_dir is not None:
        cfg["CACHE_DIR"] = args.cache_dir
    if args.num_streams is not None:
        cfg["NUM_STREAMS"] = str(args.num_streams)
    if args.load_config:
        cfg.update(load_extra_ov_config(args.load_config))
    return cfg


def _run_vlmpipeline_path(model_dir: Path, device: str, args, monitors,
                          memory_monitor: MemoryUsageMonitor,
                          image_tensor: Optional["ov.Tensor"]) -> list[GenBenchmarkResult]:
    pipe_config = _common_pipe_config(args)
    # GPU defaults: LATENCY hint + PA attention backend (match the Qwen3 dense path).
    # Each only injected when not already set via -lc/--load-config, so both stay
    # overridable, e.g. -lc '{"ATTENTION_BACKEND":"SDPA"}'.
    if device.upper() != "NPU":
        pipe_config.setdefault("PERFORMANCE_HINT", "LATENCY")
        pipe_config.setdefault("ATTENTION_BACKEND", "PA")
    print(f"  Plugin config: {pipe_config or '(empty)'}")
    initial_memory = memory_monitor.current()
    memory_monitor.reset()
    print(f"  Compiling VLMPipeline...")
    t0 = time.perf_counter()
    try:
        pipe = ov_genai.VLMPipeline(str(model_dir), device.upper(), **pipe_config)
    except Exception as e:
        print(f"  ERROR: failed to construct VLMPipeline on {device}: {e}")
        return []
    compile_time = time.perf_counter() - t0
    compile_memory = memory_monitor.summary()
    print(f"  Compiled in {compile_time:.2f}s")

    tokenizer = pipe.get_tokenizer()
    base_gen_config = pipe.get_generation_config() if hasattr(pipe, "get_generation_config") else None
    images = [image_tensor] if image_tensor is not None else None

    def count_final_tokens(raw_prompt: str) -> int:
        prompt_text = apply_chat_template_safe(
            tokenizer, raw_prompt, args.apply_chat_template
        )
        if not args.disable_prompt_permutation:
            prompt_text = permute_vlm_prompt(prompt_text, 1)
        return _genai_count_tokens(tokenizer, prompt_text)

    results = _run_loop(
        run_one=lambda **kw: run_vlm_iteration(
            pipe=pipe, gen_config=kw["gen_config"], prompt_text=kw["prompt_text"],
            iter_idx=kw["iter_idx"], is_warmup=kw["is_warmup"],
            prompt_idx=kw["prompt_idx"], images=images, permute=kw["permute"],
        ),
        model_dir=model_dir, device=device, args=args, monitors=monitors,
        memory_monitor=memory_monitor,
        tokenizer=tokenizer, base_gen_config=base_gen_config,
        compile_time=compile_time, initial_memory=initial_memory,
        compile_memory=compile_memory,
        pipeline_label="VLMPipeline",
        mode="multimodal" if images else "text-only",
        image_path=args.image or "",
        prompt_count_fn=count_final_tokens,
    )
    return results, pipe


def _run_optimum_vlm_path(model_dir: Path, device: str, args, monitors,
                          memory_monitor: MemoryUsageMonitor,
                          image_pil) -> list[GenBenchmarkResult]:
    try:
        from optimum.intel.openvino import OVModelForVisualCausalLM
        from transformers import AutoProcessor
    except ImportError:
        print("  ERROR: optimum-intel not installed. Required for GPU fallback. "
              "pip install optimum[openvino]")
        return []

    pipe_config = _common_pipe_config(args)
    print(f"  Plugin config: {pipe_config or '(empty)'}")
    initial_memory = memory_monitor.current()
    memory_monitor.reset()
    print(f"  Loading OVModelForVisualCausalLM on {device}...")
    t0 = time.perf_counter()
    try:
        model = OVModelForVisualCausalLM.from_pretrained(
            str(model_dir), device=device, ov_config=pipe_config, trust_remote_code=True,
        )
    except Exception as e:
        print(f"  ERROR: failed to load model on {device}: {e}")
        return []
    processor = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=True)
    compile_time = time.perf_counter() - t0
    compile_memory = memory_monitor.summary()
    print(f"  Loaded in {compile_time:.2f}s")

    tokenizer = processor.tokenizer

    def count_final_tokens(raw_prompt: str) -> int:
        effective_prompt = (permute_vlm_prompt(raw_prompt, 1)
                            if not args.disable_prompt_permutation else raw_prompt)
        if image_pil is not None:
            messages = [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": effective_prompt},
            ]}]
        else:
            messages = [{"role": "user", "content": [
                {"type": "text", "text": effective_prompt},
            ]}]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return _hf_count_tokens(tokenizer, text)

    def runner(**kw):
        return run_optimum_vlm_iteration(
            model=model, processor=processor,
            prompt_text=kw["prompt_text"], image=image_pil,
            max_new_tokens=args.infer_count,
            iter_idx=kw["iter_idx"], is_warmup=kw["is_warmup"],
            prompt_idx=kw["prompt_idx"], permute=kw["permute"],
            exact_output_tokens=args.exact_output_tokens,
        )

    results = _run_loop(
        run_one=runner, model_dir=model_dir, device=device, args=args,
        monitors=monitors, memory_monitor=memory_monitor,
        tokenizer=None, base_gen_config=None,
        compile_time=compile_time, initial_memory=initial_memory,
        compile_memory=compile_memory,
        pipeline_label="OptimumVLM",
        mode="multimodal" if image_pil is not None else "text-only",
        image_path=args.image or "",
        prompt_count_fn=count_final_tokens,
    )
    del model
    del processor
    gc.collect()
    return results


def _hf_count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False).input_ids)


def _genai_count_tokens(tokenizer, text: str) -> int:
    return int(tokenizer.encode(text).input_ids.shape[-1])


def resolve_prompts(args, count_fn=None) -> list[tuple[int, str]]:
    # Explicit --prompt / --prompt-file always win over the default --input-tokens.
    # (input_tokens now defaults to 1000; only used as the source when no explicit
    # prompt was given. --input-tokens 0 / <=0 falls back to --prompt/--prompt-file.)
    if args.prompt is not None or args.prompt_file:
        return load_prompts(args.prompt, args.prompt_file, args.prompt_index)
    if not args.input_tokens or args.input_tokens <= 0:
        return load_prompts(args.prompt, args.prompt_file, args.prompt_index)

    if args.prompt_index:
        sys.exit("ERROR: --prompt-index only applies to --prompt-file")
    if count_fn is None:
        sys.exit("ERROR: --input-tokens requires a tokenizer-backed count function")

    prompt, actual = build_prompt_from_token_target(
        args.input_tokens, count_fn, args.input_base_text
    )
    print(f"  Generated prompt from --input-tokens: target={args.input_tokens}, actual={actual}")
    return [(0, prompt)]


def _run_loop(*, run_one, model_dir: Path, device: str, args, monitors,
              memory_monitor: MemoryUsageMonitor,
              tokenizer, base_gen_config, compile_time: float,
              initial_memory: dict, compile_memory: dict,
              pipeline_label: str, mode: str, image_path: str,
              prompt_count_fn=None) -> list[GenBenchmarkResult]:
    """Generic prompt × iteration loop shared by both backends."""
    prompts = resolve_prompts(args, prompt_count_fn)
    print(f"  Prompts loaded: {len(prompts)} (indices: {[i for i, _ in prompts]})")

    precision = detect_precision(model_dir)
    ov_ver = ov.get_version() if hasattr(ov, "get_version") else ""
    ov_genai_ver = ov_genai.__version__ if hasattr(ov_genai, "__version__") else ""
    run_started = datetime.datetime.now().isoformat()
    permute = not args.disable_prompt_permutation

    is_optimum = (pipeline_label == "OptimumVLM")
    results: list[GenBenchmarkResult] = []

    for prompt_idx, raw_prompt in prompts:
        # Chat templating: VLMPipeline path uses GenAI tokenizer; Optimum path
        # builds its own messages structure inside run_optimum_vlm_iteration, so
        # don't double-template here.
        if not is_optimum and tokenizer is not None:
            prompt_text = apply_chat_template_safe(
                tokenizer, raw_prompt, args.apply_chat_template
            )
        else:
            prompt_text = raw_prompt

        # Count tokens for the report (best-effort).
        try:
            if is_optimum:
                in_tokens = 0
            elif tokenizer is not None:
                in_tokens = _genai_count_tokens(tokenizer, prompt_text)
            else:
                in_tokens = 0
        except Exception:
            in_tokens = 0

        token_label = "raw_text_tokens" if pipeline_label == "VLMPipeline" else "input_tokens"
        print(f"\n  [P{prompt_idx}] {token_label}={in_tokens}  "
              f"infer_count(max_new)={args.infer_count}  num_iters={args.num_iters} (+1 warmup)  "
              f"permute={permute}")

        monitors.reset()
        gen_config = build_generation_config(
            infer_count=args.infer_count, num_beams=args.num_beams,
            seed=args.seed, base_config=base_gen_config,
            exact_output_tokens=args.exact_output_tokens,
            disable_config_chat_template=(pipeline_label != "VLMPipeline"),
        )

        all_iters = []
        for num in range(args.num_iters + 1):
            memory_monitor.reset()
            it = run_one(
                prompt_text=prompt_text, gen_config=gen_config,
                iter_idx=num, is_warmup=(num == 0),
                prompt_idx=prompt_idx, permute=permute,
            )
            iteration_memory = memory_monitor.summary()
            if it is not None:
                attach_memory_metrics(it, iteration_memory)
                all_iters.append(it)
                # Optimum path doesn't have access to the GenAI tokenizer, so use
                # the streamer-observed input length.
                if in_tokens == 0 and it.input_tokens > 0:
                    in_tokens = it.input_tokens

        actual_input_tokens = [it.input_tokens for it in all_iters if not it.is_warmup]
        if not actual_input_tokens:
            actual_input_tokens = [it.input_tokens for it in all_iters]
        if actual_input_tokens:
            in_tokens = actual_input_tokens[0]
            if len(set(actual_input_tokens)) > 1:
                print(f"  [warn] measured input token counts varied across iterations: "
                      f"{sorted(set(actual_input_tokens))}; reporting first measured value")

        summary = monitors.summary()
        result = GenBenchmarkResult(
            model_dir=str(model_dir), model_name=model_dir.name,
            framework="ov", pipeline=pipeline_label, device=device.upper(),
            precision=precision, mode=mode,
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
            image_path=image_path,
            openvino_version=ov_ver, openvino_genai_version=ov_genai_ver,
            run_started_at=run_started,
        )
        fill_gen_stats(result, aggregate(all_iters))
        fill_memory_stats(result, initial_memory, compile_memory, all_iters)
        print_gen_summary(result)
        results.append(result)

    return results


def run_for_model_device(model_dir: Path, device: str, args, monitors: MonitorBundle,
                         memory_monitor: MemoryUsageMonitor,
                         image_tensor: Optional["ov.Tensor"],
                         image_pil) -> list[GenBenchmarkResult]:
    use_gpu_fallback = (
        device.upper() == "GPU" and args.optimum_fallback and not args.strict_device
    )

    print(f"\n{'#' * 80}")
    print(f"  Model: {model_dir.name} | Device: {device.upper()} | "
          f"Mode: {'multimodal' if image_tensor is not None else 'text-only'}")
    if use_gpu_fallback:
        print(f"  Backend: optimum.intel OVModelForVisualCausalLM (explicit GPU fallback)")
    else:
        print(f"  Backend: openvino_genai.VLMPipeline")
    print(f"{'#' * 80}")

    if use_gpu_fallback:
        return _run_optimum_vlm_path(model_dir, device, args, monitors, memory_monitor, image_pil)

    results, pipe = _run_vlmpipeline_path(model_dir, device, args, monitors, memory_monitor, image_tensor)
    del pipe
    gc.collect()
    return results


def main():
    args = parse_args()
    np.random.seed(args.seed)

    if args.strict_device:
        print("  [warn] --strict-device is deprecated; GenAI VLMPipeline is now the default backend.")
        if args.optimum_fallback:
            print("  [warn] --strict-device overrides --optimum-fallback.")

    # An image VLM run needs a real image+prompt, so --image forces the inline default
    # regardless of --input-tokens (a synthetic text prompt makes no sense there).
    if args.image and args.prompt is None and not args.prompt_file:
        args.prompt = ("Describe this image in detail, including objects, colors, "
                       "spatial relationships, and any visible text.")
        print(f"  [info] --image with no --prompt; using built-in image default")
    elif args.prompt is None and not args.prompt_file and (not args.input_tokens or args.input_tokens <= 0):
        args.prompt = ("Explain the key concepts of machine learning including "
                       "supervised, unsupervised, and reinforcement learning.")
        print(f"  [info] no --prompt / --prompt-file and --input-tokens<=0; using built-in default")

    model_dirs = [Path(m) for m in args.model_dir]
    for md in model_dirs:
        if not md.exists():
            sys.exit(f"ERROR: model directory not found: {md}")

    image_tensor = None
    image_pil = None
    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            sys.exit(f"ERROR: image file not found: {image_path}")
        image_tensor = load_image_tensor(str(image_path))
        image_pil = load_image_pil(str(image_path))
        print(f"  [image] {image_path.name} (shape={image_tensor.shape})")

    if not args.no_monitor:
        ensure_sudo()

    print("=" * 80)
    print("  Qwen3.5 OpenVINO GenAI VLM Benchmark")
    print("=" * 80)
    print(f"  Models:        {[md.name for md in model_dirs]}")
    print(f"  Devices:       {args.device}")
    print(f"  Mode:          {'multimodal' if image_tensor is not None else 'text-only'}")
    print(f"  infer_count:   {args.infer_count}")
    print(f"  exact_output:  {args.exact_output_tokens}")
    print(f"  num_iters:     {args.num_iters} (+1 warm-up)")
    print(f"  num_beams:     {args.num_beams}")
    print(f"  chat_template: {args.apply_chat_template}")
    print(f"  permutation:   {'OFF' if args.disable_prompt_permutation else 'ON'}")
    print(f"  optimum_fb:    {args.optimum_fallback and not args.strict_device}")
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

    all_results: list[GenBenchmarkResult] = []
    first_run = True
    for md in model_dirs:
        for dev in args.device:
            if not first_run and args.cooldown_between_runs > 0:
                print(f"\n  [cooldown] sleeping {args.cooldown_between_runs}s before next run...")
                time.sleep(args.cooldown_between_runs)
            first_run = False
            try:
                all_results.extend(run_for_model_device(
                    md, dev, args, monitors, memory_monitor, image_tensor, image_pil
                ))
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
