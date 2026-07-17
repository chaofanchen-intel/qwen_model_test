"""One-iteration helpers for OpenVINO GenAI LLMPipeline / VLMPipeline.

Both runners share:
- `build_generation_config()` — mirrors openvino.genai/llm_bench/.../text_generation.py
- `make_streamer()` — per-token timestamp callback
- `aggregate()` — summarize per-iter metrics into the dict shape consumed by
  metrics.fill_gen_stats
- `apply_chat_template()` helper
- `permute_llm_inputs()` — mutate one non-BOS input token for LLMPipeline (TokenizedInputs path)
- `permute_vlm_prompt()` — text-side mutation for VLMPipeline (str path)

Specifics:
- run_llm_iteration: text-only LLMPipeline with optional permutation.
- run_vlm_iteration: VLMPipeline with prompt + image(s).
- run_optimum_vlm_iteration: HuggingFace-style streamer for the optional
    optimum.intel legacy fallback path used to debug qwen3.5 GPU regressions.
"""

from __future__ import annotations

import datetime
import hashlib
import time
from typing import Optional

import numpy as np

import openvino as ov
import openvino_genai as ov_genai

from .metrics import GenIterationMetrics
from .stats import compute_stats


# =============================================================================
# Generation config
# =============================================================================

def build_generation_config(infer_count: int, num_beams: int, seed: int,
                            base_config=None, exact_output_tokens: bool = False,
                            disable_config_chat_template: bool = True):
    """Mirror openvino.genai/llm_bench text_generation.py:333-346."""
    cfg = base_config if base_config is not None else ov_genai.GenerationConfig()
    cfg.max_new_tokens = infer_count
    if exact_output_tokens and hasattr(cfg, "min_new_tokens"):
        cfg.min_new_tokens = infer_count
    cfg.max_length = 2**63 - 1   # don't let model's small max_length clip long inputs
    cfg.ignore_eos = True
    cfg.num_beams = num_beams
    cfg.do_sample = False
    cfg.rng_seed = seed
    if num_beams > 1:
        cfg.frequency_penalty = 0
        cfg.presence_penalty = 0
        cfg.repetition_penalty = 1
    if disable_config_chat_template and hasattr(cfg, "apply_chat_template"):
        cfg.apply_chat_template = False
    return cfg


# =============================================================================
# Streamer
# =============================================================================

def make_streamer():
    """Return (callback, timestamps_list). Each fired token appends time.perf_counter()."""
    times: list[float] = []

    def cb(_subword: str):
        times.append(time.perf_counter())
        return ov_genai.StreamingStatus.RUNNING

    return cb, times


# =============================================================================
# Chat template helper
# =============================================================================

def apply_chat_template_safe(tokenizer, prompt: str, apply: bool) -> str:
    if not apply:
        return prompt
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True
        )
    except Exception as e:
        print(f"  [warn] chat template failed ({e}); using raw prompt")
        return prompt


# =============================================================================
# Permutation: defeat KV prefix cache between iterations
# =============================================================================

def _tokenizer_id(tokenizer, getter_name: str):
    if tokenizer is None or not hasattr(tokenizer, getter_name):
        return None
    try:
        return getattr(tokenizer, getter_name)()
    except Exception:
        return None


def permute_llm_inputs(tokenized_inputs, iter_idx: int, tokenizer=None):
    """Replace one early input token with (iter_idx + 1).

    This follows the official llm_bench GenAI path: mutate token 0 only when
    the tokenizer has no BOS token; otherwise preserve BOS and mutate token 1.

    Returns a NEW openvino_genai.TokenizedInputs object suitable for
    LLMPipeline.generate(inputs=...).
    """
    from openvino_genai import TokenizedInputs
    input_ids = np.asarray(tokenized_inputs.input_ids.data).copy()
    replace_idx = 0
    bos_token_id = _tokenizer_id(tokenizer, "get_bos_token_id")
    if bos_token_id is not None and bos_token_id != -1 and input_ids.shape[1] > 1:
        replace_idx = 1

    replacement = iter_idx + 1
    eos_token_id = _tokenizer_id(tokenizer, "get_eos_token_id")
    if eos_token_id == replacement:
        replacement += 2
    input_ids[:, replace_idx] = replacement
    return TokenizedInputs(
        input_ids=ov.Tensor(input_ids),
        attention_mask=tokenized_inputs.attention_mask,
    )


def permute_vlm_prompt(prompt: str, iter_idx: int) -> str:
    """Per-iteration prompt mutation that defeats the prefix cache for the
    string-only VLMPipeline.generate API."""
    return f"{iter_idx} {prompt}"


# =============================================================================
# Result helpers
# =============================================================================

def _decode_text(result) -> str:
    if isinstance(result, str):
        return result
    if hasattr(result, "texts") and result.texts:
        return result.texts[0]
    return str(result)


def _decode_text_with_tokenizer(result, tokenizer) -> str:
    tokens = getattr(result, "tokens", None)
    if tokens is not None:
        decoded = tokenizer.decode(tokens)
        if isinstance(decoded, (list, tuple)):
            return decoded[0] if decoded else ""
        return str(decoded)
    return _decode_text(result)


def _output_size(result, fallback_n: int) -> int:
    pm = getattr(result, "perf_metrics", None)
    if pm is not None:
        try:
            n = pm.get_num_generated_tokens()
            if n and n > 0:
                return int(n)
        except Exception:
            pass
    try:
        if hasattr(result, "tokens") and result.tokens is not None:
            arr = np.array(result.tokens)
            return int(arr.shape[-1])
    except Exception:
        pass
    return fallback_n


def _refine_with_perf_metrics(result, ttft_ms: float, tpot_ms: float):
    """If the GenAI result exposes perf_metrics, prefer those values.

    This matches the official text_generation.py:398-401 path.
    """
    pm = getattr(result, "perf_metrics", None)
    if pm is None:
        return ttft_ms, tpot_ms
    try:
        v = pm.get_ttft().mean
        if v and v > 0:
            ttft_ms = float(v)
    except Exception:
        pass
    try:
        v = pm.get_tpot().mean
        if v and v > 0:
            tpot_ms = float(v)
    except Exception:
        pass
    return ttft_ms, tpot_ms


def _num_input_tokens_from_perf(result, fallback: int = 0) -> int:
    pm = getattr(result, "perf_metrics", None)
    if pm is None:
        return fallback
    try:
        n = pm.get_num_input_tokens()
        if n and n > 0:
            return int(n)
    except Exception:
        pass
    return fallback


def _mean_raw_duration_ms(raw, attr_name: str, fallback: float = 0.0) -> float:
    values = getattr(raw, attr_name, None) if raw is not None else None
    if values is None:
        return fallback
    try:
        arr = np.array(values, dtype=np.float64)
    except Exception:
        return fallback
    if arr.size == 0:
        return fallback
    return float(np.mean(arr) / 1000.0)


def _official_vlm_latency_metrics(result, fallback_ttft_ms: float,
                                  fallback_tpot_ms: float) -> tuple[float, float, float, float]:
    """Mirror official visual_language_generation.py GenAI timing.

    Official visual_text_gen reports first token latency from
    perf_metrics.get_ttft().mean minus the last tokenization duration, and
    2nd-token latency from differences in raw m_new_token_times. Units returned
    here are milliseconds.
    """
    pm = getattr(result, "perf_metrics", None)
    raw = getattr(pm, "raw_metrics", None) if pm is not None else None
    if pm is None or raw is None:
        return fallback_ttft_ms, fallback_tpot_ms, 0.0, 0.0

    tokenization_ms = _mean_raw_duration_ms(raw, "tokenization_durations")
    detokenization_ms = _mean_raw_duration_ms(raw, "detokenization_durations")

    ttft_ms = fallback_ttft_ms
    try:
        ttft_ms = float(pm.get_ttft().mean)
        tokenization_values = getattr(raw, "tokenization_durations", None)
        if tokenization_values is not None and len(tokenization_values) > 0:
            ttft_ms -= float(tokenization_values[-1]) / 1000.0
        ttft_ms = max(0.0, ttft_ms)
    except Exception:
        pass

    tpot_ms = fallback_tpot_ms
    try:
        new_token_times = np.array(raw.m_new_token_times, dtype=np.float64)
        if new_token_times.size > 1:
            second_token_durations = new_token_times[1:] - new_token_times[:-1]
            tpot_ms = float(np.mean(second_token_durations))
    except Exception:
        pass

    return ttft_ms, tpot_ms, tokenization_ms, detokenization_ms


def _infer_latency_metrics(result) -> tuple[float, float]:
    pm = getattr(result, "perf_metrics", None)
    raw = getattr(pm, "raw_metrics", None) if pm is not None else None
    durations = getattr(raw, "token_infer_durations", None) if raw is not None else None
    if durations is None:
        return -1.0, -1.0
    try:
        values_ms = (np.array(durations, dtype=np.float64) / 1000.0).tolist()
    except Exception:
        return -1.0, -1.0
    if not values_ms:
        return -1.0, -1.0
    first = float(values_ms[0])
    other = float(np.mean(values_ms[1:])) if len(values_ms) > 1 else -1.0
    return first, other


def _build_iteration_metric(iter_idx: int, is_warmup: bool, prompt_idx: int,
                            input_tokens: int, n_tokens: int, output_size: int,
                            t_start: float, t_end: float, token_times: list[float],
                            tokenization_ms: float, detokenization_ms: float,
                            ttft_ms: float, tpot_ms: float,
                            first_token_infer_ms: float,
                            other_tokens_infer_ms: float,
                            output_text: str,
                            start_iso: str, end_iso: str) -> GenIterationMetrics:
    e2e_ms = (t_end - t_start) * 1000.0
    decode_ms = max(0.0, e2e_ms - ttft_ms) if output_size > 1 else 0.0
    throughput = output_size / (e2e_ms / 1000.0) if e2e_ms > 0 else 0.0
    decode_throughput = 1000.0 / tpot_ms if tpot_ms > 0 else 0.0
    md5 = hashlib.md5(
        output_text.encode("utf-8", errors="ignore"), usedforsecurity=False
    ).hexdigest() if isinstance(output_text, str) else ""
    return GenIterationMetrics(
        iteration=iter_idx,
        is_warmup=is_warmup,
        prompt_idx=prompt_idx,
        input_tokens=input_tokens,
        output_tokens=output_size,
        infer_count=output_size,
        generate_time_s=t_end - t_start,
        ttft_ms=ttft_ms,
        tpot_ms=tpot_ms,
        first_token_infer_latency_ms=first_token_infer_ms,
        other_tokens_infer_avg_latency_ms=other_tokens_infer_ms,
        prefill_latency_ms=ttft_ms,
        decode_latency_ms=decode_ms,
        e2e_latency_ms=e2e_ms,
        throughput_tps=throughput,
        decode_throughput_tps=decode_throughput,
        tokenization_time_ms=tokenization_ms,
        detokenization_time_ms=detokenization_ms,
        result_md5=md5,
        start=start_iso,
        end=end_iso,
    )


def _print_iter_line(iter_idx: int, is_warmup: bool, prompt_idx: int,
                     in_t: int, out_t: int, requested_out: int,
                     ttft: float, tpot: float,
                     e2e: float, tps: float):
    tag = "[warm-up]" if is_warmup else f"[{iter_idx}]"
    suffix = ""
    if requested_out and out_t != requested_out:
        suffix = f"  [warn: requested {requested_out}]"
    print(f"    {tag}[P{prompt_idx}] in={in_t} out={out_t} "
          f"TTFT={ttft:.1f}ms TPOT={tpot:.2f}ms "
          f"E2E={e2e:.0f}ms TPS={tps:.1f}{suffix}")


def _requested_output_tokens(gen_config) -> int:
    try:
        return int(gen_config.max_new_tokens)
    except Exception:
        return 0


def _fallback_tpot_ms(e2e_ms: float, ttft_ms: float, output_size: int) -> float:
    if output_size <= 1:
        return 0.0
    return max(0.0, e2e_ms - ttft_ms) / (output_size - 1)


# =============================================================================
# LLM iteration (text-only)
# =============================================================================

def run_llm_iteration(pipe, prompt_text: str, gen_config, iter_idx: int,
                      is_warmup: bool, prompt_idx: int,
                      permute: bool) -> Optional[GenIterationMetrics]:
    tokenizer = pipe.get_tokenizer()
    tok_start = time.perf_counter()
    tokenized = tokenizer.encode(prompt_text)
    tok_end = time.perf_counter()
    input_tokens = int(tokenized.input_ids.shape[-1])

    inputs = permute_llm_inputs(tokenized, iter_idx, tokenizer) if permute else tokenized

    streamer_cb, token_times = make_streamer()
    start_iso = datetime.datetime.now().isoformat()
    t_start = time.perf_counter()
    result = pipe.generate(inputs, gen_config, streamer=streamer_cb)
    t_end = time.perf_counter()
    end_iso = datetime.datetime.now().isoformat()

    n_tokens = len(token_times)
    if n_tokens == 0:
        print(f"  [warn] iter {iter_idx}: 0 tokens generated (excluded)")
        return None

    detok_start = time.perf_counter()
    output_text = _decode_text_with_tokenizer(result, tokenizer)
    detok_end = time.perf_counter()

    output_size = _output_size(result, n_tokens)
    e2e_ms = (t_end - t_start) * 1000.0
    ttft_ms = (token_times[0] - t_start) * 1000.0
    tpot_ms = _fallback_tpot_ms(e2e_ms, ttft_ms, output_size)
    ttft_ms, tpot_ms = _refine_with_perf_metrics(result, ttft_ms, tpot_ms)
    first_infer_ms, other_infer_ms = _infer_latency_metrics(result)

    requested_out = _requested_output_tokens(gen_config)
    metric = _build_iteration_metric(
        iter_idx, is_warmup, prompt_idx, input_tokens, n_tokens, output_size,
        t_start, t_end, token_times,
        (tok_end - tok_start) * 1000.0, (detok_end - detok_start) * 1000.0,
        ttft_ms, tpot_ms, first_infer_ms, other_infer_ms,
        output_text, start_iso, end_iso,
    )
    _print_iter_line(iter_idx, is_warmup, prompt_idx, input_tokens, output_size,
                     requested_out, metric.ttft_ms, metric.tpot_ms, metric.e2e_latency_ms,
                     metric.throughput_tps)
    return metric


# =============================================================================
# VLM iteration (image + text)
# =============================================================================

def run_vlm_iteration(pipe, prompt_text: str, gen_config, iter_idx: int,
                      is_warmup: bool, prompt_idx: int,
                      images: Optional[list], permute: bool) -> Optional[GenIterationMetrics]:
    effective_prompt = permute_vlm_prompt(prompt_text, iter_idx) if permute else prompt_text
    start_iso = datetime.datetime.now().isoformat()
    t_start = time.perf_counter()
    if images:
        result = pipe.generate(effective_prompt, images=images,
                               generation_config=gen_config)
    else:
        result = pipe.generate(effective_prompt,
                               generation_config=gen_config)
    t_end = time.perf_counter()
    end_iso = datetime.datetime.now().isoformat()

    output_text = _decode_text(result)
    output_size = _output_size(result, 0)
    if output_size == 0:
        print(f"  [warn] iter {iter_idx}: 0 tokens generated (excluded)")
        return None

    input_tokens = _num_input_tokens_from_perf(result, 0)
    if input_tokens == 0:
        try:
            input_tokens = int(pipe.get_tokenizer().encode(effective_prompt).input_ids.shape[-1])
        except Exception:
            input_tokens = 0

    e2e_ms = (t_end - t_start) * 1000.0
    ttft_ms = e2e_ms
    tpot_ms = _fallback_tpot_ms(e2e_ms, ttft_ms, output_size)
    ttft_ms, tpot_ms, tokenization_ms, detokenization_ms = _official_vlm_latency_metrics(
        result, ttft_ms, tpot_ms
    )
    first_infer_ms, other_infer_ms = _infer_latency_metrics(result)

    requested_out = _requested_output_tokens(gen_config)
    metric = _build_iteration_metric(
        iter_idx, is_warmup, prompt_idx, input_tokens, output_size, output_size,
        t_start, t_end, [],
        tokenization_ms, detokenization_ms,
        ttft_ms, tpot_ms, first_infer_ms, other_infer_ms,
        output_text, start_iso, end_iso,
    )
    _print_iter_line(iter_idx, is_warmup, prompt_idx, input_tokens, output_size,
                     requested_out, metric.ttft_ms, metric.tpot_ms, metric.e2e_latency_ms,
                     metric.throughput_tps)
    return metric


# =============================================================================
# Optimum VLM iteration (HF streamer; used as GPU fallback for qwen3.5)
# =============================================================================

class _OptimumStreamer:
    def __init__(self, prompt_token_len: int):
        self._prompt_len = prompt_token_len
        self._prompt_seen = 0
        self.token_times: list[float] = []

    def put(self, value):
        import torch
        if isinstance(value, torch.Tensor):
            tokens = value.detach().cpu().reshape(-1).tolist()
        elif isinstance(value, np.ndarray):
            tokens = list(np.array(value).reshape(-1))
        elif isinstance(value, (list, tuple)):
            tokens = list(value)
        else:
            return
        now = time.perf_counter()
        for _ in tokens:
            if self._prompt_seen < self._prompt_len:
                self._prompt_seen += 1
                continue
            self.token_times.append(now)

    def end(self):
        pass


def run_optimum_vlm_iteration(model, processor, prompt_text: str, image,
                              max_new_tokens: int, iter_idx: int,
                              is_warmup: bool, prompt_idx: int,
                              permute: bool,
                              exact_output_tokens: bool = False) -> Optional[GenIterationMetrics]:
    import logging
    logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)

    effective_prompt = permute_vlm_prompt(prompt_text, iter_idx) if permute else prompt_text

    if image is not None:
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": effective_prompt},
        ]}]
    else:
        messages = [{"role": "user", "content": [
            {"type": "text", "text": effective_prompt},
        ]}]

    tok_start = time.perf_counter()
    text = processor.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
    if image is not None:
        inputs = processor(text=[text], images=[image], return_tensors="pt")
    else:
        inputs = processor(text=[text], return_tensors="pt")
    tok_end = time.perf_counter()
    input_tokens = int(inputs["input_ids"].shape[1])

    streamer = _OptimumStreamer(input_tokens)
    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "streamer": streamer,
    }
    if exact_output_tokens:
        generate_kwargs["min_new_tokens"] = max_new_tokens
    start_iso = datetime.datetime.now().isoformat()
    t_start = time.perf_counter()
    output_ids = model.generate(**inputs, **generate_kwargs)
    t_end = time.perf_counter()
    end_iso = datetime.datetime.now().isoformat()

    detok_start = time.perf_counter()
    try:
        output_text = processor.batch_decode(
            output_ids[:, input_tokens:], skip_special_tokens=True
        )[0]
    except Exception:
        output_text = ""
    detok_end = time.perf_counter()

    n_tokens = len(streamer.token_times)
    if n_tokens == 0:
        print(f"  [warn] iter {iter_idx}: 0 tokens generated (excluded)")
        return None

    try:
        output_size = int(output_ids.shape[1] - input_tokens)
    except Exception:
        output_size = n_tokens
    e2e_ms = (t_end - t_start) * 1000.0
    ttft_ms = (streamer.token_times[0] - t_start) * 1000.0
    tpot_ms = _fallback_tpot_ms(e2e_ms, ttft_ms, output_size)
    metric = _build_iteration_metric(
        iter_idx, is_warmup, prompt_idx, input_tokens, n_tokens, output_size,
        t_start, t_end, streamer.token_times,
        (tok_end - tok_start) * 1000.0, (detok_end - detok_start) * 1000.0,
        ttft_ms, tpot_ms, -1.0, -1.0, output_text, start_iso, end_iso,
    )
    _print_iter_line(iter_idx, is_warmup, prompt_idx, input_tokens, output_size,
                     max_new_tokens, metric.ttft_ms, metric.tpot_ms, metric.e2e_latency_ms,
                     metric.throughput_tps)
    return metric


# =============================================================================
# Aggregation
# =============================================================================

def aggregate(iterations: list[GenIterationMetrics]) -> dict:
    """Compute summary stats over MEASURED iterations (warmup excluded).

    Returns {"ttft": {mean,std,...}, "tpot": ..., "throughput": ..., "prefill": ...,
             "decode": ..., "e2e": ...}
    """
    measured = [it for it in iterations if not it.is_warmup]
    if not measured:
        return {}

    tpot_filtered = [m.tpot_ms for m in measured if m.tpot_ms > 0]
    if measured and not tpot_filtered:
        # Caller may want to know — TPOT was zeroed out (single-token outputs).
        print("  [warn] all measured iterations had <=1 generated token; TPOT will be 0")
        tpot_filtered = [m.tpot_ms for m in measured]

    return {
        "output_tokens": compute_stats([m.output_tokens for m in measured]),
        "generation_time": compute_stats([m.generate_time_s for m in measured]),
        "ttft": compute_stats([m.ttft_ms for m in measured]),
        "tpot": compute_stats(tpot_filtered),
        "first_token_infer": compute_stats([
            m.first_token_infer_latency_ms for m in measured
            if m.first_token_infer_latency_ms >= 0
        ]),
        "other_tokens_infer": compute_stats([
            m.other_tokens_infer_avg_latency_ms for m in measured
            if m.other_tokens_infer_avg_latency_ms >= 0
        ]),
        "throughput": compute_stats([m.throughput_tps for m in measured]),
        "decode_throughput": compute_stats([m.decode_throughput_tps for m in measured if m.decode_throughput_tps > 0]),
        "prefill": compute_stats([m.prefill_latency_ms for m in measured]),
        "decode": compute_stats([m.decode_latency_ms for m in measured]),
        "e2e": compute_stats([m.e2e_latency_ms for m in measured]),
        "tokenization": compute_stats([m.tokenization_time_ms for m in measured]),
        "detokenization": compute_stats([m.detokenization_time_ms for m in measured]),
    }
