"""Report writers (CSV / JSON, both append-mode) and console summary printer."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from .metrics import (
    BertBenchmarkResult, EmbedBenchmarkResult, GenBenchmarkResult,
    RerankBenchmarkResult,
)


# =============================================================================
# Generation (qwen3 + qwen3.5)
# =============================================================================

GEN_CSV_HEADER = [
    "model", "framework", "pipeline", "device", "precision", "mode",
    "num_beams", "batch_size", "apply_chat_template", "disable_prompt_permutation",
    "prompt_idx", "input_tokens", "infer_count_arg", "exact_output_tokens", "num_iters", "measured_iters",
    "compile_time_s",
    "output_tokens_mean", "output_tokens_std", "output_tokens_p50", "output_tokens_p90", "output_tokens_p95",
    "output_tokens_min", "output_tokens_max",
    "generation_time_mean_s", "generation_time_std_s", "generation_time_p50_s", "generation_time_p90_s",
    "generation_time_p95_s", "generation_time_min_s", "generation_time_max_s",
    "ttft_mean_ms", "ttft_std_ms", "ttft_p50_ms", "ttft_p90_ms", "ttft_p95_ms",
    "ttft_min_ms", "ttft_max_ms",
    "tpot_mean_ms", "tpot_std_ms", "tpot_p50_ms", "tpot_p90_ms", "tpot_p95_ms",
    "tpot_min_ms", "tpot_max_ms",
    "first_token_infer_mean_ms", "first_token_infer_std_ms", "first_token_infer_p50_ms",
    "first_token_infer_p90_ms", "first_token_infer_p95_ms", "first_token_infer_min_ms", "first_token_infer_max_ms",
    "other_tokens_infer_mean_ms", "other_tokens_infer_std_ms", "other_tokens_infer_p50_ms",
    "other_tokens_infer_p90_ms", "other_tokens_infer_p95_ms", "other_tokens_infer_min_ms", "other_tokens_infer_max_ms",
    "tps_mean", "tps_std", "tps_p50", "tps_p90", "tps_p95", "tps_min", "tps_max",
    "decode_tps_mean", "decode_tps_std", "decode_tps_p50", "decode_tps_p90", "decode_tps_p95",
    "decode_tps_min", "decode_tps_max",
    "prefill_mean_ms", "prefill_std_ms", "prefill_p50_ms", "prefill_p90_ms", "prefill_p95_ms",
    "prefill_min_ms", "prefill_max_ms",
    "decode_mean_ms", "decode_std_ms", "decode_p50_ms", "decode_p90_ms", "decode_p95_ms",
    "decode_min_ms", "decode_max_ms",
    "e2e_mean_ms", "e2e_std_ms", "e2e_p50_ms", "e2e_p90_ms", "e2e_p95_ms",
    "e2e_min_ms", "e2e_max_ms",
    "tokenization_mean_ms", "tokenization_std_ms", "tokenization_p50_ms", "tokenization_p90_ms", "tokenization_p95_ms",
    "tokenization_min_ms", "tokenization_max_ms",
    "detokenization_mean_ms", "detokenization_std_ms", "detokenization_p50_ms", "detokenization_p90_ms", "detokenization_p95_ms",
    "detokenization_min_ms", "detokenization_max_ms",
    "initial_rss_mem_mib", "initial_sys_mem_mib", "initial_rss_mem_share", "initial_sys_mem_share",
    "compile_max_rss_mem_mib", "compile_max_sys_mem_mib",
    "compile_max_rss_mem_increase_mib", "compile_max_sys_mem_increase_mib",
    "compile_max_rss_mem_share", "compile_max_sys_mem_share",
    "runtime_mean_rss_mem_mib", "runtime_max_rss_mem_mib", "runtime_mean_sys_mem_mib", "runtime_max_sys_mem_mib",
    "runtime_mean_rss_mem_increase_mib", "runtime_max_rss_mem_increase_mib",
    "runtime_mean_sys_mem_increase_mib", "runtime_max_sys_mem_increase_mib",
    "runtime_max_rss_mem_share", "runtime_max_sys_mem_share", "memory_samples",
    "gpu_avg_util", "npu_avg_util", "cpu_avg_util",
    "image_path",
    "openvino_version", "openvino_genai_version", "run_started_at",
]


def _gen_get_all_core_names(results: Sequence[GenBenchmarkResult]) -> list[str]:
    cores: set[str] = set()
    for r in results:
        cores.update(r.cpu_util.get("per_core", {}).keys())
    try:
        return sorted(cores, key=lambda x: int(x.replace("cpu", "")))
    except ValueError:
        return sorted(cores)


def _gen_row(r: GenBenchmarkResult, all_cores: Sequence[str]) -> list:
    per_core = r.cpu_util.get("per_core", {})
    return [
        r.model_name, r.framework, r.pipeline, r.device, r.precision, r.mode,
        r.num_beams, r.batch_size, r.apply_chat_template, r.disable_prompt_permutation,
        r.prompt_idx, r.input_tokens, r.infer_count_arg, r.exact_output_tokens, r.num_iters, r.measured_iters,
        f"{r.compile_time_s:.3f}",
        f"{r.output_tokens_mean:.2f}", f"{r.output_tokens_std:.2f}", f"{r.output_tokens_p50:.2f}",
        f"{r.output_tokens_p90:.2f}", f"{r.output_tokens_p95:.2f}", f"{r.output_tokens_min:.2f}",
        f"{r.output_tokens_max:.2f}",
        f"{r.generation_time_mean_s:.5f}", f"{r.generation_time_std_s:.5f}",
        f"{r.generation_time_p50_s:.5f}", f"{r.generation_time_p90_s:.5f}",
        f"{r.generation_time_p95_s:.5f}", f"{r.generation_time_min_s:.5f}",
        f"{r.generation_time_max_s:.5f}",
        f"{r.ttft_mean_ms:.3f}", f"{r.ttft_std_ms:.3f}", f"{r.ttft_p50_ms:.3f}",
        f"{r.ttft_p90_ms:.3f}", f"{r.ttft_p95_ms:.3f}", f"{r.ttft_min_ms:.3f}", f"{r.ttft_max_ms:.3f}",
        f"{r.tpot_mean_ms:.3f}", f"{r.tpot_std_ms:.3f}", f"{r.tpot_p50_ms:.3f}",
        f"{r.tpot_p90_ms:.3f}", f"{r.tpot_p95_ms:.3f}", f"{r.tpot_min_ms:.3f}", f"{r.tpot_max_ms:.3f}",
        f"{r.first_token_infer_mean_ms:.3f}", f"{r.first_token_infer_std_ms:.3f}",
        f"{r.first_token_infer_p50_ms:.3f}", f"{r.first_token_infer_p90_ms:.3f}",
        f"{r.first_token_infer_p95_ms:.3f}", f"{r.first_token_infer_min_ms:.3f}",
        f"{r.first_token_infer_max_ms:.3f}",
        f"{r.other_tokens_infer_mean_ms:.3f}", f"{r.other_tokens_infer_std_ms:.3f}",
        f"{r.other_tokens_infer_p50_ms:.3f}", f"{r.other_tokens_infer_p90_ms:.3f}",
        f"{r.other_tokens_infer_p95_ms:.3f}", f"{r.other_tokens_infer_min_ms:.3f}",
        f"{r.other_tokens_infer_max_ms:.3f}",
        f"{r.throughput_mean_tps:.2f}", f"{r.throughput_std_tps:.2f}",
        f"{r.throughput_p50_tps:.2f}", f"{r.throughput_p90_tps:.2f}", f"{r.throughput_p95_tps:.2f}",
        f"{r.throughput_min_tps:.2f}", f"{r.throughput_max_tps:.2f}",
        f"{r.decode_throughput_mean_tps:.2f}", f"{r.decode_throughput_std_tps:.2f}",
        f"{r.decode_throughput_p50_tps:.2f}", f"{r.decode_throughput_p90_tps:.2f}",
        f"{r.decode_throughput_p95_tps:.2f}", f"{r.decode_throughput_min_tps:.2f}",
        f"{r.decode_throughput_max_tps:.2f}",
        f"{r.prefill_mean_ms:.3f}", f"{r.prefill_std_ms:.3f}", f"{r.prefill_p50_ms:.3f}",
        f"{r.prefill_p90_ms:.3f}", f"{r.prefill_p95_ms:.3f}", f"{r.prefill_min_ms:.3f}",
        f"{r.prefill_max_ms:.3f}",
        f"{r.decode_mean_ms:.3f}", f"{r.decode_std_ms:.3f}", f"{r.decode_p50_ms:.3f}",
        f"{r.decode_p90_ms:.3f}", f"{r.decode_p95_ms:.3f}", f"{r.decode_min_ms:.3f}",
        f"{r.decode_max_ms:.3f}",
        f"{r.e2e_mean_ms:.3f}", f"{r.e2e_std_ms:.3f}", f"{r.e2e_p50_ms:.3f}",
        f"{r.e2e_p90_ms:.3f}", f"{r.e2e_p95_ms:.3f}", f"{r.e2e_min_ms:.3f}",
        f"{r.e2e_max_ms:.3f}",
        f"{r.tokenization_mean_ms:.3f}", f"{r.tokenization_std_ms:.3f}", f"{r.tokenization_p50_ms:.3f}",
        f"{r.tokenization_p90_ms:.3f}", f"{r.tokenization_p95_ms:.3f}", f"{r.tokenization_min_ms:.3f}",
        f"{r.tokenization_max_ms:.3f}",
        f"{r.detokenization_mean_ms:.3f}", f"{r.detokenization_std_ms:.3f}",
        f"{r.detokenization_p50_ms:.3f}", f"{r.detokenization_p90_ms:.3f}",
        f"{r.detokenization_p95_ms:.3f}", f"{r.detokenization_min_ms:.3f}",
        f"{r.detokenization_max_ms:.3f}",
        f"{r.initial_rss_mem_mib:.3f}", f"{r.initial_sys_mem_mib:.3f}",
        f"{r.initial_rss_mem_share:.3f}", f"{r.initial_sys_mem_share:.3f}",
        f"{r.compile_max_rss_mem_mib:.3f}", f"{r.compile_max_sys_mem_mib:.3f}",
        f"{r.compile_max_rss_mem_increase_mib:.3f}", f"{r.compile_max_sys_mem_increase_mib:.3f}",
        f"{r.compile_max_rss_mem_share:.3f}", f"{r.compile_max_sys_mem_share:.3f}",
        f"{r.runtime_mean_rss_mem_mib:.3f}", f"{r.runtime_max_rss_mem_mib:.3f}",
        f"{r.runtime_mean_sys_mem_mib:.3f}", f"{r.runtime_max_sys_mem_mib:.3f}",
        f"{r.runtime_mean_rss_mem_increase_mib:.3f}", f"{r.runtime_max_rss_mem_increase_mib:.3f}",
        f"{r.runtime_mean_sys_mem_increase_mib:.3f}", f"{r.runtime_max_sys_mem_increase_mib:.3f}",
        f"{r.runtime_max_rss_mem_share:.3f}", f"{r.runtime_max_sys_mem_share:.3f}",
        r.memory_samples,
        f"{r.gpu_util.get('avg_util', 0):.1f}" if r.gpu_util.get("available") else "N/A",
        f"{r.npu_util.get('avg_util', 0):.1f}" if r.npu_util.get("available") else "N/A",
        f"{r.cpu_util.get('avg_total', 0):.1f}",
        *[f"{per_core.get(c, 0.0):.1f}" for c in all_cores],
        r.image_path,
        r.openvino_version, r.openvino_genai_version, r.run_started_at,
    ]


def write_gen_csv(filepath: Path, results: Sequence[GenBenchmarkResult]) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    all_cores = _gen_get_all_core_names(results)
    header = list(GEN_CSV_HEADER)
    image_path_idx = header.index("image_path")
    header[image_path_idx:image_path_idx] = [f"cpu_{c}_util" for c in all_cores]
    file_exists = filepath.exists() and filepath.stat().st_size > 0
    write_header = not file_exists
    if file_exists:
        try:
            with open(filepath, "r", newline="", encoding="utf-8") as f:
                existing_header = next(csv.reader(f), [])
            write_header = existing_header != header
        except (OSError, StopIteration):
            write_header = True
    mode = "a" if file_exists and not write_header else "w"
    with open(filepath, mode, newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        for r in results:
            w.writerow(_gen_row(r, all_cores))
    print(f"CSV  {'appended to' if file_exists and not write_header else 'written'}: {filepath}")


# =============================================================================
# BERT (lingua)
# =============================================================================

def _bert_get_all_core_names(results: Sequence[BertBenchmarkResult]) -> list[str]:
    cores: set[str] = set()
    for r in results:
        cores.update(r.cpu_util.get("per_core", {}).keys())
    try:
        return sorted(cores, key=lambda x: int(x.replace("cpu", "")))
    except ValueError:
        return sorted(cores)


def write_bert_csv(filepath: Path, results: Sequence[BertBenchmarkResult]) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    all_cores = _bert_get_all_core_names(results)
    header = [
        "model", "framework", "device", "precision",
        "input_tokens", "compression_rate", "iters", "compile_time_s",
        "initial_rss_mem_mib", "initial_sys_mem_mib", "initial_rss_mem_share", "initial_sys_mem_share",
        "compile_max_rss_mem_mib", "compile_max_sys_mem_mib",
        "compile_max_rss_mem_increase_mib", "compile_max_sys_mem_increase_mib",
        "compile_max_rss_mem_share", "compile_max_sys_mem_share",
        "runtime_mean_rss_mem_mib", "runtime_max_rss_mem_mib", "runtime_mean_sys_mem_mib", "runtime_max_sys_mem_mib",
        "runtime_mean_rss_mem_increase_mib", "runtime_max_rss_mem_increase_mib",
        "runtime_mean_sys_mem_increase_mib", "runtime_max_sys_mem_increase_mib",
        "runtime_max_rss_mem_share", "runtime_max_sys_mem_share", "memory_samples",
        "latency_mean_ms", "latency_std_ms", "latency_p50_ms",
        "latency_p90_ms", "latency_p95_ms", "latency_min_ms", "latency_max_ms",
        "throughput_samples_mean", "throughput_samples_std", "throughput_samples_p50",
        "throughput_samples_p90", "throughput_samples_p95", "throughput_samples_min", "throughput_samples_max",
        "throughput_tokens_mean", "throughput_tokens_std", "throughput_tokens_p50",
        "throughput_tokens_p90", "throughput_tokens_p95", "throughput_tokens_min", "throughput_tokens_max",
        "avg_compressed_tokens", "avg_compression_ratio",
        "cpu_avg_util", "gpu_avg_util", "npu_avg_util",
        "openvino_version", "run_started_at",
    ]
    header.extend([f"cpu_{c}_util" for c in all_cores])

    file_exists = filepath.exists() and filepath.stat().st_size > 0
    write_header = not file_exists
    if file_exists:
        try:
            with open(filepath, "r", newline="", encoding="utf-8") as f:
                existing_header = next(csv.reader(f), [])
            write_header = existing_header != header
        except (OSError, StopIteration):
            write_header = True
    # When the header differs from an existing file (e.g. a different per-core
    # CPU column count between machines), overwrite rather than appending a
    # second header row into the middle of the file.
    mode = "a" if file_exists and not write_header else "w"
    with open(filepath, mode, newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        for r in results:
            per_core = r.cpu_util.get("per_core", {})
            row = [
                r.model_name, r.framework, r.device, r.precision,
                r.input_length, f"{r.compression_rate:.2f}", r.measure_iters,
                f"{r.compile_time_s:.3f}",
                f"{r.initial_rss_mem_mib:.3f}", f"{r.initial_sys_mem_mib:.3f}",
                f"{r.initial_rss_mem_share:.3f}", f"{r.initial_sys_mem_share:.3f}",
                f"{r.compile_max_rss_mem_mib:.3f}", f"{r.compile_max_sys_mem_mib:.3f}",
                f"{r.compile_max_rss_mem_increase_mib:.3f}", f"{r.compile_max_sys_mem_increase_mib:.3f}",
                f"{r.compile_max_rss_mem_share:.3f}", f"{r.compile_max_sys_mem_share:.3f}",
                f"{r.runtime_mean_rss_mem_mib:.3f}", f"{r.runtime_max_rss_mem_mib:.3f}",
                f"{r.runtime_mean_sys_mem_mib:.3f}", f"{r.runtime_max_sys_mem_mib:.3f}",
                f"{r.runtime_mean_rss_mem_increase_mib:.3f}", f"{r.runtime_max_rss_mem_increase_mib:.3f}",
                f"{r.runtime_mean_sys_mem_increase_mib:.3f}", f"{r.runtime_max_sys_mem_increase_mib:.3f}",
                f"{r.runtime_max_rss_mem_share:.3f}", f"{r.runtime_max_sys_mem_share:.3f}",
                r.memory_samples,
                f"{r.latency_mean_ms:.3f}", f"{r.latency_std_ms:.3f}",
                f"{r.latency_p50_ms:.3f}", f"{r.latency_p90_ms:.3f}",
                f"{r.latency_p95_ms:.3f}", f"{r.latency_min_ms:.3f}", f"{r.latency_max_ms:.3f}",
                f"{r.throughput_samples_mean:.2f}", f"{r.throughput_samples_std:.2f}",
                f"{r.throughput_samples_p50:.2f}", f"{r.throughput_samples_p90:.2f}",
                f"{r.throughput_samples_p95:.2f}", f"{r.throughput_samples_min:.2f}",
                f"{r.throughput_samples_max:.2f}",
                f"{r.throughput_tokens_mean:.2f}", f"{r.throughput_tokens_std:.2f}",
                f"{r.throughput_tokens_p50:.2f}", f"{r.throughput_tokens_p90:.2f}",
                f"{r.throughput_tokens_p95:.2f}", f"{r.throughput_tokens_min:.2f}",
                f"{r.throughput_tokens_max:.2f}",
                f"{r.avg_compressed_tokens:.0f}", f"{r.avg_compression_ratio:.4f}",
                f"{r.cpu_util.get('avg_total', 0):.1f}",
                f"{r.gpu_util.get('avg_util', 0):.1f}" if r.gpu_util.get("available") else "N/A",
                f"{r.npu_util.get('avg_util', 0):.1f}" if r.npu_util.get("available") else "N/A",
                r.openvino_version, r.run_started_at,
            ]
            row.extend([f"{per_core.get(c, 0.0):.1f}" for c in all_cores])
            w.writerow(row)
    print(f"CSV  {'appended to' if file_exists and not write_header else 'written'}: {filepath}")


# =============================================================================
# Shared append-mode CSV helper (header-aware) for encoder-style results
# =============================================================================

def _all_core_names(results) -> list[str]:
    cores: set[str] = set()
    for r in results:
        cores.update(r.cpu_util.get("per_core", {}).keys())
    try:
        return sorted(cores, key=lambda x: int(x.replace("cpu", "")))
    except ValueError:
        return sorted(cores)


def _write_csv_append(filepath: Path, header: list[str], rows: list[list]) -> None:
    """Append-mode CSV writer that re-writes the file when the header changes,
    so a differing per-core column count never injects a header row mid-file.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    file_exists = filepath.exists() and filepath.stat().st_size > 0
    write_header = not file_exists
    if file_exists:
        try:
            with open(filepath, "r", newline="", encoding="utf-8") as f:
                existing_header = next(csv.reader(f), [])
            write_header = existing_header != header
        except (OSError, StopIteration):
            write_header = True
    mode = "a" if file_exists and not write_header else "w"
    with open(filepath, mode, newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        for row in rows:
            w.writerow(row)
    print(f"CSV  {'appended to' if file_exists and not write_header else 'written'}: {filepath}")


# =============================================================================
# Text Embedding (Qwen3-Embedding)
# =============================================================================

EMBED_CSV_BASE_HEADER = [
    "model", "framework", "pipeline", "device", "precision", "mode",
    "batch_size", "pooling_type", "normalize", "max_length", "padding_side",
    "pad_to_max_length", "query_instruction", "embed_instruction",
    "prompt_idx", "input_tokens", "total_input_tokens", "embedding_dim",
    "num_iters", "measured_iters", "compile_time_s",
    "latency_mean_ms", "latency_std_ms", "latency_p50_ms", "latency_p90_ms", "latency_p95_ms",
    "latency_min_ms", "latency_max_ms",
    "e2e_mean_ms", "e2e_std_ms", "e2e_p50_ms", "e2e_p90_ms", "e2e_p95_ms",
    "e2e_min_ms", "e2e_max_ms",
    "tokenization_mean_ms", "tokenization_std_ms", "tokenization_p50_ms", "tokenization_p90_ms",
    "tokenization_p95_ms", "tokenization_min_ms", "tokenization_max_ms",
    "throughput_docs_mean", "throughput_docs_std", "throughput_docs_p50", "throughput_docs_p90",
    "throughput_docs_p95", "throughput_docs_min", "throughput_docs_max",
    "throughput_tokens_mean", "throughput_tokens_std", "throughput_tokens_p50", "throughput_tokens_p90",
    "throughput_tokens_p95", "throughput_tokens_min", "throughput_tokens_max",
    "initial_rss_mem_mib", "initial_sys_mem_mib", "initial_rss_mem_share", "initial_sys_mem_share",
    "compile_max_rss_mem_mib", "compile_max_sys_mem_mib",
    "compile_max_rss_mem_increase_mib", "compile_max_sys_mem_increase_mib",
    "compile_max_rss_mem_share", "compile_max_sys_mem_share",
    "runtime_mean_rss_mem_mib", "runtime_max_rss_mem_mib", "runtime_mean_sys_mem_mib", "runtime_max_sys_mem_mib",
    "runtime_mean_rss_mem_increase_mib", "runtime_max_rss_mem_increase_mib",
    "runtime_mean_sys_mem_increase_mib", "runtime_max_sys_mem_increase_mib",
    "runtime_max_rss_mem_share", "runtime_max_sys_mem_share", "memory_samples",
    "cpu_avg_util", "gpu_avg_util", "npu_avg_util",
]
EMBED_CSV_TAIL_HEADER = ["openvino_version", "openvino_genai_version", "run_started_at"]


def _stat_cells(r, prefix: str, suffix: str, fmt: str) -> list[str]:
    return [format(getattr(r, f"{prefix}_{stat}{suffix}"), fmt)
            for stat in ("mean", "std", "p50", "p90", "p95", "min", "max")]


def _memory_cells(r) -> list[str]:
    return [
        f"{r.initial_rss_mem_mib:.3f}", f"{r.initial_sys_mem_mib:.3f}",
        f"{r.initial_rss_mem_share:.3f}", f"{r.initial_sys_mem_share:.3f}",
        f"{r.compile_max_rss_mem_mib:.3f}", f"{r.compile_max_sys_mem_mib:.3f}",
        f"{r.compile_max_rss_mem_increase_mib:.3f}", f"{r.compile_max_sys_mem_increase_mib:.3f}",
        f"{r.compile_max_rss_mem_share:.3f}", f"{r.compile_max_sys_mem_share:.3f}",
        f"{r.runtime_mean_rss_mem_mib:.3f}", f"{r.runtime_max_rss_mem_mib:.3f}",
        f"{r.runtime_mean_sys_mem_mib:.3f}", f"{r.runtime_max_sys_mem_mib:.3f}",
        f"{r.runtime_mean_rss_mem_increase_mib:.3f}", f"{r.runtime_max_rss_mem_increase_mib:.3f}",
        f"{r.runtime_mean_sys_mem_increase_mib:.3f}", f"{r.runtime_max_sys_mem_increase_mib:.3f}",
        f"{r.runtime_max_rss_mem_share:.3f}", f"{r.runtime_max_sys_mem_share:.3f}",
        r.memory_samples,
    ]


def _util_cells(r) -> list[str]:
    return [
        f"{r.cpu_util.get('avg_total', 0):.1f}",
        f"{r.gpu_util.get('avg_util', 0):.1f}" if r.gpu_util.get("available") else "N/A",
        f"{r.npu_util.get('avg_util', 0):.1f}" if r.npu_util.get("available") else "N/A",
    ]


def write_embed_csv(filepath: Path, results: Sequence[EmbedBenchmarkResult]) -> None:
    all_cores = _all_core_names(results)
    header = list(EMBED_CSV_BASE_HEADER) + [f"cpu_{c}_util" for c in all_cores] + EMBED_CSV_TAIL_HEADER
    rows = []
    for r in results:
        per_core = r.cpu_util.get("per_core", {})
        row = [
            r.model_name, r.framework, r.pipeline, r.device, r.precision, r.mode,
            r.batch_size, r.pooling_type, r.normalize, r.max_length, r.padding_side,
            r.pad_to_max_length, r.query_instruction, r.embed_instruction,
            r.prompt_idx, r.input_tokens, r.total_input_tokens, r.embedding_dim,
            r.num_iters, r.measured_iters, f"{r.compile_time_s:.3f}",
            *_stat_cells(r, "latency", "_ms", ".3f"),
            *_stat_cells(r, "e2e", "_ms", ".3f"),
            *_stat_cells(r, "tokenization", "_ms", ".3f"),
            *_stat_cells(r, "throughput_docs", "", ".2f"),
            *_stat_cells(r, "throughput_tokens", "", ".2f"),
            *_memory_cells(r),
            *_util_cells(r),
            *[f"{per_core.get(c, 0.0):.1f}" for c in all_cores],
            r.openvino_version, r.openvino_genai_version, r.run_started_at,
        ]
        rows.append(row)
    _write_csv_append(filepath, header, rows)


# =============================================================================
# Text Reranking (Qwen3-Reranker)
# =============================================================================

RERANK_CSV_BASE_HEADER = [
    "model", "framework", "pipeline", "device", "precision",
    "num_documents", "top_n", "max_length", "padding_side", "pad_to_max_length",
    "prompt_idx", "input_tokens", "total_input_tokens",
    "num_iters", "measured_iters", "compile_time_s",
    "latency_mean_ms", "latency_std_ms", "latency_p50_ms", "latency_p90_ms", "latency_p95_ms",
    "latency_min_ms", "latency_max_ms",
    "e2e_mean_ms", "e2e_std_ms", "e2e_p50_ms", "e2e_p90_ms", "e2e_p95_ms",
    "e2e_min_ms", "e2e_max_ms",
    "tokenization_mean_ms", "tokenization_std_ms", "tokenization_p50_ms", "tokenization_p90_ms",
    "tokenization_p95_ms", "tokenization_min_ms", "tokenization_max_ms",
    "throughput_docs_mean", "throughput_docs_std", "throughput_docs_p50", "throughput_docs_p90",
    "throughput_docs_p95", "throughput_docs_min", "throughput_docs_max",
    "throughput_tokens_mean", "throughput_tokens_std", "throughput_tokens_p50", "throughput_tokens_p90",
    "throughput_tokens_p95", "throughput_tokens_min", "throughput_tokens_max",
    "initial_rss_mem_mib", "initial_sys_mem_mib", "initial_rss_mem_share", "initial_sys_mem_share",
    "compile_max_rss_mem_mib", "compile_max_sys_mem_mib",
    "compile_max_rss_mem_increase_mib", "compile_max_sys_mem_increase_mib",
    "compile_max_rss_mem_share", "compile_max_sys_mem_share",
    "runtime_mean_rss_mem_mib", "runtime_max_rss_mem_mib", "runtime_mean_sys_mem_mib", "runtime_max_sys_mem_mib",
    "runtime_mean_rss_mem_increase_mib", "runtime_max_rss_mem_increase_mib",
    "runtime_mean_sys_mem_increase_mib", "runtime_max_sys_mem_increase_mib",
    "runtime_max_rss_mem_share", "runtime_max_sys_mem_share", "memory_samples",
    "cpu_avg_util", "gpu_avg_util", "npu_avg_util",
]
RERANK_CSV_TAIL_HEADER = ["openvino_version", "openvino_genai_version", "run_started_at"]


def write_rerank_csv(filepath: Path, results: Sequence[RerankBenchmarkResult]) -> None:
    all_cores = _all_core_names(results)
    header = list(RERANK_CSV_BASE_HEADER) + [f"cpu_{c}_util" for c in all_cores] + RERANK_CSV_TAIL_HEADER
    rows = []
    for r in results:
        per_core = r.cpu_util.get("per_core", {})
        row = [
            r.model_name, r.framework, r.pipeline, r.device, r.precision,
            r.num_documents, r.top_n, r.max_length, r.padding_side, r.pad_to_max_length,
            r.prompt_idx, r.input_tokens, r.total_input_tokens,
            r.num_iters, r.measured_iters, f"{r.compile_time_s:.3f}",
            *_stat_cells(r, "latency", "_ms", ".3f"),
            *_stat_cells(r, "e2e", "_ms", ".3f"),
            *_stat_cells(r, "tokenization", "_ms", ".3f"),
            *_stat_cells(r, "throughput_docs", "", ".2f"),
            *_stat_cells(r, "throughput_tokens", "", ".2f"),
            *_memory_cells(r),
            *_util_cells(r),
            *[f"{per_core.get(c, 0.0):.1f}" for c in all_cores],
            r.openvino_version, r.openvino_genai_version, r.run_started_at,
        ]
        rows.append(row)
    _write_csv_append(filepath, header, rows)


# =============================================================================
# JSON (works for any dataclass)
# =============================================================================

def write_json(filepath: Path, results: Sequence[object]) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    new_data = [asdict(r) for r in results]
    existing: list = []
    if filepath.exists() and filepath.stat().st_size > 0:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            existing = loaded if isinstance(loaded, list) else [loaded]
        except (json.JSONDecodeError, OSError):
            existing = []
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(existing + new_data, f, indent=2, default=str)
    print(f"JSON {'appended to' if existing else 'written'}: {filepath}")


# =============================================================================
# Console summary
# =============================================================================

def _row(name: str, mean, std, p50, p90, p95, mn, mx, unit: str) -> str:
    return (f"  {name:<18} {mean:>10.2f} {std:>8.2f} {p50:>10.2f} "
            f"{p90:>10.2f} {p95:>10.2f} {mn:>8.2f} {mx:>8.2f}  [{unit}]")


def print_gen_summary(r: GenBenchmarkResult) -> None:
    sep = "=" * 80
    print(f"\n{sep}")
    print(f"  Result: {r.model_name} | {r.device} | {r.mode} | "
          f"prompt[{r.prompt_idx}] in={r.input_tokens} out_max={r.infer_count_arg}")
    print(f"  Pipeline: {r.pipeline}  measured_iters={r.measured_iters}  compile={r.compile_time_s:.2f}s")
    print(sep)
    print(f"  {'Metric':<18} {'Mean':>10} {'Std':>8} {'P50':>10} {'P90':>10} "
          f"{'P95':>10} {'Min':>8} {'Max':>8}  Unit")
    print(f"  {'-'*18} {'-'*10} {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*8}  ----")
    print(_row("TTFT", r.ttft_mean_ms, r.ttft_std_ms, r.ttft_p50_ms, r.ttft_p90_ms,
               r.ttft_p95_ms, r.ttft_min_ms, r.ttft_max_ms, "ms"))
    print(_row("TPOT", r.tpot_mean_ms, r.tpot_std_ms, r.tpot_p50_ms, r.tpot_p90_ms,
               r.tpot_p95_ms, r.tpot_min_ms, r.tpot_max_ms, "ms"))
    print(_row("1st infer", r.first_token_infer_mean_ms, r.first_token_infer_std_ms,
               r.first_token_infer_p50_ms, r.first_token_infer_p90_ms,
               r.first_token_infer_p95_ms, r.first_token_infer_min_ms,
               r.first_token_infer_max_ms, "ms"))
    print(_row("2nd infer", r.other_tokens_infer_mean_ms, r.other_tokens_infer_std_ms,
               r.other_tokens_infer_p50_ms, r.other_tokens_infer_p90_ms,
               r.other_tokens_infer_p95_ms, r.other_tokens_infer_min_ms,
               r.other_tokens_infer_max_ms, "ms"))
    print(_row("Throughput", r.throughput_mean_tps, r.throughput_std_tps,
               r.throughput_p50_tps, r.throughput_p90_tps, r.throughput_p95_tps,
               r.throughput_min_tps, r.throughput_max_tps, "tok/s"))
    print(_row("Decode TPS", r.decode_throughput_mean_tps, r.decode_throughput_std_tps,
               r.decode_throughput_p50_tps, r.decode_throughput_p90_tps,
               r.decode_throughput_p95_tps, r.decode_throughput_min_tps,
               r.decode_throughput_max_tps, "tok/s"))
    print(_row("Output tokens", r.output_tokens_mean, r.output_tokens_std,
               r.output_tokens_p50, r.output_tokens_p90, r.output_tokens_p95,
               r.output_tokens_min, r.output_tokens_max, "tok"))
    print(_row("Gen time", r.generation_time_mean_s, r.generation_time_std_s,
               r.generation_time_p50_s, r.generation_time_p90_s, r.generation_time_p95_s,
               r.generation_time_min_s, r.generation_time_max_s, "s"))
    print(_row("Prefill", r.prefill_mean_ms, r.prefill_std_ms, r.prefill_p50_ms,
               r.prefill_p90_ms, r.prefill_p95_ms, r.prefill_min_ms, r.prefill_max_ms, "ms"))
    print(_row("Decode", r.decode_mean_ms, r.decode_std_ms, r.decode_p50_ms,
               r.decode_p90_ms, r.decode_p95_ms, r.decode_min_ms, r.decode_max_ms, "ms"))
    print(_row("E2E", r.e2e_mean_ms, r.e2e_std_ms, r.e2e_p50_ms, r.e2e_p90_ms,
               r.e2e_p95_ms, r.e2e_min_ms, r.e2e_max_ms, "ms"))
    print(_row("Tokenization", r.tokenization_mean_ms, r.tokenization_std_ms,
               r.tokenization_p50_ms, r.tokenization_p90_ms, r.tokenization_p95_ms,
               r.tokenization_min_ms, r.tokenization_max_ms, "ms"))
    print(_row("Detokenization", r.detokenization_mean_ms, r.detokenization_std_ms,
               r.detokenization_p50_ms, r.detokenization_p90_ms, r.detokenization_p95_ms,
               r.detokenization_min_ms, r.detokenization_max_ms, "ms"))
    _print_memory_block(r)
    _print_util_block(r.cpu_util, r.gpu_util, r.npu_util)
    print(sep)


def print_bert_summary(r: BertBenchmarkResult) -> None:
    sep = "=" * 80
    print(f"\n{sep}")
    print(f"  Result: {r.model_name} | {r.device} | input={r.input_length} | "
          f"compression={r.compression_rate:.0%} keep")
    print(f"  measured_iters={r.measure_iters}  compile={r.compile_time_s:.2f}s")
    print(sep)
    print(f"  {'Metric':<18} {'Mean':>10} {'Std':>8} {'P50':>10} {'P90':>10} "
          f"{'P95':>10} {'Min':>8} {'Max':>8}  Unit")
    print(f"  {'-'*18} {'-'*10} {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*8}  ----")
    print(_row("Latency", r.latency_mean_ms, r.latency_std_ms, r.latency_p50_ms,
               r.latency_p90_ms, r.latency_p95_ms, r.latency_min_ms, r.latency_max_ms, "ms"))
    print(_row("Thrpt(samples)", r.throughput_samples_mean, r.throughput_samples_std,
               r.throughput_samples_p50, r.throughput_samples_p90, r.throughput_samples_p95,
               r.throughput_samples_min, r.throughput_samples_max, "samp/s"))
    print(_row("Thrpt(tokens)", r.throughput_tokens_mean, r.throughput_tokens_std,
               r.throughput_tokens_p50, r.throughput_tokens_p90, r.throughput_tokens_p95,
               r.throughput_tokens_min, r.throughput_tokens_max, "tok/s"))
    print(f"\n  --- Compression ---")
    print(f"  Avg input tokens:      {r.avg_input_tokens:.0f}")
    print(f"  Avg compressed tokens: {r.avg_compressed_tokens:.0f}")
    print(f"  Avg keep ratio:        {r.avg_compression_ratio:.2%} (lower = more compressed)")
    _print_memory_block(r)
    _print_util_block(r.cpu_util, r.gpu_util, r.npu_util)
    print(sep)


def _print_memory_block(r) -> None:
    print("\n  --- Memory ---")
    if not getattr(r, "memory_available", False):
        print("  Memory monitor: unavailable/disabled")
        return
    print(f"  Initial RSS/System: {r.initial_rss_mem_mib:.1f} / {r.initial_sys_mem_mib:.1f} MiB")
    print(f"  Compile max RSS/System: {r.compile_max_rss_mem_mib:.1f} / {r.compile_max_sys_mem_mib:.1f} MiB")
    print(f"  Runtime max RSS/System: {r.runtime_max_rss_mem_mib:.1f} / {r.runtime_max_sys_mem_mib:.1f} MiB")
    print(f"  Runtime max increase RSS/System: {r.runtime_max_rss_mem_increase_mib:.1f} / "
          f"{r.runtime_max_sys_mem_increase_mib:.1f} MiB ({r.memory_samples} samples)")


def print_embed_summary(r: EmbedBenchmarkResult) -> None:
    sep = "=" * 80
    print(f"\n{sep}")
    print(f"  Result: {r.model_name} | {r.device} | {r.precision} | mode={r.mode}")
    print(f"  Pipeline: {r.pipeline}  pooling={r.pooling_type or '(default)'} "
          f"normalize={r.normalize} batch={r.batch_size} dim={r.embedding_dim}")
    print(f"  input_tokens(max/total)={r.input_tokens}/{r.total_input_tokens}  "
          f"measured_iters={r.measured_iters}  compile={r.compile_time_s:.2f}s")
    print(sep)
    print(f"  {'Metric':<18} {'Mean':>10} {'Std':>8} {'P50':>10} {'P90':>10} "
          f"{'P95':>10} {'Min':>8} {'Max':>8}  Unit")
    print(f"  {'-'*18} {'-'*10} {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*8}  ----")
    print(_row("Latency", r.latency_mean_ms, r.latency_std_ms, r.latency_p50_ms,
               r.latency_p90_ms, r.latency_p95_ms, r.latency_min_ms, r.latency_max_ms, "ms"))
    print(_row("E2E (+token)", r.e2e_mean_ms, r.e2e_std_ms, r.e2e_p50_ms,
               r.e2e_p90_ms, r.e2e_p95_ms, r.e2e_min_ms, r.e2e_max_ms, "ms"))
    print(_row("Tokenization", r.tokenization_mean_ms, r.tokenization_std_ms, r.tokenization_p50_ms,
               r.tokenization_p90_ms, r.tokenization_p95_ms, r.tokenization_min_ms, r.tokenization_max_ms, "ms"))
    print(_row("Thrpt(docs)", r.throughput_docs_mean, r.throughput_docs_std,
               r.throughput_docs_p50, r.throughput_docs_p90, r.throughput_docs_p95,
               r.throughput_docs_min, r.throughput_docs_max, "doc/s"))
    print(_row("Thrpt(tokens)", r.throughput_tokens_mean, r.throughput_tokens_std,
               r.throughput_tokens_p50, r.throughput_tokens_p90, r.throughput_tokens_p95,
               r.throughput_tokens_min, r.throughput_tokens_max, "tok/s"))
    _print_memory_block(r)
    _print_util_block(r.cpu_util, r.gpu_util, r.npu_util)
    print(sep)


def print_rerank_summary(r: RerankBenchmarkResult) -> None:
    sep = "=" * 80
    print(f"\n{sep}")
    print(f"  Result: {r.model_name} | {r.device} | {r.precision}")
    print(f"  Pipeline: {r.pipeline}  documents={r.num_documents} top_n={r.top_n} "
          f"max_length={r.max_length or '(default)'}")
    print(f"  input_tokens(pair_max/total)={r.input_tokens}/{r.total_input_tokens}  "
          f"measured_iters={r.measured_iters}  compile={r.compile_time_s:.2f}s")
    print(sep)
    print(f"  {'Metric':<18} {'Mean':>10} {'Std':>8} {'P50':>10} {'P90':>10} "
          f"{'P95':>10} {'Min':>8} {'Max':>8}  Unit")
    print(f"  {'-'*18} {'-'*10} {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*8}  ----")
    print(_row("Latency", r.latency_mean_ms, r.latency_std_ms, r.latency_p50_ms,
               r.latency_p90_ms, r.latency_p95_ms, r.latency_min_ms, r.latency_max_ms, "ms"))
    print(_row("E2E (+token)", r.e2e_mean_ms, r.e2e_std_ms, r.e2e_p50_ms,
               r.e2e_p90_ms, r.e2e_p95_ms, r.e2e_min_ms, r.e2e_max_ms, "ms"))
    print(_row("Tokenization", r.tokenization_mean_ms, r.tokenization_std_ms, r.tokenization_p50_ms,
               r.tokenization_p90_ms, r.tokenization_p95_ms, r.tokenization_min_ms, r.tokenization_max_ms, "ms"))
    print(_row("Thrpt(docs)", r.throughput_docs_mean, r.throughput_docs_std,
               r.throughput_docs_p50, r.throughput_docs_p90, r.throughput_docs_p95,
               r.throughput_docs_min, r.throughput_docs_max, "doc/s"))
    print(_row("Thrpt(tokens)", r.throughput_tokens_mean, r.throughput_tokens_std,
               r.throughput_tokens_p50, r.throughput_tokens_p90, r.throughput_tokens_p95,
               r.throughput_tokens_min, r.throughput_tokens_max, "tok/s"))
    _print_memory_block(r)
    _print_util_block(r.cpu_util, r.gpu_util, r.npu_util)
    print(sep)


def _print_util_block(cpu: dict, gpu: dict, npu: dict) -> None:
    print("\n  --- Device Utilization ---")
    if cpu.get("avg_total", 0) > 0:
        print(f"  CPU avg: {cpu['avg_total']:.1f}%")
    if gpu.get("available"):
        print(f"  GPU avg: {gpu.get('avg_util', 0):.1f}% ({gpu.get('samples', 0)} samples)")
    if npu.get("available"):
        gen = f" [{npu.get('cpu_gen', 'unknown')}]" if npu.get("cpu_gen") else ""
        print(f"  NPU avg: {npu.get('avg_util', 0):.1f}%{gen} ({npu.get('samples', 0)} samples)")
