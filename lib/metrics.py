"""Dataclasses representing per-iteration and aggregated benchmark results.

Different task shapes are needed:
- GenIterationMetrics   / GenBenchmarkResult   (Qwen3 LLM + Qwen3.5 VLM generation)
- BertIterationMetrics  / BertBenchmarkResult  (LLMLingua-2 token classification)
- EmbedIterationMetrics / EmbedBenchmarkResult (Qwen3-Embedding feature extraction)
- RerankIterationMetrics/ RerankBenchmarkResult(Qwen3-Reranker query/document scoring)

To keep the CSV/JSON consistent across qwen3 and qwen3.5 we share most fields;
differences are kept narrow (mode / image / etc.). Embedding and reranking are
single-shot encoder/decoder forward passes (no autoregressive decode), so they
use a latency/throughput schema closer to the BERT one than to the generation
one — there is no TTFT/TPOT, and "throughput" is documents-or-tokens per second.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import statistics


# =============================================================================
# Generation (LLM + VLM share an identical schema)
# =============================================================================

@dataclass
class GenIterationMetrics:
    iteration: int = 0
    is_warmup: bool = False
    prompt_idx: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    infer_count: int = 0
    generate_time_s: float = 0.0
    ttft_ms: float = 0.0
    tpot_ms: float = 0.0
    first_token_infer_latency_ms: float = -1.0
    other_tokens_infer_avg_latency_ms: float = -1.0
    prefill_latency_ms: float = 0.0
    decode_latency_ms: float = 0.0
    e2e_latency_ms: float = 0.0
    throughput_tps: float = 0.0
    decode_throughput_tps: float = 0.0
    tokenization_time_ms: float = 0.0
    detokenization_time_ms: float = 0.0
    result_md5: str = ""
    start: str = ""
    end: str = ""
    memory_available: bool = False
    memory_samples: int = 0
    max_rss_mem_mib: float = 0.0
    max_sys_mem_mib: float = 0.0
    max_rss_mem_increase_mib: float = 0.0
    max_sys_mem_increase_mib: float = 0.0
    max_rss_mem_share: float = 0.0
    max_sys_mem_share: float = 0.0


@dataclass
class GenBenchmarkResult:
    # Identification
    model_dir: str = ""
    model_name: str = ""
    framework: str = "ov"
    pipeline: str = ""              # "LLMPipeline" | "VLMPipeline" | "OptimumVLM"
    device: str = ""
    precision: str = ""
    mode: str = "text-only"          # "text-only" | "multimodal"

    # Run config
    num_beams: int = 1
    batch_size: int = 1
    apply_chat_template: bool = False
    disable_prompt_permutation: bool = False
    prompt_idx: int = 0
    input_tokens: int = 0
    infer_count_arg: int = 0
    exact_output_tokens: bool = False
    num_iters: int = 0
    measured_iters: int = 0
    compile_time_s: float = 0.0

    # Output / preprocessing stats
    output_tokens_mean: float = 0.0; output_tokens_std: float = 0.0
    output_tokens_p50: float = 0.0; output_tokens_p90: float = 0.0; output_tokens_p95: float = 0.0
    output_tokens_min: float = 0.0; output_tokens_max: float = 0.0

    generation_time_mean_s: float = 0.0; generation_time_std_s: float = 0.0
    generation_time_p50_s: float = 0.0; generation_time_p90_s: float = 0.0; generation_time_p95_s: float = 0.0
    generation_time_min_s: float = 0.0; generation_time_max_s: float = 0.0

    tokenization_mean_ms: float = 0.0; tokenization_std_ms: float = 0.0
    tokenization_p50_ms: float = 0.0; tokenization_p90_ms: float = 0.0; tokenization_p95_ms: float = 0.0
    tokenization_min_ms: float = 0.0; tokenization_max_ms: float = 0.0

    detokenization_mean_ms: float = 0.0; detokenization_std_ms: float = 0.0
    detokenization_p50_ms: float = 0.0; detokenization_p90_ms: float = 0.0; detokenization_p95_ms: float = 0.0
    detokenization_min_ms: float = 0.0; detokenization_max_ms: float = 0.0

    # Memory summaries (MiB, process RSS includes child processes when psutil is available)
    memory_available: bool = False
    memory_samples: int = 0
    initial_rss_mem_mib: float = 0.0
    initial_sys_mem_mib: float = 0.0
    initial_rss_mem_share: float = 0.0
    initial_sys_mem_share: float = 0.0
    compile_max_rss_mem_mib: float = 0.0
    compile_max_sys_mem_mib: float = 0.0
    compile_max_rss_mem_increase_mib: float = 0.0
    compile_max_sys_mem_increase_mib: float = 0.0
    compile_max_rss_mem_share: float = 0.0
    compile_max_sys_mem_share: float = 0.0
    runtime_mean_rss_mem_mib: float = 0.0
    runtime_max_rss_mem_mib: float = 0.0
    runtime_mean_sys_mem_mib: float = 0.0
    runtime_max_sys_mem_mib: float = 0.0
    runtime_mean_rss_mem_increase_mib: float = 0.0
    runtime_max_rss_mem_increase_mib: float = 0.0
    runtime_mean_sys_mem_increase_mib: float = 0.0
    runtime_max_sys_mem_increase_mib: float = 0.0
    runtime_max_rss_mem_share: float = 0.0
    runtime_max_sys_mem_share: float = 0.0

    # Stats (filled via fill_result_stats)
    ttft_mean_ms: float = 0.0; ttft_std_ms: float = 0.0
    ttft_p50_ms: float = 0.0; ttft_p90_ms: float = 0.0; ttft_p95_ms: float = 0.0
    ttft_min_ms: float = 0.0; ttft_max_ms: float = 0.0

    tpot_mean_ms: float = 0.0; tpot_std_ms: float = 0.0
    tpot_p50_ms: float = 0.0; tpot_p90_ms: float = 0.0; tpot_p95_ms: float = 0.0
    tpot_min_ms: float = 0.0; tpot_max_ms: float = 0.0

    first_token_infer_mean_ms: float = 0.0; first_token_infer_std_ms: float = 0.0
    first_token_infer_p50_ms: float = 0.0; first_token_infer_p90_ms: float = 0.0; first_token_infer_p95_ms: float = 0.0
    first_token_infer_min_ms: float = 0.0; first_token_infer_max_ms: float = 0.0

    other_tokens_infer_mean_ms: float = 0.0; other_tokens_infer_std_ms: float = 0.0
    other_tokens_infer_p50_ms: float = 0.0; other_tokens_infer_p90_ms: float = 0.0; other_tokens_infer_p95_ms: float = 0.0
    other_tokens_infer_min_ms: float = 0.0; other_tokens_infer_max_ms: float = 0.0

    throughput_mean_tps: float = 0.0; throughput_std_tps: float = 0.0
    throughput_p50_tps: float = 0.0; throughput_p90_tps: float = 0.0; throughput_p95_tps: float = 0.0
    throughput_min_tps: float = 0.0; throughput_max_tps: float = 0.0

    decode_throughput_mean_tps: float = 0.0; decode_throughput_std_tps: float = 0.0
    decode_throughput_p50_tps: float = 0.0; decode_throughput_p90_tps: float = 0.0; decode_throughput_p95_tps: float = 0.0
    decode_throughput_min_tps: float = 0.0; decode_throughput_max_tps: float = 0.0

    prefill_mean_ms: float = 0.0; prefill_std_ms: float = 0.0
    prefill_p50_ms: float = 0.0; prefill_p90_ms: float = 0.0; prefill_p95_ms: float = 0.0
    prefill_min_ms: float = 0.0; prefill_max_ms: float = 0.0

    decode_mean_ms: float = 0.0; decode_std_ms: float = 0.0
    decode_p50_ms: float = 0.0; decode_p90_ms: float = 0.0; decode_p95_ms: float = 0.0
    decode_min_ms: float = 0.0; decode_max_ms: float = 0.0

    e2e_mean_ms: float = 0.0; e2e_std_ms: float = 0.0
    e2e_p50_ms: float = 0.0; e2e_p90_ms: float = 0.0; e2e_p95_ms: float = 0.0
    e2e_min_ms: float = 0.0; e2e_max_ms: float = 0.0

    # Resource utilization summaries
    cpu_util: dict = field(default_factory=dict)
    gpu_util: dict = field(default_factory=dict)
    npu_util: dict = field(default_factory=dict)

    # Per-iteration history
    iterations: list = field(default_factory=list)

    # Provenance
    image_path: str = ""
    openvino_version: str = ""
    openvino_genai_version: str = ""
    run_started_at: str = ""


# =============================================================================
# BERT-style classification (LLMLingua-2)
# =============================================================================

@dataclass
class BertIterationMetrics:
    iteration: int = 0
    input_tokens: int = 0
    latency_ms: float = 0.0
    throughput_samples_per_s: float = 0.0
    throughput_tokens_per_s: float = 0.0
    compressed_tokens: int = 0
    compression_ratio: float = 0.0
    memory_available: bool = False
    memory_samples: int = 0
    max_rss_mem_mib: float = 0.0
    max_sys_mem_mib: float = 0.0
    max_rss_mem_increase_mib: float = 0.0
    max_sys_mem_increase_mib: float = 0.0
    max_rss_mem_share: float = 0.0
    max_sys_mem_share: float = 0.0


@dataclass
class BertBenchmarkResult:
    model_dir: str = ""
    model_name: str = ""
    framework: str = "ov"
    device: str = ""
    precision: str = ""
    input_length: int = 0
    compression_rate: float = 0.0
    warmup_iters: int = 0
    measure_iters: int = 0
    compile_time_s: float = 0.0
    iterations: list = field(default_factory=list)

    latency_mean_ms: float = 0.0; latency_std_ms: float = 0.0
    latency_p50_ms: float = 0.0; latency_p90_ms: float = 0.0; latency_p95_ms: float = 0.0
    latency_min_ms: float = 0.0; latency_max_ms: float = 0.0

    throughput_samples_mean: float = 0.0; throughput_samples_std: float = 0.0
    throughput_samples_p50: float = 0.0; throughput_samples_p90: float = 0.0; throughput_samples_p95: float = 0.0
    throughput_samples_min: float = 0.0; throughput_samples_max: float = 0.0

    throughput_tokens_mean: float = 0.0; throughput_tokens_std: float = 0.0
    throughput_tokens_p50: float = 0.0; throughput_tokens_p90: float = 0.0; throughput_tokens_p95: float = 0.0
    throughput_tokens_min: float = 0.0; throughput_tokens_max: float = 0.0

    avg_input_tokens: float = 0.0
    avg_compressed_tokens: float = 0.0
    avg_compression_ratio: float = 0.0

    cpu_util: dict = field(default_factory=dict)
    gpu_util: dict = field(default_factory=dict)
    npu_util: dict = field(default_factory=dict)

    memory_available: bool = False
    memory_samples: int = 0
    initial_rss_mem_mib: float = 0.0
    initial_sys_mem_mib: float = 0.0
    initial_rss_mem_share: float = 0.0
    initial_sys_mem_share: float = 0.0
    compile_max_rss_mem_mib: float = 0.0
    compile_max_sys_mem_mib: float = 0.0
    compile_max_rss_mem_increase_mib: float = 0.0
    compile_max_sys_mem_increase_mib: float = 0.0
    compile_max_rss_mem_share: float = 0.0
    compile_max_sys_mem_share: float = 0.0
    runtime_mean_rss_mem_mib: float = 0.0
    runtime_max_rss_mem_mib: float = 0.0
    runtime_mean_sys_mem_mib: float = 0.0
    runtime_max_sys_mem_mib: float = 0.0
    runtime_mean_rss_mem_increase_mib: float = 0.0
    runtime_max_rss_mem_increase_mib: float = 0.0
    runtime_mean_sys_mem_increase_mib: float = 0.0
    runtime_max_sys_mem_increase_mib: float = 0.0
    runtime_max_rss_mem_share: float = 0.0
    runtime_max_sys_mem_share: float = 0.0

    openvino_version: str = ""
    run_started_at: str = ""


# =============================================================================
# Text Embedding (Qwen3-Embedding) — single forward pass, no decode loop
# =============================================================================

@dataclass
class EmbedIterationMetrics:
    iteration: int = 0
    is_warmup: bool = False
    prompt_idx: int = 0
    mode: str = "documents"          # "documents" | "query"
    batch_size: int = 1              # number of texts encoded in this call
    input_tokens: int = 0            # per-text token count (max across batch)
    total_input_tokens: int = 0      # sum of token counts across the batch
    embedding_dim: int = 0
    tokenization_time_ms: float = 0.0   # host-side encode used only for token accounting
    latency_ms: float = 0.0          # pure pipeline call (embed_documents/embed_query)
    e2e_latency_ms: float = 0.0      # tokenization accounting + latency
    throughput_docs_per_s: float = 0.0
    throughput_tokens_per_s: float = 0.0
    result_md5: str = ""
    start: str = ""
    end: str = ""
    memory_available: bool = False
    memory_samples: int = 0
    max_rss_mem_mib: float = 0.0
    max_sys_mem_mib: float = 0.0
    max_rss_mem_increase_mib: float = 0.0
    max_sys_mem_increase_mib: float = 0.0
    max_rss_mem_share: float = 0.0
    max_sys_mem_share: float = 0.0


@dataclass
class RerankIterationMetrics:
    iteration: int = 0
    is_warmup: bool = False
    prompt_idx: int = 0
    num_documents: int = 0           # candidate documents scored in this call
    top_n: int = 0                   # number of results requested back
    input_tokens: int = 0            # per query/document pair token count (max)
    total_input_tokens: int = 0      # sum across all pairs
    tokenization_time_ms: float = 0.0
    latency_ms: float = 0.0          # pure pipeline rerank() call
    e2e_latency_ms: float = 0.0
    throughput_docs_per_s: float = 0.0   # documents scored per second
    throughput_tokens_per_s: float = 0.0
    top1_index: int = -1
    top1_score: float = 0.0
    result_md5: str = ""
    start: str = ""
    end: str = ""
    memory_available: bool = False
    memory_samples: int = 0
    max_rss_mem_mib: float = 0.0
    max_sys_mem_mib: float = 0.0
    max_rss_mem_increase_mib: float = 0.0
    max_sys_mem_increase_mib: float = 0.0
    max_rss_mem_share: float = 0.0
    max_sys_mem_share: float = 0.0


def _latency_throughput_block():
    """Shared field block for the encoder-style summary dataclasses below.

    Returned as a list of (name, default) is not possible with @dataclass, so
    EmbedBenchmarkResult / RerankBenchmarkResult declare these explicitly. This
    function exists only as documentation of the intended common schema.
    """
    raise NotImplementedError  # pragma: no cover


@dataclass
class EmbedBenchmarkResult:
    model_dir: str = ""
    model_name: str = ""
    framework: str = "ov"
    pipeline: str = "TextEmbeddingPipeline"
    device: str = ""
    precision: str = ""
    mode: str = "documents"          # "documents" | "query"

    # Run config
    batch_size: int = 1
    pooling_type: str = ""
    normalize: bool = False
    max_length: int = 0              # 0 = pipeline default (unset)
    padding_side: str = ""
    pad_to_max_length: bool = False
    query_instruction: str = ""
    embed_instruction: str = ""
    prompt_idx: int = 0
    input_tokens: int = 0            # representative per-text token count
    total_input_tokens: int = 0
    embedding_dim: int = 0
    num_iters: int = 0
    measured_iters: int = 0
    compile_time_s: float = 0.0

    latency_mean_ms: float = 0.0; latency_std_ms: float = 0.0
    latency_p50_ms: float = 0.0; latency_p90_ms: float = 0.0; latency_p95_ms: float = 0.0
    latency_min_ms: float = 0.0; latency_max_ms: float = 0.0

    e2e_mean_ms: float = 0.0; e2e_std_ms: float = 0.0
    e2e_p50_ms: float = 0.0; e2e_p90_ms: float = 0.0; e2e_p95_ms: float = 0.0
    e2e_min_ms: float = 0.0; e2e_max_ms: float = 0.0

    tokenization_mean_ms: float = 0.0; tokenization_std_ms: float = 0.0
    tokenization_p50_ms: float = 0.0; tokenization_p90_ms: float = 0.0; tokenization_p95_ms: float = 0.0
    tokenization_min_ms: float = 0.0; tokenization_max_ms: float = 0.0

    throughput_docs_mean: float = 0.0; throughput_docs_std: float = 0.0
    throughput_docs_p50: float = 0.0; throughput_docs_p90: float = 0.0; throughput_docs_p95: float = 0.0
    throughput_docs_min: float = 0.0; throughput_docs_max: float = 0.0

    throughput_tokens_mean: float = 0.0; throughput_tokens_std: float = 0.0
    throughput_tokens_p50: float = 0.0; throughput_tokens_p90: float = 0.0; throughput_tokens_p95: float = 0.0
    throughput_tokens_min: float = 0.0; throughput_tokens_max: float = 0.0

    cpu_util: dict = field(default_factory=dict)
    gpu_util: dict = field(default_factory=dict)
    npu_util: dict = field(default_factory=dict)

    memory_available: bool = False
    memory_samples: int = 0
    initial_rss_mem_mib: float = 0.0
    initial_sys_mem_mib: float = 0.0
    initial_rss_mem_share: float = 0.0
    initial_sys_mem_share: float = 0.0
    compile_max_rss_mem_mib: float = 0.0
    compile_max_sys_mem_mib: float = 0.0
    compile_max_rss_mem_increase_mib: float = 0.0
    compile_max_sys_mem_increase_mib: float = 0.0
    compile_max_rss_mem_share: float = 0.0
    compile_max_sys_mem_share: float = 0.0
    runtime_mean_rss_mem_mib: float = 0.0
    runtime_max_rss_mem_mib: float = 0.0
    runtime_mean_sys_mem_mib: float = 0.0
    runtime_max_sys_mem_mib: float = 0.0
    runtime_mean_rss_mem_increase_mib: float = 0.0
    runtime_max_rss_mem_increase_mib: float = 0.0
    runtime_mean_sys_mem_increase_mib: float = 0.0
    runtime_max_sys_mem_increase_mib: float = 0.0
    runtime_max_rss_mem_share: float = 0.0
    runtime_max_sys_mem_share: float = 0.0

    iterations: list = field(default_factory=list)

    openvino_version: str = ""
    openvino_genai_version: str = ""
    run_started_at: str = ""


@dataclass
class RerankBenchmarkResult:
    model_dir: str = ""
    model_name: str = ""
    framework: str = "ov"
    pipeline: str = "TextRerankPipeline"
    device: str = ""
    precision: str = ""

    # Run config
    num_documents: int = 0
    top_n: int = 0
    max_length: int = 0              # 0 = pipeline default (unset)
    padding_side: str = ""
    pad_to_max_length: bool = False
    prompt_idx: int = 0
    input_tokens: int = 0
    total_input_tokens: int = 0
    num_iters: int = 0
    measured_iters: int = 0
    compile_time_s: float = 0.0

    latency_mean_ms: float = 0.0; latency_std_ms: float = 0.0
    latency_p50_ms: float = 0.0; latency_p90_ms: float = 0.0; latency_p95_ms: float = 0.0
    latency_min_ms: float = 0.0; latency_max_ms: float = 0.0

    e2e_mean_ms: float = 0.0; e2e_std_ms: float = 0.0
    e2e_p50_ms: float = 0.0; e2e_p90_ms: float = 0.0; e2e_p95_ms: float = 0.0
    e2e_min_ms: float = 0.0; e2e_max_ms: float = 0.0

    tokenization_mean_ms: float = 0.0; tokenization_std_ms: float = 0.0
    tokenization_p50_ms: float = 0.0; tokenization_p90_ms: float = 0.0; tokenization_p95_ms: float = 0.0
    tokenization_min_ms: float = 0.0; tokenization_max_ms: float = 0.0

    throughput_docs_mean: float = 0.0; throughput_docs_std: float = 0.0
    throughput_docs_p50: float = 0.0; throughput_docs_p90: float = 0.0; throughput_docs_p95: float = 0.0
    throughput_docs_min: float = 0.0; throughput_docs_max: float = 0.0

    throughput_tokens_mean: float = 0.0; throughput_tokens_std: float = 0.0
    throughput_tokens_p50: float = 0.0; throughput_tokens_p90: float = 0.0; throughput_tokens_p95: float = 0.0
    throughput_tokens_min: float = 0.0; throughput_tokens_max: float = 0.0

    cpu_util: dict = field(default_factory=dict)
    gpu_util: dict = field(default_factory=dict)
    npu_util: dict = field(default_factory=dict)

    memory_available: bool = False
    memory_samples: int = 0
    initial_rss_mem_mib: float = 0.0
    initial_sys_mem_mib: float = 0.0
    initial_rss_mem_share: float = 0.0
    initial_sys_mem_share: float = 0.0
    compile_max_rss_mem_mib: float = 0.0
    compile_max_sys_mem_mib: float = 0.0
    compile_max_rss_mem_increase_mib: float = 0.0
    compile_max_sys_mem_increase_mib: float = 0.0
    compile_max_rss_mem_share: float = 0.0
    compile_max_sys_mem_share: float = 0.0
    runtime_mean_rss_mem_mib: float = 0.0
    runtime_max_rss_mem_mib: float = 0.0
    runtime_mean_sys_mem_mib: float = 0.0
    runtime_max_sys_mem_mib: float = 0.0
    runtime_mean_rss_mem_increase_mib: float = 0.0
    runtime_max_rss_mem_increase_mib: float = 0.0
    runtime_mean_sys_mem_increase_mib: float = 0.0
    runtime_max_sys_mem_increase_mib: float = 0.0
    runtime_max_rss_mem_share: float = 0.0
    runtime_max_sys_mem_share: float = 0.0

    iterations: list = field(default_factory=list)

    openvino_version: str = ""
    openvino_genai_version: str = ""
    run_started_at: str = ""


# =============================================================================
# Helpers to fill stats blocks
# =============================================================================

def fill_gen_stats(result: GenBenchmarkResult, agg: dict) -> None:
    """Write aggregated stats into a GenBenchmarkResult.

    `agg` is a dict like {"ttft": {...}, "tpot": {...}, ...} produced by
    aggregating per-iteration metrics — see lib.genai_runner.aggregate.
    """
    def assign(group: str, prefix: str, unit: str = "ms"):
        s = agg.get(group)
        if not s:
            return
        for stat in ("mean", "std", "p50", "p90", "p95", "min", "max"):
            if unit == "ms":
                attr = f"{prefix}_{stat}_ms"
            else:
                attr = f"{prefix}_{stat}_tps"
            setattr(result, attr, s[stat])
    assign("ttft", "ttft")
    assign("tpot", "tpot")
    assign("first_token_infer", "first_token_infer")
    assign("other_tokens_infer", "other_tokens_infer")
    assign("throughput", "throughput", unit="tps")
    assign("decode_throughput", "decode_throughput", unit="tps")
    assign("prefill", "prefill")
    assign("decode", "decode")
    assign("e2e", "e2e")

    def assign_plain(group: str, prefix: str):
        s = agg.get(group)
        if not s:
            return
        for stat in ("mean", "std", "p50", "p90", "p95", "min", "max"):
            setattr(result, f"{prefix}_{stat}", s[stat])

    def assign_seconds(group: str, prefix: str):
        s = agg.get(group)
        if not s:
            return
        for stat in ("mean", "std", "p50", "p90", "p95", "min", "max"):
            setattr(result, f"{prefix}_{stat}_s", s[stat])

    def assign_ms(group: str, prefix: str):
        s = agg.get(group)
        if not s:
            return
        for stat in ("mean", "std", "p50", "p90", "p95", "min", "max"):
            setattr(result, f"{prefix}_{stat}_ms", s[stat])

    assign_plain("output_tokens", "output_tokens")
    assign_seconds("generation_time", "generation_time")
    assign_ms("tokenization", "tokenization")
    assign_ms("detokenization", "detokenization")


def fill_bert_stats(result: BertBenchmarkResult,
                    latency: dict, thr_samples: dict, thr_tokens: dict,
                    avg_input: float, avg_compressed: float, avg_ratio: float) -> None:
    for stat in ("mean", "std", "p50", "p90", "p95", "min", "max"):
        setattr(result, f"latency_{stat}_ms", latency[stat])
    for stat in ("mean", "std", "p50", "p90", "p95"):
        setattr(result, f"throughput_samples_{stat}", thr_samples[stat])
        setattr(result, f"throughput_tokens_{stat}", thr_tokens[stat])
    for stat in ("min", "max"):
        setattr(result, f"throughput_samples_{stat}", thr_samples[stat])
        setattr(result, f"throughput_tokens_{stat}", thr_tokens[stat])
    result.avg_input_tokens = avg_input
    result.avg_compressed_tokens = avg_compressed
    result.avg_compression_ratio = avg_ratio


def _assign_stat_block(result, agg: dict, group: str, prefix: str, suffix: str = "") -> None:
    """Generic stat assignment: result.<prefix>_<stat><suffix> = agg[group][stat]."""
    s = agg.get(group)
    if not s:
        return
    for stat in ("mean", "std", "p50", "p90", "p95", "min", "max"):
        setattr(result, f"{prefix}_{stat}{suffix}", s[stat])


def fill_encoder_stats(result, agg: dict) -> None:
    """Fill the latency / e2e / tokenization / throughput blocks shared by the
    EmbedBenchmarkResult and RerankBenchmarkResult encoder-style schemas.
    """
    _assign_stat_block(result, agg, "latency", "latency", "_ms")
    _assign_stat_block(result, agg, "e2e", "e2e", "_ms")
    _assign_stat_block(result, agg, "tokenization", "tokenization", "_ms")
    _assign_stat_block(result, agg, "throughput_docs", "throughput_docs")
    _assign_stat_block(result, agg, "throughput_tokens", "throughput_tokens")


def attach_memory_metrics(iteration, memory: dict) -> None:
    if not memory or not memory.get("available"):
        return
    iteration.memory_available = True
    iteration.memory_samples = int(memory.get("samples", 0))
    iteration.max_rss_mem_mib = float(memory.get("max_rss_mem_mib", 0.0))
    iteration.max_sys_mem_mib = float(memory.get("max_sys_mem_mib", 0.0))
    iteration.max_rss_mem_increase_mib = float(memory.get("max_rss_mem_increase_mib", 0.0))
    iteration.max_sys_mem_increase_mib = float(memory.get("max_sys_mem_increase_mib", 0.0))
    iteration.max_rss_mem_share = float(memory.get("max_rss_mem_share", 0.0))
    iteration.max_sys_mem_share = float(memory.get("max_sys_mem_share", 0.0))


def fill_memory_stats(result, initial_memory: dict, compile_memory: dict, iterations: list) -> None:
    if initial_memory and initial_memory.get("available"):
        result.memory_available = True
        result.initial_rss_mem_mib = float(initial_memory.get("rss_mem_mib", 0.0))
        result.initial_sys_mem_mib = float(initial_memory.get("sys_mem_mib", 0.0))
        result.initial_rss_mem_share = float(initial_memory.get("rss_mem_share", 0.0))
        result.initial_sys_mem_share = float(initial_memory.get("sys_mem_share", 0.0))

    if compile_memory and compile_memory.get("available"):
        result.memory_available = True
        result.memory_samples += int(compile_memory.get("samples", 0))
        result.compile_max_rss_mem_mib = float(compile_memory.get("max_rss_mem_mib", 0.0))
        result.compile_max_sys_mem_mib = float(compile_memory.get("max_sys_mem_mib", 0.0))
        result.compile_max_rss_mem_increase_mib = float(compile_memory.get("max_rss_mem_increase_mib", 0.0))
        result.compile_max_sys_mem_increase_mib = float(compile_memory.get("max_sys_mem_increase_mib", 0.0))
        result.compile_max_rss_mem_share = float(compile_memory.get("max_rss_mem_share", 0.0))
        result.compile_max_sys_mem_share = float(compile_memory.get("max_sys_mem_share", 0.0))

    measured = [it for it in iterations if not getattr(it, "is_warmup", False)]
    if not measured:
        measured = list(iterations)
    measured = [it for it in measured if getattr(it, "memory_available", False)]
    if not measured:
        return

    result.memory_available = True
    result.memory_samples += sum(int(getattr(it, "memory_samples", 0)) for it in measured)

    rss = [float(it.max_rss_mem_mib) for it in measured]
    sys_mem = [float(it.max_sys_mem_mib) for it in measured]
    rss_inc = [float(it.max_rss_mem_increase_mib) for it in measured]
    sys_inc = [float(it.max_sys_mem_increase_mib) for it in measured]
    result.runtime_mean_rss_mem_mib = statistics.fmean(rss)
    result.runtime_max_rss_mem_mib = max(rss)
    result.runtime_mean_sys_mem_mib = statistics.fmean(sys_mem)
    result.runtime_max_sys_mem_mib = max(sys_mem)
    result.runtime_mean_rss_mem_increase_mib = statistics.fmean(rss_inc)
    result.runtime_max_rss_mem_increase_mib = max(rss_inc)
    result.runtime_mean_sys_mem_increase_mib = statistics.fmean(sys_inc)
    result.runtime_max_sys_mem_increase_mib = max(sys_inc)
    result.runtime_max_rss_mem_share = max(float(it.max_rss_mem_share) for it in measured)
    result.runtime_max_sys_mem_share = max(float(it.max_sys_mem_share) for it in measured)
