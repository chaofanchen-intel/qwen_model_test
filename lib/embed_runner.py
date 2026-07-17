"""One-iteration helpers for OpenVINO GenAI TextEmbeddingPipeline / TextRerankPipeline.

These two pipelines differ fundamentally from the LLM/VLM generation pipelines:

- They perform a *single* encoder (or decoder-as-encoder) forward pass per call.
  There is no autoregressive decode loop, so there is no TTFT / TPOT / per-token
  cadence to measure — only end-to-end latency and throughput.
- The Python bindings expose **no** ``perf_metrics`` object and **no**
  ``get_tokenizer()`` (verified against openvino_genai 2026.2). So unlike the
  generation runners we cannot pull backend timings; we time the call directly
  with ``time.perf_counter()`` (this is exactly what the official llm_bench
  ``run_text_embeddings_genai`` / ``TextRerankGenAI`` paths do).
- Token accounting uses a *separate* ``openvino_genai.Tokenizer`` loaded from the
  same model dir. This is for reporting tokens/s only; tokenization is kept out
  of the pure-latency window (but folded into the reported e2e latency), matching
  ``tests/benchmark_lingua.py`` and the official tools.

Mirrors:
- openvino.genai/tools/llm_bench/task/text_embeddings.py (run_text_embeddings_genai)
- openvino.genai/tools/llm_bench/task/text_reranker.py    (TextRerankGenAI)
"""

from __future__ import annotations

import datetime
import hashlib
import time
from typing import Optional

from .metrics import EmbedIterationMetrics, RerankIterationMetrics
from .stats import compute_stats


# =============================================================================
# Token accounting (standalone tokenizer; pipelines expose no get_tokenizer)
# =============================================================================

def count_tokens(tokenizer, text: str) -> int:
    """Token count for one text via an ov_genai.Tokenizer. 0 on any failure."""
    if tokenizer is None:
        return 0
    try:
        return int(tokenizer.encode(text).input_ids.shape[-1])
    except Exception:
        return 0


def count_tokens_batch(tokenizer, texts: list[str]) -> tuple[int, int]:
    """Return (max_per_text_tokens, total_tokens) for a batch of texts."""
    if tokenizer is None or not texts:
        return 0, 0
    counts = [count_tokens(tokenizer, t) for t in texts]
    return (max(counts) if counts else 0), int(sum(counts))


def _md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8", errors="ignore"),
                       usedforsecurity=False).hexdigest()


# =============================================================================
# Embedding iteration
# =============================================================================

def run_embed_iteration(pipe, texts: list[str], mode: str, iter_idx: int,
                        is_warmup: bool, prompt_idx: int,
                        tokenizer=None) -> Optional[EmbedIterationMetrics]:
    """One embedding call.

    mode="documents" -> pipe.embed_documents(texts)  (batch of len(texts))
    mode="query"     -> pipe.embed_query(texts[0])   (single text)
    """
    # Token accounting is outside the timed window.
    tok_start = time.perf_counter()
    if mode == "query":
        per_text_tokens = count_tokens(tokenizer, texts[0])
        total_tokens = per_text_tokens
        batch = 1
    else:
        per_text_tokens, total_tokens = count_tokens_batch(tokenizer, texts)
        batch = len(texts)
    tok_ms = (time.perf_counter() - tok_start) * 1000.0

    start_iso = datetime.datetime.now().isoformat()
    t0 = time.perf_counter()
    if mode == "query":
        result = pipe.embed_query(texts[0])
    else:
        result = pipe.embed_documents(texts)
    t1 = time.perf_counter()
    end_iso = datetime.datetime.now().isoformat()

    latency_ms = (t1 - t0) * 1000.0
    e2e_ms = latency_ms + tok_ms

    # result is list[float] (query) or list[list[float]] (documents).
    if mode == "query":
        embedding_dim = len(result) if hasattr(result, "__len__") else 0
        n_vectors = 1
    else:
        n_vectors = len(result) if hasattr(result, "__len__") else 0
        embedding_dim = len(result[0]) if n_vectors and hasattr(result[0], "__len__") else 0

    docs_per_s = (batch / (latency_ms / 1000.0)) if latency_ms > 0 else 0.0
    tokens_per_s = (total_tokens / (latency_ms / 1000.0)) if latency_ms > 0 and total_tokens else 0.0

    metric = EmbedIterationMetrics(
        iteration=iter_idx, is_warmup=is_warmup, prompt_idx=prompt_idx,
        mode=mode, batch_size=batch,
        input_tokens=per_text_tokens, total_input_tokens=total_tokens,
        embedding_dim=embedding_dim,
        tokenization_time_ms=tok_ms, latency_ms=latency_ms, e2e_latency_ms=e2e_ms,
        throughput_docs_per_s=docs_per_s, throughput_tokens_per_s=tokens_per_s,
        result_md5=_md5(repr(result)[:4096]),
        start=start_iso, end=end_iso,
    )
    tag = "[warm-up]" if is_warmup else f"[{iter_idx}]"
    print(f"    {tag}[P{prompt_idx}] mode={mode} batch={batch} tokens(max/total)="
          f"{per_text_tokens}/{total_tokens} dim={embedding_dim} "
          f"lat={latency_ms:.2f}ms docs/s={docs_per_s:.1f} tok/s={tokens_per_s:.0f}")
    return metric


# =============================================================================
# Rerank iteration
# =============================================================================

def run_rerank_iteration(pipe, query: str, documents: list[str], top_n: int,
                         iter_idx: int, is_warmup: bool, prompt_idx: int,
                         tokenizer=None) -> Optional[RerankIterationMetrics]:
    """One rerank call: pipe.rerank(query, documents) -> [(idx, score), ...]."""
    # Token accounting: the reranker scores (query, document) pairs, so account
    # for query tokens repeated against each document plus the document tokens.
    tok_start = time.perf_counter()
    q_tokens = count_tokens(tokenizer, query)
    doc_max, doc_total = count_tokens_batch(tokenizer, documents)
    per_pair_max = q_tokens + doc_max
    total_tokens = q_tokens * len(documents) + doc_total
    tok_ms = (time.perf_counter() - tok_start) * 1000.0

    start_iso = datetime.datetime.now().isoformat()
    t0 = time.perf_counter()
    result = pipe.rerank(query, documents)
    t1 = time.perf_counter()
    end_iso = datetime.datetime.now().isoformat()

    latency_ms = (t1 - t0) * 1000.0
    e2e_ms = latency_ms + tok_ms
    n_docs = len(documents)
    docs_per_s = (n_docs / (latency_ms / 1000.0)) if latency_ms > 0 else 0.0
    tokens_per_s = (total_tokens / (latency_ms / 1000.0)) if latency_ms > 0 and total_tokens else 0.0

    top1_index, top1_score = -1, 0.0
    if result:
        try:
            top1_index, top1_score = int(result[0][0]), float(result[0][1])
        except (TypeError, ValueError, IndexError):
            pass

    metric = RerankIterationMetrics(
        iteration=iter_idx, is_warmup=is_warmup, prompt_idx=prompt_idx,
        num_documents=n_docs, top_n=top_n,
        input_tokens=per_pair_max, total_input_tokens=total_tokens,
        tokenization_time_ms=tok_ms, latency_ms=latency_ms, e2e_latency_ms=e2e_ms,
        throughput_docs_per_s=docs_per_s, throughput_tokens_per_s=tokens_per_s,
        top1_index=top1_index, top1_score=top1_score,
        result_md5=_md5(repr(result)),
        start=start_iso, end=end_iso,
    )
    tag = "[warm-up]" if is_warmup else f"[{iter_idx}]"
    print(f"    {tag}[P{prompt_idx}] docs={n_docs} top_n={top_n} "
          f"tokens(pair_max/total)={per_pair_max}/{total_tokens} "
          f"lat={latency_ms:.2f}ms docs/s={docs_per_s:.1f} "
          f"top1=(#{top1_index},{top1_score:.4f})")
    return metric


# =============================================================================
# Aggregation (over measured iterations only; warm-up excluded)
# =============================================================================

def aggregate_encoder(iterations: list) -> dict:
    """Summary stats for embedding/reranking iterations (warm-up excluded)."""
    measured = [it for it in iterations if not it.is_warmup]
    if not measured:
        return {}
    return {
        "latency": compute_stats([m.latency_ms for m in measured]),
        "e2e": compute_stats([m.e2e_latency_ms for m in measured]),
        "tokenization": compute_stats([m.tokenization_time_ms for m in measured]),
        "throughput_docs": compute_stats([m.throughput_docs_per_s for m in measured]),
        "throughput_tokens": compute_stats([
            m.throughput_tokens_per_s for m in measured if m.throughput_tokens_per_s > 0
        ]),
    }
