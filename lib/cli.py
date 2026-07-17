"""Shared argparse fragments and small CLI helpers.

The benchmark scripts each instantiate their own ArgumentParser, then call
add_common_gen_args(parser) to mix in the LLM/VLM shared options. This keeps
the qwen3 and qwen3.5 CLIs as similar as practical without forcing them into
a single rigid surface.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


# =============================================================================
# Common argparse fragments
# =============================================================================

def add_io_args(p: argparse.ArgumentParser, default_csv: str, default_json: str) -> None:
    p.add_argument("-r", "--csv-out", default=default_csv,
                   help="CSV report path (append mode if exists; '' to disable).")
    p.add_argument("-rj", "--json-out", default=default_json,
                   help="JSON report path (append mode if exists; '' to disable).")


def add_monitor_args(p: argparse.ArgumentParser) -> None:
    p.set_defaults(no_monitor=True)
    p.add_argument("--monitor", dest="no_monitor", action="store_false",
                   help="Enable HW utilization monitoring (CPU/GPU/NPU).")
    p.add_argument("--no-monitor", dest="no_monitor", action="store_true",
                   help="Disable HW utilization monitoring (CPU/GPU/NPU). Direct scripts default to disabled to match official llm_bench.")
    p.set_defaults(memory_monitor=False)
    p.add_argument("--memory-monitor", dest="memory_monitor", action="store_true",
                   help="Enable process/system memory monitoring.")
    p.add_argument("--no-memory-monitor", dest="memory_monitor", action="store_false",
                   help="Disable process/system memory monitoring.")
    p.add_argument("--memory-monitor-interval", type=int, default=100,
                   help="Memory monitor sampling interval in ms.")
    p.add_argument("--cpu-monitor-interval", type=int, default=100,
                   help="CPU monitor sampling interval in ms.")
    p.add_argument("--gpu-monitor-interval", type=int, default=500,
                   help="GPU monitor sampling interval in ms.")
    p.add_argument("--npu-monitor-interval", type=int, default=500,
                   help="NPU monitor sampling interval in ms.")
    p.add_argument("--gpu-command", default=None,
                   help="Custom xpu-smi command (default: 'sudo xpu-smi stats -d 1 -j').")


def add_common_gen_args(p: argparse.ArgumentParser) -> None:
    """Args shared by LLMPipeline and VLMPipeline benchmarks."""
    p.add_argument("-m", "--model-dir", required=True, nargs="+",
                   help="Path(s) to OpenVINO IR model directory.")
    p.add_argument("-d", "--device", default=["CPU"], nargs="+",
                   help="Inference device(s): CPU GPU NPU AUTO ...")
    p.add_argument("-p", "--prompt", default=None,
                   help="Single inline prompt. Mutually exclusive with --prompt-file.")
    p.add_argument("-pf", "--prompt-file", nargs="+", default=None,
                   help="One or more JSONL files; each line {\"prompt\": \"...\"}.")
    p.add_argument("-pi", "--prompt-index", type=int, nargs="+", default=None,
                   help="Subset of prompt indices to run.")
    p.add_argument("--input-tokens", type=int, default=1000,
                   help="Build one inline prompt near this final input-token count. "
                        "DEFAULT 1000 — the benchmark synthesizes a ~1000-token prompt "
                        "unless you pass --prompt / --prompt-file (or --input-tokens 0 "
                        "to fall back to the built-in default prompt). The count uses the "
                        "benchmark tokenizer and includes --apply-chat-template when enabled.")
    p.add_argument("--input-base-text", default=None,
                   help="Optional .txt source for --input-tokens prompt generation.")
    p.add_argument("-ic", "--infer-count", type=int, default=512,
                   help="Max output tokens (max_new_tokens).")
    p.add_argument("--exact-output-tokens", action="store_true",
                   help="Best-effort fixed output length: set min_new_tokens=max_new_tokens=--infer-count. "
                        "Backends may still stop early on hard model/runtime limits.")
    p.add_argument("-n", "--num-iters", type=int, default=5,
                   help="Measured iterations per prompt (warmup iter 0 always extra).")
    p.add_argument("-bs", "--batch-size", type=int, default=1,
                   help="Batch size (informational; pipeline is invoked once per prompt).")
    p.add_argument("--num-beams", type=int, default=1, help="Beam count.")
    p.add_argument("--apply-chat-template", action="store_true",
                   help="Wrap prompt in tokenizer.apply_chat_template before encoding.")
    p.add_argument("--disable-prompt-permutation", action="store_true",
                   help="Disable per-iteration prompt mutation. By DEFAULT prompts are "
                        "permuted between iterations to defeat the GenAI prefix cache and "
                        "produce realistic TTFT. Disable for cache-warm scenarios.")
    p.add_argument("-s", "--seed", type=int, default=42)
    p.add_argument("-lc", "--load-config", "--load_config", dest="load_config", default=None,
                   help="Path to JSON file with extra OV plugin properties.")
    p.add_argument("--cache-dir", default="",
                   help="OpenVINO compiled-model CACHE_DIR. Empty string matches official llm_bench default.")
    p.add_argument("--num-streams", default=None, help="OV NUM_STREAMS property.")
    p.add_argument("--cooldown-between-runs", type=int, default=5,
                   help="Seconds between (model, device) pairs.")


def add_encoder_common_args(p: argparse.ArgumentParser) -> None:
    """Args shared by the embedding and reranking encoder benchmarks.

    These pipelines do a single forward pass, so there is no infer-count /
    beam / chat-template / permutation surface. Iteration & device controls
    mirror the generation benchmarks for consistency.
    """
    p.add_argument("-m", "--model-dir", required=True, nargs="+",
                   help="Path(s) to OpenVINO IR model directory.")
    p.add_argument("-d", "--device", default=["CPU"], nargs="+",
                   help="Inference device(s): CPU GPU NPU AUTO ...")
    p.add_argument("-n", "--num-iters", type=int, default=5,
                   help="Measured iterations per input (warmup iter 0 always extra).")
    p.add_argument("--max-length", type=int, default=None,
                   help="Max token length passed to the model (pipeline default if unset). "
                        "Qwen3 encoders default to a large value internally.")
    p.add_argument("--pad-to-max-length", action="store_true",
                   help="Pad model input tensors to --max-length (fixes shape; "
                        "useful for NPU / static-shape devices).")
    p.add_argument("--padding-side", choices=["left", "right"], default=None,
                   help="Padding side. Qwen3 last-token pooling expects 'left'.")
    p.add_argument("-s", "--seed", type=int, default=42)
    p.add_argument("-lc", "--load-config", "--load_config", dest="load_config", default=None,
                   help="Path to JSON file with extra OV plugin properties.")
    p.add_argument("--cache-dir", default="",
                   help="OpenVINO compiled-model CACHE_DIR. Empty string matches official llm_bench default.")
    p.add_argument("--num-streams", default=None, help="OV NUM_STREAMS property.")
    p.add_argument("--cooldown-between-runs", type=int, default=5,
                   help="Seconds between (model, device) pairs.")


def add_embedding_args(p: argparse.ArgumentParser) -> None:
    """Text-embedding-specific options. Mirrors official benchmark.py
    --embedding_* flags plus our input-construction helpers."""
    p.add_argument("--mode", choices=["documents", "query", "both"], default="documents",
                   help="documents -> embed_documents(batch); query -> embed_query(single); "
                        "both -> run each in turn.")
    p.add_argument("-p", "--prompt", default=None,
                   help="Single inline text. Mutually exclusive with --prompt-file / --input-tokens.")
    p.add_argument("-pf", "--prompt-file", nargs="+", default=None,
                   help="One or more JSONL files; each line {\"prompt\": \"...\"} or {\"text\": \"...\"}.")
    p.add_argument("-pi", "--prompt-index", type=int, nargs="+", default=None,
                   help="Subset of prompt indices to run.")
    p.add_argument("--input-tokens", type=int, default=None,
                   help="Build one inline text near this token count (uses the model tokenizer). "
                        "Mutually exclusive with --prompt / --prompt-file.")
    p.add_argument("--input-base-text", default=None,
                   help="Optional .txt source for --input-tokens text generation.")
    p.add_argument("-bs", "--batch-size", type=int, default=1,
                   help="Number of texts encoded per embed_documents() call (documents mode). "
                        "The benchmark replicates the resolved text(s) up to this batch size.")
    p.add_argument("--pooling-type", choices=["cls", "mean", "last_token"], default="last_token",
                   help="Pooling strategy. Qwen3-Embedding is a decoder and expects last_token.")
    p.add_argument("--normalize", dest="normalize", action="store_true", default=True,
                   help="L2-normalize embeddings (default; Qwen3-Embedding recommends normalize).")
    p.add_argument("--no-normalize", dest="normalize", action="store_false",
                   help="Disable L2 normalization.")
    p.add_argument("--query-instruction", default=None,
                   help="Instruction prepended to queries (Qwen3-Embedding query format).")
    p.add_argument("--embed-instruction", default=None,
                   help="Instruction prepended to documents.")


def add_reranker_args(p: argparse.ArgumentParser) -> None:
    """Text-reranking-specific options. Mirrors official --reranking_* + --texts."""
    p.add_argument("-p", "--prompt", "--query", dest="prompt", default=None,
                   help="Inline query string. Mutually exclusive with --prompt-file.")
    p.add_argument("-pf", "--prompt-file", nargs="+", default=None,
                   help="One or more JSONL files of queries; each line {\"prompt\": \"...\"}.")
    p.add_argument("-pi", "--prompt-index", type=int, nargs="+", default=None,
                   help="Subset of query indices to run.")
    p.add_argument("--texts", nargs="+", default=None,
                   help="Candidate documents to rerank against the query.")
    p.add_argument("--texts-file", "--texts_file", dest="texts_file", nargs="+", default=None,
                   help="JSONL file(s) of candidate documents; each line {\"text\": \"...\"}.")
    p.add_argument("--num-documents", type=int, default=None,
                   help="If set, replicate/trim the document list to exactly this many "
                        "candidates (throughput sweep over candidate-set size).")
    p.add_argument("--top-n", type=int, default=3,
                   help="Number of top results to return (TextRerankPipeline.Config.top_n).")


# =============================================================================
# sudo refresh
# =============================================================================

def ensure_sudo() -> None:
    """Pre-authenticate sudo before any other output, so the password prompt
    is not interleaved with benchmark logs.
    """
    print("  [sudo] Refreshing credentials for hardware monitoring (or skip with Ctrl-C)...")
    sys.stdout.flush()
    try:
        r = subprocess.run(["sudo", "-v"], stdin=sys.stdin, timeout=60)
        if r.returncode == 0:
            print("  [sudo] OK\n")
        else:
            print("  [sudo] auth failed; GPU/NPU monitors may not work\n")
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        print(f"  [sudo] unavailable ({e}); GPU/NPU monitors may not work\n")
    sys.stdout.flush()


# =============================================================================
# Misc helpers
# =============================================================================

def detect_precision(model_dir: Path) -> str:
    name = model_dir.name.lower()
    for tag in ("int4", "int8", "fp16", "fp32", "bf16"):
        if tag in name:
            return tag.upper()
    return ""


def load_extra_ov_config(path: str) -> dict[str, str]:
    """Load and stringify extra OV plugin config from a JSON file or inline JSON."""
    import json
    if Path(path).is_file():
        with open(path, "r") as f:
            raw = json.load(f)
    else:
        raw = json.loads(path)
    return {k: (v if isinstance(v, str) else str(v)) for k, v in raw.items()}
