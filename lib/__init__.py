"""Shared benchmark library for OpenVINO model tests.

Modules:
- monitors: CPU/GPU/NPU utilization monitors (background threads).
- stats:    statistic helpers (compute_stats etc.).
- metrics:  IterationMetrics + BenchmarkResult dataclasses.
- reports:  CSV/JSON report writers (append-mode) + console summary printer.
- prompts:  JSONL prompt loader, token-targeted prompt builder.
- cli:      shared argparse fragments + ensure_sudo helper.
- genai_runner: one-iteration helpers and GenerationConfig builder for
                openvino_genai LLMPipeline / VLMPipeline.
"""
