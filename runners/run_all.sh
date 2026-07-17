#!/bin/bash
#
# Convenience entry point for the migrated benchmark suite.
# Official openvino.genai llm_bench is used for Qwen3, Qwen3.5, embedding, and
# reranker tasks. LLMLingua uses the retained direct benchmark implementation.
#
# Usage:
#   ./tests/runners/run_all.sh
#   DEVICE="GPU NPU" ITERS=10 ./tests/runners/run_all.sh
#   RUN_LINGUA=0 RUN_EMBEDDING=0 ./tests/runners/run_all.sh CPU
#
# Env toggles: RUN_QWEN3, RUN_QWEN3_5, RUN_EMBEDDING, RUN_RERANKER, RUN_LINGUA

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DEVICE="${DEVICE:-${*:-CPU GPU NPU}}"
export DEVICE

RUN_QWEN3="${RUN_QWEN3:-1}"
RUN_QWEN3_5="${RUN_QWEN3_5:-1}"
RUN_EMBEDDING="${RUN_EMBEDDING:-1}"
RUN_RERANKER="${RUN_RERANKER:-1}"
RUN_LINGUA="${RUN_LINGUA:-1}"

echo "============================================================"
echo " Migrated benchmark suite"
echo " Devices: $DEVICE"
echo "============================================================"

if [ "$RUN_QWEN3" = "1" ]; then
    "$SCRIPT_DIR/run_official_qwen3.sh"
fi

if [ "$RUN_QWEN3_5" = "1" ]; then
    "$SCRIPT_DIR/run_official_qwen3_5.sh"
fi

if [ "$RUN_EMBEDDING" = "1" ]; then
    "$SCRIPT_DIR/run_official_embedding.sh"
fi

if [ "$RUN_RERANKER" = "1" ]; then
    "$SCRIPT_DIR/run_official_reranker.sh"
fi

if [ "$RUN_LINGUA" = "1" ]; then
    "$SCRIPT_DIR/run_lingua.sh"
fi

echo "============================================================"
echo " Migrated benchmark suite complete"
echo "============================================================"