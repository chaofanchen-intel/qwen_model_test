#!/bin/bash
#
# Qwen3-Reranker Benchmark Runner (text reranking via ov_genai.TextRerankPipeline)
#
# Usage:
#   ./run_reranker.sh                        # all discovered reranker models, CPU
#   ./run_reranker.sh GPU                    # on GPU
#   ./run_reranker.sh CPU GPU                # both devices
#   MODELS="models/Qwen3-Reranker-0.6B-int8-ov" ./run_reranker.sh CPU
#   NUM_DOCUMENTS=32 TOP_N=10 ./run_reranker.sh CPU
#   TEXTS_FILE=tests/prompts/reranker/texts_64.jsonl MAX_LENGTH=1024 ./run_reranker.sh CPU
#
# Env overrides (defaults in []):
#   DEVICE              [CPU GPU NPU]       (positional args override)
#   MODELS              []                  explicit model dir(s); empty => auto-discover models/Qwen3-Reranker-*
#   QUERY               []                  inline query; empty => PROMPT_FILE queries
#   PROMPT_FILE         [tests/prompts/reranker/queries.jsonl] JSONL of queries; set empty to use QUERY/built-in
#   PROMPT_INDEX        []                  subset of query indices
#   TEXTS_FILE          [tests/prompts/reranker/texts_16.jsonl] JSONL of candidate documents ({"text": ...}); set empty for built-in defaults
#   NUM_DOCUMENTS       []                  replicate/trim candidates to exactly this count (sweep)
#   TOP_N               [3]                 number of top results to return
#   MAX_LENGTH          [512]               pad/truncate length; set empty => pipeline default (8192 for Qwen3)
#   PADDING_SIDE        []                  left | right; empty => pipeline default
#   PAD_TO_MAX_LENGTH   [0]                 1 fixes shape to MAX_LENGTH (useful for NPU)
#   ITERS               [10]                measured iters (+1 warm-up always)
#   INTERVAL            [0]                 optional cooldown sec between runs
#   CACHE_DIR           []                  empty matches official/direct default
#   LOAD_CONFIG         []                  optional JSON file passed as -lc
#   NPU_LOAD_CONFIG     [auto]              JSON string/file for NPU OV config; set empty to disable
#   NPU_CACHE_DIR       [$PROJECT_DIR/cache_npu/reranker_direct]
#   HW_MONITOR          [1]                 1 enables CPU/GPU/NPU utilization monitoring
#   MEMORY_MONITOR      [1]                 enables process/system memory monitoring
#   MEMORY_INTERVAL_MS  [100]
#   LOG_DIR             [tests/logs/reranker_<ts>]
#

set -euo pipefail

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    sed -n '2,33p' "$0"; exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_DIR="$(cd "$TESTS_DIR/.." && pwd)"
cd "$PROJECT_DIR"

source "$SCRIPT_DIR/_common.sh"
if [ -z "${PYTHON:-}" ] && [ -x "/home/user/envs/qwen_env/bin/python" ]; then
    PYTHON="/home/user/envs/qwen_env/bin/python"
fi
bench_select_python

[ -f "$TESTS_DIR/benchmark_reranker.py" ] || { echo "[ERROR] benchmark_reranker.py not found" >&2; exit 1; }

DEVICE="${DEVICE:-CPU GPU NPU}"
bench_parse_devices "$@"

# ---- model discovery ----
if [ -z "${MODELS:-}" ]; then
    MODEL_DIRS=()
    for d in models/Qwen3-Reranker-*; do
        [ -f "$d/openvino_model.xml" ] && MODEL_DIRS+=("$d")
    done
else
    read -ra MODEL_DIRS <<< "$MODELS"
fi
if [ "${#MODEL_DIRS[@]}" -eq 0 ]; then
    echo "[ERROR] no reranker models found (models/Qwen3-Reranker-*); set MODELS=..." >&2
    exit 1
fi

QUERY="${QUERY:-}"
RERANKER_PROMPT_DEFAULT="$TESTS_DIR/prompts/reranker/queries.jsonl"
RERANKER_TEXTS_DEFAULT="$TESTS_DIR/prompts/reranker/texts_16.jsonl"
PROMPT_FILE="${PROMPT_FILE-$RERANKER_PROMPT_DEFAULT}"
PROMPT_INDEX="${PROMPT_INDEX:-}"
TOP_N="${TOP_N:-3}"
NUM_DOCUMENTS="${NUM_DOCUMENTS:-}"
MAX_LENGTH="${MAX_LENGTH-512}"
PADDING_SIDE="${PADDING_SIDE:-}"
PAD_TO_MAX_LENGTH="${PAD_TO_MAX_LENGTH:-0}"
ITERS="${ITERS:-10}"
INTERVAL="${INTERVAL:-0}"
CACHE_DIR="${CACHE_DIR:-}"
LOAD_CONFIG="${LOAD_CONFIG:-}"
NPU_CACHE_DIR="${NPU_CACHE_DIR:-$PROJECT_DIR/cache_npu/reranker_direct}"
if [ -z "${NPU_LOAD_CONFIG+x}" ]; then
    NPU_LOAD_CONFIG="{\"CACHE_DIR\":\"$NPU_CACHE_DIR\"}"
fi
HW_MONITOR="${HW_MONITOR:-1}"
MEMORY_MONITOR="${MEMORY_MONITOR:-1}"
MEMORY_INTERVAL_MS="${MEMORY_INTERVAL_MS:-100}"

bench_validate_load_config "$LOAD_CONFIG" "LOAD_CONFIG"
[ -n "$LOAD_CONFIG" ] && LOAD_CONFIG="$(bench_resolve_load_config "$LOAD_CONFIG")"
bench_validate_load_config "${NPU_LOAD_CONFIG:-}" "NPU_LOAD_CONFIG"
[ -n "${NPU_LOAD_CONFIG:-}" ] && NPU_LOAD_CONFIG="$(bench_resolve_load_config "$NPU_LOAD_CONFIG")"
if [ -n "$PROMPT_FILE" ]; then
    PROMPT_FILE="$(bench_resolve_file "$PROMPT_FILE" "PROMPT_FILE")"
fi
if [ -n "${TEXTS_FILE:-}" ]; then
    TEXTS_FILE="$(bench_resolve_file "$TEXTS_FILE" "TEXTS_FILE")"
else
    TEXTS_FILE="${TEXTS_FILE-$RERANKER_TEXTS_DEFAULT}"
    [ -n "$TEXTS_FILE" ] && TEXTS_FILE="$(bench_resolve_file "$TEXTS_FILE" "TEXTS_FILE")"
fi

bench_make_log_dir "reranker"
CSV_OUT="${LOG_DIR}/results_reranker.csv"
JSON_OUT="${LOG_DIR}/results_reranker.json"

# ---- query source ----
QUERY_ARGS=()
if [ -n "${PROMPT_FILE:-}" ]; then
    [ -f "$PROMPT_FILE" ] || { echo "[ERROR] Prompt file not found: $PROMPT_FILE" >&2; exit 1; }
    QUERY_ARGS=(--prompt-file "$PROMPT_FILE")
    if [ -n "$PROMPT_INDEX" ]; then
        read -ra PROMPT_INDEX_ARR <<< "$PROMPT_INDEX"
        QUERY_ARGS+=(--prompt-index "${PROMPT_INDEX_ARR[@]}")
    fi
elif [ -n "$QUERY" ]; then
    QUERY_ARGS=(--prompt "$QUERY")
fi

# ---- documents source ----
DOC_ARGS=()
if [ -n "${TEXTS_FILE:-}" ]; then
    DOC_ARGS=(--texts-file "$TEXTS_FILE")
fi
[ -n "$NUM_DOCUMENTS" ] && DOC_ARGS+=(--num-documents "$NUM_DOCUMENTS")

echo "============================================================"
echo " Qwen3-Reranker Benchmark Runner"
echo "============================================================"
echo " Devices:      ${DEVICES[*]}"
echo " Models:       ${MODEL_DIRS[*]}"
echo " Query:        ${PROMPT_FILE:-${QUERY:-<built-in default>}}"
echo " Texts file:   ${TEXTS_FILE:-<built-in defaults>}"
echo " Num docs:     ${NUM_DOCUMENTS:-<as-is>}"
echo " top_n:        $TOP_N"
echo " Max length:   ${MAX_LENGTH:-<pipeline default>}"
echo " Iterations:   $ITERS (+1 warm-up)"
echo " Cache dir:    ${CACHE_DIR:-<empty/off>}"
echo " Load config:  ${LOAD_CONFIG:-<none>}"
echo " NPU config:   ${NPU_LOAD_CONFIG:-<none>}"
echo " HW monitor:   $HW_MONITOR"
echo " Memory mon:   $MEMORY_MONITOR (${MEMORY_INTERVAL_MS}ms)"
echo " Cooldown:     ${INTERVAL}s between runs"
echo " Log dir:      $LOG_DIR"
echo " CSV / JSON:   $CSV_OUT / $JSON_OUT"
echo "============================================================"
echo ""

bench_write_manifest "$LOG_DIR/run_info.txt" \
    "runner:           reranker" \
    "devices:          ${DEVICES[*]}" \
    "models:           ${MODEL_DIRS[*]}" \
    "query:            ${PROMPT_FILE:-${QUERY:-<default>}}" \
    "texts_file:       ${TEXTS_FILE:-<defaults>}" \
    "num_documents:    ${NUM_DOCUMENTS:-<as-is>}" \
    "top_n:            $TOP_N" \
    "max_length:       ${MAX_LENGTH:-<default>}" \
    "pad_to_max_len:   $PAD_TO_MAX_LENGTH" \
    "padding_side:     ${PADDING_SIDE:-<default>}" \
    "iters:            $ITERS" \
    "cache_dir:        ${CACHE_DIR:-<empty/off>}" \
    "load_config:      $LOAD_CONFIG" \
    "npu_load_config:  ${NPU_LOAD_CONFIG:-}" \
    "hw_monitor:       $HW_MONITOR" \
    "memory_monitor:   $MEMORY_MONITOR" \
    "memory_interval:  $MEMORY_INTERVAL_MS"

# ---- flags ----
PAD_FLAG=""
[ "$PAD_TO_MAX_LENGTH" = "1" ] && PAD_FLAG="--pad-to-max-length"
MAX_LEN_ARGS=(); [ -n "$MAX_LENGTH" ] && MAX_LEN_ARGS=(--max-length "$MAX_LENGTH")
PADSIDE_ARGS=(); [ -n "$PADDING_SIDE" ] && PADSIDE_ARGS=(--padding-side "$PADDING_SIDE")
MONITOR_ARGS=()
[ "$HW_MONITOR" = "1" ] && MONITOR_ARGS+=(--monitor) || MONITOR_ARGS+=(--no-monitor)
MEMORY_ARGS=(--memory-monitor-interval "$MEMORY_INTERVAL_MS")
[ "$MEMORY_MONITOR" = "1" ] && MEMORY_ARGS+=(--memory-monitor)

FIRST_RUN=true
for MODEL_DIR in "${MODEL_DIRS[@]}"; do
    if [ ! -d "$MODEL_DIR" ]; then
        echo "[SKIP] Model not found: $MODEL_DIR"
        continue
    fi
    MODEL_LABEL="$(bench_sanitize_label "$(basename "$MODEL_DIR")")"
    for DEV in "${DEVICES[@]}"; do
        if [ "$FIRST_RUN" = true ]; then
            FIRST_RUN=false
        elif [ "${INTERVAL:-0}" != "0" ]; then
            echo ""
            echo "[WAIT] cooling ${INTERVAL}s before next test..."
            sleep "$INTERVAL"
        fi
        RUN_NAME="${MODEL_LABEL}_${DEV}"
        LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"
        echo "============================================================"
        echo "[RUN] $RUN_NAME    (log: $LOG_FILE)"
        echo "============================================================"
        DEVICE_LOAD_CONFIG_ARGS=()
        if [ -n "$LOAD_CONFIG" ]; then
            DEVICE_LOAD_CONFIG_ARGS=(-lc "$LOAD_CONFIG")
        elif [ "$DEV" = "NPU" ] && [ -n "${NPU_LOAD_CONFIG:-}" ]; then
            DEVICE_LOAD_CONFIG_ARGS=(-lc "$NPU_LOAD_CONFIG")
        fi
        bench_run_with_log "$LOG_FILE" \
            "$BENCH_PYTHON" "$TESTS_DIR/benchmark_reranker.py" \
                -m "$MODEL_DIR" \
                -d "$DEV" \
                --top-n "$TOP_N" \
                $PAD_FLAG \
                "${MAX_LEN_ARGS[@]}" \
                "${PADSIDE_ARGS[@]}" \
                "${QUERY_ARGS[@]}" \
                "${DOC_ARGS[@]}" \
                --num-iters "$ITERS" \
                --cache-dir "$CACHE_DIR" \
                "${DEVICE_LOAD_CONFIG_ARGS[@]}" \
                "${MONITOR_ARGS[@]}" \
                "${MEMORY_ARGS[@]}" \
                --csv-out "$CSV_OUT" \
                --json-out "$JSON_OUT"
    done
done

echo ""
echo "============================================================"
echo " All tests complete!"
echo " Results: $CSV_OUT"
echo "          $JSON_OUT"
echo " Logs:    $LOG_DIR/"
echo "============================================================"
