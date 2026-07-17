#!/bin/bash
#
# Qwen3-Embedding Benchmark Runner (text embedding via ov_genai.TextEmbeddingPipeline)
#
# Usage:
#   ./run_embedding.sh                       # all discovered embedding models, CPU
#   ./run_embedding.sh GPU                   # on GPU
#   ./run_embedding.sh CPU GPU               # both devices
#   MODELS="models/Qwen3-Embedding-0.6B-int8-ov" ./run_embedding.sh CPU
#   MODE=both BATCH_SIZE=8 ./run_embedding.sh CPU
#
# Env overrides (defaults in []):
#   DEVICE              [CPU GPU NPU]       (positional args override)
#   MODELS              []                  explicit model dir(s); empty => auto-discover models/Qwen3-Embedding-*
#   MODE                [both]              documents | query | both
#   BATCH_SIZE          [8]                 texts per embed_documents() call (documents mode)
#   POOLING             [last_token]        cls | mean | last_token
#   NORMALIZE           [1]                 0 disables L2 normalization
#   MAX_LENGTH          []                  pad/truncate length; empty => pipeline default
#   PADDING_SIDE        [left]              left | right (Qwen3 last-token pooling expects left)
#   PAD_TO_MAX_LENGTH   [0]                 1 fixes shape to MAX_LENGTH (useful for NPU)
#   QUERY_INSTRUCTION   []                  instruction prepended to queries
#   INPUT_TOKENS        []                  build inline text near this token count (overrides default text)
#   PROMPT_FILE         []                  JSONL of texts; overrides INPUT_TOKENS
#   PROMPT_INDEX        []                  subset of indices
#   ITERS               [10]                measured iters (+1 warm-up always)
#   INTERVAL            [0]                 optional cooldown sec between runs
#   CACHE_DIR           []                  empty matches official/direct default
#   LOAD_CONFIG         []                  optional JSON file passed as -lc
#   NPU_LOAD_CONFIG     [auto]              JSON string/file for NPU OV config; set empty to disable
#   NPU_CACHE_DIR       [$PROJECT_DIR/cache_npu/embedding_direct]
#   HW_MONITOR          [1]                 1 enables CPU/GPU/NPU utilization monitoring
#   MEMORY_MONITOR      [1]                 enables process/system memory monitoring
#   MEMORY_INTERVAL_MS  [100]
#   LOG_DIR             [tests/logs/embedding_<ts>]
#

set -euo pipefail

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    sed -n '2,34p' "$0"; exit 0
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

[ -f "$TESTS_DIR/benchmark_embedding.py" ] || { echo "[ERROR] benchmark_embedding.py not found" >&2; exit 1; }

DEVICE="${DEVICE:-CPU GPU NPU}"
bench_parse_devices "$@"

# ---- model discovery ----
if [ -z "${MODELS:-}" ]; then
    MODEL_DIRS=()
    for d in models/Qwen3-Embedding-*; do
        [ -f "$d/openvino_model.xml" ] && MODEL_DIRS+=("$d")
    done
else
    read -ra MODEL_DIRS <<< "$MODELS"
fi
if [ "${#MODEL_DIRS[@]}" -eq 0 ]; then
    echo "[ERROR] no embedding models found (models/Qwen3-Embedding-*); set MODELS=..." >&2
    exit 1
fi

MODE="${MODE:-both}"
BATCH_SIZE="${BATCH_SIZE:-8}"
POOLING="${POOLING:-last_token}"
NORMALIZE="${NORMALIZE:-1}"
MAX_LENGTH="${MAX_LENGTH:-}"
PADDING_SIDE="${PADDING_SIDE:-left}"
PAD_TO_MAX_LENGTH="${PAD_TO_MAX_LENGTH:-0}"
QUERY_INSTRUCTION="${QUERY_INSTRUCTION:-}"
INPUT_TOKENS="${INPUT_TOKENS:-}"
ITERS="${ITERS:-10}"
INTERVAL="${INTERVAL:-0}"
CACHE_DIR="${CACHE_DIR:-}"
LOAD_CONFIG="${LOAD_CONFIG:-}"
NPU_CACHE_DIR="${NPU_CACHE_DIR:-$PROJECT_DIR/cache_npu/embedding_direct}"
if [ -z "${NPU_LOAD_CONFIG+x}" ]; then
    NPU_LOAD_CONFIG="{\"CACHE_DIR\":\"$NPU_CACHE_DIR\"}"
fi
HW_MONITOR="${HW_MONITOR:-1}"
MEMORY_MONITOR="${MEMORY_MONITOR:-1}"
MEMORY_INTERVAL_MS="${MEMORY_INTERVAL_MS:-100}"
PROMPT_INDEX="${PROMPT_INDEX:-}"

bench_validate_load_config "$LOAD_CONFIG" "LOAD_CONFIG"
[ -n "$LOAD_CONFIG" ] && LOAD_CONFIG="$(bench_resolve_load_config "$LOAD_CONFIG")"
bench_validate_load_config "${NPU_LOAD_CONFIG:-}" "NPU_LOAD_CONFIG"
[ -n "${NPU_LOAD_CONFIG:-}" ] && NPU_LOAD_CONFIG="$(bench_resolve_load_config "$NPU_LOAD_CONFIG")"

bench_make_log_dir "embedding"
CSV_OUT="${LOG_DIR}/results_embedding.csv"
JSON_OUT="${LOG_DIR}/results_embedding.json"

# ---- input source ----
INPUT_ARGS=()
if [ -n "${PROMPT_FILE:-}" ]; then
    [ -f "$PROMPT_FILE" ] || { echo "[ERROR] Prompt file not found: $PROMPT_FILE" >&2; exit 1; }
    INPUT_ARGS=(--prompt-file "$PROMPT_FILE")
    if [ -n "$PROMPT_INDEX" ]; then
        read -ra PROMPT_INDEX_ARR <<< "$PROMPT_INDEX"
        INPUT_ARGS+=(--prompt-index "${PROMPT_INDEX_ARR[@]}")
    fi
elif [ -n "$INPUT_TOKENS" ]; then
    INPUT_ARGS=(--input-tokens "$INPUT_TOKENS")
fi
# else: benchmark uses its built-in default text

echo "============================================================"
echo " Qwen3-Embedding Benchmark Runner"
echo "============================================================"
echo " Devices:      ${DEVICES[*]}"
echo " Models:       ${MODEL_DIRS[*]}"
echo " Mode:         $MODE"
echo " Batch size:   $BATCH_SIZE"
echo " Pooling:      $POOLING"
echo " Normalize:    $NORMALIZE"
echo " Padding side: $PADDING_SIDE"
echo " Max length:   ${MAX_LENGTH:-<pipeline default>}"
echo " Input:        ${PROMPT_FILE:-${INPUT_TOKENS:+input-tokens=$INPUT_TOKENS}}"
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
    "runner:           embedding" \
    "devices:          ${DEVICES[*]}" \
    "models:           ${MODEL_DIRS[*]}" \
    "mode:             $MODE" \
    "batch_size:       $BATCH_SIZE" \
    "pooling:          $POOLING" \
    "normalize:        $NORMALIZE" \
    "padding_side:     $PADDING_SIDE" \
    "max_length:       ${MAX_LENGTH:-<default>}" \
    "pad_to_max_len:   $PAD_TO_MAX_LENGTH" \
    "query_instruction:$QUERY_INSTRUCTION" \
    "input_tokens:     ${INPUT_TOKENS:-<default text>}" \
    "prompt_file:      ${PROMPT_FILE:-}" \
    "iters:            $ITERS" \
    "cache_dir:        ${CACHE_DIR:-<empty/off>}" \
    "load_config:      $LOAD_CONFIG" \
    "npu_load_config:  ${NPU_LOAD_CONFIG:-}" \
    "hw_monitor:       $HW_MONITOR" \
    "memory_monitor:   $MEMORY_MONITOR" \
    "memory_interval:  $MEMORY_INTERVAL_MS"

# ---- flags ----
NORM_FLAG=""
[ "$NORMALIZE" = "0" ] && NORM_FLAG="--no-normalize"
PAD_FLAG=""
[ "$PAD_TO_MAX_LENGTH" = "1" ] && PAD_FLAG="--pad-to-max-length"
MAX_LEN_ARGS=(); [ -n "$MAX_LENGTH" ] && MAX_LEN_ARGS=(--max-length "$MAX_LENGTH")
QINSTR_ARGS=(); [ -n "$QUERY_INSTRUCTION" ] && QINSTR_ARGS=(--query-instruction "$QUERY_INSTRUCTION")
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
            "$BENCH_PYTHON" "$TESTS_DIR/benchmark_embedding.py" \
                -m "$MODEL_DIR" \
                -d "$DEV" \
                --mode "$MODE" \
                --batch-size "$BATCH_SIZE" \
                --pooling-type "$POOLING" \
                --padding-side "$PADDING_SIDE" \
                $NORM_FLAG $PAD_FLAG \
                "${MAX_LEN_ARGS[@]}" \
                "${QINSTR_ARGS[@]}" \
                "${INPUT_ARGS[@]}" \
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
