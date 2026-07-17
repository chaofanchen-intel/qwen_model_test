#!/bin/bash
#
# LLMLingua-2 Benchmark Runner
#
# Usage:
#   ./run_lingua.sh                    # all models, CPU
#   ./run_lingua.sh GPU
#   ./run_lingua.sh CPU GPU
#   MODEL=models/llmlingua-2-bert-base-multilingual-cased-meetingbank-INT4 ./run_lingua.sh CPU
#   MODELS="bert" QUANTS="INT4 INT8" ./run_lingua.sh CPU
#
# Env overrides (defaults in []):
#   DEVICE       [CPU GPU NPU]
#   MODEL        []      explicit model dir(s), overrides auto discovery and MODELS/QUANTS
#   MODELS       []      optional family keys; empty => auto-discover models/llmlingua-2-*
#   QUANTS       [INT4]  used only with MODELS
#   WARMUP       [5]
#   ITERS        [50]
#   INTERVAL     [0]
#   LOAD_CONFIG  []      optional JSON file passed as -lc/--load-config
#   MEMORY_MONITOR      [0]    1 enables process/system memory monitoring
#   MEMORY_INTERVAL_MS  [100]
#   LOG_DIR      [tests/logs/lingua_<ts>]
#

set -euo pipefail

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    sed -n '2,20p' "$0"; exit 0
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

[ -f "$TESTS_DIR/benchmark_lingua.py" ] || { echo "[ERROR] benchmark_lingua.py not found" >&2; exit 1; }

DEVICE="${DEVICE:-CPU GPU NPU}"
bench_parse_devices "$@"

MODEL_DIRS=()
if [ -n "${MODEL:-}" ]; then
    read -ra MODEL_DIRS_RAW <<< "$MODEL"
    for model_dir in "${MODEL_DIRS_RAW[@]}"; do
        MODEL_DIRS+=("$(bench_resolve_model_dir "$model_dir" "MODEL")")
    done
elif [ -z "${MODELS:-}" ]; then
    mapfile -t MODEL_DIRS < <(bench_discover_model_dirs "llmlingua-2-*")
else
    MODEL_FAMILIES=()
    read -ra _MODELS <<< "$MODELS"
    for m in "${_MODELS[@]}"; do
        case "$m" in
            bert*)            MODEL_FAMILIES+=("llmlingua-2-bert-base-multilingual-cased-meetingbank") ;;
            xlm*|roberta*)    MODEL_FAMILIES+=("llmlingua-2-xlm-roberta-large-meetingbank") ;;
            *)                MODEL_FAMILIES+=("$m") ;;
        esac
    done
    if [ -z "${QUANTS:-}" ]; then
        QUANTS=("INT4")
    else
        read -ra QUANTS <<< "$QUANTS"
    fi
    for family in "${MODEL_FAMILIES[@]}"; do
        for quant in "${QUANTS[@]}"; do
            MODEL_DIRS+=("models/${family}-${quant}")
        done
    done
fi
if [ "${#MODEL_DIRS[@]}" -eq 0 ]; then
    echo "[ERROR] no LLMLingua models found; set MODEL=/path/to/model" >&2
    exit 1
fi
MODEL_LABELS=()
for model_dir in "${MODEL_DIRS[@]}"; do
    MODEL_LABELS+=("$(bench_sanitize_label "$(basename "$model_dir")")")
done

WARMUP="${WARMUP:-5}"
ITERS="${ITERS:-500}"
INTERVAL="${INTERVAL:-0}"
LOAD_CONFIG="${LOAD_CONFIG:-}"
MEMORY_MONITOR="${MEMORY_MONITOR:-0}"
MEMORY_INTERVAL_MS="${MEMORY_INTERVAL_MS:-100}"

bench_validate_load_config "$LOAD_CONFIG" "LOAD_CONFIG"
[ -n "$LOAD_CONFIG" ] && LOAD_CONFIG="$(bench_resolve_load_config "$LOAD_CONFIG")"

bench_make_log_dir "lingua"
CSV_OUT="${LOG_DIR}/results_lingua.csv"
JSON_OUT="${LOG_DIR}/results_lingua.json"

echo "============================================================"
echo " LLMLingua-2 Benchmark Runner"
echo "============================================================"
echo " Devices:      ${DEVICES[*]}"
echo " Models:       ${MODEL_LABELS[*]}"
echo " Warmup/Iters: $WARMUP/$ITERS"
echo " Load config:  ${LOAD_CONFIG:-<none>}"
echo " Memory mon:   $MEMORY_MONITOR (${MEMORY_INTERVAL_MS}ms)"
echo " Cooldown:     ${INTERVAL}s between runs"
echo " Log dir:      $LOG_DIR"
echo " CSV / JSON:   $CSV_OUT / $JSON_OUT"
echo "============================================================"
echo ""

bench_write_manifest "$LOG_DIR/run_info.txt" \
    "devices:     ${DEVICES[*]}" \
    "models:      ${MODEL_DIRS[*]}" \
    "warmup:      $WARMUP" \
    "iters:       $ITERS" \
    "load_config: $LOAD_CONFIG" \
    "memory_monitor: $MEMORY_MONITOR" \
    "memory_interval: $MEMORY_INTERVAL_MS"

LOAD_CONFIG_ARGS=()
MEMORY_ARGS=(--memory-monitor-interval "$MEMORY_INTERVAL_MS")
[ -n "$LOAD_CONFIG" ] && LOAD_CONFIG_ARGS=(-lc "$LOAD_CONFIG")
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
            bench_run_with_log "$LOG_FILE" \
                "$BENCH_PYTHON" "$TESTS_DIR/benchmark_lingua.py" \
                    -m "$MODEL_DIR" -d "$DEV" --monitor \
                    --warmup "$WARMUP" --iters "$ITERS" \
                    "${LOAD_CONFIG_ARGS[@]}" \
                    "${MEMORY_ARGS[@]}" \
                    --csv-out "$CSV_OUT" --json-out "$JSON_OUT"
    done
done

echo ""
echo "============================================================"
echo " All tests complete!"
echo " Results: $CSV_OUT"
echo "          $JSON_OUT"
echo " Logs:    $LOG_DIR/"
echo "============================================================"
