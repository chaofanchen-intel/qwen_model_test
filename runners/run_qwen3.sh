#!/bin/bash
#
# Qwen3 LLM Benchmark Runner (text-only)
#
# Usage:
#   ./run_qwen3.sh                     # all discovered Qwen3 models, CPU/GPU/NPU
#   ./run_qwen3.sh GPU                 # all discovered Qwen3 models on GPU
#   ./run_qwen3.sh CPU GPU             # both devices
#   MODELS="0.6B" ./run_qwen3.sh CPU
#   MODEL=models/Qwen3-1.7B-INT4 ./run_qwen3.sh CPU
#   INPUT_TOKENS=512 INFER_COUNT=256 ./run_qwen3.sh CPU
#   PROMPT_FILE=tests/prompts/qwen3_1k.jsonl INFER_COUNT=512 ./run_qwen3.sh CPU
#
# Env overrides (defaults in []):
#   DEVICE              [CPU GPU NPU]       (positional args override)
#   MODEL               []                  explicit model dir(s), overrides auto discovery and MODELS/QUANTS
#   MODELS              []                  optional size keys; empty => auto-discover models/Qwen3-* and models/qwen3-*
#   QUANTS              [INT4]              used only with MODELS
#   INPUT_TOKENS        [1000]              DEFAULT input: generated inline (~1000 tok); includes chat template if enabled
#   INPUT_BASE_TEXT     []                  optional .txt source for INPUT_TOKENS
#   PROMPT_FILE         []                  set to a JSONL path to use a real prompt file instead of INPUT_TOKENS
#   PROMPT_INDEX        [0]                 optional subset; empty runs all prompts in PROMPT_FILE (only when PROMPT_FILE set)
#   INFER_COUNT         [512]
#   EXACT_OUTPUT_TOKENS [0]                 1 sets min_new_tokens=max_new_tokens=INFER_COUNT
#   ITERS               [10]                measured iters (+1 warm-up always)
#   NUM_BEAMS           [1]
#   APPLY_CHAT_TEMPLATE [0]
#   PERMUTE             [1]                 0 to disable prompt permutation
#   INTERVAL            [0]                 optional cooldown sec between runs
#   CACHE_DIR           []                  empty string matches official/direct default
#   LOAD_CONFIG         []                  optional JSON file passed as -lc/--load-config
#   NPU_LOAD_CONFIG     [auto]              JSON string/file for NPU LLM static config; set empty to disable
#   NPU_MAX_PROMPT_LEN  [1024]
#   HW_MONITOR          [1]                 1 enables CPU/GPU/NPU utilization monitoring
#   MEMORY_MONITOR      [1]                 runner default enables process/system memory monitoring
#   MEMORY_INTERVAL_MS  [100]
#   LOG_DIR             [tests/logs/qwen3_<ts>]
#

set -euo pipefail

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    sed -n '2,30p' "$0"; exit 0
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

# Validate
[ -f "$TESTS_DIR/benchmark_qwen3.py" ] || { echo "[ERROR] benchmark_qwen3.py not found" >&2; exit 1; }

# ---- params ----
DEVICE="${DEVICE:-CPU GPU NPU}"
bench_parse_devices "$@"

MODEL_DIRS=()
if [ -n "${MODEL:-}" ]; then
    read -ra MODEL_DIRS_RAW <<< "$MODEL"
    for model_dir in "${MODEL_DIRS_RAW[@]}"; do
        MODEL_DIRS+=("$(bench_resolve_model_dir "$model_dir" "MODEL")")
    done
elif [ -n "${MODELS:-}" ] || [ -n "${QUANTS:-}" ]; then
    if [ -z "${MODELS:-}" ]; then
        MODEL_SIZES=("0.6B" "1.7B" "4B")
    else
        read -ra MODEL_SIZES <<< "$MODELS"
    fi
    if [ -z "${QUANTS:-}" ]; then
        QUANTS=("INT4")
    else
        read -ra QUANTS <<< "$QUANTS"
    fi
    for size in "${MODEL_SIZES[@]}"; do
        for quant in "${QUANTS[@]}"; do
            size_norm="$(bench_normalize_size_b "$size")"
            quant_uc=$(echo "$quant" | tr '[:lower:]' '[:upper:]')
            MODEL_DIRS+=("models/Qwen3-${size_norm}-${quant_uc}")
        done
    done
else
    mapfile -t MODEL_DIRS < <(bench_discover_model_dirs "Qwen3-[0-9]*" "qwen3-[0-9]*")
fi
if [ "${#MODEL_DIRS[@]}" -eq 0 ]; then
    echo "[ERROR] no Qwen3 text-generation models found; set MODEL=/path/to/model" >&2
    exit 1
fi
MODEL_LABELS=()
for model_dir in "${MODEL_DIRS[@]}"; do
    MODEL_LABELS+=("$(bench_sanitize_label "$(basename "$model_dir")")")
done

INFER_COUNT="${INFER_COUNT:-512}"
EXACT_OUTPUT_TOKENS="${EXACT_OUTPUT_TOKENS:-0}"
ITERS="${ITERS:-10}"
NUM_BEAMS="${NUM_BEAMS:-1}"
APPLY_CHAT_TEMPLATE="${APPLY_CHAT_TEMPLATE:-0}"
PERMUTE="${PERMUTE:-1}"
INTERVAL="${INTERVAL:-0}"
CACHE_DIR="${CACHE_DIR:-}"
LOAD_CONFIG="${LOAD_CONFIG:-}"
NPU_MAX_PROMPT_LEN="${NPU_MAX_PROMPT_LEN:-1000}"
NPU_MIN_RESPONSE_LEN="${NPU_MIN_RESPONSE_LEN:-512}"
NPU_PREFILL_CHUNK_SIZE="${NPU_PREFILL_CHUNK_SIZE:-2048}"
NPU_CACHE_DIR="${NPU_CACHE_DIR:-$PROJECT_DIR/cache_npu/qwen3_direct}"
if [ -z "${NPU_LOAD_CONFIG+x}" ]; then
    NPU_LOAD_CONFIG="{\"MAX_PROMPT_LEN\":$NPU_MAX_PROMPT_LEN,\"MIN_RESPONSE_LEN\":$NPU_MIN_RESPONSE_LEN,\"CACHE_DIR\":\"$NPU_CACHE_DIR\",\"NPUW_LLM_PREFILL_CHUNK_SIZE\":$NPU_PREFILL_CHUNK_SIZE}"
fi
HW_MONITOR="${HW_MONITOR:-1}"
MEMORY_MONITOR="${MEMORY_MONITOR:-1}"
MEMORY_INTERVAL_MS="${MEMORY_INTERVAL_MS:-100}"
# Default: synthesize a ~1000-token prompt via --input-tokens (PROMPT_FILE empty).
# Set PROMPT_FILE=<path.jsonl> to use a real prompt file instead, or INPUT_TOKENS=N
# to change the synthetic length. Both stay env-overridable.
PROMPT_FILE="${PROMPT_FILE-}"
PROMPT_INDEX="${PROMPT_INDEX:-0}"
INPUT_TOKENS="${INPUT_TOKENS:-1000}"
INPUT_BASE_TEXT="${INPUT_BASE_TEXT:-}"

bench_validate_load_config "$LOAD_CONFIG" "LOAD_CONFIG"
[ -n "$LOAD_CONFIG" ] && LOAD_CONFIG="$(bench_resolve_load_config "$LOAD_CONFIG")"
bench_validate_load_config "${NPU_LOAD_CONFIG:-}" "NPU_LOAD_CONFIG"
[ -n "${NPU_LOAD_CONFIG:-}" ] && NPU_LOAD_CONFIG="$(bench_resolve_load_config "$NPU_LOAD_CONFIG")"

bench_make_log_dir "qwen3"
CSV_OUT="${LOG_DIR}/results_qwen3.csv"
JSON_OUT="${LOG_DIR}/results_qwen3.json"

PROMPT_ARGS=()
if [ -n "${PROMPT_FILE:-}" ]; then
    PROMPT_FILE="$(bench_resolve_file "$PROMPT_FILE" "PROMPT_FILE")"
    PROMPT_ARGS=(--prompt-file "$PROMPT_FILE")
    if [ -n "$PROMPT_INDEX" ]; then
        read -ra PROMPT_INDEX_ARR <<< "$PROMPT_INDEX"
        PROMPT_ARGS+=(--prompt-index "${PROMPT_INDEX_ARR[@]}")
    fi
else
    PROMPT_ARGS=(--input-tokens "$INPUT_TOKENS")
    if [ -n "$INPUT_BASE_TEXT" ]; then
        PROMPT_ARGS+=(--input-base-text "$INPUT_BASE_TEXT")
    fi
fi

# ---- banner / manifest ----
echo "============================================================"
echo " Qwen3 LLM Benchmark Runner"
echo "============================================================"
echo " Devices:      ${DEVICES[*]}"
echo " Models:       ${MODEL_LABELS[*]}"
if [ -n "${PROMPT_FILE:-}" ]; then
    echo " Prompt file:  $PROMPT_FILE"
    echo " Prompt idx:   ${PROMPT_INDEX:-<all>}"
else
    echo " Input tokens: $INPUT_TOKENS"
    [ -n "$INPUT_BASE_TEXT" ] && echo " Base text:    $INPUT_BASE_TEXT"
fi
echo " infer_count:  $INFER_COUNT"
echo " exact_out:    $EXACT_OUTPUT_TOKENS"
echo " Iterations:   $ITERS (+1 warm-up)"
echo " num_beams:    $NUM_BEAMS"
echo " chat_tmpl:    $APPLY_CHAT_TEMPLATE"
echo " permute:      $PERMUTE"
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
    "devices:          ${DEVICES[*]}" \
    "models:           ${MODEL_DIRS[*]}" \
    "prompt_file:      ${PROMPT_FILE:-}" \
    "prompt_index:     ${PROMPT_INDEX:-<all>}" \
    "input_tokens:     $INPUT_TOKENS" \
    "input_base_text:  $INPUT_BASE_TEXT" \
    "infer_count:      $INFER_COUNT" \
    "exact_output:     $EXACT_OUTPUT_TOKENS" \
    "iters:            $ITERS" \
    "num_beams:        $NUM_BEAMS" \
    "apply_chat_tpl:   $APPLY_CHAT_TEMPLATE" \
    "permute:          $PERMUTE" \
    "cache_dir:        ${CACHE_DIR:-<empty/off>}" \
    "load_config:      $LOAD_CONFIG" \
    "npu_load_config:  ${NPU_LOAD_CONFIG:-}" \
    "hw_monitor:       $HW_MONITOR" \
    "memory_monitor:   $MEMORY_MONITOR" \
    "memory_interval:  $MEMORY_INTERVAL_MS"

# ---- main loop ----
FIRST_RUN=true
CHAT_FLAG=""
PERM_FLAG=""
EXACT_FLAG=""
MONITOR_ARGS=()
MEMORY_ARGS=(--memory-monitor-interval "$MEMORY_INTERVAL_MS")
[ "$APPLY_CHAT_TEMPLATE" = "1" ] && CHAT_FLAG="--apply-chat-template"
[ "$PERMUTE" = "0" ] && PERM_FLAG="--disable-prompt-permutation"
[ "$EXACT_OUTPUT_TOKENS" = "1" ] && EXACT_FLAG="--exact-output-tokens"
[ "$HW_MONITOR" = "1" ] && MONITOR_ARGS+=(--monitor) || MONITOR_ARGS+=(--no-monitor)
[ "$MEMORY_MONITOR" = "1" ] && MEMORY_ARGS+=(--memory-monitor)

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
            elif [ "$DEV" = "NPU" ] && [ -n "$NPU_LOAD_CONFIG" ]; then
                DEVICE_LOAD_CONFIG_ARGS=(-lc "$NPU_LOAD_CONFIG")
            fi
            bench_run_with_log "$LOG_FILE" \
                "$BENCH_PYTHON" "$TESTS_DIR/benchmark_qwen3.py" \
                    -m "$MODEL_DIR" \
                    -d "$DEV" \
                    "${PROMPT_ARGS[@]}" \
                    --infer-count "$INFER_COUNT" \
                    --num-iters "$ITERS" \
                    --num-beams "$NUM_BEAMS" \
                    $CHAT_FLAG $PERM_FLAG $EXACT_FLAG \
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
