#!/bin/bash
#
# Official llm_bench runner for Qwen3-Embedding (--task text_embed).
#
# Usage:
#   ./tests/runners/run_official_embedding.sh
#   ./tests/runners/run_official_embedding.sh GPU
#   MODEL=/abs/path/to/Qwen3-Embedding-0.6B-int8-ov ITERS=10 ./tests/runners/run_official_embedding.sh GPU
#   PROMPT_FILE=/abs/path/texts.jsonl EMBEDDING_MAX_LENGTH=1024 ./tests/runners/run_official_embedding.sh GPU
#   LENGTHS="100 200 400 600 800 1k 2k 4k 6k 8k" BATCH_SIZES="1 2" ./tests/runners/run_official_embedding.sh GPU
#
# Env overrides (defaults in []):
#   DEVICE                    [CPU GPU NPU]       positional args override
#   MODEL                     []                  explicit model path(s); empty => auto-discover models/Qwen3-Embedding-*
#   ITERS                     [10]                measured iterations; official warm-up iter 0 is extra
#   PROMPT_FILE               [tests/prompts/embedding/texts.jsonl]; set empty for benchmark default text
#   PROMPT_INDEX              [0]                 e.g. "0 2"; empty means all prompts
#   BATCH_SIZE                [1]                 -bs (texts per embed call)
#   BATCH_SIZES               []                  optional sweep list, e.g. "1 2"; overrides BATCH_SIZE per run
#   LENGTHS                   []                  optional token-length sweep. 1k/2k/4k/6k/8k map to
#                                                   1024/2048/4096/6144/8192; also supports 100/200/400/600/800
#   POOLING                   [last_token]        --embedding_pooling: cls|mean|last_token
#   NORMALIZE                 [1]                 1 adds --embedding_normalize
#   EMBEDDING_MAX_LENGTH      [512]               --embedding_max_length; set empty for pipeline default
#   EMBEDDING_PADDING_SIDE    [left]              --embedding_padding_side: left|right
#   PAD_TO_MAX_LENGTH         [1]                 1 adds --embedding_pad_to_max_length
#   LOAD_CONFIG               []                  optional JSON file passed as -lc
#   NPU_LOAD_CONFIG           [auto]              JSON string/file for NPU OV config; set empty to disable
#   NPU_CACHE_DIR             [$PROJECT_DIR/cache_npu/embedding]
#   RESOURCE_MONITOR          [1]
#   RESOURCE_INTERVAL_MS      [500]
#   RESOURCE_GPU_COMMAND      []                  overrides auto sudo -n xpu-smi command
#   SUDO_REFRESH              [1]                 run sudo -v before resource monitoring
#   MEMORY_CONSUMPTION        [0]                 official -mc
#   MEMORY_INTERVAL           [0.05]
#   INTERVAL                  [0]                 optional cooldown seconds between runs
#   LOG_DIR / REPORT_DIR      [tests/logs/official_embedding_<model>_<device>_<ts>]
#   BENCHMARK_DIR             [results]
#   BENCHMARK_CSV             [$BENCHMARK_DIR/benchmark_embedding_official.csv]
#   BENCHMARK_JSON            [$BENCHMARK_DIR/benchmark_embedding_official.json]
#   EXTRA_BENCH_ARGS          []                  appended to benchmark.py

set -euo pipefail

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    sed -n '2,40p' "$0"
    exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_DIR="$(cd "$TESTS_DIR/.." && pwd)"
LLM_BENCH_DIR="$PROJECT_DIR/openvino.genai/tools/llm_bench"

source "$SCRIPT_DIR/_common.sh"
if [ -z "${PYTHON:-}" ] && [ -x "/home/user/envs/qwen_env/bin/python" ]; then
    PYTHON="/home/user/envs/qwen_env/bin/python"
fi
bench_select_python

DEVICE="${DEVICE:-CPU GPU NPU}"
bench_parse_devices "$@"

ITERS="${ITERS:-10}"
EMBEDDING_PROMPT_DEFAULT="$PROJECT_DIR/tests/prompts/embedding/texts.jsonl"
EMBEDDING_LENGTH_PROMPT_DEFAULT="$PROJECT_DIR/tests/prompts/embedding/texts_len_sweep.jsonl"
PROMPT_FILE_WAS_SET=0
if [ -n "${PROMPT_FILE+x}" ]; then
    PROMPT_FILE_WAS_SET=1
fi
PROMPT_FILE="${PROMPT_FILE-$EMBEDDING_PROMPT_DEFAULT}"
PROMPT_INDEX="${PROMPT_INDEX:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
BATCH_SIZES="${BATCH_SIZES:-}"
LENGTHS="${LENGTHS:-}"
POOLING="${POOLING:-last_token}"
NORMALIZE="${NORMALIZE:-1}"
EMBEDDING_MAX_LENGTH="${EMBEDDING_MAX_LENGTH-512}"
EMBEDDING_PADDING_SIDE="${EMBEDDING_PADDING_SIDE:-left}"
PAD_TO_MAX_LENGTH="${PAD_TO_MAX_LENGTH:-1}"
RESOURCE_MONITOR="${RESOURCE_MONITOR:-1}"
RESOURCE_INTERVAL_MS="${RESOURCE_INTERVAL_MS:-500}"
RESOURCE_GPU_COMMAND="${RESOURCE_GPU_COMMAND:-}"
SUDO_REFRESH="${SUDO_REFRESH:-1}"
MEMORY_CONSUMPTION="${MEMORY_CONSUMPTION:-0}"
MEMORY_INTERVAL="${MEMORY_INTERVAL:-0.05}"
LOAD_CONFIG="${LOAD_CONFIG:-}"
NPU_CACHE_DIR="${NPU_CACHE_DIR:-$PROJECT_DIR/cache_npu/embedding}"
if [ -z "${NPU_LOAD_CONFIG+x}" ]; then
    NPU_LOAD_CONFIG="{\"CACHE_DIR\":\"$NPU_CACHE_DIR\"}"
fi
INTERVAL="${INTERVAL:-0}"
EXTRA_BENCH_ARGS="${EXTRA_BENCH_ARGS:-}"

bench_validate_load_config "$LOAD_CONFIG" "LOAD_CONFIG"
[ -n "$LOAD_CONFIG" ] && LOAD_CONFIG="$(bench_resolve_load_config "$LOAD_CONFIG")"
bench_validate_load_config "${NPU_LOAD_CONFIG:-}" "NPU_LOAD_CONFIG"
[ -n "${NPU_LOAD_CONFIG:-}" ] && NPU_LOAD_CONFIG="$(bench_resolve_load_config "$NPU_LOAD_CONFIG")"
if [ -n "$LENGTHS" ] && [ "$PROMPT_FILE_WAS_SET" = "0" ]; then
    PROMPT_FILE="$EMBEDDING_LENGTH_PROMPT_DEFAULT"
fi
if [ -n "$PROMPT_FILE" ]; then
    PROMPT_FILE="$(bench_resolve_file "$PROMPT_FILE" "PROMPT_FILE")"
fi

MODEL_PATHS=()
MODEL_LABELS=()
if [ -n "${MODEL:-}" ]; then
    read -ra MODEL_PATHS_RAW <<< "$MODEL"
    for model_path in "${MODEL_PATHS_RAW[@]}"; do
        MODEL_PATHS+=("$(bench_resolve_model_dir "$model_path" "MODEL")")
    done
else
    for d in "$PROJECT_DIR"/models/Qwen3-Embedding-*; do
        [ -f "$d/openvino_model.xml" ] && MODEL_PATHS+=("$d")
    done
fi
if [ "${#MODEL_PATHS[@]}" -eq 0 ]; then
    echo "[ERROR] no embedding models found; set MODEL=/path/to/model" >&2
    exit 1
fi
for model_path in "${MODEL_PATHS[@]}"; do
    MODEL_LABELS+=("$(bench_sanitize_label "$(basename "$model_path")")")
done

if [ "${#MODEL_LABELS[@]}" -eq 1 ]; then
    MODEL_SET_LABEL="${MODEL_LABELS[0]}"
else
    MODEL_SET_LABEL="multi"
fi
DEVICE_SET_LABEL="$(bench_sanitize_label "${DEVICES[*]}")"
if [ -n "${REPORT_DIR:-}" ] && [ -z "${LOG_DIR:-}" ]; then
    LOG_DIR="$REPORT_DIR"
fi
bench_make_log_dir "official_embedding_${MODEL_SET_LABEL}_${DEVICE_SET_LABEL}"
REPORT_DIR="$LOG_DIR"
BENCHMARK_DIR="${BENCHMARK_DIR:-$PROJECT_DIR/results}"
mkdir -p "$BENCHMARK_DIR"
BENCHMARK_CSV="${BENCHMARK_CSV:-$BENCHMARK_DIR/benchmark_embedding_official.csv}"
BENCHMARK_JSON="${BENCHMARK_JSON:-$BENCHMARK_DIR/benchmark_embedding_official.json}"

BASE_PROMPT_ARGS=()
if [ -n "$PROMPT_FILE" ]; then
    BASE_PROMPT_ARGS=(-pf "$PROMPT_FILE")
fi

BASE_EXTRA_ARGS=(--embedding_pooling "$POOLING")
[ "$NORMALIZE" = "1" ] && BASE_EXTRA_ARGS+=(--embedding_normalize)
[ -n "$EMBEDDING_PADDING_SIDE" ] && BASE_EXTRA_ARGS+=(--embedding_padding_side "$EMBEDDING_PADDING_SIDE")
[ "$PAD_TO_MAX_LENGTH" = "1" ] && BASE_EXTRA_ARGS+=(--embedding_pad_to_max_length)
[ -n "$RESOURCE_GPU_COMMAND" ] && BASE_EXTRA_ARGS+=(--resource_gpu_command "$RESOURCE_GPU_COMMAND")
if [ -n "$EXTRA_BENCH_ARGS" ]; then
    read -ra EXTRA_BENCH_ARGS_ARR <<< "$EXTRA_BENCH_ARGS"
    BASE_EXTRA_ARGS+=("${EXTRA_BENCH_ARGS_ARR[@]}")
fi

if [ -n "$BATCH_SIZES" ]; then
    read -ra BATCH_SIZE_RUNS <<< "$BATCH_SIZES"
else
    BATCH_SIZE_RUNS=("$BATCH_SIZE")
fi

if [ -n "$LENGTHS" ]; then
    read -ra LENGTH_RUNS <<< "$LENGTHS"
else
    LENGTH_RUNS=("")
fi

embedding_prompt_index_for_length() {
    case "$1" in
        100) echo 5 ;;
        200) echo 6 ;;
        400) echo 7 ;;
        600) echo 8 ;;
        800) echo 9 ;;
        1024|1k|1K) echo 0 ;;
        2048|2k|2K) echo 1 ;;
        4096|4k|4K) echo 2 ;;
        6144|6k|6K) echo 3 ;;
        8192|8k|8K) echo 4 ;;
        *) echo "[ERROR] unsupported LENGTHS value '$1' (supported: 100 200 400 600 800 1024/1k 2048/2k 4096/4k 6144/6k 8192/8k)" >&2; return 1 ;;
    esac
}

embedding_prompt_length_value() {
    case "$1" in
        1k|1K) echo 1024 ;;
        2k|2K) echo 2048 ;;
        4k|4K) echo 4096 ;;
        6k|6K) echo 6144 ;;
        8k|8K) echo 8192 ;;
        *) echo "$1" ;;
    esac
}

if [ "$RESOURCE_MONITOR" = "1" ] && [ "$SUDO_REFRESH" = "1" ]; then
    bench_refresh_sudo
fi

echo "============================================================"
echo " Official Qwen3-Embedding text_embed benchmark.py Runner"
echo "============================================================"
echo " Devices:        ${DEVICES[*]}"
echo " Models:         ${MODEL_LABELS[*]}"
echo " Prompt file:    ${PROMPT_FILE:-<benchmark default>}"
if [ -n "$LENGTHS" ]; then
    echo " Length sweep:   $LENGTHS"
    echo " Batch sweep:    ${BATCH_SIZE_RUNS[*]}"
else
    echo " Prompt idx:     ${PROMPT_INDEX:-<all>}"
fi
echo " Iterations:     $ITERS (+1 warm-up)"
echo " batch_size:     $BATCH_SIZE"
echo " pooling:        $POOLING"
echo " normalize:      $NORMALIZE"
echo " padding_side:   $EMBEDDING_PADDING_SIDE"
echo " max_length:     ${EMBEDDING_MAX_LENGTH:-<pipeline default>}"
echo " pad_to_max:     $PAD_TO_MAX_LENGTH"
echo " Load config:    ${LOAD_CONFIG:-<none>}"
echo " NPU config:     ${NPU_LOAD_CONFIG:-<none>}"
echo " Resource mon:   $RESOURCE_MONITOR (${RESOURCE_INTERVAL_MS}ms)"
echo " Memory mode:    $MEMORY_CONSUMPTION (${MEMORY_INTERVAL}s)"
echo " Log dir:        $LOG_DIR"
echo " Benchmark CSV:  $BENCHMARK_CSV"
echo " Benchmark JSON: $BENCHMARK_JSON"
echo "============================================================"
echo ""

bench_write_manifest "$LOG_DIR/run_info.txt" \
    "runner:           official_embedding" \
    "task:             text_embed" \
    "devices:          ${DEVICES[*]}" \
    "models:           ${MODEL_PATHS[*]}" \
    "prompt_file:      ${PROMPT_FILE:-<benchmark default>}" \
    "prompt_index:     ${PROMPT_INDEX:-<all>}" \
    "lengths:          ${LENGTHS:-}" \
    "batch_sizes:      ${BATCH_SIZES:-}" \
    "iters:            $ITERS" \
    "batch_size:       $BATCH_SIZE" \
    "pooling:          $POOLING" \
    "normalize:        $NORMALIZE" \
    "padding_side:     $EMBEDDING_PADDING_SIDE" \
    "max_length:       ${EMBEDDING_MAX_LENGTH:-<pipeline default>}" \
    "pad_to_max_len:   $PAD_TO_MAX_LENGTH" \
    "load_config:      $LOAD_CONFIG" \
    "npu_load_config:  ${NPU_LOAD_CONFIG:-}" \
    "resource_monitor: $RESOURCE_MONITOR" \
    "resource_interval:$RESOURCE_INTERVAL_MS" \
    "memory_mode:      $MEMORY_CONSUMPTION" \
    "benchmark_csv:    $BENCHMARK_CSV" \
    "benchmark_json:   $BENCHMARK_JSON"

FIRST_RUN=true
for idx in "${!MODEL_PATHS[@]}"; do
    model_path="${MODEL_PATHS[$idx]}"
    model_label="${MODEL_LABELS[$idx]}"
    if [ ! -d "$model_path" ]; then
        echo "[SKIP] Model not found: $model_path"
        continue
    fi
    for dev in "${DEVICES[@]}"; do
        DEVICE_EXTRA_ARGS=()
        if [ -n "$LOAD_CONFIG" ]; then
            DEVICE_EXTRA_ARGS+=(-lc "$LOAD_CONFIG")
        elif [ "$dev" = "NPU" ] && [ -n "${NPU_LOAD_CONFIG:-}" ]; then
            DEVICE_EXTRA_ARGS+=(-lc "$NPU_LOAD_CONFIG")
        fi

        for length_label in "${LENGTH_RUNS[@]}"; do
            RUN_PROMPT_ARGS=("${BASE_PROMPT_ARGS[@]}")
            RUN_EXTRA_ARGS=("${BASE_EXTRA_ARGS[@]}")
            length_value=""
            length_run_label="defaultlen"
            if [ -n "$length_label" ]; then
                length_value="$(embedding_prompt_length_value "$length_label")"
                prompt_idx="$(embedding_prompt_index_for_length "$length_label")"
                RUN_PROMPT_ARGS+=(-pi "$prompt_idx")
                RUN_EXTRA_ARGS+=(--embedding_max_length "$length_value")
                length_run_label="len${length_value}"
            else
                if [ -n "$PROMPT_INDEX" ]; then
                    read -ra PROMPT_INDEX_ARR <<< "$PROMPT_INDEX"
                    RUN_PROMPT_ARGS+=(-pi "${PROMPT_INDEX_ARR[@]}")
                fi
                [ -n "$EMBEDDING_MAX_LENGTH" ] && RUN_EXTRA_ARGS+=(--embedding_max_length "$EMBEDDING_MAX_LENGTH")
            fi
            for bs in "${BATCH_SIZE_RUNS[@]}"; do
                if [ "$FIRST_RUN" = true ]; then
                    FIRST_RUN=false
                elif [ "${INTERVAL:-0}" != "0" ]; then
                    echo ""
                    echo "[WAIT] cooling ${INTERVAL}s before next test..."
                    sleep "$INTERVAL"
                fi
                run_name="$(bench_sanitize_label "${model_label}_${dev}_${length_run_label}_bs${bs}_n${ITERS}")"
                csv_out="$LOG_DIR/${run_name}_iters.csv"
                json_out="$LOG_DIR/${run_name}_iters.json"
                log_file="$LOG_DIR/${run_name}.log"
                echo "============================================================"
                echo "[RUN] $run_name"
                echo "      model:      $model_path"
                echo "      length:     ${length_value:-${EMBEDDING_MAX_LENGTH:-<pipeline default>}}"
                echo "      batch_size: $bs"
                echo "      log:        $log_file"
                echo "============================================================"

                cd "$LLM_BENCH_DIR"
                bench_run_with_log "$log_file" \
                    "$BENCH_PYTHON" benchmark.py \
                        -m "$model_path" \
                        -d "$dev" \
                        -n "$ITERS" \
                        "${RUN_PROMPT_ARGS[@]}" \
                        --task text_embed \
                        -bs "$bs" \
                        -r "$csv_out" \
                        -rj "$json_out" \
                        --benchmark_csv "$BENCHMARK_CSV" \
                        --benchmark_json "$BENCHMARK_JSON" \
                        --resource_monitor "$RESOURCE_MONITOR" \
                        --resource_monitor_interval "$RESOURCE_INTERVAL_MS" \
                        -mc "$MEMORY_CONSUMPTION" \
                        --memory_consumption_interval "$MEMORY_INTERVAL" \
                        "${DEVICE_EXTRA_ARGS[@]}" \
                        "${RUN_EXTRA_ARGS[@]}"
            done
        done
    done
done

echo ""
echo "============================================================"
echo " Official Qwen3-Embedding benchmark complete"
echo " Benchmark CSV:  $BENCHMARK_CSV"
echo " Benchmark JSON: $BENCHMARK_JSON"
echo " Logs:           $LOG_DIR/"
echo "============================================================"
