#!/bin/bash
#
# Official llm_bench runner for Qwen3-Reranker (--task text_rerank).
#
# Usage:
#   ./tests/runners/run_official_reranker.sh
#   ./tests/runners/run_official_reranker.sh GPU
#   MODEL=/abs/path/to/Qwen3-Reranker-0.6B-int8-ov ITERS=10 ./tests/runners/run_official_reranker.sh GPU
#   TEXTS_FILE=/abs/path/texts_64.jsonl RERANKING_MAX_LENGTH=1024 ./tests/runners/run_official_reranker.sh CPU
#
# Env overrides (defaults in []):
#   DEVICE                    [CPU GPU NPU]       positional args override
#   MODEL                     []                  explicit model path(s); empty => auto-discover models/Qwen3-Reranker-*
#   ITERS                     [10]                measured iterations; official warm-up iter 0 is extra
#   PROMPT_FILE               [tests/prompts/reranker/queries.jsonl] query file; set empty to use benchmark default query
#   PROMPT_INDEX              []                  e.g. "0 2"; empty means all queries
#   TEXTS_FILE                [tests/prompts/reranker/texts_16.jsonl] candidate documents ({"text": ...}); set empty for benchmark defaults
#   TOP_N                     [3]                 --reranking_top_n
#   RERANKING_MAX_LENGTH      [512]               --reranking_max_length; set empty for Qwen3 default 8192
#   LOAD_CONFIG               []                  optional JSON file passed as -lc
#   NPU_LOAD_CONFIG           [auto]              JSON string/file for NPU OV config; set empty to disable
#   NPU_CACHE_DIR             [$PROJECT_DIR/cache_npu/reranker]
#   RESOURCE_MONITOR          [1]
#   RESOURCE_INTERVAL_MS      [500]
#   RESOURCE_GPU_COMMAND      []
#   SUDO_REFRESH              [1]
#   MEMORY_CONSUMPTION        [0]                 official -mc
#   MEMORY_INTERVAL           [0.05]
#   INTERVAL                  [0]                 optional cooldown seconds between runs
#   LOG_DIR / REPORT_DIR      [tests/logs/official_reranker_<model>_<device>_<ts>]
#   BENCHMARK_DIR             [results]
#   BENCHMARK_CSV             [$BENCHMARK_DIR/benchmark_reranker_official.csv]
#   BENCHMARK_JSON            [$BENCHMARK_DIR/benchmark_reranker_official.json]
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
RERANKER_PROMPT_DEFAULT="$PROJECT_DIR/tests/prompts/reranker/queries.jsonl"
RERANKER_TEXTS_DEFAULT="$PROJECT_DIR/tests/prompts/reranker/texts_16.jsonl"
PROMPT_FILE="${PROMPT_FILE-$RERANKER_PROMPT_DEFAULT}"
PROMPT_INDEX="${PROMPT_INDEX:-}"
TEXTS_FILE="${TEXTS_FILE-$RERANKER_TEXTS_DEFAULT}"
TOP_N="${TOP_N:-3}"
RERANKING_MAX_LENGTH="${RERANKING_MAX_LENGTH-512}"
RESOURCE_MONITOR="${RESOURCE_MONITOR:-1}"
RESOURCE_INTERVAL_MS="${RESOURCE_INTERVAL_MS:-500}"
RESOURCE_GPU_COMMAND="${RESOURCE_GPU_COMMAND:-}"
SUDO_REFRESH="${SUDO_REFRESH:-1}"
MEMORY_CONSUMPTION="${MEMORY_CONSUMPTION:-0}"
MEMORY_INTERVAL="${MEMORY_INTERVAL:-0.05}"
LOAD_CONFIG="${LOAD_CONFIG:-}"
NPU_CACHE_DIR="${NPU_CACHE_DIR:-$PROJECT_DIR/cache_npu/reranker}"
if [ -z "${NPU_LOAD_CONFIG+x}" ]; then
    NPU_LOAD_CONFIG="{\"CACHE_DIR\":\"$NPU_CACHE_DIR\"}"
fi
# NPU rejects the dynamic-shape reranker IR. Set NPU_STATIC_RESHAPE=1 to auto-generate a
# static [num_candidates, RERANKING_MAX_LENGTH] copy per NPU run (tests/reshape_reranker_static.py)
# and run it with pad_to_max_length=True (forced on NPU in ov_utils.create_genai_text_reranker_model).
# WARNING: NPU compile time/RAM explode with candidate count and max_length. Measured on this box
# (Qwen3-Reranker-0.6B-fp16): [2x256]=64s, [2x512]=132s, [1x2048]=462s, [2x2048] did not finish in
# 15min, [15x512] did not finish in ~16min and used ~26GB. Practical NPU use is small candidate
# counts (a few) with RERANKING_MAX_LENGTH<=512. Larger workloads belong on GPU/CPU.
NPU_STATIC_RESHAPE="${NPU_STATIC_RESHAPE:-0}"
NPU_STATIC_DIR="${NPU_STATIC_DIR:-$PROJECT_DIR/models/_reranker_npu_static}"
INTERVAL="${INTERVAL:-0}"
EXTRA_BENCH_ARGS="${EXTRA_BENCH_ARGS:-}"

bench_validate_load_config "$LOAD_CONFIG" "LOAD_CONFIG"
[ -n "$LOAD_CONFIG" ] && LOAD_CONFIG="$(bench_resolve_load_config "$LOAD_CONFIG")"
bench_validate_load_config "${NPU_LOAD_CONFIG:-}" "NPU_LOAD_CONFIG"
[ -n "${NPU_LOAD_CONFIG:-}" ] && NPU_LOAD_CONFIG="$(bench_resolve_load_config "$NPU_LOAD_CONFIG")"
if [ -n "$PROMPT_FILE" ]; then
    PROMPT_FILE="$(bench_resolve_file "$PROMPT_FILE" "PROMPT_FILE")"
fi
if [ -n "$TEXTS_FILE" ]; then
    TEXTS_FILE="$(bench_resolve_file "$TEXTS_FILE" "TEXTS_FILE")"
fi

# Candidate count = reranker batch dim (all candidates scored in one forward pass).
# With TEXTS_FILE set it is the non-empty line count; with TEXTS_FILE empty it is benchmark.py's
# built-in default of 2 candidates. Needed to build a correctly-shaped static model for NPU.
if [ -n "$TEXTS_FILE" ]; then
    NUM_CANDIDATES="$(grep -c '[^[:space:]]' "$TEXTS_FILE")"
else
    NUM_CANDIDATES=2
fi

MODEL_PATHS=()
MODEL_LABELS=()
if [ -n "${MODEL:-}" ]; then
    read -ra MODEL_PATHS_RAW <<< "$MODEL"
    for model_path in "${MODEL_PATHS_RAW[@]}"; do
        MODEL_PATHS+=("$(bench_resolve_model_dir "$model_path" "MODEL")")
    done
else
    for d in "$PROJECT_DIR"/models/Qwen3-Reranker-*; do
        [ -f "$d/openvino_model.xml" ] && MODEL_PATHS+=("$d")
    done
fi
if [ "${#MODEL_PATHS[@]}" -eq 0 ]; then
    echo "[ERROR] no reranker models found; set MODEL=/path/to/model" >&2
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
bench_make_log_dir "official_reranker_${MODEL_SET_LABEL}_${DEVICE_SET_LABEL}"
REPORT_DIR="$LOG_DIR"
BENCHMARK_DIR="${BENCHMARK_DIR:-$PROJECT_DIR/results}"
mkdir -p "$BENCHMARK_DIR"
BENCHMARK_CSV="${BENCHMARK_CSV:-$BENCHMARK_DIR/benchmark_reranker_official.csv}"
BENCHMARK_JSON="${BENCHMARK_JSON:-$BENCHMARK_DIR/benchmark_reranker_official.json}"

PROMPT_ARGS=()
if [ -n "$PROMPT_FILE" ]; then
    PROMPT_ARGS=(-pf "$PROMPT_FILE")
fi
if [ -n "$PROMPT_INDEX" ]; then
    read -ra PROMPT_INDEX_ARR <<< "$PROMPT_INDEX"
    PROMPT_ARGS+=(-pi "${PROMPT_INDEX_ARR[@]}")
fi

EXTRA_ARGS=(--reranking_top_n "$TOP_N")
[ -n "$RERANKING_MAX_LENGTH" ] && EXTRA_ARGS+=(--reranking_max_length "$RERANKING_MAX_LENGTH")
[ -n "$TEXTS_FILE" ] && EXTRA_ARGS+=(--texts_file "$TEXTS_FILE")
[ -n "$RESOURCE_GPU_COMMAND" ] && EXTRA_ARGS+=(--resource_gpu_command "$RESOURCE_GPU_COMMAND")
if [ -n "$EXTRA_BENCH_ARGS" ]; then
    read -ra EXTRA_BENCH_ARGS_ARR <<< "$EXTRA_BENCH_ARGS"
    EXTRA_ARGS+=("${EXTRA_BENCH_ARGS_ARR[@]}")
fi

if [ "$RESOURCE_MONITOR" = "1" ] && [ "$SUDO_REFRESH" = "1" ]; then
    bench_refresh_sudo
fi

echo "============================================================"
echo " Official Qwen3-Reranker text_rerank benchmark.py Runner"
echo "============================================================"
echo " Devices:        ${DEVICES[*]}"
echo " Models:         ${MODEL_LABELS[*]}"
echo " Query file:     ${PROMPT_FILE:-<benchmark default>}"
echo " Query idx:      ${PROMPT_INDEX:-<all>}"
echo " Texts file:     ${TEXTS_FILE:-<built-in defaults>}"
echo " top_n:          $TOP_N"
echo " max_length:     ${RERANKING_MAX_LENGTH:-<model default>}"
echo " Iterations:     $ITERS (+1 warm-up)"
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
    "runner:           official_reranker" \
    "task:             text_rerank" \
    "devices:          ${DEVICES[*]}" \
    "models:           ${MODEL_PATHS[*]}" \
    "query_file:       ${PROMPT_FILE:-<benchmark default>}" \
    "query_index:      ${PROMPT_INDEX:-<all>}" \
    "texts_file:       ${TEXTS_FILE:-<defaults>}" \
    "top_n:            $TOP_N" \
    "reranking_max_len:${RERANKING_MAX_LENGTH:-<model default>}" \
    "iters:            $ITERS" \
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
        if [ "$FIRST_RUN" = true ]; then
            FIRST_RUN=false
        elif [ "${INTERVAL:-0}" != "0" ]; then
            echo ""
            echo "[WAIT] cooling ${INTERVAL}s before next test..."
            sleep "$INTERVAL"
        fi
        run_name="$(bench_sanitize_label "${model_label}_${dev}_n${ITERS}")"
        csv_out="$LOG_DIR/${run_name}_iters.csv"
        json_out="$LOG_DIR/${run_name}_iters.json"
        log_file="$LOG_DIR/${run_name}.log"
        echo "============================================================"
        echo "[RUN] $run_name"
        echo "      model: $model_path"
        echo "      log:   $log_file"
        echo "============================================================"

        DEVICE_EXTRA_ARGS=()
        if [ -n "$LOAD_CONFIG" ]; then
            DEVICE_EXTRA_ARGS+=(-lc "$LOAD_CONFIG")
        elif [ "$dev" = "NPU" ] && [ -n "${NPU_LOAD_CONFIG:-}" ]; then
            DEVICE_EXTRA_ARGS+=(-lc "$NPU_LOAD_CONFIG")
        fi

        # For NPU, optionally swap in a static [NUM_CANDIDATES, RERANKING_MAX_LENGTH] model so the
        # NPU compiler accepts the IR. Reused across runs with the same shape.
        run_model_path="$model_path"
        if [ "$dev" = "NPU" ] && [ "$NPU_STATIC_RESHAPE" = "1" ]; then
            if [ -z "$RERANKING_MAX_LENGTH" ]; then
                echo "[ERROR] NPU_STATIC_RESHAPE=1 requires RERANKING_MAX_LENGTH to be set (static seq length)." >&2
                exit 1
            fi
            static_dir="$NPU_STATIC_DIR/${model_label}-b${NUM_CANDIDATES}-s${RERANKING_MAX_LENGTH}"
            if [ ! -f "$static_dir/openvino_model.xml" ]; then
                echo "[NPU] generating static reranker model: candidates=$NUM_CANDIDATES max_length=$RERANKING_MAX_LENGTH"
                echo "      (NPU compile of this shape may take minutes; large candidate counts / max_length may be infeasible)"
                "$BENCH_PYTHON" "$TESTS_DIR/reshape_reranker_static.py" \
                    --src "$model_path" \
                    --num-candidates "$NUM_CANDIDATES" \
                    --max-length "$RERANKING_MAX_LENGTH" \
                    --out "$static_dir" --force
            else
                echo "[NPU] reusing static reranker model: $static_dir"
            fi
            run_model_path="$static_dir"
        fi

        cd "$LLM_BENCH_DIR"
        bench_run_with_log "$log_file" \
            "$BENCH_PYTHON" benchmark.py \
                -m "$run_model_path" \
                -d "$dev" \
                -n "$ITERS" \
                "${PROMPT_ARGS[@]}" \
                --task text_rerank \
                -r "$csv_out" \
                -rj "$json_out" \
                --benchmark_csv "$BENCHMARK_CSV" \
                --benchmark_json "$BENCHMARK_JSON" \
                --resource_monitor "$RESOURCE_MONITOR" \
                --resource_monitor_interval "$RESOURCE_INTERVAL_MS" \
                -mc "$MEMORY_CONSUMPTION" \
                --memory_consumption_interval "$MEMORY_INTERVAL" \
                "${DEVICE_EXTRA_ARGS[@]}" \
                "${EXTRA_ARGS[@]}"
    done
done

echo ""
echo "============================================================"
echo " Official Qwen3-Reranker benchmark complete"
echo " Benchmark CSV:  $BENCHMARK_CSV"
echo " Benchmark JSON: $BENCHMARK_JSON"
echo " Logs:           $LOG_DIR/"
echo "============================================================"
