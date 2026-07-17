#!/bin/bash
# Common helpers sourced by all benchmark runner scripts under tests/runners/.

# bench_select_python -> sets BENCH_PYTHON.
# Honors PYTHON=... first, then falls back to python, then python3.
bench_select_python() {
    if [ -n "${PYTHON:-}" ]; then
        BENCH_PYTHON="$PYTHON"
    elif command -v python >/dev/null 2>&1; then
        BENCH_PYTHON="python"
    elif command -v python3 >/dev/null 2>&1; then
        BENCH_PYTHON="python3"
    else
        echo "[ERROR] neither python nor python3 was found; set PYTHON=/path/to/python" >&2
        exit 1
    fi
}

# bench_run_with_log <log_file> <command...>
# Runs `command...` capturing stdout+stderr into the log via tee, then prints
# a [WARN] line if the command itself exited non-zero. Uses ${PIPESTATUS[0]}
# so `tee` exit code never masks a real failure.
bench_run_with_log() {
    local log_file="$1"
    shift
    set +e
    "$@" 2>&1 | tee "$log_file"
    local exit_code=${PIPESTATUS[0]}
    set -e
    if [ "$exit_code" -ne 0 ]; then
        echo "[WARN] command exited non-zero (code=$exit_code); continuing." | tee -a "$log_file"
    fi
    return 0
}

# bench_make_log_dir <prefix> -> sets LOG_DIR (no collisions)
bench_make_log_dir() {
    local prefix="$1"
    local timestamp
    timestamp=$(date +%Y%m%d_%H%M%S)
    LOG_DIR="${LOG_DIR:-${TESTS_DIR}/logs/${prefix}_${timestamp}}"
    local suffix=0
    while [ -d "$LOG_DIR" ]; do
        suffix=$((suffix + 1))
        LOG_DIR="${TESTS_DIR}/logs/${prefix}_${timestamp}_${suffix}"
    done
    mkdir -p "$LOG_DIR"
}

# bench_write_manifest <manifest_path> <kv pairs...>
# Each kv pair is "key=value"
bench_write_manifest() {
    local path="$1"
    local py="${BENCH_PYTHON:-${PYTHON:-python}}"
    shift
    {
        for kv in "$@"; do
            echo "$kv"
        done
        echo "python_cmd:       $py"
        echo "python:           $($py --version 2>&1)"
        echo "openvino:         $($py -c "import openvino as ov; print(ov.get_version())" 2>&1)"
        echo "openvino_genai:   $($py -c "import openvino_genai as g; print(g.__version__)" 2>&1 || echo 'n/a')"
    } > "$path"
}

# bench_parse_devices: positional args > $DEVICE > "CPU"
# Sets DEVICES array.
bench_parse_devices() {
    if [ $# -gt 0 ]; then
        DEVICES=("$@")
    elif [ -n "${DEVICE:-}" ]; then
        read -ra DEVICES <<< "$DEVICE"
    else
        DEVICES=("CPU")
    fi
}

# bench_sanitize_label <text>
# Converts model/device labels into filesystem-friendly names.
bench_sanitize_label() {
    printf '%s' "$1" | tr ' /:' '___' | tr -cd '[:alnum:]_.-'
}

# bench_normalize_size_b <size>
# Normalizes Qwen size suffixes to match exported directory names such as 0.6B.
bench_normalize_size_b() {
    local size="$1"
    if [[ "$size" =~ ^([0-9.]+)[bB]$ ]]; then
        printf '%sB\n' "${BASH_REMATCH[1]}"
    else
        printf '%s\n' "$size"
    fi
}

# bench_resolve_model_dir <path> <label>
# Resolves absolute and project-root-relative model directories.
bench_resolve_model_dir() {
    local path="$1"
    local label="${2:-MODEL}"
    if [ -d "$path" ]; then
        (cd "$path" && pwd)
        return 0
    fi
    if [ -n "${PROJECT_DIR:-}" ] && [ -d "$PROJECT_DIR/$path" ]; then
        (cd "$PROJECT_DIR/$path" && pwd)
        return 0
    fi
    if [ -n "${PROJECT_DIR:-}" ] && [ -d "$PROJECT_DIR/models/$path" ]; then
        (cd "$PROJECT_DIR/models/$path" && pwd)
        return 0
    fi
    echo "[ERROR] $label not found: $path" >&2
    return 1
}

# bench_append_unique_model_path <array_name> <model_path>
# Appends a model path to an array only if it is not already present.
bench_append_unique_model_path() {
    local -n target_array="$1"
    local model_path="$2"
    local existing
    for existing in "${target_array[@]}"; do
        [ "$existing" = "$model_path" ] && return 0
    done
    target_array+=("$model_path")
}

# bench_discover_model_dirs <glob> [<glob>...]
# Prints model directories under $PROJECT_DIR/models matching one or more globs.
bench_discover_model_dirs() {
    local pattern d
    local -a dirs=()
    shopt -s nullglob
    for pattern in "$@"; do
        for d in "$PROJECT_DIR"/models/$pattern; do
            if [ -f "$d/openvino_model.xml" ] || [ -f "$d/openvino_language_model.xml" ]; then
                dirs+=("$d")
            fi
        done
    done
    shopt -u nullglob
    if [ "${#dirs[@]}" -gt 0 ]; then
        printf '%s\n' "${dirs[@]}" | sort -uV
    fi
}

# bench_validate_load_config <value> <label>
# Accepts either an existing file path or an inline JSON object/array string.
bench_validate_load_config() {
    local value="$1"
    local label="${2:-LOAD_CONFIG}"
    if [ -z "$value" ] || [ -f "$value" ]; then
        return 0
    fi
    if [ -n "${PROJECT_DIR:-}" ] && [ -f "$PROJECT_DIR/$value" ]; then
        return 0
    fi
    case "$value" in
        \{*|\[*) return 0 ;;
    esac
    echo "[ERROR] $label must be an existing file or inline JSON: $value" >&2
    return 1
}

# bench_resolve_load_config <value>
# Prints an absolute file path when value names a JSON file, otherwise prints
# the inline JSON string unchanged.
bench_resolve_load_config() {
    local value="$1"
    if [ -z "$value" ]; then
        printf '\n'
        return 0
    fi
    if [ -f "$value" ]; then
        local dir base
        dir="$(cd "$(dirname "$value")" && pwd)"
        base="$(basename "$value")"
        printf '%s/%s\n' "$dir" "$base"
        return 0
    fi
    if [ -n "${PROJECT_DIR:-}" ] && [ -f "$PROJECT_DIR/$value" ]; then
        printf '%s\n' "$PROJECT_DIR/$value"
        return 0
    fi
    printf '%s\n' "$value"
}

# bench_resolve_file <path> <label>
# Resolves absolute paths, project-root-relative paths, and llm_bench-relative
# paths. Prints the resolved path on success.
bench_resolve_file() {
    local path="$1"
    local label="${2:-file}"
    if [ -f "$path" ]; then
        local dir base
        dir="$(cd "$(dirname "$path")" && pwd)"
        base="$(basename "$path")"
        printf '%s/%s\n' "$dir" "$base"
        return 0
    fi
    if [ -n "${PROJECT_DIR:-}" ] && [ -f "$PROJECT_DIR/$path" ]; then
        printf '%s\n' "$PROJECT_DIR/$path"
        return 0
    fi
    if [ -n "${LLM_BENCH_DIR:-}" ] && [ -f "$LLM_BENCH_DIR/$path" ]; then
        printf '%s\n' "$LLM_BENCH_DIR/$path"
        return 0
    fi
    echo "[ERROR] $label not found: $path" >&2
    return 1
}

# bench_refresh_sudo
# Pre-authenticates sudo before monitor threads start calling sudo -n.
bench_refresh_sudo() {
    if ! command -v sudo >/dev/null 2>&1; then
        echo "[WARN] sudo not found; GPU/NPU utilization may be unavailable."
        return 0
    fi
    echo "[INFO] Refreshing sudo credentials for xpu-smi/NPU telemetry..."
    if sudo -v; then
        echo "[INFO] sudo credentials ready."
    else
        echo "[WARN] sudo authentication failed; GPU/NPU utilization may be unavailable."
    fi
}
