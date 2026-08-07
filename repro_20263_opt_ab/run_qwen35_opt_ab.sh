#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/chaofan/model_benchmark
OV=$ROOT/benchmark/openvino
BUILD=$OV/build
INSTALL_PLUGIN=$OV/install/runtime/lib/intel64/libopenvino_intel_gpu_plugin.so
BUILT_PLUGIN=$OV/bin/intel64/Release/libopenvino_intel_gpu_plugin.so
PATCH_DIR=$OV/qwen3.5_patch
OUT=$ROOT/benchmark/tests/repro_20263_opt_ab/results
MODEL=$ROOT/benchmark/models/Qwen3.5-2B-INT4
PROMPT=$ROOT/model_bench/prompts/qwen3.5_1k.jsonl
MB=$ROOT/model_bench/build-local/model_bench
mkdir -p "$OUT"

files=(
  src/plugins/intel_gpu/src/graph/layout_optimizer.cpp
  src/plugins/intel_gpu/src/kernel_selector/kernels/permute/permute_kernel_ref.cpp
  src/plugins/intel_gpu/src/graph/dynamic_quantize.cpp
  src/plugins/intel_gpu/src/plugin/transformations/dynamic_quantize_fully_connected.cpp
  src/plugins/intel_gpu/src/graph/impls/ocl_v2/paged_causal_conv1d_ref.cl
  src/plugins/intel_gpu/src/graph/impls/ocl_v2/paged_causal_conv1d_ref.cpp
  src/plugins/intel_gpu/src/kernel_selector/cl_kernels/activation_opt.cl
  src/plugins/intel_gpu/src/kernel_selector/kernels/activation/activation_kernel_opt.cpp
)
backup=$(mktemp -d)
installed_backup=$(mktemp)
cleanup() {
  for f in "${files[@]}"; do cp "$backup/$f" "$OV/$f"; done
  cp "$installed_backup" "$INSTALL_PLUGIN"
  rm -rf "$backup" "$installed_backup"
}
trap cleanup EXIT
for f in "${files[@]}"; do mkdir -p "$backup/$(dirname "$f")"; cp "$OV/$f" "$backup/$f"; done
cp "$INSTALL_PLUGIN" "$installed_backup"

apply_state() {
  local state=$1; shift
  for f in "${files[@]}"; do git -C "$OV" show HEAD:"$f" > "$OV/$f"; done
  for p in "$@"; do git -C "$OV" apply "$PATCH_DIR/$p"; done
  cmake --build "$BUILD" --target openvino_intel_gpu_plugin -j2 >/dev/null
  cp "$BUILT_PLUGIN" "$INSTALL_PLUGIN"
  cp "$BUILT_PLUGIN" "$OUT/libopenvino_intel_gpu_plugin_${state}.so"
}
run_state() {
  local state=$1 backend=$2
  echo "[$(date -Is)] start $state backend=$backend"
  apply_state "$state" "${@:3}"
  "$MB" vlm -m "$MODEL" -d GPU -n 5 -ic 512 -pf "$PROMPT" \
    --enable-prompt-permutation --no-monitor \
    -lc <(printf '{"PERFORMANCE_HINT":"LATENCY","ATTENTION_BACKEND":"%s"}\n' "$backend") \
    -rj "$OUT/${state}.json" -r "$OUT/${state}.csv" > "$OUT/${state}.log" 2>&1
  grep -E '^  TTFT|^  TPOT|^  Decode TPS' "$OUT/${state}.log" || true
}

# SDPA: isolate opt-1, opt-2, opt-3 cumulatively.
run_state sdpa_pristine SDPA
run_state sdpa_opt1 SDPA opt1_depthwise_conv_planar.patch
run_state sdpa_opt1_opt2 SDPA opt1_depthwise_conv_planar.patch opt2_permute_lws_f_axis.patch
run_state sdpa_opt1_opt2_opt3 SDPA opt1_depthwise_conv_planar.patch opt2_permute_lws_f_axis.patch opt3_dynamic_quantize_reuse.patch
# PA: isolate opt-3, opt-4, opt-5 cumulatively.
run_state pa_pristine PA
run_state pa_opt3 PA opt3_dynamic_quantize_reuse.patch
run_state pa_opt3_opt4 PA opt3_dynamic_quantize_reuse.patch opt4_conv1d_token_tiled.patch
run_state pa_opt3_opt4_opt5 PA opt3_dynamic_quantize_reuse.patch opt4_conv1d_token_tiled.patch opt5_dynamic_activation_opt.patch

echo "[$(date -Is)] ALL_DONE"
