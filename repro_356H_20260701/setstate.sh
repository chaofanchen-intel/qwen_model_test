#!/bin/bash
# Materialize one of the 6 plugin build-states into the 6 opt source files, then rebuild.
# Usage: setstate.sh <pristine|opt1|opt12|opt123|opt4|opt4_3>
set -eo pipefail
OV=/home/chaofan/model_benchmark/benchmark/openvino
PA_ARCH="$OV/../tests/opt_pa_archive"
SDPA_ARCH="$OV/../tests/opt_sdpa_archive"
cd "$OV"

OPTS=(
  src/plugins/intel_gpu/src/graph/layout_optimizer.cpp
  src/plugins/intel_gpu/src/kernel_selector/kernels/permute/permute_kernel_ref.cpp
  src/plugins/intel_gpu/src/plugin/transformations/dynamic_quantize_fully_connected.cpp
  src/plugins/intel_gpu/src/graph/dynamic_quantize.cpp
  src/plugins/intel_gpu/src/graph/impls/ocl_v2/paged_causal_conv1d_ref.cl
  src/plugins/intel_gpu/src/graph/impls/ocl_v2/paged_causal_conv1d_ref.cpp
)

restore_pristine() {
  for f in "${OPTS[@]}"; do
    cp -f "/tmp/opt_pristine/$(echo "$f" | tr '/' '_')" "$f"
  done
}

STATE="$1"
echo "=== setstate: $STATE ==="
restore_pristine
case "$STATE" in
  pristine) ;;
  opt1)
    git apply --include='*layout_optimizer.cpp' "$SDPA_ARCH/sdpa_opt123.patch" ;;
  opt12)
    git apply --include='*layout_optimizer.cpp' "$SDPA_ARCH/sdpa_opt123.patch"
    git apply --include='*permute_kernel_ref.cpp' "$SDPA_ARCH/sdpa_opt123.patch" ;;
  opt123)
    git apply "$SDPA_ARCH/sdpa_opt123.patch" ;;
  opt4)
    git apply "$PA_ARCH/pa_opt4_conv1d.patch" ;;
  opt4_3)
    git apply "$PA_ARCH/pa_opt4_conv1d.patch"
    git apply "$PA_ARCH/pa_opt3_dq_reuse.patch" ;;
  *) echo "unknown state $STATE"; exit 1 ;;
esac

echo "--- markers ---"
echo "opt1(layout depthwise short-circuit): $(grep -c 'TTFT-opt: depthwise convolutions' src/plugins/intel_gpu/src/graph/layout_optimizer.cpp || true)"
echo "opt2(permute LWS F-axis):            $(grep -c 'gws_f >= gws_y' src/plugins/intel_gpu/src/kernel_selector/kernels/permute/permute_kernel_ref.cpp || true)"
echo "opt3(DQ reuse):                      $(grep -c 'find_reusable\|reusable_dq' src/plugins/intel_gpu/src/plugin/transformations/dynamic_quantize_fully_connected.cpp || true)"
echo "opt4(conv1d token-parallel):         $(grep -c 'total_tokens' src/plugins/intel_gpu/src/graph/impls/ocl_v2/paged_causal_conv1d_ref.cpp || true)"

echo "--- rebuild ---"
cd "$OV/build"
make openvino_intel_gpu_plugin -j"$(nproc)" > /tmp/repro/build_${STATE}.log 2>&1 && echo "build OK" || { echo "BUILD FAILED, tail:"; tail -20 /tmp/repro/build_${STATE}.log; exit 1; }
cp -f "$OV/bin/intel64/Release/libopenvino_intel_gpu_plugin.so" \
      "$OV/install/runtime/lib/intel64/libopenvino_intel_gpu_plugin.so"
echo "=== setstate $STATE done, .so installed ==="
