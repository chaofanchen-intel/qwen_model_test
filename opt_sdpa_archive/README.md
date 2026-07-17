# SDPA-era GPU source optimizations (opt-1/2/3) — archive

These three optimizations target the **SDPA** attention path's Qwen3.5 prefill.
They are **inert under PA** (the current OpenVINO default); see the PA doc.

- `sdpa_opt123.patch` — git patch of all 3 opts (4 files), base = OpenVINO 2026.2.1 HEAD `ede283a88e`.
- `OV源码优化降低TTFT_SDPA.md` — full reproduction doc (method + data, SDPA口径).

## Apply
    cd benchmark/openvino
    git apply ../tests/opt_sdpa_archive/sdpa_opt123.patch
    bash rebuild_gpu_plugin.sh

## Revert
    cd benchmark/openvino
    git checkout -- \
      src/plugins/intel_gpu/src/graph/layout_optimizer.cpp \
      src/plugins/intel_gpu/src/kernel_selector/kernels/permute/permute_kernel_ref.cpp \
      src/plugins/intel_gpu/src/plugin/transformations/dynamic_quantize_fully_connected.cpp \
      src/plugins/intel_gpu/src/graph/dynamic_quantize.cpp

Files touched: layout_optimizer.cpp (opt-1), permute_kernel_ref.cpp (opt-2),
dynamic_quantize_fully_connected.cpp + dynamic_quantize.cpp (opt-3).
