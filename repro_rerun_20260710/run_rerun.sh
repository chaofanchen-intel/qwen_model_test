#!/bin/bash
# Clean re-test driver for the Qwen3/Qwen3.5 optimization report (20260710).
#
# Why this exists: the original repro_leader_report_20260710 GPU data was
# collected in one continuous ~15-min session and shows a monotonic thermal/
# contention ramp (r0 cold -> r4 hot) that the 5-round Latin-square design did
# NOT cancel. NPU-4B shows a milder within-run ramp. This driver re-measures
# everything on the idle machine as n=8 (+1 warm-up) per state/config, with a
# cooldown between runs and per-run temp logging, so any residual drift is
# visible. Permutation is ON for BOTH GPU and NPU (deterministic, iter-indexed).
#
# GPU states are reproduced by swapping ONLY the intel_gpu plugin .so; the
# genai lib never changes (opt7-genai is byte-identical to pristine-genai, so
# "final" == "opt3_opt5_opt6"). The installed plugin is restored on exit.
set -uo pipefail

ROOT=/home/chaofan/model_benchmark
cd "$ROOT"
MB=./model_bench/build-local/model_bench
NEW=benchmark/tests/repro_rerun_20260710
BIN=benchmark/tests/repro_leader_report_20260710/ic512/binaries
INSTALL=benchmark/openvino/install/runtime/lib/intel64/libopenvino_intel_gpu_plugin.so
GPU_PROMPT=./model_bench/prompts/qwen3.5_1k.jsonl
NPU_PROMPT=./benchmark/tests/prompts/qwen3_1k.jsonl
COOL=${COOL:-45}          # cooldown seconds between runs
PROG=$NEW/logs/progress.log

# --- restore the original installed plugin no matter how we exit ---
restore_plugin() {
  cp -f "$NEW/backup/libopenvino_intel_gpu_plugin.INSTALLED.so" "$INSTALL"
  echo "[$(now)] RESTORED installed plugin -> $(md5sum "$INSTALL" | cut -c1-12) (expect de95bee0c6c6)" | tee -a "$PROG"
}
trap restore_plugin EXIT

now() { date +%H:%M:%S; }
temp() { awk '{printf "%d", $1/1000}' /sys/class/thermal/thermal_zone*/temp 2>/dev/null | head -c3; cat /sys/class/thermal/thermal_zone6/temp 2>/dev/null | awk '{printf " pkg=%dC", $1/1000}'; }
cool() { echo "[$(now)] cooldown ${COOL}s (pkg=$(cat /sys/class/thermal/thermal_zone6/temp 2>/dev/null | awk '{printf "%dC",$1/1000}'))" | tee -a "$PROG"; sleep "$COOL"; }

log() { echo "[$(now)] $*" | tee -a "$PROG"; }

swap_gpu() {  # $1 = state key (matches binaries/libopenvino_intel_gpu_plugin_<key>.so)
  cp -f "$BIN/libopenvino_intel_gpu_plugin_$1.so" "$INSTALL"
  log "  swapped GPU plugin -> $1 ($(md5sum "$INSTALL" | cut -c1-12))"
}

gpu_run() {  # $1=size dir (0.8B-INT4)  $2=label  $3=plugin-state-key
  local size=$1 label=$2 state=$3
  local out=$NEW/gpu_pa/$size/$label
  swap_gpu "$state"
  cool
  log "GPU RUN size=$size label=$label pkg_before=$(cat /sys/class/thermal/thermal_zone6/temp 2>/dev/null | awk '{printf "%dC",$1/1000}')"
  $MB vlm -m ./benchmark/models/Qwen3.5-$size -d GPU -n 8 -ic 512 \
     -pf "$GPU_PROMPT" --enable-prompt-permutation --no-monitor \
     -lc "$NEW/gpu_pa.json" -rj "$out.json" -r "$out.csv" \
     > "$NEW/logs/gpu_${size}_${label}.log" 2>&1
  log "  done -> $out.json  ($(grep -m1 'TTFT ' "$NEW/logs/gpu_${size}_${label}.log" | awk '{print "TTFT_mean="$2" std="$3}'))"
}

npu_run() {  # $1=model (Qwen3-0.6B-INT4)  $2=config label  $3=config json
  local model=$1 label=$2 cfg=$3
  local out=$NEW/npu/${model}_${label}
  cool
  log "NPU RUN model=$model config=$label pkg_before=$(cat /sys/class/thermal/thermal_zone6/temp 2>/dev/null | awk '{printf "%dC",$1/1000}')"
  $MB llm -m ./benchmark/models/$model -d NPU -n 8 -ic 512 \
     -pf "$NPU_PROMPT" --enable-prompt-permutation --no-monitor \
     -lc "$NEW/$cfg" -rj "$out.json" -r "$out.csv" \
     > "$NEW/logs/npu_${model}_${label}.log" 2>&1
  log "  done -> $out.json  ($(grep -m1 'TTFT ' "$NEW/logs/npu_${model}_${label}.log" | awk '{print "TTFT_mean="$2" std="$3}'))"
}

log "================ RERUN START (COOL=${COOL}s, permute ON both) ================"

# ---------------- GPU: 3 sizes x 4 states + pristine recheck ----------------
for size in 0.8B-INT4 2B-INT4 4B-INT4; do
  log "---- GPU size $size ----"
  gpu_run "$size" pristine        pristine
  gpu_run "$size" opt3            opt3
  gpu_run "$size" opt3_opt5       opt3_opt5
  gpu_run "$size" opt3_opt5_opt6  opt3_opt5_opt6   # == "final" (opt7 genai is a no-op binary)
  gpu_run "$size" pristine_recheck pristine        # gradient control: compare to first pristine
done
restore_plugin   # back to installed opt3_opt5_opt6 before NPU (NPU ignores it, but keep clean)

# ---------------- NPU: 3 sizes x 4 configs (+ base recheck on 4B) -----------
for model in Qwen3-0.6B-INT4 Qwen3-1.7B-INT4 Qwen3-4B-INT4; do
  log "---- NPU model $model ----"
  npu_run "$model" base            npu_base.json
  npu_run "$model" turbo           npu_turbo.json
  npu_run "$model" best_perf       npu_best_perf.json
  npu_run "$model" best_perf_turbo npu_best_perf_turbo.json
done
npu_run Qwen3-4B-INT4 base_recheck npu_base.json   # gradient control for the hottest NPU model

log "================ RERUN DONE ================"
