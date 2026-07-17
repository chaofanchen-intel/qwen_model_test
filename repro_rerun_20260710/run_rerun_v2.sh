#!/bin/bash
# Clean re-test driver v2 — TEMPERATURE-GATED cooldown.
#
# v1 used a fixed 45s cooldown; that was far too short (SoC cools only ~2-4 C/min)
# so the package temperature ramped cumulatively across the session and re-created
# the exact thermal bias we are trying to remove. v2 waits until x86_pkg_temp
# returns to TARGET before every measured run, so each state starts from the same
# cold-ish baseline. Calibration: 4B started @60C -> mean 577 (gold cold-start
# ~565); @81C -> 824 (throttled). TARGET=60C balances fidelity vs cooldown time.
#
# n=8 (+1 warm-up), permutation ON for BOTH GPU and NPU (deterministic, iter-indexed).
# GPU states swap ONLY the intel_gpu plugin .so (genai is a no-op: opt7==pristine).
# Installed plugin restored on exit.
set -uo pipefail
ROOT=/home/chaofan/model_benchmark; cd "$ROOT"
MB=./model_bench/build-local/model_bench
NEW=benchmark/tests/repro_rerun_20260710
BIN=benchmark/tests/repro_leader_report_20260710/ic512/binaries
INSTALL=benchmark/openvino/install/runtime/lib/intel64/libopenvino_intel_gpu_plugin.so
GPU_PROMPT=./model_bench/prompts/qwen3.5_1k.jsonl
NPU_PROMPT=./benchmark/tests/prompts/qwen3_1k.jsonl
TARGET=${TARGET:-45}      # cool to this pkg temp (C) before each run
CAP=${CAP:-420}           # max cooldown wait (s)
PROG=$NEW/logs/progress.log
ZONE=/sys/class/thermal/thermal_zone6/temp   # x86_pkg_temp

pkg(){ awk '{printf "%d",$1/1000}' "$ZONE"; }
now(){ date +%H:%M:%S; }
log(){ echo "[$(now)] $*" | tee -a "$PROG"; }
restore_plugin(){ cp -f "$NEW/backup/libopenvino_intel_gpu_plugin.INSTALLED.so" "$INSTALL"; log "RESTORED plugin -> $(md5sum "$INSTALL"|cut -c1-12) (want de95bee0c6c6)"; }
trap restore_plugin EXIT

cool_to(){ local s=$(pkg) w=0; while [ "$(pkg)" -gt "$TARGET" ] && [ "$w" -lt "$CAP" ]; do sleep 10; w=$((w+10)); done; log "  cooldown ${s}C -> $(pkg)C in ${w}s (target ${TARGET}C)"; }
swap_gpu(){ cp -f "$BIN/libopenvino_intel_gpu_plugin_$1.so" "$INSTALL"; log "  plugin -> $1 ($(md5sum "$INSTALL"|cut -c1-12))"; }

gpu_run(){ local size=$1 label=$2 state=$3; local out=$NEW/gpu_pa/$size/$label
  swap_gpu "$state"; cool_to
  log "GPU size=$size label=$label start_pkg=$(pkg)C"
  $MB vlm -m ./benchmark/models/Qwen3.5-$size -d GPU -n 8 -ic 512 \
     -pf "$GPU_PROMPT" --enable-prompt-permutation --no-monitor \
     -lc "$NEW/gpu_pa.json" -rj "$out.json" -r "$out.csv" > "$NEW/logs/gpu_${size}_${label}.log" 2>&1
  log "  done end_pkg=$(pkg)C  $(grep -m1 '  TTFT ' "$NEW/logs/gpu_${size}_${label}.log" | awk '{print "TTFT="$2" std="$3" p50="$5}')"
}
npu_run(){ local model=$1 label=$2 cfg=$3; local out=$NEW/npu/${model}_${label}
  cool_to
  log "NPU model=$model config=$label start_pkg=$(pkg)C"
  $MB llm -m ./benchmark/models/$model -d NPU -n 8 -ic 512 \
     -pf "$NPU_PROMPT" --enable-prompt-permutation --no-monitor \
     -lc "$NEW/$cfg" -rj "$out.json" -r "$out.csv" > "$NEW/logs/npu_${model}_${label}.log" 2>&1
  log "  done end_pkg=$(pkg)C  $(grep -m1 '  TTFT ' "$NEW/logs/npu_${model}_${label}.log" | awk '{print "TTFT="$2" std="$3" p50="$5}')"
}

log "============ RERUN v2 START (TARGET=${TARGET}C CAP=${CAP}s, permute ON) ============"
for size in 0.8B-INT4 2B-INT4 4B-INT4; do
  log "---- GPU $size ----"
  gpu_run "$size" pristine         pristine
  gpu_run "$size" opt3             opt3
  gpu_run "$size" opt3_opt5        opt3_opt5
  gpu_run "$size" opt3_opt5_opt6   opt3_opt5_opt6     # == final (opt7 genai no-op)
  gpu_run "$size" pristine_recheck pristine           # gradient control
done
restore_plugin
for model in Qwen3-0.6B-INT4 Qwen3-1.7B-INT4 Qwen3-4B-INT4; do
  log "---- NPU $model ----"
  npu_run "$model" base            npu_base.json
  npu_run "$model" turbo           npu_turbo.json
  npu_run "$model" best_perf       npu_best_perf.json
  npu_run "$model" best_perf_turbo npu_best_perf_turbo.json
done
npu_run Qwen3-4B-INT4 base_recheck npu_base.json        # gradient control (hottest NPU)
log "============ RERUN v2 DONE ============"
