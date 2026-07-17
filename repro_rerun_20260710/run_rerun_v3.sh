#!/bin/bash
# Clean re-test driver v3 — cools to the COLDEST ACHIEVABLE baseline before each run.
#
# History: v1 (fixed 45s cooldown) let package temp ramp cumulatively -> re-created
# the thermal bias. v2 (gate 60C) was better but 60C still leaves a ~+12ms/4B offset
# vs a true cold start. v3 cools to the real idle floor via plateau detection:
# wait until pkg<=TARGET OR no new temperature low for PLATEAU seconds (=floor
# reached), with CAP as a hard backstop. This reproduces the gold cold-start numbers
# (user's fresh 4B=565ms) for EVERY state, so both absolute values and inter-state
# deltas are trustworthy.
#
# n=5 (+1 warm-up); permutation ON for BOTH GPU and NPU (deterministic, iter-indexed,
# so md5 at iteration k is comparable across states/configs). GPU states swap ONLY the
# intel_gpu plugin .so (genai is a no-op: opt7-genai == pristine-genai, byte-identical).
# The installed plugin is restored on exit.  Designed to run unattended (overnight).
set -uo pipefail
ROOT=/home/chaofan/model_benchmark; cd "$ROOT"
MB=./model_bench/build-local/model_bench
NEW=benchmark/tests/repro_rerun_20260710
BIN=benchmark/tests/repro_leader_report_20260710/ic512/binaries
INSTALL=benchmark/openvino/install/runtime/lib/intel64/libopenvino_intel_gpu_plugin.so
GPU_PROMPT=./model_bench/prompts/qwen3.5_1k.jsonl
NPU_PROMPT=./benchmark/tests/prompts/qwen3_1k.jsonl
TARGET=${TARGET:-50}      # ideal target (C); real floor is ~48-50C so stall-detect governs
CAP=${CAP:-1200}          # hard max cooldown wait per run (s)
PLATEAU=${PLATEAU:-120}   # stall window (s): if temp dropped <2C over this window -> floor reached
INT=15                    # temp sample interval (s)
PROG=$NEW/logs/progress.log
ZONE=/sys/class/thermal/thermal_zone6/temp   # x86_pkg_temp (SoC package incl. iGPU/NPU)

pkg(){ awk '{printf "%d",$1/1000}' "$ZONE"; }
now(){ date '+%m-%d %H:%M:%S'; }
log(){ echo "[$(now)] $*" | tee -a "$PROG"; }
restore_plugin(){ cp -f "$NEW/backup/libopenvino_intel_gpu_plugin.INSTALLED.so" "$INSTALL"; log "RESTORED plugin -> $(md5sum "$INSTALL"|cut -c1-12) (want de95bee0c6c6)"; }
trap restore_plugin EXIT

cool_to(){ local s=$(pkg) w=0 chk=$(pkg) reason=cap
  while [ "$w" -lt "$CAP" ]; do
    local c=$(pkg)
    if [ "$c" -le "$TARGET" ]; then reason=target; break; fi
    if [ "$w" -gt 0 ] && [ $((w % PLATEAU)) -eq 0 ]; then
      if [ $((chk - c)) -lt 2 ]; then reason=floor; break; fi   # <2C drop over last window
      chk=$c
    fi
    sleep "$INT"; w=$((w+INT))
  done
  log "  cooldown ${s}C -> $(pkg)C in ${w}s (stop=${reason}, target ${TARGET}C)"
}
swap_gpu(){ cp -f "$BIN/libopenvino_intel_gpu_plugin_$1.so" "$INSTALL"; log "  plugin -> $1 ($(md5sum "$INSTALL"|cut -c1-12))"; }

gpu_run(){ local size=$1 label=$2 state=$3; local out=$NEW/gpu_pa/$size/$label
  swap_gpu "$state"; cool_to
  log "GPU size=$size label=$label start_pkg=$(pkg)C"
  if $MB vlm -m ./benchmark/models/Qwen3.5-$size -d GPU -n 5 -ic 512 \
       -pf "$GPU_PROMPT" --enable-prompt-permutation --no-monitor \
       -lc "$NEW/gpu_pa.json" -rj "$out.json" -r "$out.csv" > "$NEW/logs/gpu_${size}_${label}.log" 2>&1; then
    log "  done end_pkg=$(pkg)C  $(grep -m1 '  TTFT ' "$NEW/logs/gpu_${size}_${label}.log" | awk '{print "TTFT="$2" std="$3" p50="$5}')"
  else
    log "  !! FAILED GPU $size $label (see logs/gpu_${size}_${label}.log)"
  fi
}
npu_run(){ local model=$1 label=$2 cfg=$3; local out=$NEW/npu/${model}_${label}
  cool_to
  log "NPU model=$model config=$label start_pkg=$(pkg)C"
  if $MB llm -m ./benchmark/models/$model -d NPU -n 5 -ic 512 \
       -pf "$NPU_PROMPT" --enable-prompt-permutation --no-monitor \
       -lc "$NEW/$cfg" -rj "$out.json" -r "$out.csv" > "$NEW/logs/npu_${model}_${label}.log" 2>&1; then
    log "  done end_pkg=$(pkg)C  $(grep -m1 '  TTFT ' "$NEW/logs/npu_${model}_${label}.log" | awk '{print "TTFT="$2" std="$3" p50="$5}')"
  else
    log "  !! FAILED NPU $model $label (see logs/npu_${model}_${label}.log)"
  fi
}

log "========= RERUN v3 START (TARGET=${TARGET}C PLATEAU=${PLATEAU}s CAP=${CAP}s, permute ON both) ========="
log "start pkg=$(pkg)C"
for size in 0.8B-INT4 2B-INT4 4B-INT4; do
  log "---- GPU $size ----"
  gpu_run "$size" pristine         pristine
  gpu_run "$size" opt3             opt3
  gpu_run "$size" opt3_opt5        opt3_opt5
  gpu_run "$size" opt3_opt5_opt6   opt3_opt5_opt6     # == final (opt7 genai is a byte-identical no-op)
  gpu_run "$size" pristine_recheck pristine           # gradient control: compare to first pristine
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
log "========= RERUN v3 DONE ========="
# final summary for easy review tomorrow
python3 "$NEW/monitor.py" > "$NEW/logs/FINAL_SUMMARY.txt" 2>&1 || true
log "wrote logs/FINAL_SUMMARY.txt"
