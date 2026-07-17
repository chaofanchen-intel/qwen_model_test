#!/bin/bash
# Calibration: cool to TARGET, then run one 4B GPU n=8 and observe whether it
# stays flat. Determines the highest start-temp that yields clean cold-start data.
set -uo pipefail
cd /home/chaofan/model_benchmark
MB=./model_bench/build-local/model_bench
NEW=benchmark/tests/repro_rerun_20260710
TARGET=${TARGET:-60}
PROG=$NEW/logs/calib.log
zone=/sys/class/thermal/thermal_zone6/temp
pkg(){ awk '{printf "%d",$1/1000}' $zone; }
: > "$PROG"
echo "[$(date +%H:%M:%S)] calibration start, cooling to <=${TARGET}C (now $(pkg)C)" | tee -a "$PROG"
waited=0
while [ "$(pkg)" -gt "$TARGET" ] && [ "$waited" -lt 600 ]; do sleep 10; waited=$((waited+10)); done
echo "[$(date +%H:%M:%S)] reached $(pkg)C after ${waited}s; running 4B (installed plugin=opt3_opt5_opt6/final)" | tee -a "$PROG"
$MB vlm -m ./benchmark/models/Qwen3.5-4B-INT4 -d GPU -n 8 -ic 512 \
  -pf ./model_bench/prompts/qwen3.5_1k.jsonl --enable-prompt-permutation --no-monitor \
  -lc "$NEW/gpu_pa.json" -rj "$NEW/logs/calib_4B_final_at${TARGET}C.json" -r /dev/null \
  > "$NEW/logs/calib_4B_run.log" 2>&1
echo "[$(date +%H:%M:%S)] done, end pkg=$(pkg)C" | tee -a "$PROG"
# per-iter summary
python3 - "$NEW/logs/calib_4B_final_at${TARGET}C.json" << 'PY' | tee -a "$PROG"
import json,sys
d=json.load(open(sys.argv[1]))[0]
its=[x for x in d["iterations"] if not x["is_warmup"]]
tt=[x["ttft_ms"] for x in its]
print(f"TTFT mean={d['ttft_mean_ms']:.2f} std={d['ttft_std_ms']:.2f} min={min(tt):.0f} max={max(tt):.0f} first->last={tt[0]:.0f}->{tt[-1]:.0f}")
print("per-iter TTFT:", " ".join(f"{x:.0f}" for x in tt))
print(f"TPOT={d['tpot_mean_ms']:.2f} TPS={d['throughput_mean_tps']:.2f} E2E={d['e2e_mean_ms']:.0f}")
PY
