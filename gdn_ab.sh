#!/bin/bash
# Temp-gated cold-start A/B for GDN experiments on 2B PA isolated prefill (ic=1).
# Usage: gdn_ab.sh <label> [ENVVAR=val ...]
#   e.g. gdn_ab.sh vblock8 OV_GDN_VBLOCK=8
# Cools SoC to <=TARGET before running, then runs n=5 ic=1 and prints
# first_token_infer / ttft / md5-seq. Appends a row to gdn_ab_results.tsv.
set -uo pipefail
cd /home/chaofan/model_benchmark
MB=./model_bench/build-local/model_bench
OUT=benchmark/tests/gdn_ab_runs; mkdir -p "$OUT"
TARGET=${TARGET:-50}; CAP=${CAP:-420}; PLATEAU=${PLATEAU:-120}; INT=15
ZONE=/sys/class/thermal/thermal_zone6/temp
pkg(){ awk '{printf "%d",$1/1000}' "$ZONE"; }
label=$1; shift
env_kv=("$@")

# cool to floor
s=$(pkg); w=0; chk=$(pkg)
while [ "$w" -lt "$CAP" ]; do c=$(pkg); [ "$c" -le "$TARGET" ] && break
  if [ "$w" -gt 0 ] && [ $((w % PLATEAU)) -eq 0 ]; then [ $((chk-c)) -lt 2 ] && break; chk=$c; fi
  sleep "$INT"; w=$((w+INT)); done
echo "[$(date +%H:%M:%S)] $label: cooled ${s}C->$(pkg)C in ${w}s; env=${env_kv[*]:-none}"

env "${env_kv[@]}" $MB vlm -m ./benchmark/models/Qwen3.5-2B-INT4 -d GPU -n 5 -ic 1 \
  -pf ./model_bench/prompts/qwen3.5_1k.jsonl --enable-prompt-permutation --no-monitor \
  -lc benchmark/tests/repro_rerun_20260710/gpu_pa.json \
  -rj "$OUT/$label.json" -r /dev/null > "$OUT/$label.log" 2>&1

python3 - "$OUT/$label.json" "$label" << 'PY'
import json,sys
d=json.load(open(sys.argv[1]))[0]; lbl=sys.argv[2]
its=[x for x in d["iterations"] if not x["is_warmup"]]
fti=d["first_token_infer_mean_ms"]; ftistd=d["first_token_infer_std_ms"]
ttft=d["ttft_mean_ms"]
md5=",".join(x["result_md5"][:8] for x in its[:3])
line=f"{lbl}\t{fti:.2f}\t{ftistd:.2f}\t{ttft:.2f}\t{md5}"
print(f"  first_token_infer={fti:.2f}±{ftistd:.2f}  ttft={ttft:.2f}  md5[:3]={md5}")
open("benchmark/tests/gdn_ab_results.tsv","a").write(line+"\n")
PY
