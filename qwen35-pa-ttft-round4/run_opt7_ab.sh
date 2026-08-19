#!/usr/bin/env bash
# Interleaved A/B for opt-7 (RMSNorm folded into DynamicQuantize) on the local iGPU.
#
# Interleaved rather than one-pass-per-stage: this machine drifts thermally by ~1 ms over a
# session, which is a sizeable fraction of the increment being measured. Every REP visits every
# stage, so the drift hits both stages roughly equally and the medians stay comparable.
#
# Stages are hot-swapped plugin binaries built from the same tree:
#   opt3456      = opt-3 + opt-4 + opt-5 + opt-6            (the round-3 stack, this round's base)
#   opt3456_opt7 = the same plus opt-7
set -euo pipefail

ROOT=/home/chaofan/model_benchmark
R4=$ROOT/benchmark/tests/qwen35-pa-ttft-round4
INSTALL_SO=$ROOT/benchmark/openvino/install/runtime/lib/intel64/libopenvino_intel_gpu_plugin.so
MB=$ROOT/model_bench/build-local/model_bench
PROMPT=$ROOT/model_bench/prompts/qwen3.5_1k.jsonl
MODEL=${MODEL:-$ROOT/benchmark/models/Qwen3.5-2B-INT4}
OUT=${OUT:-$R4/results/opt7_ab}
REPS=${REPS:-4}
N=${N:-5}
IC=${IC:-1}

mkdir -p "$OUT"
BACKUP=$(mktemp /tmp/gpu_plugin_backup.XXXXXX.so)
cp "$INSTALL_SO" "$BACKUP"
restore() { cp "$BACKUP" "$INSTALL_SO"; rm -f "$BACKUP"; }
trap restore EXIT

STAGES=(opt3456 opt3456_opt7)

CFG=$(mktemp /tmp/pa_cfg.XXXXXX.json)
printf '{"PERFORMANCE_HINT":"LATENCY","ATTENTION_BACKEND":"PA"}\n' > "$CFG"

echo "model=$MODEL  n=$N ic=$IC reps=$REPS"
for rep in $(seq 1 "$REPS"); do
    for st in "${STAGES[@]}"; do
        cp "$R4/results/so_pa_${st}.so" "$INSTALL_SO"
        # opt-6's extra z-gate fusion is opt-in; enable it so both stages carry the full round-3 stack.
        env OV_SWISH_HOIST_ALL=1 "$MB" vlm -m "$MODEL" -d GPU -n "$N" -ic "$IC" \
            -pf "$PROMPT" --no-monitor -lc "$CFG" \
            -r "$OUT/${st}_r${rep}.csv" -rj "$OUT/${st}_r${rep}.json" \
            > "$OUT/${st}_r${rep}.log" 2>&1
        infer=$(grep "1st infer" "$OUT/${st}_r${rep}.log" | awk '{print $3}')
        ttft=$(grep -E "^  TTFT" "$OUT/${st}_r${rep}.log" | awk '{print $2}')
        echo "rep$rep ${st}: infer=${infer} ttft=${ttft}"
    done
done
rm -f "$CFG"
echo "results in $OUT"
