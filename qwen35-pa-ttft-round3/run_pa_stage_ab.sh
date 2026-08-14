#!/usr/bin/env bash
# Interleaved PA stage A/B for Qwen3.5-2B on the local 32-EU iGPU.
#
# Why interleaved rather than one-pass-per-stage: this machine drifts thermally over a
# session (three back-to-back identical runs measured 274.4 / 275.4 / 276.6 ms), which is
# the same order as the per-stage increments being measured. Running every stage once in
# sequence therefore biases later stages. Here every REP visits every stage, so drift
# affects all stages roughly equally and the medians stay comparable.
#
# Stages are hot-swapped plugin binaries. pristine/opt3/opt3+4/opt3+4+5 are reused from
# benchmark/tests/repro_20263_opt_ab/results (built 2026-08-07 from the same 2026.3 tree);
# opt3+4+5+6 is built by this round.
set -euo pipefail

ROOT=/home/chaofan/model_benchmark
ROUND3=$ROOT/benchmark/tests/qwen35-pa-ttft-round3
PREV=$ROOT/benchmark/tests/repro_20263_opt_ab/results
INSTALL_SO=$ROOT/benchmark/openvino/install/runtime/lib/intel64/libopenvino_intel_gpu_plugin.so
MB=$ROOT/model_bench/build-local/model_bench
PROMPT=$ROOT/model_bench/prompts/qwen3.5_1k.jsonl
MODEL=${MODEL:-$ROOT/benchmark/models/Qwen3.5-2B-INT4}
OUT=$ROUND3/results/stage_ab
REPS=${REPS:-3}
N=${N:-5}
IC=${IC:-1}

mkdir -p "$OUT"
BACKUP=$(mktemp /tmp/gpu_plugin_backup.XXXXXX.so)
cp "$INSTALL_SO" "$BACKUP"
restore() { cp "$BACKUP" "$INSTALL_SO"; rm -f "$BACKUP"; }
trap restore EXIT

declare -A SO=(
    [pristine]="$PREV/libopenvino_intel_gpu_plugin_pa_pristine.so"
    [opt3]="$PREV/libopenvino_intel_gpu_plugin_pa_opt3.so"
    [opt34]="$PREV/libopenvino_intel_gpu_plugin_pa_opt3_opt4.so"
    [opt345]="$PREV/libopenvino_intel_gpu_plugin_pa_opt3_opt4_opt5.so"
    [opt3456]="$ROUND3/results/so_pa_opt3_opt4_opt5_opt6.so"
)
STAGES=(pristine opt3 opt34 opt345 opt3456)

CFG=$(mktemp /tmp/pa_cfg.XXXXXX.json)
printf '{"PERFORMANCE_HINT":"LATENCY","ATTENTION_BACKEND":"PA"}\n' > "$CFG"

echo "model=$MODEL  n=$N ic=$IC reps=$REPS"
for rep in $(seq 1 "$REPS"); do
    for st in "${STAGES[@]}"; do
        cp "${SO[$st]}" "$INSTALL_SO"
        # opt-6's extra z-gate fusion is opt-in; enable it so the stage measures the full patch.
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
