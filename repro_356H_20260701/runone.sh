#!/bin/bash
# Run one model_bench measurement and print a single result line.
# Usage: runone.sh <state> <task:vlm|llm> <modeldir> <backend:PA|SDPA> <ic> <label>
set -eo pipefail
# Capture args FIRST — sourcing setupvars.sh clobbers positional params.
STATE="$1"; TASK="$2"; MODEL="$3"; BACKEND="$4"; IC="$5"; LABEL="$6"

ROOT=/home/chaofan/model_benchmark
source "$ROOT/benchmark/openvino/install/setupvars.sh" >/dev/null 2>&1
export LD_LIBRARY_PATH="$ROOT/benchmark/openvino/install/runtime/lib/intel64:${LD_LIBRARY_PATH:-}"
cd "$ROOT/model_bench"

MODELNAME="$(basename "$MODEL")"
if [ "$BACKEND" = "PA" ]; then LC=/tmp/pa.json; else LC=/tmp/sdpa.json; fi

# prompt file: qwen3.5 for vlm, qwen3 for llm
if [ "$TASK" = "vlm" ]; then PF="$ROOT/benchmark/tests/prompts/qwen3.5_1k.jsonl";
else PF="$ROOT/benchmark/tests/prompts/qwen3_1k.jsonl"; fi

OUT=/tmp/repro/out_${STATE}_${TASK}_${MODELNAME}_${BACKEND}_ic${IC}.json
# dense LLM path uses prompt permutation ON per SDPA doc §7.2; VLM stays OFF (default).
PERM=""
if [ "$TASK" = "llm" ]; then PERM="--enable-prompt-permutation"; fi

"$ROOT/model_bench/build-local/model_bench" "$TASK" -m "$MODEL" -d GPU -n 8 -ic "$IC" \
  -pf "$PF" --no-monitor $PERM -lc "$LC" \
  -r /tmp/repro/_x.csv -rj "$OUT" > /tmp/repro/run_${STATE}_${TASK}_${MODELNAME}_${BACKEND}_ic${IC}.log 2>&1

python3 - "$OUT" "$LABEL" "$MODELNAME" "$BACKEND" "$IC" <<'PY'
import json,sys
from collections import Counter
out,label,model,backend,ic=sys.argv[1:6]
r=json.load(open(out))[0]
md5s=[it['result_md5'] for it in r['iterations'] if not it.get('is_warmup')]
c=Counter(md5s)
sig="|".join(f"{k[:8]}x{v}" for k,v in sorted(c.items()))
print(f"RESULT\t{label}\t{model}\t{backend}\tic={ic}\tinfer={r['first_token_infer_mean_ms']:.2f}\te2e={r['ttft_mean_ms']:.2f}\tmd5={sig}")
PY
