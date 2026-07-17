#!/bin/bash
# Hot-swap a prebuilt .so into install/, then run one model_bench measurement.
# Usage: swaprun.sh <so_file> <task> <modeldir> <backend> <ic> <label>
set -eo pipefail
SO="$1"; TASK="$2"; MODEL="$3"; BACKEND="$4"; IC="$5"; LABEL="$6"
ROOT=/home/chaofan/model_benchmark
INSTALL="$ROOT/benchmark/openvino/install/runtime/lib/intel64/libopenvino_intel_gpu_plugin.so"
cp -f "$SO" "$INSTALL"
source "$ROOT/benchmark/openvino/install/setupvars.sh" >/dev/null 2>&1
export LD_LIBRARY_PATH="$ROOT/benchmark/openvino/install/runtime/lib/intel64:${LD_LIBRARY_PATH:-}"
cd "$ROOT/model_bench"
MODELNAME="$(basename "$MODEL")"
if [ "$BACKEND" = "PA" ]; then LC=/tmp/pa.json; else LC=/tmp/sdpa.json; fi
if [ "$TASK" = "vlm" ]; then PF="$ROOT/benchmark/tests/prompts/qwen3.5_1k.jsonl"; else PF="$ROOT/benchmark/tests/prompts/qwen3_1k.jsonl"; fi
PERM=""; [ "$TASK" = "llm" ] && PERM="--enable-prompt-permutation"
OUT=/tmp/repro/swap_${LABEL}_${MODELNAME}_${BACKEND}.json
"$ROOT/model_bench/build-local/model_bench" "$TASK" -m "$MODEL" -d GPU -n 8 -ic "$IC" \
  -pf "$PF" --no-monitor $PERM -lc "$LC" -r /tmp/repro/_x.csv -rj "$OUT" \
  > /tmp/repro/swap_${LABEL}_${MODELNAME}_${BACKEND}.log 2>&1
python3 - "$OUT" "$LABEL" "$MODELNAME" "$BACKEND" "$IC" <<'PY'
import json,sys
from collections import Counter
out,label,model,backend,ic=sys.argv[1:6]
r=json.load(open(out))[0]
c=Counter(it['result_md5'] for it in r['iterations'] if not it.get('is_warmup'))
sig="|".join(f"{k[:8]}x{v}" for k,v in sorted(c.items()))
print(f"SWAP\t{label}\t{model}\t{backend}\tic={ic}\tinfer={r['first_token_infer_mean_ms']:.2f}\te2e={r['ttft_mean_ms']:.2f}\tmd5={sig}")
PY
