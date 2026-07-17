#!/usr/bin/env python
"""Offline-reshape a Qwen3-Reranker OpenVINO IR to a static [num_candidates, max_length] shape.

The NPU plugin rejects the dynamic-shape ([?,?]) reranker IR exported by optimum
(see tests/README.md "Qwen3-Embedding / Qwen3-Reranker on NPU"). This helper writes a
static-shape copy that the NPU compiler accepts. Pair it with the GenAI TextRerankPipeline
running with pad_to_max_length=True so every inference feeds exactly [num_candidates, max_length].

IMPORTANT: the reranker scores ALL candidate documents for a query in ONE forward pass, so the
batch dimension equals the number of candidate texts. A static model is therefore valid only for
that exact candidate count. Re-run this helper per (num_candidates, max_length) pair.

NPU compile time grows steeply with max_length (and with batch). Measured on Core Ultra X7 358H:
  [2,512]=132s   [1,2048]=462s   [2,2048]=did-not-finish-in-15min.
Prefer small max_length (256/512) on NPU.

Usage:
  python tests/reshape_reranker_static.py \
      --src models/Qwen3-Reranker-0.6B-fp16-ov \
      --num-candidates 15 --max-length 512 \
      --out models/Qwen3-Reranker-0.6B-fp16-ov-npu-b15-s512
"""
import argparse
import shutil
import sys
from pathlib import Path

import openvino as ov


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="source reranker model dir (with openvino_model.xml)")
    ap.add_argument("--num-candidates", type=int, required=True,
                    help="batch dim = number of candidate documents scored per query")
    ap.add_argument("--max-length", type=int, required=True, help="static sequence length (pad/truncate target)")
    ap.add_argument("--out", help="output dir; default <src>-npu-b<N>-s<L>")
    ap.add_argument("--force", action="store_true", help="overwrite output dir if it exists")
    args = ap.parse_args()

    src = Path(args.src)
    xml = src / "openvino_model.xml"
    if not xml.exists():
        print(f"[ERROR] {xml} not found", file=sys.stderr)
        return 1

    out = Path(args.out) if args.out else src.parent / f"{src.name}-npu-b{args.num_candidates}-s{args.max_length}"
    if out.exists():
        if not args.force:
            print(f"[ERROR] {out} exists; pass --force to overwrite", file=sys.stderr)
            return 1
        shutil.rmtree(out)

    # Copy the whole dir (tokenizer/detokenizer/config/...) then overwrite the model IR with the
    # static-reshaped version so the dir stays a valid GenAI TextRerankPipeline model directory.
    shutil.copytree(src, out)
    model = ov.Core().read_model(str(xml))
    shape = ov.PartialShape([args.num_candidates, args.max_length])
    reshape_map = {inp.get_any_name(): shape for inp in model.inputs}
    print(f"reshaping inputs to {shape}: {sorted(reshape_map)}")
    model.reshape(reshape_map)
    ov.save_model(model, str(out / "openvino_model.xml"))
    print(f"[OK] static reranker model written to {out}")
    print(f"     run with: DEVICE=NPU, RERANKING_MAX_LENGTH={args.max_length}, "
          f"and exactly {args.num_candidates} candidate texts, pad_to_max_length=True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
