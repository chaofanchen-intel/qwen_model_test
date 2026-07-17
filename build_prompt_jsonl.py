#!/usr/bin/env python3
"""
Build a JSONL prompt file with a target input-token count.

Each output line is {"prompt": "..."} -- the schema both the official
openvino.genai/tools/llm_bench/benchmark.py (-pf flag) and our
benchmark_qwen3.py / benchmark_qwen3_5.py (--prompt-file) consume.

Usage:
    python tests/build_prompt_jsonl.py \
        --model models/Qwen3-1.7B-INT4 \
        --tokens 1024 \\
        --out tests/prompts/qwen3_1k.jsonl

The tokenizer used to count tokens is loaded via openvino_genai.Tokenizer
(the same class the benchmarks use), so the reported count exactly matches
what the benchmark sees -- no HuggingFace-vs-OpenVINO drift.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import openvino_genai as ov_genai
except ImportError:
    sys.exit("ERROR: openvino-genai not installed.")

# Allow `python build_prompt_jsonl.py` from any CWD.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from lib.prompts import build_prompt_to_target, load_base_text


def main():
    p = argparse.ArgumentParser(description="Build a JSONL prompt with a target token count.")
    p.add_argument("--model", required=True,
                   help="Path to OV model dir (used only to load the tokenizer).")
    p.add_argument("--tokens", type=int, default=1024, help="Target input token count.")
    p.add_argument("--out", required=True, help="Output JSONL path.")
    p.add_argument("--base-text", default=None,
                   help="Optional path to a .txt file used as source.")
    p.add_argument("--apply-chat-template", action="store_true",
                   help="Count tokens after tokenizer.apply_chat_template(...).")
    args = p.parse_args()

    base = load_base_text(args.base_text)

    print(f"Loading tokenizer from: {args.model}")
    tokenizer = ov_genai.Tokenizer(args.model)

    def count(text: str) -> int:
        if args.apply_chat_template:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": text}], add_generation_prompt=True
            )
        return int(tokenizer.encode(text).input_ids.shape[-1])

    print(f"Building prompt of ~{args.tokens} tokens...")
    prompt, actual = build_prompt_to_target(base, args.tokens, count)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"prompt": prompt}, f, ensure_ascii=False)
        f.write("\n")
    print(f"Wrote: {out_path}")
    print(f"  target tokens:  {args.tokens}")
    print(f"  actual tokens:  {actual}")
    print(f"  chat template:  {args.apply_chat_template}")
    print(f"  prompt chars:   {len(prompt)}")


if __name__ == "__main__":
    main()
