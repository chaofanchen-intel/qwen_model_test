"""Prompt loading from JSONL + token-targeted prompt construction.

The JSONL schema is the same one the official openvino.genai/llm_bench
benchmark.py consumes: each line is a JSON object with at least a "prompt"
string field.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional


_PROJECT_DIR = Path(__file__).resolve().parents[2]
_REF_PROMPT_FILE = (
    _PROJECT_DIR
    / "openvino.genai/tools/llm_bench/prompts/llama-2-7b-chat_l.jsonl"
)

_FALLBACK_TEXT = (
    "Machine learning is a branch of artificial intelligence that uses statistical "
    "techniques to give computer systems the ability to learn from data, without being "
    "explicitly programmed. The discipline encompasses several paradigms: in supervised "
    "learning a model is trained on labeled examples and learns a mapping from input "
    "features to output targets; in unsupervised learning the algorithm discovers "
    "structure or patterns in unlabeled data, often via clustering or dimensionality "
    "reduction; in reinforcement learning an agent learns to act in an environment by "
    "receiving rewards or penalties for its choices and gradually optimizes its policy. "
    "Modern neural networks are layered systems of differentiable transformations whose "
    "parameters are tuned by backpropagation. Convolutional networks excel at spatial "
    "data such as images, recurrent and attention-based networks model sequences such as "
    "text, and transformer architectures form the backbone of contemporary large "
    "language models. Training a useful model requires a careful balance of "
    "regularization, optimization, data quality, and evaluation: overfit a model and it "
    "memorizes the training set instead of generalizing; underfit it and the function it "
    "represents is too simple to capture the underlying signal. Practitioners therefore "
    "use techniques such as dropout, weight decay, early stopping, and cross validation "
    "to find the right capacity. Beyond core training, the lifecycle of a machine "
    "learning system includes data ingestion, feature engineering, deployment, monitoring, "
    "and continuous retraining as data distributions shift over time. "
)


def load_prompts(prompt: Optional[str],
                 prompt_files: Optional[list[str]],
                 prompt_indices: Optional[list[int]]) -> list[tuple[int, str]]:
    """Return a list of (original_index, prompt_text).

    - --prompt and --prompt-file are mutually exclusive.
    - If neither is provided, raises (caller should error).
    - --prompt-index is applied AFTER concatenation of all -pf files;
      duplicates are silently dropped, order preserved.
    """
    if prompt is not None and prompt_files:
        sys.exit("ERROR: --prompt and --prompt-file are mutually exclusive")
    if prompt is not None:
        return [(0, prompt)]
    if not prompt_files:
        sys.exit("ERROR: provide --prompt or --prompt-file")

    all_prompts: list[str] = []
    for pf in prompt_files:
        path = Path(pf)
        if not path.exists():
            sys.exit(f"ERROR: prompt file not found: {pf}")
        with open(path, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    sys.exit(f"ERROR: {pf}:{lineno}: invalid JSON ({e})")
                if "prompt" not in obj:
                    sys.exit(f"ERROR: {pf}:{lineno}: missing 'prompt' key")
                all_prompts.append(str(obj["prompt"]))

    if prompt_indices:
        seen: set[int] = set()
        out: list[tuple[int, str]] = []
        for i in prompt_indices:
            if i in seen:
                continue
            seen.add(i)
            if not (0 <= i < len(all_prompts)):
                sys.exit(f"ERROR: prompt_index {i} out of range "
                         f"[0, {len(all_prompts)})")
            out.append((i, all_prompts[i]))
        return out
    return list(enumerate(all_prompts))


def load_texts_jsonl(files: list[str], key_candidates=("text", "prompt")) -> list[str]:
    """Load a flat list of strings from one or more JSONL files.

    Accepts either {"text": ...} or {"prompt": ...} lines (first matching key
    wins). Used for reranker candidate-document files, mirroring the official
    llm_bench --texts_file which expects a {"text": ...} schema.
    """
    out: list[str] = []
    for pf in files:
        path = Path(pf)
        if not path.exists():
            sys.exit(f"ERROR: texts file not found: {pf}")
        with open(path, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    sys.exit(f"ERROR: {pf}:{lineno}: invalid JSON ({e})")
                for k in key_candidates:
                    if k in obj:
                        out.append(str(obj[k]))
                        break
                else:
                    sys.exit(f"ERROR: {pf}:{lineno}: missing any of keys {key_candidates}")
    return out


def load_base_text(base_text_path: Optional[str] = None) -> str:
    """Load source text used for token-targeted prompt generation."""
    if base_text_path:
        return Path(base_text_path).read_text(encoding="utf-8")
    if _REF_PROMPT_FILE.exists():
        try:
            with open(_REF_PROMPT_FILE, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
            obj = json.loads(first_line)
            if "prompt" in obj and isinstance(obj["prompt"], str):
                return obj["prompt"]
        except (OSError, json.JSONDecodeError):
            pass
    return _FALLBACK_TEXT


# =============================================================================
# Token-targeted prompt construction
# =============================================================================

def build_prompt_to_target(base_text: str, target_tokens: int,
                           count_fn) -> tuple[str, int]:
    """Binary-search over character split to land near target_tokens.

    `count_fn(text) -> int` is the tokenizer-specific length probe. We never
    re-tokenize a re-joined word list (that's the O(n^2) trap); we always
    slice the original character buffer.
    """
    base_len = count_fn(base_text)
    if base_len == 0:
        return base_text, 0

    if base_len >= target_tokens:
        lo, hi = 1, len(base_text)
        best = base_text
        while lo <= hi:
            mid = (lo + hi) // 2
            cand = base_text[:mid]
            if count_fn(cand) <= target_tokens:
                best = cand
                lo = mid + 1
            else:
                hi = mid - 1
        return best, count_fn(best)

    repeat = (target_tokens // base_len) + 2
    long_text = (base_text + " ") * repeat
    lo, hi = 1, len(long_text)
    best = long_text
    while lo <= hi:
        mid = (lo + hi) // 2
        cand = long_text[:mid]
        if count_fn(cand) <= target_tokens:
            best = cand
            lo = mid + 1
        else:
            hi = mid - 1
    return best, count_fn(best)


def build_prompt_from_token_target(target_tokens: int, count_fn,
                                   base_text_path: Optional[str] = None) -> tuple[str, int]:
    """Build a raw prompt whose caller-defined token count is near target_tokens.

    `count_fn(raw_prompt)` lets callers count exactly what they will benchmark,
    for example after applying a chat template.
    """
    if target_tokens < 1:
        sys.exit("ERROR: --input-tokens must be >= 1")
    base_text = load_base_text(base_text_path)
    return build_prompt_to_target(base_text, target_tokens, count_fn)


# =============================================================================
# Anti-prefix-cache permutation for VLM (text-side)
# =============================================================================

def vlm_permute_prompt(prompt: str, iter_idx: int) -> str:
    """Return a per-iteration variant of `prompt` that defeats KV-prefix cache.

    The official llm_bench achieves this for LLMs by mutating
    input_ids[:, 0]. VLMPipeline's generate accepts only `str`, not
    TokenizedInputs, so we emulate by prepending a per-iteration sentinel
    token-equivalent ASCII char. The string is ASCII-printable but
    semantically negligible (numbers + space). This guarantees the first
    real token differs across iterations so the prefix cache misses.
    """
    return f"{iter_idx} {prompt}"
