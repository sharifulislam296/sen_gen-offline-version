from __future__ import annotations
import argparse, os, sys
from typing import Any, Dict, List, Sequence

import src.bootstrap  # ensures oxford_a2.txt exists first
from .filters import (
    filter_sentences,
    dedupe,
    normalize_word_list,
    ALLOWED_WORDS,
)
from .templates import fallback_sentence, fallback_sentences

# Optional OpenAI import (graceful degradation)
try:
    import openai  # type: ignore
except ImportError:  # pragma: no cover
    openai = None  # type: ignore

# ---------------- Defaults & presets -----------------
DEFAULT_MODE    = "per-word"
DEFAULT_LEVEL   = "A2"
DEFAULT_MAX_LEN = 15
LEVELS = {
    "A1": {"max_len": 8,  "max_unknown": 1},
    "A2": {"max_len": 12, "max_unknown": 2},
    "B1": {"max_len": 20, "max_unknown": 4},
}

# ---------------- OpenAI helper ----------------------
def _openai_client() -> Any | None:
    """
    Returns an OpenAI client.
    Handles both secret keys (sk-...) and project keys (sk-proj-...).
    For project keys we also need OPENAI_PROJECT env var.
    """
    if openai is None:
        return None

    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        return None

    project_id = os.getenv("OPENAI_PROJECT")
    if key.startswith("sk-proj-") and not project_id:
        print("[error] Project key detected but OPENAI_PROJECT var is missing.", file=sys.stderr)
        return None

    try:
        return openai.OpenAI(api_key=key, project=project_id)  # type: ignore[attr-defined]
    except Exception as e:
        print(f"[warn] OpenAI init error: {e}", file=sys.stderr)
        return None

# ---------------- Prompt helpers ---------------------
def _prompt_per_word(word: str, n: int, max_len: int, level: str) -> str:
    return (
        "You are a patient English tutor.\n"
        f"Write {n} DIFFERENT sentences in English.\n"
        f"Each MUST include the word: {word}\n"
        f"Keep each sentence <= {max_len} words (CEFR {level} or easier).\n"
        "No numbering or quotes; one sentence per line."
    )

def _prompt_mixed(plan: List[List[str]], max_len: int, level: str) -> str:
    lines = [f"{i+1}) {', '.join(req)}" for i, req in enumerate(plan)]
    return (
        "You are a patient English tutor.\n"
        f"Write {len(plan)} short English sentences.\n"
        "Each sentence MUST include the required word(s) shown.\n"
        f"Keep each sentence <= {max_len} words (CEFR {level} or easier).\n"
        "Return exactly the number of sentences requested.\n"
        "No numbering, bullets, or quotes.\n\n"
        "Required word(s) by line:\n" + "\n".join(lines)
    )

# ---------------- LLM wrapper ------------------------
def call_llm(prompt: str, max_tokens: int = 400) -> str:
    client = _openai_client()
    if client is None:
        return ""
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[warn] LLM call failed: {e}", file=sys.stderr)
        return ""

# ---------------- Helpers ----------------------------
def _strip_leading_num(s: str) -> str:
    import re
    return re.sub(r"^\d+[.)]\s*", "", s.lstrip("-*•").strip())

def _split_lines(raw: str) -> List[str]:
    if not raw:
        return []
    out = []
    for l in raw.splitlines():
        l = _strip_leading_num(l)
        if l and l[-1].isalnum():
            l += "."
        if l:
            out.append(l)
    return out

def _plan_mixed(targets: Sequence[str], n: int) -> List[List[str]]:
    plan = [[] for _ in range(n)]
    for i, w in enumerate([t for t in targets if t]):
        plan[i % n].append(w)
    return plan

# ---------------- Per-word core ----------------------
def _generate_single(
    word: str, n: int, max_len: int, max_unknown: int, level: str
) -> List[str]:
    prompt = _prompt_per_word(word, n * 3, max_len, level)
    cand = _split_lines(call_llm(prompt))
    kept = dedupe(
        filter_sentences(cand, [word], ALLOWED_WORDS, max_len, max_unknown)
    )
    if len(kept) < n:
        kept.extend(fallback_sentences([word], n - len(kept), max_len=max_len))
    return kept[:n]

# ---------------- Mixed core -------------------------
def _generate_mixed(
    targets: Sequence[str], n: int, max_len: int, max_unknown: int, level: str
) -> List[str]:
    plan = _plan_mixed(targets, n)
    prompt = _prompt_mixed(plan, max_len, level)
    cand = _split_lines(call_llm(prompt))

    out: List[str] = []
    for i in range(n):
        line = cand[i] if i < len(cand) else ""
        req = plan[i]
        keep = filter_sentences([line], req, ALLOWED_WORDS, max_len, max_unknown)
        out.append(keep[0] if keep else fallback_sentence(req, max_len=max_len))
    return out

# ---------------- Public generate() ------------------
def generate(
    words: Sequence[str],
    n: int = 5,
    mode: str = DEFAULT_MODE,
    level: str = DEFAULT_LEVEL,
    max_len: int | None = None,
    max_unknown: int | None = None,
) -> List[str] | Dict[str, List[str]]:
    targets = normalize_word_list(words)
    level_u = level.upper()
    params = LEVELS[level_u]

    max_len = max_len or (DEFAULT_MAX_LEN if level_u == "A2" else params["max_len"])
    max_unknown = max_unknown or params["max_unknown"]

    if mode == "per-word":
        return {w: _generate_single(w, n, max_len, max_unknown, level_u) for w in targets}
    else:
        return _generate_mixed(targets, n, max_len, max_unknown, level_u)

# ---------------- CLI entry --------------------------
def _parse_args(argv: Sequence[str] | None = None):
    p = argparse.ArgumentParser(description="Beginner Sentence Generator (API-only)")
    p.add_argument("--words", nargs="+", required=True)
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--mode", choices=["per-word", "mixed"], default=DEFAULT_MODE)
    p.add_argument("--level", choices=["A1", "A2", "B1"], default=DEFAULT_LEVEL)
    p.add_argument("--max-len", type=int, default=None)
    p.add_argument("--max-unknown", type=int, default=None)
    return p.parse_args(argv)

def main(argv: Sequence[str] | None = None):
    args = _parse_args(argv)
    res = generate(
        args.words, args.n, args.mode, args.level, args.max_len, args.max_unknown
    )
    if isinstance(res, dict):
        for w, sents in res.items():
            print(f"# {w}")
            for s in sents:
                print(s)
            print()
    else:
        for s in res:
            print(s)

if __name__ == "__main__":
    main()
