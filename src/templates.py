"""
Template fallback utilities for the Beginner Sentence Generator project.

If the language model output is missing, unusable, or fails a filter
(checks in filters.py), we call these routines to generate *simple,
guaranteed-valid* sentences that include the required target word(s).

Public entry points:
    fallback_sentence(required_words, max_len=15)
    fallback_sentences(words, n, max_len=15)

`required_words` is a list/tuple of 1 or more target words that *must*
appear in the returned sentence. When more than one is given, a 'pair'
template (or generic join) is used.
"""

from __future__ import annotations
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence

from .filters import guess_pos, normalize_word

# ------------------------------------------------------------------
# Locate data directory (../data relative to this file)
# ------------------------------------------------------------------
_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_TPL_PATH = _DATA_DIR / "templates.json"

# In-memory cache
_TPL_CACHE: Dict[str, List[str]] | None = None


# ------------------------------------------------------------------
# Load template JSON (cached)
# ------------------------------------------------------------------
def _load_templates() -> Dict[str, List[str]]:
    global _TPL_CACHE
    if _TPL_CACHE is None:
        with _TPL_PATH.open(encoding="utf8") as f:
            data = json.load(f)
        # Drop meta if present
        _TPL_CACHE = {k: v for k, v in data.items() if isinstance(v, list)}
    return _TPL_CACHE


# ------------------------------------------------------------------
# Utility: pick correct English article
# ------------------------------------------------------------------
def _fix_article(text: str) -> str:
    """
    Replace occurrences of 'a {word}' with 'an {word}' when the next
    alphabetic character is a vowel sound heuristic (a,e,i,o,u,8,h-on-silence).
    Crude but good enough for beginner material.
    """
    tokens = text.split()
    out = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.lower() == "a" and i + 1 < len(tokens):
            nxt = tokens[i + 1]
            first = nxt[0].lower()
            if first in "aeiou":
                tok = "an"
        out.append(tok)
        i += 1
    return " ".join(out)


# ------------------------------------------------------------------
# Utility: truncate to max_len tokens (last-resort safety)
# ------------------------------------------------------------------
def _enforce_max_len(text: str, max_len: int) -> str:
    tokens = text.split()
    if len(tokens) <= max_len:
        return text
    return " ".join(tokens[:max_len]).rstrip(".,;:") + "."


# ------------------------------------------------------------------
# Single-word template selection & fill
# ------------------------------------------------------------------
def _fill_single(word: str, max_len: int = 15) -> str:
    tpl_data = _load_templates()
    pos = guess_pos(word)
    choices = tpl_data.get(pos, None)
    if not choices:
        choices = tpl_data.get("generic", ["This is {word}."])
    tpl = random.choice(choices)
    s = tpl.format(
        word=word,
        word_cap=word[:1].upper() + word[1:],
    )
    s = _fix_article(s)
    s = _enforce_max_len(s, max_len)
    return s


# ------------------------------------------------------------------
# Pair (two required words) template selection & fill
# ------------------------------------------------------------------
def _fill_pair(w1: str, w2: str, max_len: int = 15) -> str:
    tpl_data = _load_templates()
    choices = tpl_data.get("pair") or [
        "I see {w1} and {w2}.",
        "{w1_cap} with {w2_cap}.",
        "We use {w1} and {w2}.",
    ]
    tpl = random.choice(choices)
    s = tpl.format(
        w1=w1,
        w2=w2,
        w1_cap=w1[:1].upper() + w1[1:],
        w2_cap=w2[:1].upper() + w2[1:],
    )
    s = _fix_article(s)
    s = _enforce_max_len(s, max_len)
    return s


# ------------------------------------------------------------------
# Multi-word (3+) fallback: simple comma join
# ------------------------------------------------------------------
def _fill_multi(words: Sequence[str], max_len: int = 15) -> str:
    # Try definitional template if available
    tpl_data = _load_templates()
    defs = tpl_data.get("definitional")
    if defs:
        # Use first word as term; treat rest as context
        w = words[0]
        tpl = random.choice(defs)
        s = tpl.format(word=w, word_cap=w[:1].upper() + w[1:])
    else:
        s = ", ".join(words)
        s = f"{s}."
    s = _fix_article(s)
    s = _enforce_max_len(s, max_len)
    return s


# ------------------------------------------------------------------
# Public: build one fallback sentence for required_words (1+)
# ------------------------------------------------------------------
def fallback_sentence(required_words: Sequence[str], max_len: int = 15) -> str:
    """
    Return one guaranteed sentence that includes ALL required words
    (or at least does not drop them—shorter than max_len).

    Rules:
    - 1 word  -> pick template by guessed POS
    - 2 words -> pair template
    - 3+ words-> definitional or comma-join fallback
    """
    req = [normalize_word(w) for w in required_words if w]
    req = [w for w in req if w]  # drop empties
    if not req:
        return "I do not have a word."
    if len(req) == 1:
        return _fill_single(req[0], max_len=max_len)
    if len(req) == 2:
        return _fill_pair(req[0], req[1], max_len=max_len)
    return _fill_multi(req, max_len=max_len)


# ------------------------------------------------------------------
# Public: build multiple fallback sentences cycling through words
# ------------------------------------------------------------------
def fallback_sentences(words: Sequence[str], n: int, max_len: int = 15) -> List[str]:
    """
    Convenience: return up to `n` fallback sentences, each using one of the
    input words (cycled if needed). Used when LLM output is too short.
    """
    req = [normalize_word(w) for w in words if w]
    if not req:
        return ["I do not have a word."] * n

    out: List[str] = []
    i = 0
    while len(out) < n:
        w = req[i % len(req)]
        out.append(fallback_sentence([w], max_len=max_len))
        i += 1
    return out[:n]
