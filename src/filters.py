from __future__ import annotations
import re, string
from pathlib import Path
from typing import List, Sequence

import spacy
from wordfreq import zipf_frequency
from textstat import flesch_reading_ease

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
A1_PATH  = DATA_DIR / "oxford_a2.txt"          # created by bootstrap.py


def _load_allowed_words(path: Path = A1_PATH) -> List[str]:
    words: List[str] = []
    if path.exists():
        with path.open(encoding="utf8") as f:
            for line in f:
                line = line.strip()
                if line and line.isalpha():
                    words.append(line.lower())
    return words


ALLOWED_WORDS = set(_load_allowed_words())

NLP = spacy.load("en_core_web_sm", disable=["ner", "parser"])

_PUNCT_RX = re.compile(rf"[{re.escape(string.punctuation)}]+")


def normalize_word(word: str) -> str:
    """lower-case + strip punctuation"""
    return _PUNCT_RX.sub("", word).lower()


def normalize_word_list(words: Sequence[str]) -> List[str]:
    """Apply normalize_word & dedupe while preserving order."""
    seen, out = set(), []
    for w in words:
        w = normalize_word(w)
        if w and w not in seen:
            seen.add(w)
            out.append(w)
    return out


def guess_pos(word: str) -> str:
    """Return coarse POS tag for *word* (NOUN, VERB, ADJ, etc.)."""
    doc = NLP(word)
    return doc[0].pos_ if doc else "NOUN"


def dedupe(seq: Sequence[str]) -> List[str]:
    """Remove exact duplicates, keep first occurrence."""
    seen, out = set(), []
    for s in seq:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out

def contains_all_required(sentence: str, required: Sequence[str]) -> bool:
    low = sentence.lower()
    return all(r.lower() in low for r in required)


def filter_sentences(
    sentences: Sequence[str],
    required: Sequence[str],
    allowed_words: set[str],
    max_len: int,
    max_unknown: int,
) -> List[str]:
    """Return sentences that pass length + vocab + required-word checks."""
    keep: List[str] = []
    req_low = [r.lower() for r in required]

    for s in sentences:
        tokens = s.split()
        if len(tokens) > max_len:
            continue

        low_tokens = [t.lower().strip(".,!?") for t in tokens]
        if not all(r in low_tokens for r in req_low):
            continue

        unknown = sum(
            1
            for t in low_tokens
            if t not in allowed_words and t not in req_low
        )
        if unknown > max_unknown:
            continue

        keep.append(s)

    return keep


def passes_filters(
    sent: str,
    target: str,
    allowed_words: set[str],
    max_len: int,
    max_unknown: int,
    min_re: int = 80,
) -> bool:

    tokens = sent.split()
    if len(tokens) > max_len:
        return False
    if target.lower() not in [t.lower() for t in tokens]:
        return False
    if flesch_reading_ease(sent) < min_re:
        return False

    rare = 0
    for t in tokens:
        tl = t.lower().strip(".,!?")
        if tl == target.lower():
            continue
        if tl in allowed_words:
            continue
        if zipf_frequency(tl, "en") < 3.5:  # rarer than ~1/10 000
            rare += 1
            if rare > max_unknown:
                return False
    return True
