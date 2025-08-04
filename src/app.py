"""
app.py
------

Streamlit demo UI for the Beginner Sentence Generator project.

Run from project root:
    streamlit run src/app.py
or
    python -m streamlit run src/app.py
"""

from __future__ import annotations
import os
from typing import List
import streamlit as st

# ---------- robust import shim -----------------------------------
try:
    from .generator import generate, DEFAULT_MODE, DEFAULT_LEVEL, DEFAULT_MAX_LEN, LEVELS
except ImportError:
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parent.parent          # .../sentence_gen
    sys.path.insert(0, str(ROOT))
    from src.generator import generate, DEFAULT_MODE, DEFAULT_LEVEL, DEFAULT_MAX_LEN, LEVELS
# -----------------------------------------------------------------


# ----------------- helper functions ------------------------------
def _parse_words(raw: str) -> List[str]:
    if not raw.strip():
        return []
    parts = []
    for chunk in raw.split(","):
        parts.extend(chunk.strip().split())
    seen, words = set(), []
    for p in parts:
        low = p.lower()
        if low not in seen:
            seen.add(low)
            words.append(p)
    return words


def _warn_if_no_key():
    if not os.environ.get("OPENAI_API_KEY"):
        st.warning("No OPENAI_API_KEY found. Output will use fallback templates only.", icon="⚠️")
# -----------------------------------------------------------------


# ----------------- UI layout -------------------------------------
st.set_page_config(page_title="Beginner Sentence Generator", page_icon="✏️", layout="centered")
st.title("Beginner Sentence Generator")
st.caption("Generate simple English sentences for any word(s).")

raw_words = st.text_input(
    "Enter word(s):",
    value="",                           # start empty every session
    placeholder="apple red eat",
    help="Separate by space or comma.",
    key="raw_words"                     # key lets us reset later
)

n = st.slider("Number of sentences", 1, 20, 5, help="Per word (per‑word) or total (mixed).")
mode = st.radio("Mode", ["per-word", "mixed"], horizontal=True)
level = st.selectbox("Difficulty level", ["A1", "A2", "B1"], index=["A1", "A2", "B1"].index(DEFAULT_LEVEL))

# always‑visible sliders
level_defaults = LEVELS[level]
default_len = DEFAULT_MAX_LEN if level == "A2" else level_defaults["max_len"]
max_len = st.slider(
    "Max words per sentence",    # <— leave this
    min_value=5,                 # leave this
    max_value=50,                # leave this
    value=20,                    # ← set default you want
    step=1,
    help="Upper limit for each sentence.",
)
max_unknown = st.slider(
    "Max allowed non-beginner words",
    min_value=0,
    max_value=10,
    value=5,                     # ← set default you want
    step=1,
    help="Words not in the beginner list (excluding your target words).",
)
# -----------------------------------------------------------------


# ----------------- Generate & display ----------------------------
if st.button("Generate"):
    _warn_if_no_key()
    words = _parse_words(raw_words)
    if not words:
        st.error("Please enter at least one word.")
        st.stop()

    with st.spinner("Generating sentences…"):
        result = generate(
            words=words,
            n=n,
            mode=mode,
            level=level,
            max_len=int(max_len),
            max_unknown=int(max_unknown),
        )

    st.success("Done!")

    if mode == "per-word" and isinstance(result, dict):
        for w, sentences in result.items():
            st.markdown(f"### {w}")
            for s in sentences:
                st.write("• " + s)
    else:  # mixed mode
        st.markdown("### Sentences")
        for s in result:                   # type: ignore[arg-type]
            st.write("• " + s)
# -----------------------------------------------------------------
