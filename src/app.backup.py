from __future__ import annotations
import os, concurrent.futures as futures
from typing import List, Tuple
import streamlit as st

# ---------- imports from generator -------------------
try:
    from .generator import (
        generate,
        generate_french_like_english,
        generate_fr_from_english,
        generate_image,
        make_image_prompt,
        DEFAULT_LEVEL, LEVELS
    )
except ImportError:
    import sys
    from pathlib import Path
    ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(ROOT))
    from src.generator import (
        generate,
        generate_french_like_english,
        generate_fr_from_english,
        generate_image,
        make_image_prompt,
        DEFAULT_LEVEL, LEVELS
    )

def _parse_words(raw: str) -> List[str]:
    out, seen = [], set()
    for p in raw.replace(",", " ").split():
        low = p.lower()
        if low and low not in seen:
            seen.add(low); out.append(p)
    return out


def warn_no_key():
    if not os.getenv("OPENAI_API_KEY"):
        st.warning("OPENAI_API_KEY missing. Fallback sentences only.", icon="⚠️")


st.set_page_config(page_title="Sentence Generator", page_icon="✏️")
st.title("Beginner Sentence Generator")

language_mode = st.selectbox(
    "Language Mode",
    ["English → English", "Français → Français", "English → Français"],
    index=0
)

raw_words = st.text_input("Enter word(s) / Entrez des mot(s)", "")
n      = st.slider("Sentences per word", 1, 10, 5)
mode   = st.radio("Generation mode", ["per-word", "mixed"], horizontal=True)
level  = st.selectbox("Difficulty", ["A1", "A2", "B1"], index=0)

style  = st.selectbox(
    "Image style",
    ["Photorealistic", "Kid-friendly illustration", "Watercolor", "Pixel art"],
    index=0,
    help="Photorealistic offers most real life photos"
)
image_size = st.selectbox(
    "Image size (supported by gpt-image-1)",
    ["1024x1024", "1024x1536", "1536x1024", "auto"],
    index=0,
    help= "Choose the auto size for cost effective usage"
)

# 💰 NEW: cost controls
image_quantity = st.radio(
    "Image generation",
    ["Per sentence", "Per word", "No images"],
    index=0,
    help="Per word = 1 image per initial word. No images = text only (almost free)."
)
alert_limit = st.slider("Alert me after … images (this session)", 10, 300, 50)

# English filtering sliders
max_len = st.slider("Max words per sentence", 5, 50, 20)
max_unknown = st.slider("Max unknown words", 0, 10, 5)

# session-state counter for budget alert
if "images_generated" not in st.session_state:
    st.session_state["images_generated"] = 0


# ----------------- Generate --------------------------
if st.button("Generate"):
    warn_no_key()
    words = _parse_words(raw_words)
    if not words:
        st.error("Please enter at least one word."); st.stop()
    if language_mode == "Français → Français" and mode != "per-word":
        st.warning("French supports per-word only."); st.stop()

    with st.spinner("Generating…"):
        if language_mode == "Français → Français":
            raw = generate_french_like_english(words, n)
        elif language_mode == "English → Français":
            raw = generate_fr_from_english(words, n)
        else:
            raw = generate(words, n, mode, level, max_len, max_unknown)

    # ---------- helper: choose whether to queue an image ----------
    def should_add_image(idx_in_word: int) -> bool:
        if image_quantity == "No images":
            return False
        if image_quantity == "Per word":
            return idx_in_word == 0
        return True  # Per sentence

    # helper: async fetch with fallback to kid-style
    def fetch(prompt: str):
        primary = make_image_prompt(prompt, style)
        path = generate_image(primary, image_size)
        if not path and style != "Kid-friendly illustration":
            kid_prompt = make_image_prompt(prompt, "Kid-friendly illustration")
            path = generate_image(kid_prompt, image_size)
        return path

    ex = futures.ThreadPoolExecutor(max_workers=4)
    tasks: List[Tuple[futures.Future, st.delta_generator.DeltaGenerator]] = []

    # ---------- Render + queue images -----------------------------
    word_index = 0  # for per-word logic
    if language_mode == "Français → Français":
        for w, sents in raw.items():
            st.markdown(f"### {w}")
            for i, s in enumerate(sents):
                st.write("• " + s)
                ph = st.empty()
                if should_add_image(i):
                    fut = ex.submit(fetch, s)
                    tasks.append((fut, ph))
            word_index += 1

    elif language_mode == "English → Français":
        for item in raw.get("items", []):
            st.markdown(f"### {item.get('en_word')}")
            for i, sent in enumerate(item.get("sentences", [])):
                fr, en = sent["fr"], sent["en"]
                st.write(f"• **{fr}**  \n  _{en}_")
                ph = st.empty()
                if should_add_image(i):
                    fut = ex.submit(fetch, fr)
                    tasks.append((fut, ph))
            word_index += 1

    else:  # English → English
        if mode == "per-word" and isinstance(raw, dict):
            for w, sents in raw.items():
                st.markdown(f"### {w}")
                for i, s in enumerate(sents):
                    st.write("• " + s)
                    ph = st.empty()
                    if should_add_image(i):
                        fut = ex.submit(fetch, s)
                        tasks.append((fut, ph))
                word_index += 1
        else:  # mixed list
            for i, s in enumerate(raw):  # type: ignore
                st.write("• " + s)
                ph = st.empty()
                if should_add_image(i):
                    fut = ex.submit(fetch, s)
                    tasks.append((fut, ph))
            word_index = 1  # treat as one “word” block

    # ---------- fill images + budget alert ------------------------
    for fut, ph in tasks:
        path = fut.result()
        if path:
            ph.image(path, use_container_width=False)
            st.session_state.images_generated += 1
        else:
            ph.caption("⚠️ Image unavailable due to low balance.")

    # ----------- budget alert in dollars -----------------
    if st.session_state.images_generated > alert_limit:
        dollars = st.session_state.images_generated * 0.011  # Low-tier 1024²
        st.warning(
            f"💸 **Heads-up!** This session has generated "
            f"{st.session_state.images_generated} images ≈ "
            f"${dollars:,.2f}. (Limit was {alert_limit} images.)",
            icon="💸"
        )



