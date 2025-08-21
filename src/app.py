from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent

# ----- generator imports -----
try:
    from .generator import (
        generate,                      # EN→EN
        generate_french_like_english,  # FR→FR
        generate_fr_from_english,      # EN→FR
        generate_image,
        make_image_prompt,
        DEFAULT_LEVEL, LEVELS,
    )
except ImportError:
    import sys
    sys.path.insert(0, str(ROOT))
    from src.generator import (  # type: ignore
        generate,
        generate_french_like_english,
        generate_fr_from_english,
        generate_image,
        make_image_prompt,
        DEFAULT_LEVEL, LEVELS,
    )

# ----- DB -----
try:
    from . import db as cachedb
except ImportError:
    import sys
    sys.path.insert(0, str(ROOT))
    from src import db as cachedb  # type: ignore


# ========= helpers =========
def _parse_words(raw: str) -> List[str]:
    out, seen = [], set()
    for p in raw.replace(",", " ").split():
        low = p.lower().strip()
        if low and low not in seen:
            seen.add(low)
            out.append(low)
    return out


def _online_available() -> bool:
    """True only when OFFLINE unset and a usable key is present."""
    if os.getenv("OFFLINE") == "1":
        return False
    key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not key:
        return False
    # If project-scoped, project must be set
    if key.startswith("sk-proj-") and not (os.getenv("OPENAI_PROJECT") or "").strip():
        return False
    return True


def _normalize_size(size: str) -> str:
    return "1024x1024" if size == "auto" else size


def _gen_image_safe(prompt: str, size: str, style: str):
    """Call generate_image robustly; return path or None."""
    size = _normalize_size(size)
    try:
        res = generate_image(prompt=prompt, size=size, style=style)
    except TypeError:
        try:
            res = generate_image(prompt, size, style)
        except TypeError:
            res = generate_image(prompt, size)
    if isinstance(res, dict):
        return res.get("path")
    return res


def _resolve_img_path(p: str) -> str:
    pth = Path(p)
    if not pth.is_absolute():
        pth = ROOT / pth
    return str(pth)


def _normalize_en_fr_items(res: Any, headword: str) -> List[Dict[str, Any]]:
    """
    Make EN→FR generator results consistent:
    Returns a list of {"en_word": headword, "fr_targets": [...], "sentences":[{"fr":...,"en":...}]}
    Accepts shapes like:
      {"items":[{"en_word":"x","sentences":[..., ...]}]}
      {"x": [ "fr only", {"fr":"..","en":".."} ]}
      or other mild variants.
    """
    out: List[Dict[str, Any]] = []
    hw = (headword or "").lower().strip()

    if not res:
        return out

    # Case 1: expected {"items":[...]}
    if isinstance(res, dict) and isinstance(res.get("items"), list):
        for it in res["items"]:
            if not isinstance(it, dict):
                continue
            enw = (it.get("en_word") or it.get("word") or "").lower().strip()
            if enw and enw != hw:
                continue
            sents_norm: List[Dict[str, str]] = []
            for s in it.get("sentences", []):
                if isinstance(s, dict):
                    fr = (s.get("fr") or s.get("fr_text") or "").strip()
                    en = (s.get("en") or s.get("en_text") or "").strip()
                elif isinstance(s, str):
                    fr, en = s.strip(), ""
                else:
                    continue
                if fr:
                    sents_norm.append({"fr": fr, "en": en})
            if sents_norm:
                out.append({"en_word": headword, "fr_targets": it.get("fr_targets", []), "sentences": sents_norm})
        return out

    # Case 2: mapping: {"headword": [ ... ]}
    if isinstance(res, dict) and hw in {k.lower(): None for k in res.keys()}:
        # find key equal to headword case-insensitively
        target_key = None
        for k in res.keys():
            if (k or "").lower().strip() == hw:
                target_key = k
                break
        vals = res.get(target_key, [])
        sents_norm: List[Dict[str, str]] = []
        for s in vals or []:
            if isinstance(s, dict):
                fr = (s.get("fr") or s.get("fr_text") or "").strip()
                en = (s.get("en") or s.get("en_text") or "").strip()
            elif isinstance(s, str):
                fr, en = s.strip(), ""
            else:
                continue
            if fr:
                sents_norm.append({"fr": fr, "en": en})
        if sents_norm:
            out.append({"en_word": headword, "fr_targets": [], "sentences": sents_norm})
        return out

    # Case 3: plain list
    if isinstance(res, list):
        sents_norm: List[Dict[str, str]] = []
        for s in res:
            if isinstance(s, dict):
                fr = (s.get("fr") or s.get("fr_text") or "").strip()
                en = (s.get("en") or s.get("en_text") or "").strip()
            elif isinstance(s, str):
                fr, en = s.strip(), ""
            else:
                continue
            if fr:
                sents_norm.append({"fr": fr, "en": en})
        if sents_norm:
            out.append({"en_word": headword, "fr_targets": [], "sentences": sents_norm})
        return out

    return out


# ========== UI ==========
st.set_page_config(page_title="Sentence Generator", page_icon="✏️")
st.title("Beginner Sentence Generator")

colA, colB, colC, colD = st.columns([1.4, 1, 0.9, 1.1])
with colA:
    language_mode = st.selectbox(
        "Language Mode",
        ["English → English", "Français → Français", "English → Français"], index=0
    )
with colB:
    strict_pack = st.checkbox("Strict pack mode", value=True, help="No randomness/fallback; show locked sentences only.")
with colC:
    pack_name = st.text_input("Pack name", "main")
with colD:
    # Live health panel
    st.caption("Status / Health")
    st.code(
        f"OFFLINE={os.getenv('OFFLINE') or '∅'}\n"
        f"KEY={'✅' if (os.getenv('OPENAI_API_KEY') or '').strip() else '✗'}\n"
        f"PROJECT={'✅' if (os.getenv('OPENAI_PROJECT') or '').strip() else '∅'}\n"
        f"Pack={pack_name}",
        language="text",
    )

raw_words = st.text_input("Enter word(s) / Entrez des mot(s)", "")
n = st.slider("Sentences per word (display in non-pack mode)", 1, 10, 5)
mode = st.radio("Generation mode (non-pack only)", ["per-word", "mixed"], horizontal=True)
level = st.selectbox("Difficulty", ["A1", "A2", "B1"], index=0)

style = st.selectbox(
    "Image style",
    ["Photorealistic", "Kid-friendly illustration", "Watercolor", "Pixel art"],
    index=0,
)
image_size = st.selectbox("Image size", ["1024x1024", "1024x1536", "1536x1024", "auto"], index=0)

image_quantity = st.radio(
    "When generating new images… (non-pack mode only)",
    ["Per sentence", "Per word", "No images"],
    index=0,
)
alert_limit = st.slider("Alert me after … images", 10, 300, 50)

max_len = st.slider("Max words per EN sentence", 5, 50, 20)
max_unknown = st.slider("Max unknown words (EN filter)", 0, 10, 5)

if "images_generated" not in st.session_state:
    st.session_state["images_generated"] = 0


def _mode_key() -> str:
    return {
        "English → English": "en-en",
        "Français → Français": "fr-fr",
        "English → Français": "en-fr",
    }[language_mode]


def _should_add_image(idx_in_word: int) -> bool:
    if image_quantity == "No images":
        return False
    if image_quantity == "Per word":
        return idx_in_word == 0
    return True


def _render_one(conn, en_head: str, fr_text: str | None, en_text: str | None, idx_in_word: int, sentence_mode: str, *, strict: bool):
    if strict:
        img = cachedb.fetch_image_path_for_sentence(conn, en_head, fr=fr_text, en=en_text, mode=sentence_mode)
    else:
        img = cachedb.fetch_image_path_for_sentence_or_word(conn, en_head, fr=fr_text, en=en_text, mode=sentence_mode)

    if img and os.path.exists(_resolve_img_path(img)):
        st.image(_resolve_img_path(img), use_container_width=False)
        return

    if not strict and _should_add_image(idx_in_word) and _online_available():
        scene = (fr_text or en_text or "").strip()
        if scene:
            prompt = make_image_prompt(scene, style)
            path = _gen_image_safe(prompt, image_size, style)
            if not path and style != "Kid-friendly illustration":
                prompt2 = make_image_prompt(scene, "Kid-friendly illustration")
                path = _gen_image_safe(prompt2, image_size, "Kid-friendly illustration")
            if path:
                sid = cachedb._find_sentence_id_by_text(conn, en_head, fr=fr_text, en=en_text, mode=sentence_mode)
                if sid:
                    cachedb.save_image_for_sentence(
                        conn,
                        sid,
                        path=Path(path).as_posix(),
                        style=style,
                        size=_normalize_size(image_size),
                        model="gpt-image-1",
                        prompt=scene,
                    )
                    st.image(_resolve_img_path(path), use_container_width=False)
                    st.session_state.images_generated += 1
                else:
                    st.error("Generated an image but could not link it to a sentence (sid not found).")
                return

    st.caption("🗂️ No cached image." if strict else "⚠️ Image unavailable (offline or low balance).")


# ====== AUTHORING (Seed 5 now) ======
st.divider()
st.subheader("✍️ Authoring (online seeding)")
st.caption("Type **exactly one** word above, then click a Seed button. This will generate 5 sentences + 5 images and lock them into your pack for offline use.")

col_seed1, col_seed2, col_seed3, col_seed4 = st.columns([1, 1, 1, 1])
with col_seed1:
    do_en_en = st.button("Seed 5 EN→EN now", use_container_width=True)
with col_seed2:
    do_fr_fr = st.button("Seed 5 FR→FR now", use_container_width=True)
with col_seed3:
    do_en_fr = st.button("Seed 5 EN→FR now", use_container_width=True)
with col_seed4:
    do_all = st.button("Seed 5 for ALL modes", use_container_width=True)


def _seed_word_mode(conn, word: str, mode_key: str, k: int = 5) -> int:
    """Generate k sentences + 1 image each, save, and lock exactly those k into the pack. Returns locked count."""
    with st.status(f"Seeding **{word}** / **{mode_key}** …", expanded=True) as status:
        try:
            if not _online_available():
                st.write("❌ Online generation unavailable (missing key or OFFLINE=1).")
                status.update(state="error")
                return 0

            st.write("• Generating sentences…")
            batch_tag = f"author-{mode_key}-{int(time.time())}"
            wid = cachedb.upsert_word(conn, word, "en")

            if mode_key == "en-en":
                res = generate([word], k, "per-word", "A1", 20, 5)
                if isinstance(res, dict):
                    cachedb.save_en_en(conn, word, res.get(word, []), level="A1", model=batch_tag)
                else:
                    st.write("❌ Unexpected EN→EN generator return.")
                    status.update(state="error")
                    return 0

            elif mode_key == "fr-fr":
                res = generate_french_like_english([word], k)
                cachedb.save_fr_fr(conn, word, res.get(word, []), level="A1", model=batch_tag)

            else:  # en-fr
                raw = generate_fr_from_english([word], k) or {}
                items = _normalize_en_fr_items(raw, word)
                found = False
                for it in items:
                    if it.get("sentences"):
                        cachedb.save_en_fr(conn, word, it, level="A1", model=batch_tag)
                        found = True
                if not found:
                    st.write("❌ EN→FR generator returned no usable items for this word.")
                    status.update(state="error")
                    return 0

            st.write("• Selecting the newly saved sentences…")
            rows = conn.execute(
                "SELECT id, fr, en FROM sentences WHERE word_id=? AND mode=? AND model=? ORDER BY id ASC LIMIT ?",
                (wid, mode_key, batch_tag, k),
            ).fetchall()
            if not rows:
                st.write("❌ No sentences saved; aborting.")
                status.update(state="error")
                return 0

            st.write("• Generating 1 image per sentence (if missing)…")
            created = 0
            for r in rows:
                sid = int(r["id"])
                fr = (r["fr"] or "").strip()
                en = (r["en"] or "").strip()
                scene = fr or en
                if not cachedb.fetch_image_path_for_sentence_id(conn, sid):
                    prompt = make_image_prompt(scene, style)
                    path = _gen_image_safe(prompt, image_size, style)
                    if not path and style != "Kid-friendly illustration":
                        prompt2 = make_image_prompt(scene, "Kid-friendly illustration")
                        path = _gen_image_safe(prompt2, image_size, "Kid-friendly illustration")
                    if path:
                        cachedb.save_image_for_sentence(
                            conn,
                            sid,
                            path=Path(path).as_posix(),
                            style=style,
                            size=_normalize_size(image_size),
                            model="gpt-image-1",
                            prompt=scene,
                        )
                        created += 1
                        st.write(f"  ✅ Image saved for sentence {sid}")
                    else:
                        st.write(f"  ⚠️ Image failed for sentence {sid}")

            st.write("• Locking sentences into the pack…")
            sids = [int(r["id"]) for r in rows]
            cachedb.set_pack_for_word_mode(conn, word, mode_key, sids, pack=pack_name)

            st.write(f"✅ Done. Locked {len(sids)} sentences (created {created} image(s)).")
            status.update(state="complete")
            return len(sids)
        except Exception as e:
            st.exception(e)
            status.update(state="error")
            return 0


# Handle button clicks
if do_en_en or do_fr_fr or do_en_fr or do_all:
    words = _parse_words(raw_words)
    st.write(f"**Parsed words:** {words}")
    if not words or len(words) != 1:
        st.error("Enter exactly **one** word to seed.")
    else:
        conn = cachedb.connect()
        try:
            w = words[0]
            total = 0
            if do_all:
                for mk in ("en-en", "fr-fr", "en-fr"):
                    total += _seed_word_mode(conn, w, mk, k=5)
            else:
                if do_en_en:
                    total += _seed_word_mode(conn, w, "en-en", k=5)
                if do_fr_fr:
                    total += _seed_word_mode(conn, w, "fr-fr", k=5)
                if do_en_fr:
                    total += _seed_word_mode(conn, w, "en-fr", k=5)
            if total:
                st.success(f"✅ Authoring finished for '{w}'. Switch to **Strict pack mode** to view offline later.")
            else:
                st.warning("No sentences were locked. Check the status panel above for details.")
        finally:
            try:
                conn.close()
            except Exception:
                pass


# ====== VIEWER ======
st.divider()
st.subheader("📖 Viewer")

if st.button("Generate / Refresh View", use_container_width=True):
    words = _parse_words(raw_words)
    if not words:
        st.error("Please enter at least one word.")
        st.stop()

    conn = cachedb.connect()
    mode_key = _mode_key()

    if strict_pack:
        for w in words:
            rows = cachedb.fetch_pack_sentences(conn, w, mode_key, pack=pack_name)
            if not rows:
                st.warning(f"No pack rows for '{w}' ({mode_key}, pack='{pack_name}').")
                continue
            st.markdown(f"### {w}")
            for i, r in enumerate(rows):
                fr = r["fr"] or ""
                en = r["en"] or ""
                if mode_key == "en-fr":
                    st.write(f"• **{fr}**  \n  _{en}_")
                elif mode_key == "fr-fr":
                    st.write("• " + fr)
                else:
                    st.write("• " + en)
                _render_one(
                    conn,
                    w,
                    fr if mode_key != "en-en" else None,
                    en if mode_key != "fr-fr" else None,
                    i,
                    mode_key,
                    strict=True,
                )
    else:
        if not _online_available():
            st.info("Offline mode or missing key. Showing cache; on-demand generation disabled.")

        if language_mode == "English → Français":
            merged: Dict[str, Any] = {"items": []}
            cached = cachedb.fetch_en_fr_items(conn, words, n)
            merged["items"].extend(cached.get("items", []))
            for item in merged["items"]:
                en_head = item.get("en_word", "")
                st.markdown(f"### {en_head}")
                lemmas = item.get("fr_targets") or []
                if lemmas:
                    st.caption("Lemmas: " + ", ".join(lemmas))
                for i, sent in enumerate(item.get("sentences", [])):
                    fr, en = sent.get("fr", ""), sent.get("en", "")
                    if en:
                        st.write(f"• **{fr}**  \n  _{en}_")
                    else:
                        st.write(f"• **{fr}**")
                    _render_one(conn, en_head, fr, en, i, "en-fr", strict=False)

        elif language_mode == "Français → Français":
            for w in words:
                sents = cachedb.fetch_fr_fr(conn, w, n)
                st.markdown(f"### {w}")
                for i, s in enumerate(sents):
                    st.write("• " + s)
                    _render_one(conn, w, s, None, i, "fr-fr", strict=False)

        else:  # English → English
            for w in words:
                sents = cachedb.fetch_en_en(conn, w, n)
                st.markdown(f"### {w}")
            for i, s in enumerate(sents):
                st.write("• " + s)
                _render_one(conn, w, None, s, i, "en-en", strict=False)

    try:
        conn.close()
    except Exception:
        pass

    if st.session_state.images_generated > alert_limit:
        dollars = st.session_state.images_generated * 0.011
        st.warning(
            f"💸 Heads-up: {st.session_state.images_generated} images ≈ ${dollars:,.2f}.",
            icon="💸",
        )


# =============== FR-lemma offline search ==========
st.divider()
st.subheader("🔎 Search by French word (offline)")

fr_query = st.text_input("Type a French lemma (e.g., école). Accents optional.", key="fr_search_input")
fr_k = st.slider("How many sentences", 1, 20, 5, key="fr_search_k")

if st.button("Search FR offline", use_container_width=True):
    if not fr_query.strip():
        st.error("Please type a French word (lemma).")
    else:
        try:
            conn = cachedb.connect()
            rows = cachedb.fetch_fr_by_french_word(conn, fr_query.strip(), fr_k)
        except Exception as e:
            rows = []
            st.exception(e)

        if not rows:
            st.info("No cached matches. Try another lemma or seed EN→FR for this word.")
        else:
            for r in rows:
                fr_text = r.get("fr", "")
                en_gloss = r.get("en", "")
                en_head = r.get("en_word", "")
                mode_val = r.get("mode", "")
                if en_gloss:
                    st.write(f"• **{fr_text}**  \n  _{en_gloss}_  \n  *(headword: {en_head}, {mode_val})*")
                else:
                    st.write(f"• **{fr_text}**  \n  *(headword: {en_head}, {mode_val})*")

                try:
                    img_path = cachedb.fetch_image_path_for_sentence(
                        conn, en_head, fr=fr_text, en=en_gloss or None, mode=mode_val
                    )
                    if img_path:
                        abp = _resolve_img_path(img_path)
                        if os.path.exists(abp):
                            st.image(abp, use_container_width=False)
                except Exception:
                    pass

        try:
            conn.close()
        except Exception:
            pass
