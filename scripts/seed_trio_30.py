from __future__ import annotations

import os
import sys
import json
import time
import pathlib
from typing import List

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import db
from src.generator import (
    generate,
    generate_fr_from_english,
    generate_french_like_english,
    generate_image,
    make_image_prompt,
)

MODEL_TAG = "trio_seed:v1"
IMG_SIZE  = "1024x1024"
IMG_STYLE = "Photorealistic"

# Top 30 A1–A2-ish “classroom” words (edit as you like)
WORDS: List[str] = [
    "school","family","friend","home","water","food",
    "book","teacher","student","class","room","city",
    "country","day","night","morning","time","phone",
    "computer","music","movie","car","bus","train",
    "weather","work","help","question","answer","problem",
]


def require_online() -> None:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise SystemExit("OPENAI_API_KEY missing. Set it before seeding.")
    if key.startswith("sk-proj-") and not os.getenv("OPENAI_PROJECT"):
        raise SystemExit("OPENAI_PROJECT is required for project-scoped keys.")


def _coerce_en_fr_items(res):
    """
    Accepts many shapes from generate_fr_from_english:
      - dict with "items": [...]
      - single dict {"en_word","fr_targets","sentences":[...]}
      - list of dicts
      - raw JSON string of any of the above
    Returns a list[dict] with normalized fields:
      {"en_word": str, "fr_targets": list[str], "sentences": list[{"fr","en"}]}
    """
    # If it's a string, try to parse JSON
    if isinstance(res, str):
        try:
            res = json.loads(res)
        except Exception:
            return []

    items = []
    if isinstance(res, dict):
        if isinstance(res.get("items"), list):
            items = res["items"]
        elif ("en_word" in res) or ("sentences" in res):
            items = [res]
    elif isinstance(res, list):
        items = res

    out = []
    for it in items:
        # If any element is a string, try to parse it as JSON; otherwise skip
        if isinstance(it, str):
            try:
                it = json.loads(it)
            except Exception:
                continue
        if not isinstance(it, dict):
            continue

        # Normalize fr_targets -> list[str]
        norm_targets = []
        for t in (it.get("fr_targets") or []):
            if isinstance(t, str) and t.strip():
                norm_targets.append(t.strip())
            elif isinstance(t, dict) and t.get("lemma"):
                norm_targets.append(t["lemma"].strip())
        it["fr_targets"] = norm_targets

        # Normalize sentences -> list[{"fr","en"}] (skip empties)
        norm_pairs = []
        for p in (it.get("sentences") or []):
            if isinstance(p, str):
                # Unknown string shape — ignore rather than guessing
                continue
            fr = (p.get("fr") or "").strip()
            en = (p.get("en") or "").strip()
            if fr:
                norm_pairs.append({"fr": fr, "en": en})
        it["sentences"] = norm_pairs

        # Ensure en_word present (best-effort)
        if not it.get("en_word"):
            it["en_word"] = ""

        out.append(it)

    return out


def ensure_en_fr(conn, word: str, n: int) -> int:
    """At least n EN→FR pairs (and lemmas). Accepts dict/list/string payloads."""
    have = conn.execute("""
        SELECT COUNT(*)
        FROM sentences s
        JOIN words w ON w.id=s.word_id
        WHERE w.text=? AND s.mode='en-fr' AND s.approved=1
    """, (word,)).fetchone()[0]

    need = max(0, n - int(have))
    if need <= 0:
        return 0

    res = generate_fr_from_english([word], n=need) or {}
    items = _coerce_en_fr_items(res)

    saved = 0
    for it in items:
        db.save_en_fr(conn, word, it, level="A2", model=MODEL_TAG)
        saved += len(it.get("sentences", []))
    return saved


def ensure_en_en(conn, word: str, n: int) -> int:
    """At least n EN→EN sentences, slightly richer A2–B1."""
    have = conn.execute("""
        SELECT COUNT(*)
        FROM sentences s
        JOIN words w ON w.id=s.word_id
        WHERE w.text=? AND s.mode='en-en' AND s.approved=1
    """, (word,)).fetchone()[0]

    need = max(0, n - int(have))
    if need <= 0:
        return 0

    res = generate([word], n=need, mode="per-word", level="B1", max_len=22, max_unknown=6)
    if isinstance(res, dict):
        db.save_en_en(conn, word, res.get(word, []), level="B1", model=MODEL_TAG)
        return len(res.get(word, []))
    return 0


def ensure_fr_fr(conn, word: str, n: int) -> int:
    """
    At least n FR→FR sentences under the same English headword.
    We use FR lemmas saved from EN→FR (e.g., école, scolarité)
    and ask the model to produce FR sentences with those lemmas.
    """
    have = conn.execute("""
        SELECT COUNT(*)
        FROM sentences s
        JOIN words w ON w.id=s.word_id
        WHERE w.text=? AND s.mode='fr-fr' AND s.approved=1
    """, (word,)).fetchone()[0]

    need = max(0, n - int(have))
    if need <= 0:
        return 0

    # get lemmas; if none, try to create via a small EN→FR call
    lrows = conn.execute("""
        SELECT lemma FROM fr_lemmas fl
        JOIN words w ON w.id=fl.word_id
        WHERE w.text=?
        ORDER BY lemma
    """, (word,)).fetchall()
    lemmas = [r[0] for r in lrows]
    if not lemmas:
        ensure_en_fr(conn, word, 2)
        lrows = conn.execute("""
            SELECT lemma FROM fr_lemmas fl
            JOIN words w ON w.id=fl.word_id
            WHERE w.text=?
            ORDER BY lemma
        """, (word,)).fetchall()
        lemmas = [r[0] for r in lrows]

    seeds = lemmas[:2] if lemmas else [word]
    per = max(1, need // max(1, len(seeds)))
    total = 0
    for L in seeds:
        gen = generate_french_like_english([L], n=per)  # returns {lemma: [fr,...]}
        for _, fr_sents in gen.items():
            db.save_fr_fr(conn, word, fr_sents, level="A2", model=MODEL_TAG)
            total += len(fr_sents)
    return total


def ensure_images_for_word(conn, word: str, n: int) -> int:
    """
    Attach up to n **Photorealistic** images to this word, 1 per sentence,
    preferring EN→FR FR text, then FR→FR, then EN→EN.
    """
    # how many sentence-linked images already?
    have = conn.execute("""
      SELECT COUNT(*) FROM images i
      JOIN words w ON w.id=i.word_id
      WHERE w.text=? AND i.sentence_id IS NOT NULL
    """, (word,)).fetchone()[0]
    need = max(0, n - int(have))
    if need <= 0:
        return 0

    picked = 0
    for mode in ("en-fr", "fr-fr", "en-en"):
        rows = conn.execute("""
           SELECT s.id, s.fr, s.en
           FROM sentences s
           JOIN words w ON w.id=s.word_id
           WHERE w.text=? AND s.mode=? AND s.approved=1
           ORDER BY RANDOM()
        """, (word, mode)).fetchall()

        for sid, fr, en in rows:
            if picked >= need:
                break
            if db.has_image_for_sentence(conn, sid):
                continue
            text = fr or en or ""
            if not text:
                continue
            prompt = make_image_prompt(text, "Photorealistic")
            path = generate_image(prompt, IMG_SIZE)  # cached on disk if same prompt/size
            if not path:
                continue
            db.save_image_for_sentence(
                conn, sid,
                path=path, style=IMG_STYLE, size=IMG_SIZE,
                model="gpt-image-1", prompt=prompt
            )
            picked += 1

        if picked >= need:
            break
    return picked


def main():
    require_online()
    conn = db.connect()
    TARGET_EACH = 5

    print(f"[trio_seed] Seeding {len(WORDS)} words…")
    total_enfr = total_enen = total_frfr = total_imgs = 0

    for w in WORDS:
        print(f"\n=== {w} ===")
        try:
            a = ensure_en_fr(conn, w, TARGET_EACH)
            print(f"EN→FR +lemmas: +{a}")
        except Exception as e:
            print(f"EN→FR failed: {e}")

        try:
            b = ensure_en_en(conn, w, TARGET_EACH)
            print(f"EN→EN        : +{b}")
        except Exception as e:
            print(f"EN→EN failed : {e}")

        try:
            c = ensure_fr_fr(conn, w, TARGET_EACH)
            print(f"FR→FR        : +{c}")
        except Exception as e:
            print(f"FR→FR failed : {e}")

        try:
            d = ensure_images_for_word(conn, w, TARGET_EACH)
            print(f"Images (photo): +{d}")
        except Exception as e:
            print(f"Images failed : {e}")

        total_enfr += a if isinstance(a, int) else 0
        total_enen += b if isinstance(b, int) else 0
        total_frfr += c if isinstance(c, int) else 0
        total_imgs += d if isinstance(d, int) else 0

        time.sleep(0.2)  # be polite to API

    print("\n[trio_seed] DONE.")
    print(f"Added EN→FR: {total_enfr}, EN→EN: {total_enen}, FR→FR: {total_frfr}, Images: {total_imgs}")


if __name__ == "__main__":
    main()
