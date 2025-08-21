from __future__ import annotations
import sys, time, os, pathlib
from typing import List, Tuple

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import db
from src.generator import (
    generate,                       # EN→EN
    generate_fr_from_english,       # EN→FR pairs
    generate_french_like_english,   # FR→FR
    make_image_prompt,
    generate_image,
)

PER_MODE = 5
IMAGES_PER_WORD = 5
STYLE = "Photorealistic"       # REQUIREMENT
SIZE  = "1024x1024"
TEXT_TAG = "seed_one_text:v1"
IMG_TAG  = "seed_one_photo:v1"

def online_available() -> bool:
    if os.getenv("OFFLINE") == "1":
        return False
    key = os.getenv("OPENAI_API_KEY","").strip()
    if not key:
        return False
    if key.startswith("sk-proj-") and not os.getenv("OPENAI_PROJECT"):
        return False
    return True

def count_mode(conn, word: str, mode: str) -> int:
    wid = db.upsert_word(conn, word, "en")
    return conn.execute("SELECT COUNT(*) FROM sentences WHERE word_id=? AND mode=?", (wid, mode)).fetchone()[0]

def ensure_en_en(conn, word: str):
    before = count_mode(conn, word, "en-en")
    need = max(0, PER_MODE - before)
    if need:
        res = generate([word], n=need, mode="per-word", level="B1", max_len=24, max_unknown=6)
        if isinstance(res, dict):
            db.save_en_en(conn, word, res.get(word, []), level="A2", model=TEXT_TAG)
    after = count_mode(conn, word, "en-en")
    return max(0, after - before)

def ensure_en_fr(conn, word: str) -> List[str]:
    before = count_mode(conn, word, "en-fr")
    need = max(0, PER_MODE - before)
    fr_boost: List[str] = []
    if need:
        resp = generate_fr_from_english([word], n=need) or {}
        items = resp.get("items", []) if isinstance(resp, dict) else []
        for it in items:
            db.save_en_fr(conn, word, it, level="A2", model=TEXT_TAG)
            for p in it.get("sentences", []):
                fr = (p.get("fr") or "").strip()
                if fr: fr_boost.append(fr)
    after = count_mode(conn, word, "en-fr")
    return fr_boost

def ensure_fr_fr(conn, word: str, fr_boost: List[str]):
    before = count_mode(conn, word, "fr-fr")
    need = max(0, PER_MODE - before)
    used = 0
    if need and fr_boost:
        use = fr_boost[:need]
        db.save_fr_fr(conn, word, use, level="A2", model=TEXT_TAG)
        used = len(use)
    remain = max(0, need - used)
    if remain:
        gen = generate_french_like_english([word], n=remain)
        db.save_fr_fr(conn, word, gen.get(word, []), level="A2", model=TEXT_TAG)
    after = count_mode(conn, word, "fr-fr")
    return max(0, after - before)

def pick_sentence_rows_for_imaging(conn, word: str, k: int) -> List[Tuple[int,str,str,str]]:
    wid = db.upsert_word(conn, word, "en")
    picked: List[Tuple[int,str,str,str]] = []
    def add(mode: str):
        nonlocal picked
        rows = conn.execute(
            """
            SELECT s.id, s.fr, s.en, s.mode
            FROM sentences s
            LEFT JOIN images i ON i.sentence_id = s.id
            WHERE s.word_id=? AND s.mode=? AND s.approved=1 AND i.id IS NULL
            ORDER BY RANDOM()
            LIMIT ?
            """,
            (wid, mode, k - len(picked))
        ).fetchall()
        picked.extend([(r[0], r[1] or "", r[2] or "", r[3]) for r in rows])
    for mode in ("en-fr","fr-fr","en-en"):
        if len(picked) >= k: break
        add(mode)
    return picked[:k]

def ensure_photorealistic_images(conn, word: str):
    rows = pick_sentence_rows_for_imaging(conn, word, IMAGES_PER_WORD)
    if not rows:
        print("[img] no sentences available to image yet"); return 0
    saved = 0
    word_cover_exists = bool(db.fetch_image_path(conn, word))
    cover_path = None
    for sid, fr, en, mode in rows:
        # Photorealistic prompt: prefer English gloss if available for clarity
        caption = (en.strip() if (mode=="en-fr" and en.strip()) else (fr or en).strip())
        if not caption: continue
        prompt = make_image_prompt(caption, STYLE)   # NO kid fallback
        path = generate_image(prompt, SIZE)
        if not path:
            print(f"[img] failed for sentence {sid}: {caption!r}")
            continue
        db.save_image_for_sentence(conn, sid, path=path, style=STYLE, size=SIZE, model=IMG_TAG, prompt=prompt)
        saved += 1
        if not word_cover_exists and cover_path is None:
            cover_path = path
        time.sleep(0.08)
    if cover_path:
        db.save_image_for_word(conn, word=word, path=cover_path, style=STYLE, size=SIZE, model=IMG_TAG, prompt="(word-cover)")
    return saved

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/seed_one_photo.py <word>"); sys.exit(1)
    word = sys.argv[1].strip().lower()
    if not online_available():
        print("[seed_one] Missing OPENAI credentials or OFFLINE=1. Aborting."); return
    conn = db.connect()
    print(f"=== Seeding '{word}' ===")
    add1 = ensure_en_en(conn, word)
    boost = ensure_en_fr(conn, word)
    add3 = ensure_fr_fr(conn, word, boost)
    print(f"[text] added en-en {add1}, fr-fr {add3}; en-fr boost {len(boost)}")
    imgs = ensure_photorealistic_images(conn, word)
    print(f"[img] added {imgs} photorealistic images")
    print("[done]")

if __name__ == "__main__":
    main()
