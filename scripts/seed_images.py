from __future__ import annotations
import sys, time, argparse, pathlib
from typing import List, Dict, Any

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import db
from src.generator import make_image_prompt, generate_image

DEFAULT_STYLE = "Kid-friendly illustration"
DEFAULT_SIZE  = "1024x1024"
MODEL_TAG     = "seed_images:v1"

def words_needing_images(conn, first_letter: str, per_letter_limit: int) -> List[str]:
    like = (first_letter.lower() + "%")
    rows = conn.execute(
        """
        SELECT w.text
        FROM words w
        LEFT JOIN images i ON i.word_id = w.id
        WHERE w.lang='en'
          AND w.text LIKE ?
        GROUP BY w.id
        HAVING COUNT(i.id)=0
        ORDER BY w.text
        LIMIT ?
        """,
        (like, per_letter_limit),
    ).fetchall()
    return [r[0] for r in rows]

def pick_caption(conn, word: str) -> str | None:
    # Prefer EN→FR French sentence; else EN→EN sentence; else None
    row = conn.execute("""
        SELECT s.fr FROM sentences s
        JOIN words w ON w.id=s.word_id
        WHERE w.text=? AND s.mode='en-fr'
        ORDER BY RANDOM() LIMIT 1
    """, (word,)).fetchone()
    if row and row[0]:
        return row[0]

    row = conn.execute("""
        SELECT s.en FROM sentences s
        JOIN words w ON w.id=s.word_id
        WHERE w.text=? AND s.mode='en-en'
        ORDER BY RANDOM() LIMIT 1
    """, (word,)).fetchone()
    if row and row[0]:
        return row[0]

    return None

def main():
    ap = argparse.ArgumentParser(description="Seed one image per word (if missing).")
    ap.add_argument("--letters", default="abcdefghijklmnopqrstuvwxyz",
                    help="Initial letters to process (e.g. 'abc', default a..z).")
    ap.add_argument("--per-letter-limit", type=int, default=200, help="Max words per initial.")
    ap.add_argument("--style", default=DEFAULT_STYLE, help="Image style.")
    ap.add_argument("--size", default=DEFAULT_SIZE, help="Image size.")
    ap.add_argument("--sleep", type=float, default=0.15, help="Pause between words.")
    args = ap.parse_args()

    conn = db.connect()
    total = 0

    for ch in args.letters:
        if not ch.isalpha(): continue
        batch = words_needing_images(conn, ch, args.per_letter_limit)
        print(f"[seed_images] Letter '{ch}': {len(batch)} words")
        for w in batch:
            cap = pick_caption(conn, w)
            if not cap:
                print(f"[skip] {w}: no caption found yet")
                time.sleep(args.sleep)
                continue

            prompt = make_image_prompt(cap, args.style)
            path = generate_image(prompt, args.size)
            if path:
                try:
                    db.save_image_for_word(
                        conn, w, path=path, style=args.style, size=args.size,
                        model=MODEL_TAG, prompt=prompt
                    )
                    print(f"[ok] {w} → {path}")
                    total += 1
                except Exception as e:
                    print(f"[warn] save failed for {w}: {e}")
            else:
                print(f"[warn] gen failed for {w}")
            time.sleep(args.sleep)

    print(f"[seed_images] Done. Saved {total} images.")

if __name__ == "__main__":
    main()
