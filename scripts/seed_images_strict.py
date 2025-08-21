from __future__ import annotations
import sys, time, argparse, pathlib, os
from typing import List, Tuple

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import db
from src.generator import make_image_prompt, generate_image

DEFAULT_STYLE = "Kid-friendly illustration"
DEFAULT_SIZE  = "1024x1024"
MODEL_TAG     = "seed_images_strict:v1"

def online_available() -> bool:
    if os.getenv("OFFLINE") == "1": return False
    key = os.getenv("OPENAI_API_KEY","").strip()
    if not key: return False
    if key.startswith("sk-proj-") and not os.getenv("OPENAI_PROJECT"):
        return False
    return True

def words_for_letter(conn, ch: str, limit: int) -> List[str]:
    like = (ch.lower() + "%")
    rows = conn.execute(
        "SELECT text FROM words WHERE lang='en' AND text LIKE ? ORDER BY text LIMIT ?",
        (like, limit)
    ).fetchall()
    return [r[0] for r in rows]

def build_caption(mode: str, fr: str|None, en: str|None) -> str:
    """
    We favor English content for image models when we have it.
    If EN gloss exists (en-fr mode), use EN gloss; else use FR sentence directly.
    """
    if mode == "en-fr" and en:
        return en.strip()
    # If fr-fr or no gloss, use the French sentence
    txt = (fr or en or "").strip()
    return txt

def main():
    ap = argparse.ArgumentParser(description="Seed images per sentence (strict matching to sentences).")
    ap.add_argument("--letters", default="abcdefghijklmnopqrstuvwxyz", help="Initial letters to process (e.g. 'abc').")
    ap.add_argument("--per-letter-words", type=int, default=200, help="Max headwords per letter.")
    ap.add_argument("--per-word-sentences", type=int, default=1, help="How many sentences per word to image.")
    ap.add_argument("--style", default=DEFAULT_STYLE, help="Image style.")
    ap.add_argument("--size", default=DEFAULT_SIZE, help="Image size.")
    ap.add_argument("--sleep", type=float, default=0.15, help="Pause between generations.")
    args = ap.parse_args()

    if not online_available():
        print("[seed_images_strict] Offline / missing key. Aborting.")
        return

    conn = db.connect()
    total = 0

    for ch in args.letters:
        if not ch.isalpha(): continue
        batch = words_for_letter(conn, ch, args.per_letter_words)
        print(f"[seed_images_strict] Letter '{ch}': {len(batch)} headwords")

        for w in batch:
            mode, rows = db.pick_sentence_rows_for_word(conn, w, per_word=args.per_word_sentences)
            if not rows:
                # nothing to image yet for this word
                time.sleep(args.sleep)
                continue

            for sid, fr, en in rows:
                if db.has_image_for_sentence(conn, sid):
                    # already imaged this sentence
                    continue

                caption = build_caption(mode or "", fr, en)
                if not caption:
                    continue

                # Build a strict, classroom-safe prompt tied to the chosen sentence
                # (make_image_prompt already adds safety constraints for Photorealistic + kid-friendly)
                prompt = make_image_prompt(caption, args.style)
                path = generate_image(prompt, args.size)
                if path:
                    try:
                        db.save_image_for_sentence(
                            conn, sid,
                            path=path, style=args.style, size=args.size,
                            model=MODEL_TAG, prompt=prompt
                        )
                        print(f"[ok] {w} (sid={sid}, mode={mode}) → {path}")
                        total += 1
                    except Exception as e:
                        print(f"[warn] save failed for sid={sid}: {e}")
                else:
                    print(f"[warn] image gen failed for sid={sid} ({w})")

                time.sleep(args.sleep)

    print(f"[seed_images_strict] Done. Saved {total} images.")

if __name__ == "__main__":
    main()
