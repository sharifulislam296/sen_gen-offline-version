# scripts/pack_build.py
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List

# --- Make project root importable no matter where we run this file from ---
ROOT = Path(__file__).resolve().parents[1]  # repo root (…/sentence_gen)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Now we can import from src/ reliably
try:
    from src import db as cachedb
    from src.generator import (
        generate,                      # not used here but handy if you extend
        generate_french_like_english,  # not used here but handy if you extend
        generate_fr_from_english,      # not used here but handy if you extend
        generate_image,
        make_image_prompt,
    )
except Exception as e:
    raise SystemExit(
        f"Failed to import project modules from {ROOT}.\n"
        f"Error: {e}\n"
        "Tip: ensure this file lives in <repo>/scripts/ and the code in <repo>/src/."
    )


def _norm_size(sz: str) -> str:
    return "1024x1024" if sz == "auto" else sz


def ensure_sentence_image(conn, sentence_id: int, scene: str, *, style="Photorealistic", size="1024x1024"):
    """Generate & save exactly-one image for this sentence if missing."""
    if cachedb.fetch_image_path_for_sentence_id(conn, sentence_id):
        return
    size = _norm_size(size)
    prompt = make_image_prompt(scene, style)
    path = None
    try:
        # support both keyword and positional signatures
        path = generate_image(prompt=prompt, size=size, style=style)
    except TypeError:
        try:
            path = generate_image(prompt, size, style)
        except TypeError:
            path = generate_image(prompt, size)
    if path:
        cachedb.save_image_for_sentence(
            conn, sentence_id,
            path=Path(path).as_posix(), style=style, size=size,
            model="gpt-image-1", prompt=scene
        )


def pick_candidates(conn, word: str, mode: str, k: int) -> List[int]:
    """
    Choose up to k sentences for (word, mode), preferring ones that already have images,
    then oldest first for stability.
    """
    wid = cachedb.upsert_word(conn, word, "en")
    rows = conn.execute(
        """
        SELECT s.id, COALESCE(MAX(i.id),0) AS has_img
        FROM sentences s
        LEFT JOIN images i ON i.sentence_id = s.id
        WHERE s.word_id=? AND s.mode=? AND s.approved=1
        GROUP BY s.id
        ORDER BY has_img DESC, s.id ASC
        LIMIT ?
        """,
        (wid, mode, k),
    ).fetchall()
    return [int(r[0]) for r in rows]


def build_pack(words: List[str], k: int = 5, pack: str = "main", *, style="Photorealistic", size="1024x1024"):
    """
    Lock a deterministic pack: for each (word, mode) select K sentence_ids (stable order),
    ensure each has exactly one sentence-level image, and write them into featured_sentences.
    """
    if not words:
        print("No words provided.", file=sys.stderr)
        return
    conn = cachedb.connect()
    try:
        for w in words:
            print(f"== {w} ==")
            for mode in ("en-en", "fr-fr", "en-fr"):
                sids = pick_candidates(conn, w, mode, k)
                if not sids:
                    print(f"  [{mode}] 0 sentences available — seed/generate text first.")
                    continue

                # Save deterministic pack order
                cachedb.set_pack_for_word_mode(conn, w, mode, sids, pack=pack)
                print(f"  [{mode}] locked {len(sids)} sentences into pack '{pack}'")

                # Ensure sentence-level images; generate missing
                for sid in sids:
                    row = conn.execute("SELECT fr, en FROM sentences WHERE id=?", (sid,)).fetchone()
                    scene = (row["fr"] or row["en"] or "").strip()
                    ensure_sentence_image(conn, sid, scene, style=style, size=size)

        print("\nPack built successfully.")
    finally:
        try:
            conn.close()
        except Exception:
            pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a deterministic 'featured' pack for specific words.")
    p.add_argument("--pack", default="main", help="Pack name to write to (default: main)")
    p.add_argument("--k", type=int, default=5, help="Sentences per word per mode (default: 5)")
    p.add_argument("--style", default="Photorealistic", help="Image style (default: Photorealistic)")
    p.add_argument("--size", default="1024x1024", help="Image size or 'auto' (default: 1024x1024)")
    p.add_argument("words", nargs="*", help="Words to include in the pack")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Default demo list if none provided
    words = args.words or ["school","family","teacher","book","friend","classroom","homework","play","lunch","bus"]
    build_pack(words, k=args.k, pack=args.pack, style=args.style, size=args.size)
