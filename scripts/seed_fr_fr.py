from __future__ import annotations

import sys
import pathlib
import time
import argparse
import sqlite3

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import db
from src.generator import generate_french_like_english

MODEL_TAG = "seed_fr_fr:v1"   # stored in DB


def pick_words_needing_fr(conn: sqlite3.Connection, start_letter: str, limit: int, target_count: int) -> list[str]:
    """
    Return up to `limit` English headwords whose FR→FR sentence count is < target_count,
    filtered by first letter if provided.
    """
    like = (start_letter.lower() + "%") if start_letter else "%"
    rows = conn.execute(
        """
        SELECT w.text, COALESCE(COUNT(s.id), 0) AS fr_cnt
        FROM words w
        LEFT JOIN sentences s
          ON s.word_id = w.id
         AND s.mode    = 'fr-fr'
        WHERE w.lang='en' AND w.text LIKE ?
        GROUP BY w.id
        HAVING fr_cnt < ?
        ORDER BY w.text
        LIMIT ?
        """,
        (like, target_count, limit)
    ).fetchall()
    return [r[0] for r in rows]


def main():
    ap = argparse.ArgumentParser(description="Seed FR→FR sentences into DB")
    ap.add_argument("--start", default="a", help="first letter filter (e.g., a, m, u). Use '' for all.")
    ap.add_argument("--limit", type=int, default=200, help="max words this run")
    ap.add_argument("--count", type=int, default=2, help="sentences per word to ensure")
    ap.add_argument("--level", default="A2", choices=["A1","A2","B1"], help="stored level tag")
    ap.add_argument("--sleep", type=float, default=0.2, help="pause between words (seconds)")
    args = ap.parse_args()

    conn = db.connect()

    words = pick_words_needing_fr(conn, args.start, args.limit, args.count)
    print(f"[seed_fr_fr] {len(words)} words need FR→FR (< {args.count}) starting at '{args.start or '*'}'.")

    for w in words:
        print(f"[debug] FR seeding: {w}")
        try:
            # Generate n French sentences for this English headword
            res = generate_french_like_english([w], n=args.count)
            sents = res.get(w, [])  # list[str]
            if not sents:
                print(f"[debug] ⚠️ no FR sentences returned for {w}; skipping")
                continue

            # Persist
            db.save_fr_fr(conn, w, sents, level=args.level, model=MODEL_TAG)
            print(f"[debug] ✅ {w} → {len(sents)} FR sentences saved")
        except Exception as e:
            print(f"[debug] ❌ {w} failed: {e}")
        time.sleep(args.sleep)

    print("[seed_fr_fr] Done.")


if __name__ == "__main__":
    main()
