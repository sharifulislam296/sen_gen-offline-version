from __future__ import annotations

import sys
import time
import pathlib
import argparse
import sqlite3
from typing import Dict, Any, List

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import db
from src.generator import generate_fr_from_english

MODEL_TAG = "seed_en_fr:v1"


def words_needing_en_fr(conn: sqlite3.Connection, first_letter: str, target_pairs: int, per_letter_limit: int) -> List[str]:
    """
    Return up to `per_letter_limit` English headwords whose EN→FR pair count is < target_pairs,
    filtered by initial letter (case-insensitive).
    """
    like = (first_letter.lower() + "%")
    rows = conn.execute(
        """
        SELECT w.text, COALESCE(SUM(CASE WHEN s.mode='en-fr' THEN 1 ELSE 0 END), 0) AS pair_cnt
        FROM words w
        LEFT JOIN sentences s
               ON s.word_id = w.id
        WHERE w.lang='en' AND w.text LIKE ?
        GROUP BY w.id
        HAVING pair_cnt < ?
        ORDER BY w.text
        LIMIT ?
        """,
        (like, target_pairs, per_letter_limit),
    ).fetchall()
    return [r[0] for r in rows]


def _normalize_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure fr_targets is a list[str]. If dicts are present, pull 'lemma'.
    Keep sentences as [{'fr','en'}].
    """
    targets = item.get("fr_targets") or []
    norm_targets: List[str] = []
    for x in targets:
        if isinstance(x, str):
            t = x.strip()
            if t:
                norm_targets.append(t)
        elif isinstance(x, dict):
            t = (x.get("lemma") or "").strip()
            if t:
                norm_targets.append(t)
    item["fr_targets"] = norm_targets

    sent_list = []
    for p in item.get("sentences", []):
        fr = (p.get("fr") or "").strip()
        en = (p.get("en") or "").strip()
        if fr:
            sent_list.append({"fr": fr, "en": en})
    item["sentences"] = sent_list
    return item


def main():
    ap = argparse.ArgumentParser(description="Seed EN→FR lemmas + sentence pairs (A→Z sequentially).")
    ap.add_argument("--letters", default="abcdefghijklmnopqrstuvwxyz",
                    help="Initials to process in order (e.g., 'abc', 'mnop', default: a..z).")
    ap.add_argument("--pairs", type=int, default=2, help="Ensure at least this many EN→FR pairs per word.")
    ap.add_argument("--per-letter-limit", type=int, default=500, help="Max words per initial to process.")
    ap.add_argument("--level", default="A2", choices=["A1","A2","B1"], help="Level tag stored in DB.")
    ap.add_argument("--sleep", type=float, default=0.2, help="Pause between words (seconds).")
    args = ap.parse_args()

    conn = db.connect()

    total_pairs_saved = 0
    for ch in args.letters:
        if not ch.isalpha():
            continue
        print(f"\n[seed_en_fr] === Letter '{ch.lower()}' ===")
        batch = words_needing_en_fr(conn, ch, args.pairs, args.per_letter_limit)
        print(f"[seed_en_fr] {len(batch)} words need EN→FR (< {args.pairs})")

        for w in batch:
            print(f"[debug] EN→FR seeding: {w}")
            try:
                # Generate pairs for this single word; function returns {"items":[...]}
                res = generate_fr_from_english([w], n=args.pairs) or {}
                items = res.get("items", []) if isinstance(res, dict) else []

                saved_for_word = 0
                for it in items:
                    it = _normalize_item(it)
                    db.save_en_fr(conn, w, it, level=args.level, model=MODEL_TAG)
                    saved_for_word += len(it.get("sentences", []))

                print(f"[debug] ✅ {w} → {saved_for_word} EN→FR pairs saved")
                total_pairs_saved += saved_for_word
            except Exception as e:
                print(f"[debug] ❌ {w} failed: {e}")
            time.sleep(args.sleep)

    print(f"\n[seed_en_fr] Done. Total EN→FR pairs saved this run: {total_pairs_saved}")


if __name__ == "__main__":
    main()
