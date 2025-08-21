from __future__ import annotations

import sys
import pathlib
import re
import time
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import db
from src.generator import generate  # uses your existing generator

# -------- config you can tweak ----------
START_LETTER = "u"        # change to v/w/x/y/z for the next batches
LIMIT        = 500        # how many words to take from CSV
LEVEL        = "A2"       # stored into DB
MODEL_TAG    = "seed_small:v1"  # stored into DB
MODE         = "per-word" # 'per-word' returns {word: [sents]}
COUNT        = 2          # sentences per word
CSV_PATH     = ROOT / "data" / "oxford5000.csv"
# ---------------------------------------


def load_a1_a2_words(limit: int, start_letter: str, already_seeded: set[str]) -> list[str]:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Missing file: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    df.columns = [str(c).strip().lower() for c in df.columns]
    word_col  = next((c for c in ("word","headword","lemma","entry","term") if c in df.columns), None)
    level_col = next((c for c in ("level","cefr","band","cefr level") if c in df.columns), None)
    if word_col is None or level_col is None:
        raise KeyError(f"CSV must have word and level columns. Got: {df.columns.tolist()}")

    df[word_col]  = df[word_col].astype(str).str.lower()
    df[level_col] = df[level_col].astype(str).str.upper()

    # A1/A2 only, start from given letter onward; make it deterministic
    df = df[(df[level_col].isin(["A1","A2"])) & (df[word_col].str[0] >= start_letter.lower())]
    df = df.sort_values(by=word_col)

    words, seen = [], set()
    for w in df[word_col].tolist():
        w = re.sub(r"[^a-z]", "", w)
        if not w or w in seen or w in already_seeded:
            continue
        seen.add(w)
        words.append(w)
        if len(words) >= limit:
            break

    print(f"[seed_small] Loaded {len(words)} words from CSV (from '{start_letter}').")
    return words


def get_already_seeded(conn) -> set[str]:
    # Anything that appears in 'words' table (lang='en') counts as seeded
    rows = conn.execute("SELECT text FROM words WHERE lang='en'").fetchall()
    return {r[0] for r in rows}


def main():
    # Always use the DB path from src/db.py (sentences.db)
    conn = db.connect()

    already_seeded = get_already_seeded(conn)
    words = load_a1_a2_words(LIMIT, START_LETTER, already_seeded)

    print(f"[seed_small] Seeding {len(words)} words into DB…")
    for w in words:
        print(f"[debug] trying word: {w}")
        try:
            res = generate([w], n=COUNT, mode=MODE, level=LEVEL)
            # res is either {word:[...]} for 'per-word', or a list for 'mixed'
            if isinstance(res, dict):
                sents = res.get(w, [])
            else:
                sents = list(res)

            # WRITE to DB  ✅
            db.save_en_en(conn, w, sents, level=LEVEL, model=MODEL_TAG)

            print(f"[debug] ✅ {w} → {len(sents)} sentences saved")
        except Exception as e:
            print(f"[debug] ❌ {w} failed: {e}")
        time.sleep(0.2)

    print("[seed_small] Done.")


if __name__ == "__main__":
    main()
