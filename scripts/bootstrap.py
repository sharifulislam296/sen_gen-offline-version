from __future__ import annotations
import pathlib, csv, sys

# Paths
DATA = pathlib.Path(__file__).resolve().parent.parent / "data"
DATA.mkdir(exist_ok=True)

OXFORD_TXT = DATA / "oxford_a2.txt"
OXFORD_CSV = DATA / "oxford5000.csv"

def _write_from_csv() -> None:
    print(f"[bootstrap] reading {OXFORD_CSV}", file=sys.stderr)

    with OXFORD_CSV.open(encoding="utf8") as f:
        reader = csv.DictReader(f)
        count = 0
        with OXFORD_TXT.open("w", encoding="utf8") as out:
            for row in reader:
                level = row.get("level", "").strip().upper()  # ← FIXED HERE
                word = row.get("word", "").strip().lower()
                if level in ("A1", "A2") and word.isalpha():
                    out.write(word + "\n")
                    count += 1

    print(f"[bootstrap] wrote {count} words → {OXFORD_TXT}", file=sys.stderr)

def _ensure_oxford() -> None:
    if OXFORD_TXT.exists():
        print("[bootstrap] A1/A2 list already exists, skipping…", file=sys.stderr)
        return

    if OXFORD_CSV.exists():
        _write_from_csv()
    else:
        print(f"[bootstrap] ERROR: Missing {OXFORD_CSV}", file=sys.stderr)

if __name__ == "__main__":
    _ensure_oxford()
