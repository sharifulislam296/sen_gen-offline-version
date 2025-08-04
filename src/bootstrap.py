from __future__ import annotations
import pathlib, urllib.request, csv, io, sys

DATA = pathlib.Path(__file__).resolve().parent.parent / "data"
DATA.mkdir(exist_ok=True)

OXFORD_TXT  = DATA / "oxford_a2.txt"
OXFORD_CSV  = DATA / "oxford5000.csv"
OXFORD_XLSX = DATA / "oxford5000.xlsx"

def _write_from_reader(reader: csv.DictReader) -> None:
    with OXFORD_TXT.open("w", encoding="utf8") as out:
        for row in reader:
            if row.get("cefr", "").upper() in ("A1", "A2"):
                w = row["word"].lower()
                if w.isalpha():
                    out.write(w + "\n")
    print(f"[bootstrap] wrote {OXFORD_TXT}", file=sys.stderr)

def _ensure_oxford() -> None:
    if OXFORD_TXT.exists():
        return

    # 1) local CSV
    if OXFORD_CSV.exists():
        print("[bootstrap] using local oxford5000.csv …", file=sys.stderr)
        with OXFORD_CSV.open(encoding="utf8") as f:
            _write_from_reader(csv.DictReader(f))
        return

    # 2) local XLSX
    if OXFORD_XLSX.exists():
        print("[bootstrap] using local oxford5000.xlsx …", file=sys.stderr)
        import pandas as pd
        df = pd.read_excel(OXFORD_XLSX)
        df.columns = [c.strip().lower() for c in df.columns]
        with OXFORD_TXT.open("w", encoding="utf8") as out:
            for _, r in df.iterrows():
                if str(r["cefr"]).upper() in ("A1", "A2"):
                    w = str(r["word"]).lower()
                    if w.isalpha():
                        out.write(w + "\n")
        return

    # 3) download CSV
    print("[bootstrap] downloading Oxford list …", file=sys.stderr)
    url = (
        "https://raw.githubusercontent.com/taffy-code/oxford-3000-5000/"
        "main/oxford5000.csv"
    )
    with urllib.request.urlopen(url) as resp:
        _write_from_reader(csv.DictReader(io.TextIOWrapper(resp, encoding="utf8")))

_ensure_oxford()
