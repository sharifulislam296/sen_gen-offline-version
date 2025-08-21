from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .hashutil import stable_hash

# -------------------- paths --------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_DB_PATH = DATA_DIR / "sentences.db"

# Expose a stable name for external imports / scripts
DB_PATH = DEFAULT_DB_PATH


# ---------------- connection & schema ----------------
def connect(path: str | os.PathLike = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """
    Open a connection to the SQLite DB, ensure schema exists,
    and enable pragmas for reliability/perf.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p))
    conn.row_factory = sqlite3.Row
    with conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
    ensure_schema(conn)
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()

    # Track headwords (we store them once; 'lang' is the headword language)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS words (
          id    INTEGER PRIMARY KEY,
          text  TEXT NOT NULL,
          lang  TEXT NOT NULL CHECK (lang IN ('en','fr')),
          UNIQUE(text, lang)
        );
        """
    )

    # Sentences: supports EN→EN, FR→FR, and EN→FR pairs
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS sentences (
          id           INTEGER PRIMARY KEY,
          word_id      INTEGER NOT NULL,
          lang         TEXT NOT NULL CHECK (lang IN ('en','fr')),
          fr           TEXT,          -- french sentence (if applicable)
          en           TEXT,          -- english sentence or gloss (if applicable)
          level        TEXT,          -- A1/A2/B1
          mode         TEXT,          -- 'en-en','fr-fr','en-fr'
          prompt_hash  TEXT NOT NULL, -- dedupe key we control
          model        TEXT,
          created_at   TEXT DEFAULT (datetime('now')),
          approved     INTEGER DEFAULT 1,
          FOREIGN KEY(word_id) REFERENCES words(id),
          UNIQUE(prompt_hash)         -- keep it simple & portable
        );
        """
    )

    # FR lemmas for EN→FR (e.g., école, scolarité)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS fr_lemmas (
          id       INTEGER PRIMARY KEY,
          word_id  INTEGER NOT NULL,
          lemma    TEXT NOT NULL,
          UNIQUE(word_id, lemma),
          FOREIGN KEY(word_id) REFERENCES words(id)
        );
        """
    )

    # Images (usually one per word; can also link a specific sentence)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS images (
          id          INTEGER PRIMARY KEY,
          word_id     INTEGER,
          sentence_id INTEGER,
          path        TEXT NOT NULL,
          style       TEXT,
          size        TEXT,
          prompt_hash TEXT NOT NULL,  -- dedupe key
          model       TEXT,
          created_at  TEXT DEFAULT (datetime('now')),
          FOREIGN KEY(word_id) REFERENCES words(id),
          FOREIGN KEY(sentence_id) REFERENCES sentences(id),
          UNIQUE(prompt_hash)
        );
        """
    )

    # helpful indexes
    cur.execute("CREATE INDEX IF NOT EXISTS idx_sentences_word ON sentences(word_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_sentences_mode ON sentences(mode);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_images_word   ON images(word_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_fr_lemmas_lemma ON fr_lemmas(lemma);")  # NEW

    conn.commit()


def init_db() -> None:
    """
    Create the database file (if missing) and ensure the schema exists.
    Safe to call multiple times.
    """
    conn = connect(DB_PATH)
    conn.close()


# -------------------- basic helpers --------------------
def upsert_word(conn: sqlite3.Connection, text: str, lang: str) -> int:
    text, lang = text.strip(), lang.strip()
    row = conn.execute(
        "SELECT id FROM words WHERE text=? AND lang=?", (text, lang)
    ).fetchone()
    if row:
        return int(row[0])
    cur = conn.execute("INSERT INTO words(text, lang) VALUES(?,?)", (text, lang))
    return int(cur.lastrowid)


def insert_lemmas(conn: sqlite3.Connection, word_id: int, lemmas: Iterable[str]) -> None:
    with conn:
        for l in lemmas or []:
            l2 = (l or "").strip()
            if l2:
                conn.execute(
                    "INSERT OR IGNORE INTO fr_lemmas(word_id, lemma) VALUES(?,?)",
                    (word_id, l2),
                )


# -------------------- save APIs --------------------
def save_en_en(
    conn: sqlite3.Connection,
    word: str,
    sentences: Sequence[str],
    *,
    level: str,
    model: str,
) -> None:
    wid = upsert_word(conn, word, "en")
    for s in sentences or []:
        s2 = (s or "").strip()
        if not s2:
            continue
        key = stable_hash({"k": "en-en", "w": word, "t": s2, "lvl": level, "m": model})
        with conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO sentences(word_id, lang, fr, en, level, mode, prompt_hash, model)
                VALUES(?, 'en', NULL, ?, ?, 'en-en', ?, ?)
                """,
                (wid, s2, level, key, model),
            )


def save_fr_fr(
    conn: sqlite3.Connection,
    word: str,
    sentences: Sequence[str],
    *,
    level: str,
    model: str,
) -> None:
    # We use the same EN headword row for FR sentences (word list lives in EN)
    wid = upsert_word(conn, word, "en")
    for s in sentences or []:
        s2 = (s or "").strip()
        if not s2:
            continue
        key = stable_hash({"k": "fr-fr", "w": word, "t": s2, "lvl": level, "m": model})
        with conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO sentences(word_id, lang, fr, en, level, mode, prompt_hash, model)
                VALUES(?, 'fr', ?, NULL, ?, 'fr-fr', ?, ?)
                """,
                (wid, s2, level, key, model),
            )


def save_en_fr(
    conn: sqlite3.Connection,
    word: str,
    item: Dict[str, Any],
    *,
    level: str,
    model: str,
) -> None:
    """
    EN→FR item shape:
      {"en_word": "...",
       "fr_targets": ["école", ...],
       "sentences": [{"fr": "...", "en": "..."}, ...]}
    """
    wid = upsert_word(conn, word, "en")
    insert_lemmas(conn, wid, [t for t in item.get("fr_targets", []) if isinstance(t, str)])
    for pair in item.get("sentences", []):
        fr = (pair.get("fr") or "").strip()
        en = (pair.get("en") or "").strip()
        if not fr:
            continue
        key = stable_hash({"k": "en-fr", "w": word, "fr": fr, "en": en, "lvl": level, "m": model})
        with conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO sentences(word_id, lang, fr, en, level, mode, prompt_hash, model)
                VALUES(?, 'fr', ?, ?, ?, 'en-fr', ?, ?)
                """,
                (wid, fr, en or None, level, key, model),
            )


def save_image_for_word(
    conn: sqlite3.Connection,
    word: str,
    *,
    path: str,
    style: str,
    size: str,
    model: str,
    prompt: str,
) -> None:
    wid = upsert_word(conn, word, "en")
    key = stable_hash({"k": "img", "w": word, "p": path, "st": style, "sz": size, "m": model, "pr": prompt})
    with conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO images(word_id, sentence_id, path, style, size, prompt_hash, model)
            VALUES(?, NULL, ?, ?, ?, ?, ?)
            """,
            (wid, path, style, size, key, model),
        )


# -------------------- fetch APIs --------------------
def fetch_en_en(conn: sqlite3.Connection, word: str, n: int) -> List[str]:
    wid = upsert_word(conn, word, "en")
    rows = conn.execute(
        "SELECT en FROM sentences WHERE word_id=? AND mode='en-en' AND approved=1 ORDER BY RANDOM() LIMIT ?",
        (wid, n),
    ).fetchall()
    return [r[0] for r in rows]


def fetch_fr_fr(conn: sqlite3.Connection, word: str, n: int) -> List[str]:
    wid = upsert_word(conn, word, "en")
    rows = conn.execute(
        "SELECT fr FROM sentences WHERE word_id=? AND mode='fr-fr' AND approved=1 ORDER BY RANDOM() LIMIT ?",
        (wid, n),
    ).fetchall()
    return [r[0] for r in rows]


def fetch_en_fr_items(conn: sqlite3.Connection, words: Sequence[str], n: int) -> Dict[str, Any]:
    items: List[Dict[str, Any]] = []
    for w in words:
        wid = upsert_word(conn, w, "en")
        # lemmas
        lrows = conn.execute(
            "SELECT lemma FROM fr_lemmas WHERE word_id=? ORDER BY lemma",
            (wid,),
        ).fetchall()
        lemmas = [r[0] for r in lrows]
        # sentence pairs
        srows = conn.execute(
            "SELECT fr, en FROM sentences WHERE word_id=? AND mode='en-fr' AND approved=1 ORDER BY RANDOM() LIMIT ?",
            (wid, n),
        ).fetchall()
        sentences = [{"fr": r[0], "en": r[1] or ""} for r in srows]
        if sentences:
            items.append(
                {
                    "en_word": w,
                    "fr_targets": lemmas,
                    "sentences": sentences,
                    "needs_clarification": False,
                }
            )
    return {"items": items}


def fetch_image_path(conn: sqlite3.Connection, word: str) -> Optional[str]:
    row = conn.execute(
        "SELECT images.path FROM images JOIN words ON images.word_id = words.id "
        "WHERE words.text=? AND words.lang='en' ORDER BY images.id DESC LIMIT 1",
        (word,),
    ).fetchone()
    return row[0] if row else None


def fetch_fr_by_french_word(conn: sqlite3.Connection, fr_word: str, n: int = 10) -> List[Dict[str, str]]:
    """
    Look up FR sentences by French lemma (from fr_lemmas).
    Returns FR sentences (from both FR→FR and EN→FR), plus the English headword and gloss.
    """
    lemma = (fr_word or "").strip().lower()
    rows = conn.execute(
        """
        SELECT
          s.fr                    AS fr,
          COALESCE(s.en, '')      AS en_gloss,
          w.text                  AS en_head,
          s.mode                  AS mode
        FROM fr_lemmas      fl
        JOIN words          w  ON w.id = fl.word_id
        JOIN sentences      s  ON s.word_id = w.id
        WHERE fl.lemma = ?
          AND s.approved = 1
          AND s.lang = 'fr'
          AND s.mode IN ('fr-fr','en-fr')
        ORDER BY RANDOM()
        LIMIT ?
        """,
        (lemma, n),
    ).fetchall()
    return [
        {"fr": r["fr"], "en": r["en_gloss"], "en_word": r["en_head"], "mode": r["mode"]}
        for r in rows
        if r["fr"]
    ]
