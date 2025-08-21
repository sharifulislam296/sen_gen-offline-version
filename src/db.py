from __future__ import annotations

import os
import sqlite3
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .hashutil import stable_hash

# -------------------- paths --------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_DB_PATH = DATA_DIR / "sentences.db"

DB_PATH = DEFAULT_DB_PATH  # stable import name


# -------------------- normalization --------------------
def _lemma_norm(s: str) -> str:
    if not s:
        return ""
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch)).lower().strip()


# ---------------- connection & schema ----------------
def connect(path: str | os.PathLike = DEFAULT_DB_PATH) -> sqlite3.Connection:
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

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS sentences (
          id           INTEGER PRIMARY KEY,
          word_id      INTEGER NOT NULL,
          lang         TEXT NOT NULL CHECK (lang IN ('en','fr')),
          fr           TEXT,
          en           TEXT,
          level        TEXT,
          mode         TEXT CHECK (mode IN ('en-en','fr-fr','en-fr')),
          prompt_hash  TEXT NOT NULL,
          model        TEXT,
          created_at   TEXT DEFAULT (datetime('now')),
          approved     INTEGER DEFAULT 1,
          FOREIGN KEY(word_id) REFERENCES words(id),
          UNIQUE(prompt_hash)
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS fr_lemmas (
          id         INTEGER PRIMARY KEY,
          word_id    INTEGER NOT NULL,
          lemma      TEXT NOT NULL,
          UNIQUE(word_id, lemma),
          FOREIGN KEY(word_id) REFERENCES words(id)
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS images (
          id          INTEGER PRIMARY KEY,
          word_id     INTEGER,
          sentence_id INTEGER,
          path        TEXT NOT NULL,
          style       TEXT,
          size        TEXT,
          prompt_hash TEXT NOT NULL,
          model       TEXT,
          created_at  TEXT DEFAULT (datetime('now')),
          FOREIGN KEY(word_id) REFERENCES words(id),
          FOREIGN KEY(sentence_id) REFERENCES sentences(id),
          UNIQUE(prompt_hash)
        );
        """
    )

    # Deterministic packs
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS featured_sentences (
          id          INTEGER PRIMARY KEY,
          word_id     INTEGER NOT NULL,
          mode        TEXT NOT NULL CHECK (mode IN ('en-en','fr-fr','en-fr')),
          sentence_id INTEGER NOT NULL UNIQUE,
          slot        INTEGER NOT NULL,
          pack        TEXT NOT NULL DEFAULT 'main',
          FOREIGN KEY(word_id) REFERENCES words(id),
          FOREIGN KEY(sentence_id) REFERENCES sentences(id),
          UNIQUE(word_id, mode, pack, slot)
        );
        """
    )

    # helpful indexes
    cur.execute("CREATE INDEX IF NOT EXISTS idx_sentences_word ON sentences(word_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_sentences_mode ON sentences(mode);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_images_word ON images(word_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_images_sentence ON images(sentence_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_fr_lemmas_lemma ON fr_lemmas(lemma);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_feat_word_mode ON featured_sentences(word_id, mode);")

    # migration: lemma_norm for accent-insensitive search
    ti = conn.execute("PRAGMA table_info(fr_lemmas)").fetchall()
    cols = [r[1] for r in ti]
    if "lemma_norm" not in cols:
        conn.execute("ALTER TABLE fr_lemmas ADD COLUMN lemma_norm TEXT")
        rows = conn.execute("SELECT id, lemma FROM fr_lemmas").fetchall()
        for r in rows:
            ln = _lemma_norm(r["lemma"] if isinstance(r, sqlite3.Row) else r[1])
            conn.execute("UPDATE fr_lemmas SET lemma_norm=? WHERE id=?", (ln, r["id"] if isinstance(r, sqlite3.Row) else r[0]))
        conn.execute("CREATE INDEX IF NOT EXISTS idx_fr_lemmas_lemma_norm ON fr_lemmas(lemma_norm)")

    conn.commit()


def init_db() -> None:
    conn = connect(DB_PATH)
    conn.close()


# -------------------- basic helpers --------------------
def upsert_word(conn: sqlite3.Connection, text: str, lang: str) -> int:
    text, lang = text.strip(), lang.strip()
    row = conn.execute("SELECT id FROM words WHERE text=? AND lang=?", (text, lang)).fetchone()
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
                    "INSERT OR IGNORE INTO fr_lemmas(word_id, lemma, lemma_norm) VALUES(?,?,?)",
                    (word_id, l2, _lemma_norm(l2)),
                )


# -------------------- save APIs --------------------
def save_en_en(conn: sqlite3.Connection, word: str, sentences: Sequence[str], *, level: str, model: str) -> None:
    wid = upsert_word(conn, word, "en")
    for s in sentences or []:
        s2 = (s or "").strip()
        if not s2:
            continue
        key = stable_hash({"k": "en-en", "w": word, "t": s2, "lvl": level, "m": model})
        with conn:
            conn.execute(
                "INSERT OR IGNORE INTO sentences(word_id, lang, fr, en, level, mode, prompt_hash, model)"
                " VALUES(?, 'en', NULL, ?, ?, 'en-en', ?, ?)",
                (wid, s2, level, key, model),
            )


def save_fr_fr(conn: sqlite3.Connection, word: str, sentences: Sequence[str], *, level: str, model: str) -> None:
    wid = upsert_word(conn, word, "en")
    for s in sentences or []:
        s2 = (s or "").strip()
        if not s2:
            continue
        key = stable_hash({"k": "fr-fr", "w": word, "t": s2, "lvl": level, "m": model})
        with conn:
            conn.execute(
                "INSERT OR IGNORE INTO sentences(word_id, lang, fr, en, level, mode, prompt_hash, model)"
                " VALUES(?, 'fr', ?, NULL, ?, 'fr-fr', ?, ?)",
                (wid, s2, level, key, model),
            )


def save_en_fr(
    conn: sqlite3.Connection,
    word: str,
    item: Dict[str, Any] | List[Any],
    *,
    level: str,
    model: str,
) -> None:
    """
    Accepts multiple shapes:
    - {"en_word": "school", "fr_targets": [...], "sentences": [ {"fr":"...","en":"..."}, "just fr string", ... ]}
    - Or a bare list of sentences: ["fr only", {"fr":"...","en":"..."}]
    """
    wid = upsert_word(conn, word, "en")

    # Lemmas (when provided)
    if isinstance(item, dict):
        insert_lemmas(conn, wid, [t for t in item.get("fr_targets", []) if isinstance(t, str)])

    # Normalize to a list of {"fr":..., "en":...}
    if isinstance(item, dict):
        src = item.get("sentences", [])
    elif isinstance(item, list):
        src = item
    else:
        src = []

    norm_pairs: List[Dict[str, str]] = []
    for pair in src or []:
        if isinstance(pair, dict):
            fr = (pair.get("fr") or pair.get("fr_text") or "").strip()
            en = (pair.get("en") or pair.get("en_text") or "").strip()
        elif isinstance(pair, str):
            fr, en = pair.strip(), ""
        else:
            continue
        if fr:
            norm_pairs.append({"fr": fr, "en": en})

    for p in norm_pairs:
        key = stable_hash({"k": "en-fr", "w": word, "fr": p["fr"], "en": p["en"], "lvl": level, "m": model})
        with conn:
            conn.execute(
                "INSERT OR IGNORE INTO sentences(word_id, lang, fr, en, level, mode, prompt_hash, model)"
                " VALUES(?, 'fr', ?, ?, ?, 'en-fr', ?, ?)",
                (wid, p["fr"], p["en"] or None, level, key, model),
            )


def save_image_for_word(conn: sqlite3.Connection, word: str, *, path: str, style: str, size: str, model: str, prompt: str) -> None:
    wid = upsert_word(conn, word, "en")
    key = stable_hash({"k": "img", "w": word, "p": path, "st": style, "sz": size, "m": model, "pr": prompt})
    with conn:
        conn.execute(
            "INSERT OR IGNORE INTO images(word_id, sentence_id, path, style, size, prompt_hash, model)"
            " VALUES(?, NULL, ?, ?, ?, ?, ?)",
            (wid, path, style, size, key, model),
        )


# ---------- sentence-level image helpers ----------
def has_image_for_sentence(conn: sqlite3.Connection, sentence_id: int) -> bool:
    row = conn.execute("SELECT 1 FROM images WHERE sentence_id=? LIMIT 1", (sentence_id,)).fetchone()
    return bool(row)


def save_image_for_sentence(
    conn: sqlite3.Connection,
    sentence_id: int,
    *,
    path: str,
    style: str,
    size: str,
    model: str,
    prompt: str,
) -> None:
    if has_image_for_sentence(conn, sentence_id):
        return
    row = conn.execute("SELECT word_id FROM sentences WHERE id=?", (sentence_id,)).fetchone()
    if not row:
        return
    word_id = int(row[0])
    key = stable_hash({"k": "img_sent", "sid": sentence_id, "p": path, "st": style, "sz": size, "m": model, "pr": prompt})
    with conn:
        conn.execute(
            "INSERT OR IGNORE INTO images(word_id, sentence_id, path, style, size, prompt_hash, model)"
            " VALUES(?, ?, ?, ?, ?, ?, ?)",
            (word_id, sentence_id, path, style, size, key, model),
        )


# -------------------- fetch APIs --------------------
def fetch_en_en(conn: sqlite3.Connection, word: str, n: int) -> List[str]:
    wid = upsert_word(conn, word, "en")
    rows = conn.execute(
        """
        SELECT s.en, MAX(i.id) AS has_img
        FROM sentences s
        LEFT JOIN images i ON i.sentence_id = s.id
        WHERE s.word_id=? AND s.mode='en-en' AND s.approved=1
        GROUP BY s.id
        ORDER BY (has_img IS NULL) ASC, s.id ASC
        LIMIT ?
        """,
        (wid, n),
    ).fetchall()
    return [r[0] for r in rows if r[0]]


def fetch_fr_fr(conn: sqlite3.Connection, word: str, n: int) -> List[str]:
    wid = upsert_word(conn, word, "en")
    rows = conn.execute(
        """
        SELECT s.fr, MAX(i.id) AS has_img
        FROM sentences s
        LEFT JOIN images i ON i.sentence_id = s.id
        WHERE s.word_id=? AND s.mode='fr-fr' AND s.approved=1
        GROUP BY s.id
        ORDER BY (has_img IS NULL) ASC, s.id ASC
        LIMIT ?
        """,
        (wid, n),
    ).fetchall()
    return [r[0] for r in rows if r[0]]


def fetch_en_fr_items(conn: sqlite3.Connection, words: Sequence[str], n: int) -> Dict[str, Any]:
    items: List[Dict[str, Any]] = []
    for w in words:
        wid = upsert_word(conn, w, "en")
        lrows = conn.execute("SELECT lemma FROM fr_lemmas WHERE word_id=? ORDER BY lemma", (wid,)).fetchall()
        lemmas = [r[0] for r in lrows]
        srows = conn.execute(
            "SELECT fr, en FROM sentences WHERE word_id=? AND mode='en-fr' AND approved=1 ORDER BY id ASC LIMIT ?",
            (wid, n),
        ).fetchall()
        sentences = [{"fr": r[0], "en": r[1] or ""} for r in srows if r[0]]
        if sentences:
            items.append({"en_word": w, "fr_targets": lemmas, "sentences": sentences, "needs_clarification": False})
    return {"items": items}


def fetch_image_path(conn: sqlite3.Connection, word: str) -> Optional[str]:
    # ONLY true word-level images to avoid cross-sentence reuse
    row = conn.execute(
        "SELECT i.path FROM images i JOIN words w ON i.word_id = w.id "
        "WHERE w.text=? AND w.lang='en' AND i.sentence_id IS NULL "
        "ORDER BY i.id DESC LIMIT 1",
        (word,),
    ).fetchone()
    return row[0] if row else None


def fetch_image_path_for_sentence_id(conn: sqlite3.Connection, sentence_id: int) -> Optional[str]:
    row = conn.execute("SELECT path FROM images WHERE sentence_id=? ORDER BY id DESC LIMIT 1", (sentence_id,)).fetchone()
    return row[0] if row else None


def _canon_text(s: Optional[str]) -> str:
    return (s or "").strip()


def _find_sentence_id_by_text(
    conn: sqlite3.Connection,
    word: str,
    *,
    fr: Optional[str] = None,
    en: Optional[str] = None,
    mode: Optional[str] = None,
) -> Optional[int]:
    wid = upsert_word(conn, word, "en")
    fr_c, en_c = _canon_text(fr), _canon_text(en)

    clauses = ["word_id=?"]
    params: List[Any] = [wid]
    if fr is not None:
        clauses.append("TRIM(COALESCE(fr,'')) = ?")
        params.append(fr_c)
    if en is not None:
        clauses.append("TRIM(COALESCE(en,'')) = ?")
        params.append(en_c)
    if mode is not None:
        clauses.append("mode=?")
        params.append(mode)

    row = conn.execute(
        "SELECT id FROM sentences WHERE "
        + " AND ".join(clauses)
        + " AND approved=1 ORDER BY id DESC LIMIT 1",
        params,
    ).fetchone()
    return int(row[0]) if row else None


def fetch_image_path_for_sentence(
    conn: sqlite3.Connection, word: str, *, fr: Optional[str] = None, en: Optional[str] = None, mode: Optional[str] = None
) -> Optional[str]:
    sid = _find_sentence_id_by_text(conn, word, fr=fr, en=en, mode=mode)
    if not sid:
        return None
    return fetch_image_path_for_sentence_id(conn, sid)


def fetch_image_path_for_sentence_or_word(
    conn: sqlite3.Connection, word: str, *, fr: Optional[str] = None, en: Optional[str] = None, mode: Optional[str] = None
) -> Optional[str]:
    p = fetch_image_path_for_sentence(conn, word, fr=fr, en=en, mode=mode)
    if p:
        return p
    return fetch_image_path(conn, word)


def fetch_fr_by_french_word(conn: sqlite3.Connection, fr_word: str, n: int = 10) -> List[Dict[str, str]]:
    key = _lemma_norm(fr_word)
    rows = conn.execute(
        """
        SELECT s.fr AS fr, COALESCE(s.en, '') AS en_gloss, w.text AS en_head, s.mode AS mode
        FROM fr_lemmas fl
        JOIN words w ON w.id = fl.word_id
        JOIN sentences s ON s.word_id = w.id
        WHERE fl.lemma_norm = ?
          AND s.approved = 1 AND s.lang = 'fr'
          AND s.mode IN ('fr-fr','en-fr')
        ORDER BY s.id ASC
        LIMIT ?
        """,
        (key, n),
    ).fetchall()
    return [{"fr": r["fr"], "en": r["en_gloss"], "en_word": r["en_head"], "mode": r["mode"]} for r in rows if r["fr"]]


# ---------- deterministic packs ----------
def set_pack_for_word_mode(conn: sqlite3.Connection, word: str, mode: str, sentence_ids: Sequence[int], *, pack: str = "main") -> None:
    wid = upsert_word(conn, word, "en")
    with conn:
        conn.execute("DELETE FROM featured_sentences WHERE word_id=? AND mode=? AND pack=?", (wid, mode, pack))
        for idx, sid in enumerate(sentence_ids, start=1):
            conn.execute(
                "INSERT OR REPLACE INTO featured_sentences(word_id, mode, sentence_id, slot, pack)"
                " VALUES(?,?,?,?,?)",
                (wid, mode, int(sid), idx, pack),
            )


def fetch_pack_sentences(conn: sqlite3.Connection, word: str, mode: str, *, pack: str = "main") -> List[sqlite3.Row]:
    wid = upsert_word(conn, word, "en")
    rows = conn.execute(
        """
        SELECT fs.slot, s.id AS sentence_id, s.fr, s.en
        FROM featured_sentences fs
        JOIN sentences s ON s.id = fs.sentence_id
        WHERE fs.word_id=? AND fs.mode=? AND fs.pack=?
        ORDER BY fs.slot ASC
        """,
        (wid, mode, pack),
    ).fetchall()
    return list(rows)


def count_pack_coverage(conn: sqlite3.Connection, word: str, *, pack: str = "main") -> Dict[str, int]:
    wid = upsert_word(conn, word, "en")
    out = {}
    for md in ("en-en", "fr-fr", "en-fr"):
        out[md] = conn.execute(
            "SELECT COUNT(*) FROM featured_sentences WHERE word_id=? AND mode=? AND pack=?",
            (wid, md, pack),
        ).fetchone()[0]
    return out
