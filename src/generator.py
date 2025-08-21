from __future__ import annotations
import argparse, base64, hashlib, json, os, re, sys
from pathlib import Path
from typing import Any, Dict, List, Sequence
from enum import Enum

class SentenceMode(str, Enum):
    PER_WORD = "per-word"
    MIXED = "mixed"

class SentenceLevel(str, Enum):
    A1 = "A1"
    A2 = "A2"
    B1 = "B1"

# local deps
import src.bootstrap  # ensures oxford_a2.txt is downloaded
from .filters   import filter_sentences, dedupe, normalize_word_list, ALLOWED_WORDS
from .templates import fallback_sentence, fallback_sentences

# OpenAI client (graceful if SDK absent)
try:
    import openai  # type: ignore
except ImportError:  # pragma: no cover
    openai = None  # type: ignore

# ---------------------- CONSTANTS ---------------------------------
DEFAULT_MODE    = "per-word"
DEFAULT_LEVEL   = "A2"
DEFAULT_MAX_LEN = 15
LEVELS = {
    "A1": {"max_len": 8,  "max_unknown": 1},
    "A2": {"max_len": 12, "max_unknown": 2},
    "B1": {"max_len": 20, "max_unknown": 4},
}

ALLOWED_IMAGE_SIZES = {"1024x1024", "1024x1536", "1536x1024", "auto"}

BASE_DIR = Path(__file__).resolve().parent.parent
IMG_DIR  = BASE_DIR / "assets" / "imgcache"
IMG_DIR.mkdir(parents=True, exist_ok=True)

# ====================== OPENAI HELPER =============================
def _openai_client() -> Any | None:
    if openai is None:
        return None
    # Force offline if set
    if os.getenv("OFFLINE") == "1":
        print("[info] OFFLINE=1 → skipping OpenAI client init", file=sys.stderr)
        return None
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        return None
    project = os.getenv("OPENAI_PROJECT")
    if key.startswith("sk-proj-") and not project:
        print("[error] OPENAI_PROJECT missing for project key.", file=sys.stderr)
        return None
    try:
        return openai.OpenAI(api_key=key, project=project)  # type: ignore[attr-defined]
    except Exception as e:
        print(f"[warn] OpenAI init error: {e}", file=sys.stderr)
        return None

# ====================== IMAGE HELPERS =============================
def _normalize_size(size: str | None) -> str:
    if not size:
        return "1024x1024"
    size = size.strip()
    if size not in ALLOWED_IMAGE_SIZES:
        if size in {"256x256", "512x512"}:
            print(f"[warn] size {size} unsupported; using 1024x1024", file=sys.stderr)
            return "1024x1024"
        print(f"[warn] size {size} unsupported; using auto", file=sys.stderr)
        return "auto"
    return size

def _img_path(prompt: str, size: str) -> Path:
    h = hashlib.sha256((size + "|" + prompt).encode("utf-8")).hexdigest()[:24]
    return IMG_DIR / f"{h}_{size}.png"

def generate_image(prompt: str, size: str = "1024x1024") -> str:
    size = _normalize_size(size)
    path = _img_path(prompt, size)
    if path.exists():
        return str(path)

    client = _openai_client()
    if client is None:
        print("[warn] image generation skipped: no OpenAI client", file=sys.stderr)
        return ""

    try:
        img = client.images.generate(model="gpt-image-1", prompt=prompt, size=size)
        b64 = img.data[0].b64_json
        with open(path, "wb") as f:
            f.write(base64.b64decode(b64))
        return str(path) if path.exists() else ""
    except Exception as e:
        print(f"[warn] image generation failed ({type(e).__name__}): {e}", file=sys.stderr)
        return ""

def make_image_prompt(sentence: str, style: str) -> str:
    s = sentence.strip()
    if style == "Photorealistic":
        return (
            "High-quality photorealistic studio photo, modest clothing, no cleavage, no intimate photo "
            "no shorts above the knee, suitable for classroom use. " + s
        )
    if style == "Watercolor":
        return f"Soft watercolor painting. {s}"
    if style == "Pixel art":
        return f"Retro 16-bit pixel-art sprite. {s}"
    return ("Simple kid-friendly flat illustration, single subject, white background, "
            "no text or logos. Clearly depict: " + s)

def call_llm(prompt: str, max_tokens: int = 400) -> str:
    client = _openai_client()
    if client is None:
        return ""
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=max_tokens,
        )
        return (r.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"[warn] LLM call failed: {e}", file=sys.stderr)
        return ""

def _strip_num(s: str) -> str:
    return re.sub(r"^\d+[.)]\s*", "", s.lstrip("-*•").strip())

def _split_lines(raw: str) -> List[str]:
    lines = []
    for l in (raw or "").splitlines():
        l = _strip_num(l)
        if l and l[-1].isalnum():
            l += "."
        if l:
            lines.append(l)
    return lines

def _prompt_per_word(w: str, n: int, max_len: int, level: str) -> str:
    return (f"You are a patient English tutor.\nWrite {n} DIFFERENT sentences in English.\n"
            f"Each MUST include the word: {w}\nKeep each sentence <= {max_len} words "
            f"(CEFR {level} or easier).\nNo numbering or quotes; one sentence per line.")

def _prompt_mixed(plan: List[List[str]], max_len: int, level: str) -> str:
    req = [f"{i+1}) {', '.join(r)}" for i, r in enumerate(plan)]
    return (f"You are a patient English tutor.\nWrite {len(plan)} short English sentences.\n"
            "Each sentence MUST include the required word(s) shown.\n"
            f"Keep each sentence <= {max_len} words (CEFR {level} or easier).\n"
            "Return exactly the number of sentences requested.\nNo numbering or quotes.\n\n"
            "Required word(s) by line:\n" + "\n".join(req))

def _plan_mixed(tgt: Sequence[str], n: int) -> List[List[str]]:
    plan = [[] for _ in range(n)]
    for i, w in enumerate([t for t in tgt if t]):
        plan[i % n].append(w)
    return plan

def _gen_single(w: str, n: int, max_len: int, max_unknown: int, level: str) -> List[str]:
    raw = call_llm(_prompt_per_word(w, n * 3, max_len, level))
    cand = _split_lines(raw)
    kept = dedupe(filter_sentences(cand, [w], ALLOWED_WORDS, max_len, max_unknown))
    if len(kept) < n:
        kept.extend(fallback_sentences([w], n - len(kept), max_len))
    return kept[:n]

def _gen_mixed(tgt: Sequence[str], n: int, ml: int, mu: int, level: str) -> List[str]:
    plan = _plan_mixed(tgt, n)
    cand = _split_lines(call_llm(_prompt_mixed(plan, ml, level)))
    out: List[str] = []
    for i in range(n):
        line = cand[i] if i < len(cand) else ""
        req = plan[i]
        keep = filter_sentences([line], req, ALLOWED_WORDS, ml, mu)
        out.append(keep[0] if keep else fallback_sentence(req, ml))
    return out

# ------------------ FRENCH → FRENCH ------------------
def generate_french_like_english(words: List[str], n: int = 3) -> Dict[str, List[str]]:
    targ = [w.strip() for w in words if w.strip()]
    out: Dict[str, List[str]] = {}
    for w in targ:
        prompt = ("Tu es un professeur de français patient.\n"
                  f"Écris {n*3} phrases DIFFÉRENTES en français.\n"
                  f"Chaque phrase DOIT inclure le mot : {w}\n"
                  "Niveau A1–A2 ; ≤ 12 mots. Une phrase par ligne.")
        lines = _split_lines(call_llm(prompt)) or [
            f"Je vois le mot « {w} ».","J'utilise le mot « {w} » aujourd'hui.","Nous lisons le mot « {w} » en classe."
        ]
        out[w] = lines[:n]
    return out

# ------------------ ENGLISH → FRENCH ------------------
def generate_fr_from_english(words: List[str], n: int = 3) -> Dict[str, Any]:
    targets = [w.strip() for w in words if w.strip()]
    if not targets:
        return {"items": []}

    client = _openai_client()
    if client is None:
        print("[warn] no OpenAI client — using fallback sentences", file=sys.stderr)
        return _fallback_items(targets, n)

    try:
        raw = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Return strict JSON only."},
                {"role": "user",
                 "content": (
                     f"English headwords: {', '.join(targets)}\n"
                     f"Give 1–2 beginner French lemmas and {n} French sentences with English gloss.\n"
                     "Target difficulty: A2–B1 (still simple, but slightly richer than A1).\n"
                     "Enforce VARIETY across the sentences for each word:\n"
                     " - one uses a location/preposition (à, dans, sous, près de, etc.)\n"
                     " - one uses possession (mon/ma/mes, son/sa/ses, etc.)\n"
                     " - one uses a time expression (hier, demain, ce matin, le soir, souvent)\n"
                     " - one is a negation or a question (ne … pas / ?)\n"
                     " - one uses a connector (mais, parce que, quand)\n"
                     "Keep each sentence ≤ 14 words; natural classroom-safe content.\n"
                     "Return JSON as a single object with keys: en_word, fr_targets, sentences[]."
                 )},
            ],
            temperature=0.6,
            max_tokens=1600,
        ).choices[0].message.content or ""
    except Exception as e:
        print(f"[warn] EN→FR LLM call failed: {e}", file=sys.stderr)
        return _fallback_items(targets, n)

    try:
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.S)
        parsed = json.loads(m.group(1) if m else raw)
        return {"items": [parsed]}  # ✅ wrap always
    except Exception as e:
        print(f"[warn] EN→FR JSON parse failed: {e}", file=sys.stderr)
        return _fallback_items(targets, n)

def _fallback_items(targets: List[str], n: int) -> Dict[str, Any]:
    items = []
    for w in targets:
        sentences = [{"fr": f"Je vois le mot « {w} ».", "en": f"I see the word {w}."}]
        while len(sentences) < n:
            sentences.append({"fr": f"Je répète le mot « {w} ».", "en": f"I repeat the word {w}."})
        items.append({
            "en_word": w,
            "fr_targets": [{"lemma": w, "pos": "expr", "gender": "", "note": "fallback"}],
            "sentences": sentences[:n],
            "needs_clarification": False
        })
    return {"items": items}

# ------------------ MAIN GENERATOR ------------------
def generate(
    words: Sequence[str], n: int = 5, mode: str = DEFAULT_MODE,
    level: str = DEFAULT_LEVEL, max_len: int | None = None, max_unknown: int | None = None
) -> List[str] | Dict[str, List[str]]:
    t = normalize_word_list(words)
    level_u = level.upper()
    params = LEVELS[level_u]
    ml = max_len or (DEFAULT_MAX_LEN if level_u == "A2" else params["max_len"])
    mu = max_unknown or params["max_unknown"]
    if mode == "per-word":
        return {w: _gen_single(w, n, ml, mu, level_u) for w in t}
    return _gen_mixed(t, n, ml, mu, level_u)

# ------------------ CLI ------------------
def _parse(argv: Sequence[str] | None = None):
    p = argparse.ArgumentParser(description="Sentence Generator CLI")
    p.add_argument("--words", nargs="+", required=True)
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--mode", choices=["per-word", "mixed"], default=DEFAULT_MODE)
    p.add_argument("--level", choices=["A1", "A2", "B1"], default=DEFAULT_LEVEL)
    p.add_argument("--max-len", type=int)
    p.add_argument("--max-unknown", type=int)
    return p.parse_args(argv)

def main(argv: Sequence[str] | None = None):
    a = _parse(argv)
    res = generate(a.words, a.n, a.mode, a.level, a.max_len, a.max_unknown)
    if isinstance(res, dict):
        for w, s in res.items():
            print(f"# {w}")
            [print(x) for x in s]
            print()
    else:
        [print(x) for x in res]

if __name__ == "__main__":
    main()
