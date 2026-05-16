"""
CORE EMBEDDING MODULE (A3)
--------------------------
Pre-rank jobs by semantic similarity against the candidate's CV before deciding
which ones deserve a full Gemini verdict. The CV embedding is content-hashed
and cached so we only re-embed when cv_text.txt actually changes.

Public API:
- get_cv_embedding(cv_text, api_key)         -> list[float]
- embed_jobs(rows, api_key)                  -> list[list[float] | None]
- cosine_similarity(a, b)                    -> float
- rank_by_similarity(cv_vec, job_vecs)       -> list[float]
- attach_similarity(combined_jobs, cv_text, api_key, throttle_seconds=0.05)
                                             -> dataframe with 'similarity' column

Heavy SDK + numpy imports are lazy so the parser/renderer tests can import this
module without the full runtime stack installed.
"""
import os
import json
import hashlib
import time
import math


CV_EMBEDDING_CACHE = "data/cv_embedding.json"
EMBED_MODEL = "gemini-embedding-001"     # Gemini embeddings, free tier, ~1500 RPM
EMBED_THROTTLE_SECONDS = 0.05            # ~20 RPS — well under the limit
JOB_TEXT_MAX_CHARS = 1000                # truncate long descriptions for embedding


def _cv_hash(text):
    """Stable 16-char hex hash of the CV text content."""
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


def _read_cached_embedding(text):
    """Return cached embedding if its hash matches the current CV; else None."""
    if not os.path.exists(CV_EMBEDDING_CACHE):
        return None
    try:
        with open(CV_EMBEDDING_CACHE, "r", encoding="utf-8") as f:
            cached = json.load(f)
        if cached.get("cv_hash") == _cv_hash(text):
            return cached.get("embedding")
    except Exception as e:
        print(f"CV embedding cache read failed: {e}")
    return None


def _write_cached_embedding(text, embedding):
    """Persist embedding + hash to disk. Best-effort; ignored on failure."""
    try:
        d = os.path.dirname(CV_EMBEDDING_CACHE)
        if d:                                   # skip makedirs when cache is at cwd root
            os.makedirs(d, exist_ok=True)
        with open(CV_EMBEDDING_CACHE, "w", encoding="utf-8") as f:
            json.dump({"cv_hash": _cv_hash(text), "embedding": list(embedding)}, f)
    except Exception as e:
        print(f"CV embedding cache write failed: {e}")


def _embed_text(client, text):
    """Single embed call. Returns list[float] or None on failure."""
    try:
        response = client.models.embed_content(
            model=EMBED_MODEL,
            contents=text,
        )
        # google-genai's response shape: response.embeddings[0].values
        return list(response.embeddings[0].values)
    except Exception as e:
        print(f"Embedding call failed: {str(e)[:200]}")
        return None


def get_cv_embedding(cv_text, api_key):
    """Return the CV embedding vector, reading from disk cache when the hash matches."""
    cached = _read_cached_embedding(cv_text)
    if cached:
        print("CV embedding: cache hit.")
        return cached

    if not api_key:
        print("CV embedding: no API key, returning None.")
        return None

    from google import genai  # lazy: don't require this at import time
    client = genai.Client(api_key=api_key)
    print("CV embedding: regenerating (CV changed or first run).")
    vec = _embed_text(client, cv_text)
    if vec is not None:
        _write_cached_embedding(cv_text, vec)
    return vec


def embed_jobs(rows, api_key, throttle_seconds=EMBED_THROTTLE_SECONDS):
    """Generate one embedding per row using title + truncated description.

    Returns a list aligned with `rows`; entries are None if the call failed
    or the row had no usable text. Throttles gently between calls.
    """
    if not rows or not api_key:
        return [None] * len(rows or [])

    from google import genai
    client = genai.Client(api_key=api_key)
    embeddings = []
    for row in rows:
        title = str(row.get("title", "")).strip()
        description = str(row.get("description", "")).strip()[:JOB_TEXT_MAX_CHARS]
        text = f"{title}\n\n{description}".strip()
        if not text:
            embeddings.append(None)
            continue
        embeddings.append(_embed_text(client, text))
        time.sleep(throttle_seconds)
    return embeddings


def cosine_similarity(a, b):
    """Plain-Python cosine sim. Returns 0.0 for None or zero-norm inputs."""
    if a is None or b is None or len(a) == 0 or len(b) == 0:
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def rank_by_similarity(cv_vec, job_vecs):
    """Cosine sim of cv_vec against each job_vec. Returns list aligned with job_vecs."""
    return [cosine_similarity(cv_vec, jv) for jv in job_vecs]


def attach_similarity(combined_jobs, cv_text, api_key, throttle_seconds=EMBED_THROTTLE_SECONDS):
    """Add a `similarity` column to `combined_jobs` and return it sorted DESC.

    Best-effort: if embedding fails entirely (no API key, no network), every
    row gets similarity=0.0 and the original order is preserved — callers can
    still take the top-N and fall through to full AI eval on everyone.
    """
    if combined_jobs.empty:
        return combined_jobs

    cv_vec = get_cv_embedding(cv_text, api_key)
    if cv_vec is None:
        combined_jobs = combined_jobs.copy()
        combined_jobs["similarity"] = 0.0
        return combined_jobs

    rows = combined_jobs.to_dict("records")
    job_vecs = embed_jobs(rows, api_key, throttle_seconds=throttle_seconds)
    sims = rank_by_similarity(cv_vec, job_vecs)

    out = combined_jobs.copy()
    out["similarity"] = sims
    out = out.sort_values("similarity", ascending=False).reset_index(drop=True)
    return out
