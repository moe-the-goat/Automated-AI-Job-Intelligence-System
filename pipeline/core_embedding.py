"""
CORE EMBEDDING MODULE (A3)
--------------------------
Pre-rank jobs by semantic similarity against the candidate's CV before deciding
which ones deserve a full Gemini verdict. The CV embedding is content-hashed
and cached so we only re-embed when cv_text.txt actually changes.

Ranking key (since 2026-05-17):
  weighted_score = similarity * region_weight * trust_weight * role_weight

where region_weight is in [0.50, 1.30] (heavy deweight on India / sanctioned
regions, boost on EU / Americas / Middle East / fully-remote), trust_weight
is 1.25 for `pre_flagged_trusted=True` rows, 1.00 otherwise, and role_weight
is 1.20 for AI/ML / 1.10 for SWE / 1.00 for everything else. The raw
`similarity` column is kept for visibility/debug; `weighted_score` drives the
sort order and the top-N cutoff for AI evaluation.

Public API:
- get_cv_embedding(cv_text, api_key)         -> list[float]
- embed_jobs(rows, api_key)                  -> list[list[float] | None]
- cosine_similarity(a, b)                    -> float
- rank_by_similarity(cv_vec, job_vecs)       -> list[float]
- attach_similarity(combined_jobs, cv_text, api_key, throttle_seconds=0.65)
                                             -> (df, url_to_embedding_dict)
                                                df gets `similarity` AND
                                                `weighted_score` columns and is
                                                sorted by `weighted_score` DESC.
                                                The dict maps job_url -> raw
                                                768-dim embedding vector, so the
                                                caller can run semantic dedup.
- drop_semantic_duplicates(df, embeddings)   -> df with rows ≥0.97 similar to
                                                anything in the 14-day history
                                                cache removed
- update_embedding_history(embeddings)       -> persist these embeddings into
                                                data/embedding_history.json so
                                                the next run can dedup against
                                                them

Heavy SDK + numpy imports are lazy so the parser/renderer tests can import this
module without the full runtime stack installed.
"""
import os
import json
import hashlib
import time
import math
from datetime import datetime, timezone, timedelta

from pipeline.logging_setup import get_logger

logger = get_logger(__name__)


CV_EMBEDDING_CACHE = "data/cv_embedding.json"
EMBED_MODEL = "gemini-embedding-001"     # Gemini embeddings, free tier: 100 RPM / 30K TPM / 1K RPD
# 100 RPM means 1 call per 0.6s minimum. Set to 0.65s for a small safety
# buffer. The previous value of 0.05s (~20 RPS = 1200 RPM) burst past the
# 100 RPM cap on every run and triggered ~60 429 errors per peak. The 1500
# RPM figure in the old comment was wrong (probably the paid-tier limit).
EMBED_THROTTLE_SECONDS = 0.65            # ~92 RPM — just under the 100 RPM free-tier cap
JOB_TEXT_MAX_CHARS = 7000                # include full requirements section, not just HR intro

# Semantic dedup config (2026-05-21). Stores 768-dim job embeddings keyed by
# URL so a job posted again next week under a different URL gets caught by
# cosine similarity even though URL-based dedup misses it.
EMBEDDING_HISTORY_CACHE = "data/embedding_history.json"
SEMANTIC_DEDUP_THRESHOLD = 0.97          # conservative — only catch near-identical jobs
EMBEDDING_HISTORY_TTL_DAYS = 14          # roll off entries older than this on every save


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
        logger.warning("CV embedding cache read failed: %s", e)
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
        logger.warning("CV embedding cache write failed: %s", e)


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
        logger.warning("Embedding call failed: %s", str(e)[:200])
        return None


def get_cv_embedding(cv_text, api_key):
    """Return the CV embedding vector, reading from disk cache when the hash matches."""
    cached = _read_cached_embedding(cv_text)
    if cached:
        logger.info("CV embedding: cache hit.")
        return cached

    if not api_key:
        logger.warning("CV embedding: no API key, returning None.")
        return None

    from google import genai  # lazy: don't require this at import time
    client = genai.Client(api_key=api_key)
    logger.info("CV embedding: regenerating (CV changed or first run).")
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
    """Add `similarity` + `weighted_score` columns; return (df, url->embedding dict).

    `similarity`     — raw cosine sim against the CV embedding (for debugging /
                       email visibility).
    `weighted_score` — similarity * region_weight * trust_weight * role_weight.
                       Used for the top-N cutoff that decides which jobs reach
                       the Cerebras+Groq verdict.

    `api_key` should be the embedding-dedicated key (GEMINI_EMBED_API_KEY in
    production) so the embedding burst (~100 RPM peak) doesn't poison the main
    Gemini key's quota. Falls back to the main GEMINI_API_KEY if unset.

    Returns a 2-tuple: (sorted_dataframe, {job_url: embedding_vector, ...}).
    The embeddings dict is what the caller feeds into drop_semantic_duplicates
    and update_embedding_history. URLs whose embedding call failed (None) are
    omitted from the dict.

    Best-effort: if embedding fails entirely (no API key, no network), every
    row gets similarity=0.0 and weighted_score=0.0, and the embeddings dict
    comes back empty — region/trust/role weights still order the result.
    """
    if combined_jobs.empty:
        return combined_jobs, {}

    from pipeline.region_weighting import compute_combined_weight

    cv_vec = get_cv_embedding(cv_text, api_key)
    if cv_vec is None:
        out = combined_jobs.copy()
        out["similarity"] = 0.0
        weights = [compute_combined_weight(r) for r in out.to_dict("records")]
        out["weighted_score"] = [0.0 for _ in weights]
        return out, {}

    rows = combined_jobs.to_dict("records")
    job_vecs = embed_jobs(rows, api_key, throttle_seconds=throttle_seconds)
    sims = rank_by_similarity(cv_vec, job_vecs)
    weights = [compute_combined_weight(r) for r in rows]
    weighted = [s * w for s, w in zip(sims, weights)]

    # Build url -> embedding mapping (skip rows whose embedding call failed).
    url_to_emb = {}
    for row, vec in zip(rows, job_vecs):
        url = str(row.get("job_url", "")).strip()
        if url and vec is not None:
            url_to_emb[url] = list(vec)

    out = combined_jobs.copy()
    out["similarity"] = sims
    out["weighted_score"] = weighted
    out = out.sort_values("weighted_score", ascending=False).reset_index(drop=True)
    return out, url_to_emb


# ---------------------------------------------------------------------------
# Semantic dedup against a rolling 14-day embedding history (#5)
# ---------------------------------------------------------------------------

def _now_iso():
    return datetime.now(tz=timezone.utc).isoformat()


def load_embedding_history():
    """Return {url: {embedding: [...], added_at: ISO}} from disk, or {} on miss/error."""
    if not os.path.exists(EMBEDDING_HISTORY_CACHE):
        return {}
    try:
        with open(EMBEDDING_HISTORY_CACHE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        logger.warning("Embedding history load failed (starting empty): %s", e)
        return {}


def _prune_history(history, now=None):
    """Drop entries older than EMBEDDING_HISTORY_TTL_DAYS from history (mutates + returns)."""
    if not history:
        return history
    now = now or datetime.now(tz=timezone.utc)
    cutoff = now - timedelta(days=EMBEDDING_HISTORY_TTL_DAYS)
    stale = []
    for url, entry in history.items():
        added_at = entry.get("added_at", "") if isinstance(entry, dict) else ""
        try:
            ts = datetime.fromisoformat(added_at)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
        except Exception:
            stale.append(url)            # unparsable timestamp -> drop defensively
            continue
        if ts < cutoff:
            stale.append(url)
    for url in stale:
        history.pop(url, None)
    return history


def save_embedding_history(history):
    """Write history to disk, pruning stale entries first. Best-effort."""
    try:
        d = os.path.dirname(EMBEDDING_HISTORY_CACHE)
        if d:
            os.makedirs(d, exist_ok=True)
        _prune_history(history)
        with open(EMBEDDING_HISTORY_CACHE, "w", encoding="utf-8") as f:
            json.dump(history, f)
    except Exception as e:
        logger.warning("Embedding history save failed: %s", e)


def drop_semantic_duplicates(df, new_embeddings, threshold=SEMANTIC_DEDUP_THRESHOLD):
    """Remove rows whose embedding is ≥threshold cosine-similar to any history entry.

    `new_embeddings` is the {url: embedding_vector} dict from attach_similarity.
    Only considers the rolling 14-day history (already TTL-pruned on save).
    Returns the filtered dataframe. Rows whose URL isn't in `new_embeddings`
    (e.g. embedding API failure) are kept by default — we don't drop them just
    because we couldn't verify, since URL-dedup already handled exact repeats.
    """
    if df is None or df.empty or not new_embeddings:
        return df

    history = load_embedding_history()
    if not history:
        return df

    # Pre-extract historical vectors as a flat list for the inner loop.
    hist_pairs = []
    for url, entry in history.items():
        if not isinstance(entry, dict):
            continue
        vec = entry.get("embedding")
        if vec:
            hist_pairs.append((url, vec))
    if not hist_pairs:
        return df

    duplicate_urls = set()
    for new_url, new_vec in new_embeddings.items():
        if new_vec is None:
            continue
        for hist_url, hist_vec in hist_pairs:
            if hist_url == new_url:
                continue                # same URL already handled by URL-dedup
            if cosine_similarity(new_vec, hist_vec) >= threshold:
                duplicate_urls.add(new_url)
                break

    if not duplicate_urls:
        return df

    if "job_url" not in df.columns:
        return df
    return df[~df["job_url"].astype(str).isin(duplicate_urls)].reset_index(drop=True)


def update_embedding_history(new_embeddings):
    """Add today's embeddings to disk so future runs can dedup against them.

    Pass in the {url: embedding_vector} dict from attach_similarity, filtered
    down to only URLs you actually want to remember (typically those that
    survived all the way to email rendering).
    """
    if not new_embeddings:
        return
    history = load_embedding_history()
    now_str = _now_iso()
    for url, vec in new_embeddings.items():
        if vec is None:
            continue
        history[url] = {"embedding": list(vec), "added_at": now_str}
    save_embedding_history(history)
