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

# Feedback RAG (2026-05-25). Runs on a dedicated Gemini project so its quota
# doesn't compete with the CV/job pre-rank above. Model is Gemini Embedding 2
# (8K input, 3072 dims) for stronger retrieval than the v1 used for CV ranking.
# The actual switch from digest → RAG happens in core_feedback once the log
# crosses RAG_FEEDBACK_THRESHOLD entries.
FEEDBACK_EMBED_MODEL = "gemini-embedding-2"
FEEDBACK_JOB_TEXT_MAX_CHARS = 1500       # short context window — title + intro is enough for retrieval

# Semantic dedup config (2026-05-21). Stores 768-dim job embeddings keyed by
# URL so a job posted again next week under a different URL gets caught by
# cosine similarity even though URL-based dedup misses it.
EMBEDDING_HISTORY_CACHE = "data/embedding_history.json"
SEMANTIC_DEDUP_THRESHOLD = 0.97          # conservative — only catch near-identical jobs
EMBEDDING_HISTORY_TTL_DAYS = 14          # roll off entries older than this on every save


def _cv_hash(text):
    """Stable 16-char hex hash of the CV text content."""
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


# Cap on how many distinct CV embeddings we keep cached. The cache is a map
# keyed by CV-content hash, so each user's CV is its own entry — multi-user runs
# no longer evict each other (the old single-slot file made every user re-embed,
# defeating the cache). 50 covers a comfortably large beta; the oldest entries
# are dropped when the cap is exceeded.
CV_EMBEDDING_CACHE_MAX_ENTRIES = 50


def _read_cache_map():
    """Load the {cv_hash: {"embedding": [...], "ts": float}} cache map.

    Tolerates the LEGACY single-entry shape ({"cv_hash", "embedding"}) by
    migrating it into the map form on read, so an existing cache file isn't lost.
    Returns {} on any failure.
    """
    if not os.path.exists(CV_EMBEDDING_CACHE):
        return {}
    try:
        with open(CV_EMBEDDING_CACHE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.warning("CV embedding cache read failed: %s", e)
        return {}
    if not isinstance(data, dict):
        return {}
    # Legacy single-entry file → wrap it into the keyed map.
    if "cv_hash" in data and "embedding" in data:
        return {data["cv_hash"]: {"embedding": data["embedding"], "ts": 0}}
    return data


def _read_cached_embedding(text):
    """Return this CV's cached embedding (keyed by content hash); else None."""
    entry = _read_cache_map().get(_cv_hash(text))
    if isinstance(entry, dict):
        return entry.get("embedding")
    return None


def _write_cached_embedding(text, embedding):
    """Persist this CV's embedding into the keyed cache map. Best-effort.

    Per-user safe: writing one CV's vector never evicts another's, unlike the
    old single-slot file. Caps the map size, dropping the oldest entries first.
    """
    try:
        cache = _read_cache_map()
        cache[_cv_hash(text)] = {"embedding": list(embedding), "ts": time.time()}
        if len(cache) > CV_EMBEDDING_CACHE_MAX_ENTRIES:
            # Drop oldest by timestamp until back under the cap.
            for h, _ in sorted(cache.items(), key=lambda kv: kv[1].get("ts", 0))[
                : len(cache) - CV_EMBEDDING_CACHE_MAX_ENTRIES
            ]:
                cache.pop(h, None)
        d = os.path.dirname(CV_EMBEDDING_CACHE)
        if d:                                   # skip makedirs when cache is at cwd root
            os.makedirs(d, exist_ok=True)
        with open(CV_EMBEDDING_CACHE, "w", encoding="utf-8") as f:
            json.dump(cache, f)
    except Exception as e:
        logger.warning("CV embedding cache write failed: %s", e)


def _embed_text(client, text, model=EMBED_MODEL):
    """Single embed call. Returns list[float] or None on failure.

    `model` defaults to the CV/job pre-rank model (Gemini Embedding 1). Pass
    FEEDBACK_EMBED_MODEL when computing vectors that need to live in the
    feedback-retrieval space (different model = different vector space —
    you cannot compare across them).
    """
    try:
        response = client.models.embed_content(
            model=model,
            contents=text,
        )
        # Tally usage (best-effort; never let tracking break an embed call).
        try:
            from pipeline.core_llm_usage import get_tracker, extract_tokens
            get_tracker().record("Gemini", model, ok=True, tokens=extract_tokens(response))
        except Exception:
            pass
        # google-genai's response shape: response.embeddings[0].values
        return list(response.embeddings[0].values)
    except Exception as e:
        try:
            from pipeline.core_llm_usage import get_tracker
            get_tracker().record("Gemini", model, ok=False)
        except Exception:
            pass
        logger.warning("Embedding call failed (model=%s): %s", model, str(e)[:200])
        return None


def _embed_keys(api_key):
    """Normalize the embedding key input to a list of accounts.

    `api_key` may be a single key or a comma-separated list (multiple Gemini
    embedding accounts). Embeddings are the daily-RPD bottleneck, so rotating
    across accounts is the main scaling lever here."""
    if not api_key:
        return []
    if isinstance(api_key, (list, tuple)):
        items = api_key
    else:
        items = str(api_key).split(",")
    return [k.strip() for k in items if k and k.strip()]


def get_cv_embedding(cv_text, api_key):
    """Return the CV embedding vector, reading from disk cache when the hash matches."""
    cached = _read_cached_embedding(cv_text)
    if cached:
        logger.info("CV embedding: cache hit.")
        return cached

    keys = _embed_keys(api_key)
    if not keys:
        logger.warning("CV embedding: no API key, returning None.")
        return None

    from google import genai  # lazy: don't require this at import time
    # One CV embed per run (and it's cached) — account choice is immaterial; use
    # the first. The high-volume job embeds below are what rotate across accounts.
    client = genai.Client(api_key=keys[0])
    logger.info("CV embedding: regenerating (CV changed or first run).")
    vec = _embed_text(client, cv_text)
    if vec is not None:
        _write_cached_embedding(cv_text, vec)
    return vec


def embed_jobs(rows, api_key, throttle_seconds=EMBED_THROTTLE_SECONDS, cache=None):
    """Generate one embedding per row using title + truncated description.

    Returns a list aligned with `rows`; entries are None if the call failed
    or the row had no usable text. Throttles gently between calls.

    `cache` is an optional {text_hash: embedding} dict shared across users in a
    tick. A job's embedding is a pure function of its text, so when the SAME job
    is seen by several users in one tick (the global scrape is shared via the API
    cache) it's embedded ONCE and reused — cutting embedding RPD and run time,
    which is the binding constraint as the user count grows. A cache hit also
    skips the throttle sleep (no API call was made). cache=None preserves the old
    per-call behavior exactly.
    """
    keys = _embed_keys(api_key)
    if not rows or not keys:
        return [None] * len(rows or [])

    # Round-robin across embedding accounts: each account is hit every Nth call,
    # so its effective rate is 1/N — meaning we can pace N× faster while keeping
    # each account under its RPM. N accounts therefore add BOTH daily-RPD ceiling
    # (the real bottleneck) AND speed. One account reproduces the old behavior.
    n_accounts = len(keys)
    per_call_pace = throttle_seconds / n_accounts
    clients = {}  # api_key -> genai.Client, created lazily on first use

    def _client_for(k):
        c = clients.get(k)
        if c is None:
            from google import genai
            c = genai.Client(api_key=k)
            clients[k] = c
        return c

    embeddings = []
    rot = 0  # local rotation index across the accounts
    for row in rows:
        title = str(row.get("title", "")).strip()
        description = str(row.get("description", "")).strip()[:JOB_TEXT_MAX_CHARS]
        text = f"{title}\n\n{description}".strip()
        if not text:
            embeddings.append(None)
            continue

        ck = hashlib.sha256(text.encode("utf-8")).hexdigest() if cache is not None else None
        if ck is not None and ck in cache:
            embeddings.append(cache[ck])
            continue

        key = keys[rot % n_accounts]
        rot += 1
        vec = _embed_text(_client_for(key), text)
        embeddings.append(vec)
        if ck is not None and vec is not None:
            cache[ck] = vec
        time.sleep(per_call_pace)
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


def attach_similarity(combined_jobs, cv_text, api_key, throttle_seconds=EMBED_THROTTLE_SECONDS,
                      job_embed_cache=None, paths=None):
    """Add `similarity` + `weighted_score` columns; return (df, url->embedding dict).

    `similarity`     — raw cosine sim against the CV embedding (for debugging /
                       email visibility).
    `weighted_score` — similarity * region_weight * trust_weight * role_weight.
                       Used for the top-N cutoff that decides which jobs reach
                       the Cerebras+Groq verdict.

    `paths` (the user's chosen career tracks) makes the role weight per-user —
    titles matching a chosen path are boosted. None/empty keeps the legacy
    hardcoded role tiers.

    `api_key` should be the embedding-dedicated key (GEMINI_EMBED_API_KEY in
    production) so the embedding burst (~100 RPM peak) doesn't poison the main
    Gemini key's quota. Falls back to the main GEMINI_API_KEY if unset.

    `job_embed_cache` is an optional dict shared across users within one tick so
    a job seen by several users is embedded once (see embed_jobs). Pass the same
    dict for every user in a tick; leave None for single-user / one-off calls.

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
        weights = [compute_combined_weight(r, paths) for r in out.to_dict("records")]
        out["weighted_score"] = [0.0 for _ in weights]
        return out, {}

    rows = combined_jobs.to_dict("records")
    job_vecs = embed_jobs(rows, api_key, throttle_seconds=throttle_seconds, cache=job_embed_cache)
    sims = rank_by_similarity(cv_vec, job_vecs)
    weights = [compute_combined_weight(r, paths) for r in rows]
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


def drop_semantic_duplicates(df, new_embeddings, threshold=SEMANTIC_DEDUP_THRESHOLD, history=None):
    """Remove rows whose embedding is ≥threshold cosine-similar to any history entry.

    `new_embeddings` is the {url: embedding_vector} dict from attach_similarity.
    Only considers the rolling 14-day history (already TTL-pruned on save).
    Returns the filtered dataframe. Rows whose URL isn't in `new_embeddings`
    (e.g. embedding API failure) are kept by default — we don't drop them just
    because we couldn't verify, since URL-dedup already handled exact repeats.

    `history` lets a caller inject the comparison set directly. Two accepted
    shapes:
      * legacy disk form  {url: {"embedding": [...], "added_at": ...}}
      * flat form         {url: [...]}   (the multi-user Supabase history)
    When None (the single-user default) it's loaded from the local disk cache.
    """
    if df is None or df.empty or not new_embeddings:
        return df

    if history is None:
        history = load_embedding_history()
    if not history:
        return df

    # Pre-extract historical vectors as a flat list for the inner loop. Tolerate
    # both the disk shape ({"embedding": [...]}) and the flat shape ([...]).
    hist_pairs = []
    for url, entry in history.items():
        if isinstance(entry, dict):
            vec = entry.get("embedding")
        elif isinstance(entry, list):
            vec = entry
        else:
            vec = None
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


# ---------------------------------------------------------------------------
# Cross-tick job-embedding cache — persist embeddings by TEXT hash so a job whose
# text we've already embedded isn't re-embedded on the next tick. The per-tick
# `cache` in embed_jobs already dedupes WITHIN a tick (shared public feed); this
# seeds that same dict from disk so the (slowly-churning) feeds don't get
# re-embedded every tick. The workflow persists data/job_embedding_cache.json
# across runs (like the CV cache), directly relieving the Gemini embedding RPD
# bottleneck. Keyed identically to embed_jobs: sha256(title + "\n\n" + description).
# ---------------------------------------------------------------------------

JOB_EMBEDDING_CACHE_FILE = "data/job_embedding_cache.json"
JOB_EMBEDDING_CACHE_TTL_DAYS = 14           # roll off texts not seen in this window
JOB_EMBEDDING_CACHE_MAX_ENTRIES = 6000      # hard cap so the cached file stays small


def _read_job_embed_cache_raw(path=None):
    """Load the on-disk {text_hash: {"embedding": [...], "ts": epoch}} map, or {}."""
    path = path or JOB_EMBEDDING_CACHE_FILE          # resolve at call time (test-overridable)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        logger.warning("Job-embedding cache read failed (starting empty): %s", e)
        return {}


def load_job_embedding_cache(path=None):
    """Return a flat {text_hash: embedding} dict to SEED embed_jobs' per-tick cache.

    embed_jobs keys its cache by sha256(title + description); seeding that dict
    with what we embedded on prior ticks means a repeat job (the public feeds
    churn slowly) skips the Gemini call entirely. Returns {} on miss/error, so
    it degrades to the old per-tick-only behavior.
    """
    raw = _read_job_embed_cache_raw(path)
    out = {}
    for h, entry in raw.items():
        if isinstance(entry, dict):
            vec = entry.get("embedding")
        elif isinstance(entry, list):
            vec = entry
        else:
            vec = None
        if vec:
            out[h] = vec
    return out


def save_job_embedding_cache(cache, path=None, now=None):
    """Persist the {text_hash: embedding} cache to disk (TTL-pruned + capped).

    `cache` is the per-tick dict embed_jobs accumulated (seeded from
    load_job_embedding_cache + any newly-embedded jobs). Original timestamps are
    PRESERVED for hashes already on disk, so an entry rolls off ~TTL days after it
    FIRST appeared (not perpetually refreshed just for being reloaded), and the
    hard cap keeps the cached file small. Best-effort — never raises.
    """
    if not cache:
        return
    path = path or JOB_EMBEDDING_CACHE_FILE          # resolve at call time (test-overridable)
    try:
        ts_now = (now or datetime.now(tz=timezone.utc)).timestamp()
        prior = _read_job_embed_cache_raw(path)
        merged = {}
        for h, vec in cache.items():
            if not vec:
                continue
            old = prior.get(h)
            ts = old.get("ts", ts_now) if isinstance(old, dict) else ts_now
            merged[h] = {"embedding": list(vec), "ts": ts}
        # TTL prune, then cap to the most-recently-first-seen entries.
        cutoff = ts_now - JOB_EMBEDDING_CACHE_TTL_DAYS * 86400
        merged = {h: e for h, e in merged.items() if e.get("ts", 0) >= cutoff}
        if len(merged) > JOB_EMBEDDING_CACHE_MAX_ENTRIES:
            keep = sorted(merged.items(), key=lambda kv: kv[1].get("ts", 0), reverse=True)
            merged = dict(keep[:JOB_EMBEDDING_CACHE_MAX_ENTRIES])
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(merged, f)
        logger.info("Job-embedding cache: saved %d entr%s to disk.",
                    len(merged), "y" if len(merged) == 1 else "ies")
    except Exception as e:
        logger.warning("Job-embedding cache write failed: %s", e)


# ---------------------------------------------------------------------------
# Feedback RAG: per-job retrieval against past user feedback (2026-05-25)
# ---------------------------------------------------------------------------
#
# Once the feedback log crosses RAG_FEEDBACK_THRESHOLD entries (see
# core_feedback), the pipeline stops feeding every verdict prompt the same
# AI-summarized profile. It instead embeds the current job, finds the most
# similar past feedback entries, and injects THOSE as the candidate-preference
# context. This makes the signal job-specific rather than averaged across an
# increasingly heterogeneous history.

def embed_feedback_text(text, api_key):
    """Embed a single piece of text with the feedback-retrieval model.

    Used both to embed each archived feedback entry once on ingest, and to
    embed every incoming job at retrieval time. Returns None when the API key
    is missing or the call fails — callers treat None as "skip / no context".
    """
    if not text or not api_key:
        return None
    from google import genai  # lazy: SDK only loaded when actually called
    client = genai.Client(api_key=api_key)
    return _embed_text(client, text, model=FEEDBACK_EMBED_MODEL)


def _format_job_for_retrieval(job_row):
    """Build the text representation used to embed a job at retrieval time.

    Kept short on purpose — the embedding model is paid per token and the
    title + company + first ~1500 chars of description carries the signal
    that maps the job to similar past feedback. Including the full 7000-char
    description (as the CV pre-rank does) would dilute the match against
    short feedback-entry texts like '[applied] Backend Engineer @ Stripe'.
    """
    title = str(job_row.get("title", "")).strip()
    company = str(job_row.get("company", "")).strip()
    location = str(job_row.get("location", "")).strip()
    description = str(job_row.get("description", "")).strip()[:FEEDBACK_JOB_TEXT_MAX_CHARS]
    header = f"{title} @ {company}"
    if location:
        header += f" ({location})"
    return f"{header}\n\n{description}".strip()


def retrieve_relevant_feedback(job_row, feedback_embeddings, api_key, top_k=5):
    """Return a short formatted list of past feedback entries most similar to this job.

    `feedback_embeddings` is the {"entries": [{"text": ..., "embedding": ...}, ...]}
    dict produced by core_feedback.ensure_feedback_embeddings. Embeds the job
    with FEEDBACK_EMBED_MODEL, ranks every archived feedback entry by cosine
    similarity, and returns the top-K formatted for direct injection into the
    verdict prompt's CANDIDATE LEARNED PREFERENCES block.

    Returns "" when there's no API key, no archived entries, or the job-embed
    call fails — the verdict still runs, just without RAG context.
    """
    entries = (feedback_embeddings or {}).get("entries", []) if isinstance(feedback_embeddings, dict) else []
    if not entries or not api_key:
        return ""

    job_text = _format_job_for_retrieval(job_row)
    if not job_text:
        return ""
    job_vec = embed_feedback_text(job_text, api_key)
    if job_vec is None:
        return ""

    scored = []
    for entry in entries:
        vec = entry.get("embedding") if isinstance(entry, dict) else None
        text = entry.get("text", "") if isinstance(entry, dict) else ""
        if not vec or not text:
            continue
        sim = cosine_similarity(job_vec, vec)
        scored.append((sim, text))

    if not scored:
        return ""

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:max(1, int(top_k))]
    lines = [f"- (similarity {sim:.2f}) {text}" for sim, text in top]
    return "Most similar past feedback entries from this candidate:\n" + "\n".join(lines)
