import pandas as pd
import re
import json
import os

from pipeline.logging_setup import get_logger

logger = get_logger(__name__)

# --- Reputation modifier rules (A1) ---
# Loaded once at import time. A blacklisted company name (or post URL handle)
# tags the row so downstream rendering shows a 🚫 badge and core_ai.py caps
# match_percentage at 55. trust_boost is tagged but currently informational only.
REPUTATION_FILE = "data/reputation.json"

def _load_reputation():
    try:
        if os.path.exists(REPUTATION_FILE):
            with open(REPUTATION_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {
                "blacklist_name":   [p.lower() for p in data.get("blacklist_name_patterns", [])],
                "blacklist_handle": [p.lower() for p in data.get("blacklist_handle_patterns", [])],
                "trust_boost":      [p.lower() for p in data.get("trust_boost", [])],
            }
    except Exception as e:
        logger.warning("Reputation file load failed (using empty): %s", e)
    return {"blacklist_name": [], "blacklist_handle": [], "trust_boost": []}

_REPUTATION = _load_reputation()

def _pre_flag_reputation(df):
    """Tag rows whose company name or post URL matches the reputation lists.

    Adds two boolean columns: pre_flagged_low_quality, pre_flagged_trusted.
    Does NOT drop rows (flag-and-pass is friendlier to auditing).
    """
    if df.empty:
        return df
    df = df.copy()
    name_lower = df.get("company", pd.Series(dtype=str)).astype(str).str.lower()
    url_lower = df.get("job_url", pd.Series(dtype=str)).astype(str).str.lower()

    low_q = pd.Series(False, index=df.index)
    for pat in _REPUTATION["blacklist_name"]:
        low_q |= name_lower.str.contains(pat, na=False, regex=False)
    for pat in _REPUTATION["blacklist_handle"]:
        low_q |= url_lower.str.contains(pat, na=False, regex=False)

    trusted = pd.Series(False, index=df.index)
    for pat in _REPUTATION["trust_boost"]:
        trusted |= name_lower.str.contains(pat, na=False, regex=False)

    df["pre_flagged_low_quality"] = low_q
    df["pre_flagged_trusted"] = trusted
    flagged = int(low_q.sum())
    if flagged:
        logger.info("Reputation filter: flagged %d row(s) as low-quality.", flagged)
    return df

# langdetect catches non-English titles (Italian "Posizioni", German "Entwickler", etc.)
# that the CJK Unicode pre-filter misses. Optional dependency — if missing, we
# silently keep every title.
try:
    from langdetect import detect, DetectorFactory, LangDetectException
    DetectorFactory.seed = 0
    _HAS_LANGDETECT = True
except ImportError:
    _HAS_LANGDETECT = False

def _is_english_title(title):
    """Returns False only when langdetect is confident the title is non-English.

    langdetect's accuracy collapses on short strings — its own docs say it needs
    ~50 chars to be reliable. Tech titles are also loaded with proper nouns
    (PyTorch, FastAPI, RAG, etc.) that confuse it into reporting Italian / Welsh
    on titles like "AI Engineer". 30 chars is a calibrated trade-off: keeps real
    tech titles, still catches obviously non-English long-form postings like
    "Sviluppatore Senior PHP per fintech italiana". The CJK regex filter (step 3)
    catches truly non-English short titles in Asian languages.
    """
    if not _HAS_LANGDETECT or not isinstance(title, str) or len(title.strip()) < 30:
        return True
    try:
        return detect(title) == 'en'
    except LangDetectException:
        return True


# Description-text langdetect threshold. Lowered 400 -> 300 (2026-05-17),
# then 300 -> 100 (2026-05-22) after a Spanish-description job slipped
# through. 100 chars gives langdetect enough signal for confident detection
# of major languages while still avoiding false positives on very short
# English boilerplate snippets.
_DESCRIPTION_LANGDETECT_MIN_CHARS = 100

# Strip HTML so descriptions like "<p>Wir suchen einen Entwickler...</p>" don't
# get the tags confusing langdetect into "English" because of `<p>` / `<div>`.
_HTML_TAG_RE_FOR_LANGDETECT = re.compile(r"<[^>]+>")


def _is_english_description(description):
    """Returns False only when langdetect is confident the description is non-English.

    Conservative by design — we only drop on a high-confidence non-English read
    of substantial text. False positives here delete real jobs from the daily
    email, so the threshold is high (400 chars). Below that, we keep the row
    and let the AI handle whatever language confusion remains.

    Also strips HTML tags first; raw HTML can fool langdetect into "English"
    purely on tag names regardless of the actual body text language.
    """
    if not _HAS_LANGDETECT or not isinstance(description, str):
        return True
    cleaned = _HTML_TAG_RE_FOR_LANGDETECT.sub(" ", description).strip()
    # Heavy whitespace collapse to make length check meaningful.
    cleaned = re.sub(r"\s+", " ", cleaned)
    if len(cleaned) < _DESCRIPTION_LANGDETECT_MIN_CHARS:
        return True
    # Cap the text we feed to langdetect; it doesn't need the whole essay,
    # and a sampled 2KB chunk runs ~10x faster than 20KB descriptions.
    sample = cleaned[:2000]
    try:
        return detect(sample) == 'en'
    except LangDetectException:
        return True

"""
CORE FILTER MODULE
------------------
This module is the deterministic "bouncer" of the pipeline. It kicks out jobs 
that are obviously bad (non-tech roles, foreign languages, explicit non-remote 
locations) BEFORE we spend time and API quota sending them to the AI.
It also tracks which jobs we've already seen to prevent duplicate evaluations across daily runs.
"""

class JobTracker:
    """Tracks jobs we've already evaluated so we don't waste API calls on them tomorrow."""
    def __init__(self, filepath="seen_jobs.json"):
        self.filepath = filepath
        self.seen_urls = set()
        self.load()

    def load(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r") as f:
                    data = json.load(f)
                    self.seen_urls = set(data.get("urls", []))
            except Exception as e:
                logger.warning("Error loading seen jobs: %s", e)

    def save(self):
        try:
            with open(self.filepath, "w") as f:
                json.dump({"urls": list(self.seen_urls)}, f)
        except Exception as e:
            logger.warning("Error saving seen jobs: %s", e)

    def is_seen(self, url):
        return url in self.seen_urls

    def mark_seen(self, url):
        if url:
            self.seen_urls.add(url)

def filter_api_jobs(df, hours_old):
    """Initial light filter for API jobs to ensure they are somewhat relevant and fresh."""
    if df.empty:
        return df
    
    # Very loose filter here, just making sure it's in the realm of tech.
    role_keywords = ['software', 'developer', 'engineer', 'ai', 'data', 'machine learning', 'backend', 'frontend', 'fullstack', 'python', 'java']
    
    title_lower = df['title'].str.lower()
    has_role = title_lower.str.contains('|'.join(role_keywords), na=False)
    df = df[has_role].copy()
    
    # Filter by recency (hours_old). Arbeitnow returns Unix timestamps; others return ISO strings.
    try:
        # First try numeric (Unix seconds); whatever fails will be NaT and re-tried as ISO below.
        numeric = pd.to_numeric(df['date_posted'], errors='coerce')
        dt_from_unix = pd.to_datetime(numeric, unit='s', utc=True, errors='coerce')
        dt_from_iso = pd.to_datetime(df['date_posted'], utc=True, errors='coerce')
        df['date_posted_dt'] = dt_from_unix.fillna(dt_from_iso)
        now = pd.Timestamp.utcnow()
        cutoff = now - pd.Timedelta(hours=hours_old)
        df = df[df['date_posted_dt'].isna() | (df['date_posted_dt'] >= cutoff)]
        df = df.drop(columns=['date_posted_dt'])
    except Exception as e:
        logger.warning("Date filtering error: %s", e)
        
    return df

def apply_pipeline_filters(combined_jobs, tracker=None):
    """
    The main gauntlet. Runs all the Tier 2 and Tier 3 deterministic filters
    to protect the AI from evaluating garbage data.

    If a JobTracker is passed in, previously-evaluated URLs are dropped first
    so we don't burn API quota re-evaluating them.
    """
    if combined_jobs.empty:
        return combined_jobs

    # 0. Drop URLs we've already evaluated on prior runs (cheapest filter -> runs first)
    if tracker is not None and "job_url" in combined_jobs.columns:
        before = len(combined_jobs)
        combined_jobs = combined_jobs[~combined_jobs['job_url'].astype(str).apply(tracker.is_seen)]
        logger.info("Seen-jobs filter: dropped %d previously-evaluated jobs.", before - len(combined_jobs))

    # 0b. Tag rows against the reputation list (flag, don't drop).
    combined_jobs = _pre_flag_reputation(combined_jobs)

    # 1. URL Deduplication
    if "job_url" in combined_jobs.columns:
        combined_jobs = combined_jobs.drop_duplicates(subset=["job_url"])
        
    # 2. Smarter deduplication: Drop duplicates by normalized Title + Company
    if "title" in combined_jobs.columns and "company" in combined_jobs.columns:
        combined_jobs['norm_title'] = combined_jobs['title'].astype(str).str.replace(r'\s*\(.*?\)', '', regex=True).str.strip().str.lower()
        combined_jobs['norm_company'] = combined_jobs['company'].astype(str).str.strip().str.lower()
        combined_jobs = combined_jobs.drop_duplicates(subset=["norm_title", "norm_company"])
        combined_jobs = combined_jobs.drop(columns=['norm_title', 'norm_company'])
        
    # 3. Language Pre-filter: Reject Chinese/Korean/Japanese titles to save AI calls
    if "title" in combined_jobs.columns:
        combined_jobs = combined_jobs[~combined_jobs['title'].astype(str).str.contains(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]', na=False)]

    # 3b. Catch non-English titles that aren't CJK (e.g. Italian "Posizioni", German "Entwickler").
    if "title" in combined_jobs.columns and _HAS_LANGDETECT:
        before = len(combined_jobs)
        combined_jobs = combined_jobs[combined_jobs['title'].apply(_is_english_title)]
        dropped = before - len(combined_jobs)
        if dropped:
            logger.info("Language filter (title): dropped %d non-English titles.", dropped)

    # 3c. Catch non-English DESCRIPTIONS even when the title looks English.
    # Real failure mode: a job titled "Software Engineer (m/w/d)" passes the
    # title filter, but its body is entirely in German — the AI then evaluates
    # partial information and can score it spuriously high. We catch this only
    # for descriptions long enough to be confident in the language detection.
    if "description" in combined_jobs.columns and _HAS_LANGDETECT:
        before = len(combined_jobs)
        combined_jobs = combined_jobs[combined_jobs['description'].apply(_is_english_description)]
        dropped = before - len(combined_jobs)
        if dropped:
            logger.info("Language filter (description): dropped %d non-English descriptions.", dropped)

    # 4. Location Pre-filter: Drop clearly location-locked jobs that don't say remote
    if "location" in combined_jobs.columns:
        explicit_non_remote = ['shanghai', 'beijing', 'mumbai', 'bangalore', 'moscow', 'tx', 'ca', 'ny', 'california', 'texas', 'new york', 'india', 'china', 'russia']
        pattern = '|'.join([rf'\b{loc}\b' for loc in explicit_non_remote])
        remote_in_loc = combined_jobs['location'].astype(str).str.lower().str.contains('remote')
        remote_in_title = combined_jobs['title'].astype(str).str.lower().str.contains('remote')
        bad_loc = combined_jobs['location'].astype(str).str.lower().str.contains(pattern, na=False)
        combined_jobs = combined_jobs[~(bad_loc & ~remote_in_loc & ~remote_in_title)]
        
    # 5. Filter out senior/lead roles by title — covers ordinary seniority words
    # AND company-internal level codes used by FAANG-style ladders.
    # Examples we want to catch:
    #   "Software Engineer (L5)" — Netflix/Google
    #   "Senior Engineer, IC6"   — Meta
    #   "Software Engineer E5"   — Stripe
    #   "Sr Engineer II"         — explicit Roman suffix on a senior role
    # The level-code patterns require digits in the senior range to avoid
    # false positives on titles that happen to contain "L1" or "E2" tokens.
    exclude_words = ['senior', 'sr', 'sr.', 'lead', 'principal', 'manager', 'director',
                     'staff', 'head', 'vp', 'president', 'architect']
    exclude_level_codes = [
        r'\bl[4-9]\b', r'\bl1[0-2]\b',           # L4-L12 (Google, Netflix)
        r'\bic[4-9]\b', r'\bic1[0-2]\b',         # IC4-IC12 (Meta)
        r'\be[4-9]\b',                            # E4-E9 (Stripe, some banks)
        r'\bg[7-9]\b', r'\bg1[0-2]\b',           # G7-G12 (Amazon)
        r'\bsde\s*[2-9]\b', r'\bsde-?[2-9]\b',   # SDE II / SDE-3 (Amazon)
        r'\bswe\s*[2-9]\b', r'\bswe-?[2-9]\b',   # SWE II / SWE-3
    ]
    if "title" in combined_jobs.columns:
        word_pattern = '|'.join([rf'\b{w}\b' for w in exclude_words])
        level_pattern = '|'.join(exclude_level_codes)
        combined_pattern = f'{word_pattern}|{level_pattern}'
        combined_jobs = combined_jobs[~combined_jobs['title'].str.lower().str.contains(combined_pattern, na=False)]
        
    # 6. Role keyword filter — must contain at least one signal that it's a tech role.
    # Wide enough to catch every legitimate target role; narrow enough that sales /
    # marketing / HR / data-entry / customer-success cannot sneak through (those are
    # blocked by the non-tech title signals earlier and by the seniority filter).
    # The embedding ranker and AI handle final relevance — this layer just drops
    # obviously irrelevant stuff before it burns any quota.
    role_keywords = [
        # Core software / web roles
        'software engineer', 'software developer', 'backend', 'frontend', 'fullstack',
        'full-stack', 'full stack', 'web developer', 'web engineer', 'programmer',
        # Languages and frameworks (specific enough to avoid "Java project manager" etc.)
        'python', 'java developer', 'java engineer', 'javascript', 'typescript',
        'c\\+\\+', 'c#', 'golang', 'rust', 'kotlin', 'swift',
        'react', 'node.js', 'django', 'fastapi', 'flask',
        # ML / AI / Data
        'ml engineer', 'ai engineer', 'machine learning', 'deep learning',
        'data scientist', 'data engineer', 'data analyst',
        'artificial intelligence', 'neural', 'nlp', 'natural language',
        'computer vision', 'llm', 'large language model', 'generative ai',
        'research engineer', 'research scientist',
        # Platform / Infrastructure
        'devops', 'site reliability', 'sre', 'platform engineer',
        'cloud engineer', 'cloud developer', 'infrastructure engineer',
        'systems engineer', 'embedded',
        # Distinguished IC titles
        'member of technical staff',
        # Entry-level catch-all
        'intern',
    ]
    if "title" in combined_jobs.columns:
        pattern = '|'.join([rf'{w}' for w in role_keywords])
        combined_jobs = combined_jobs[combined_jobs['title'].str.lower().str.contains(pattern, na=False)]

    # 7. Non-tech intern blockers. The "intern" catch-all in step 6 is broad —
    # too broad on its own. Real failures observed on 2026-05-17:
    #   - "Graduate Research Intern, Biology"     (pure science, not SWE)
    #   - "Business Analyst Intern (Entry Level)" (business analytics, not data)
    # Strategy: if "intern" is in the title AND any of these non-tech signals
    # ALSO appears, drop the row. Tech-keyword positives in step 6 alone aren't
    # enough — the title has to be a tech role on its own merits.
    nontech_intern_blockers = [
        'biology', 'biomed', 'biotech', 'biochem', 'pharma', 'medical', 'nursing',
        'business analyst', 'business analytics',
        'social media', 'communications intern', 'pr intern', 'public relations',
        'hr intern', 'human resources', 'recruiting intern',
        'accounting', 'finance intern', 'tax intern', 'audit intern',
        'legal intern', 'law intern',
        'marketing analyst intern', 'brand intern',
    ]
    if "title" in combined_jobs.columns:
        title_lower = combined_jobs['title'].astype(str).str.lower()
        has_intern = title_lower.str.contains(r'\bintern\b', na=False, regex=True)
        blocker_pattern = '|'.join(nontech_intern_blockers)
        has_blocker = title_lower.str.contains(blocker_pattern, na=False)
        before = len(combined_jobs)
        combined_jobs = combined_jobs[~(has_intern & has_blocker)]
        dropped = before - len(combined_jobs)
        if dropped:
            logger.info("Non-tech intern filter: dropped %d title(s) (biology / business / etc.).", dropped)

    return combined_jobs
