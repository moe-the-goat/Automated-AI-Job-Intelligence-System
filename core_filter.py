import pandas as pd
import re
import json
import os

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
        print(f"Reputation file load failed (using empty): {e}")
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
        print(f"Reputation filter: flagged {flagged} row(s) as low-quality.")
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
    """Returns False only when langdetect is confident the title is non-English."""
    if not _HAS_LANGDETECT or not isinstance(title, str) or len(title.strip()) < 10:
        return True
    try:
        return detect(title) == 'en'
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
                print(f"Error loading seen jobs: {e}")

    def save(self):
        try:
            with open(self.filepath, "w") as f:
                json.dump({"urls": list(self.seen_urls)}, f)
        except Exception as e:
            print(f"Error saving seen jobs: {e}")

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
        print(f"Date filtering error: {e}")
        
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
        print(f"Seen-jobs filter: dropped {before - len(combined_jobs)} previously-evaluated jobs.")

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
            print(f"Language filter: dropped {dropped} non-English titles.")

    # 4. Location Pre-filter: Drop clearly location-locked jobs that don't say remote
    if "location" in combined_jobs.columns:
        explicit_non_remote = ['shanghai', 'beijing', 'mumbai', 'bangalore', 'moscow', 'tx', 'ca', 'ny', 'california', 'texas', 'new york', 'india', 'china', 'russia']
        pattern = '|'.join([rf'\b{loc}\b' for loc in explicit_non_remote])
        remote_in_loc = combined_jobs['location'].astype(str).str.lower().str.contains('remote')
        remote_in_title = combined_jobs['title'].astype(str).str.lower().str.contains('remote')
        bad_loc = combined_jobs['location'].astype(str).str.lower().str.contains(pattern, na=False)
        combined_jobs = combined_jobs[~(bad_loc & ~remote_in_loc & ~remote_in_title)]
        
    # 5. Filter out senior/lead roles
    exclude_words = ['senior', 'sr', 'sr.', 'lead', 'principal', 'manager', 'director', 'staff', 'head', 'vp', 'president']
    if "title" in combined_jobs.columns:
        pattern = '|'.join([rf'\b{w}\b' for w in exclude_words])
        combined_jobs = combined_jobs[~combined_jobs['title'].str.lower().str.contains(pattern, na=False)]
        
    # 6. TIGHTENED Role Keywords (Tier 3 Item 15)
    # Replaced broad 'engineer' and 'data' with specific titles so we don't catch "Sales Engineer" or "Data Entry".
    role_keywords = [
        'software engineer', 'software developer', 'backend', 'frontend', 'fullstack', 
        'web developer', 'python', 'java ', 'c\\+\\+', 'c#', 'programmer',
        'ml engineer', 'ai engineer', 'machine learning', 'data scientist', 'data engineer',
        'artificial intelligence', 'intern'
    ]
    if "title" in combined_jobs.columns:
        pattern = '|'.join([rf'{w}' for w in role_keywords])
        combined_jobs = combined_jobs[combined_jobs['title'].str.lower().str.contains(pattern, na=False)]

    return combined_jobs
