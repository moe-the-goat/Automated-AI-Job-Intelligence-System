import pandas as pd
import re
import json
import os

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
