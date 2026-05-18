import requests
import pandas as pd
from jobspy import scrape_jobs

"""
CORE SEARCH MODULE
------------------
This module is responsible for hunting down jobs across the internet.
It talks directly to JobSpy (for LinkedIn + Indeed; Glassdoor removed because
JobSpy's connector errors on every call) and to several free public APIs:
Remotive, Arbeitnow, Jobicy, RemoteOK, Himalayas, The Muse, WeWorkRemotely.
The fetchers each return a pandas DataFrame with the same shape so the rest
of the pipeline can concat them without per-source casing.

Common output schema:
    title, company, location, job_url, description (optional), date_posted

Pure functions — pure transformations of HTTP/RSS responses. Each fetcher
is wrapped in try/except so a single dead source can never block the rest
of the pipeline.
"""

USER_AGENT = "Mozilla/5.0 (compatible; JobAlertsBot/1.0)"
HTTP_TIMEOUT = 10

def fetch_remotive_jobs():
    """Fetches purely remote software development jobs from Remotive's public API."""
    try:
        url = "https://remotive.com/api/remote-jobs?category=software-dev"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        jobs = response.json().get("jobs", [])
        
        parsed_jobs = []
        for j in jobs:
            parsed_jobs.append({
                "title": j.get("title", ""),
                "company": j.get("company_name", ""),
                "location": j.get("candidate_required_location", "Remote"),
                "job_url": j.get("url", ""),
                "date_posted": j.get("publication_date", "")
            })
        return pd.DataFrame(parsed_jobs)
    except Exception as e:
        print(f"Failed to fetch Remotive jobs: {e}")
        return pd.DataFrame()

def fetch_arbeitnow_jobs():
    """Fetches jobs from Arbeitnow, strictly filtering for those marked as remote."""
    try:
        url = "https://www.arbeitnow.com/api/job-board-api"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        jobs = response.json().get("data", [])
        
        parsed_jobs = []
        for j in jobs:
            if j.get("remote"):
                parsed_jobs.append({
                    "title": j.get("title", ""),
                    "company": j.get("company_name", ""),
                    "location": "Remote",
                    "job_url": j.get("url", ""),
                    "date_posted": str(j.get("created_at", ""))
                })
        return pd.DataFrame(parsed_jobs)
    except Exception as e:
        print(f"Failed to fetch Arbeitnow jobs: {e}")
        return pd.DataFrame()

def fetch_jobicy_jobs():
    """Fetches remote jobs from the Jobicy API and filters down to tech industries.

    Jobicy's v2 API does NOT accept a `tag` query parameter (returns 0 results),
    so we pull the unfiltered feed and filter by `jobIndustry` client-side.
    """
    try:
        url = "https://jobicy.com/api/v2/remote-jobs?count=100"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        jobs = response.json().get("jobs", [])

        tech_industry_keywords = ("tech", "engineer", "software", "data", "ai", "develop", "it ", "information")
        parsed_jobs = []
        for j in jobs:
            industries = " ".join(j.get("jobIndustry", []) if isinstance(j.get("jobIndustry"), list) else []).lower()
            if industries and not any(k in industries for k in tech_industry_keywords):
                continue
            parsed_jobs.append({
                "title": j.get("jobTitle", ""),
                "company": j.get("companyName", ""),
                "location": j.get("jobGeo", "Remote"),
                "job_url": j.get("url", ""),
                "description": j.get("jobDescription", ""),
                "date_posted": j.get("pubDate", "")
            })
        return pd.DataFrame(parsed_jobs)
    except Exception as e:
        print(f"Failed to fetch Jobicy jobs: {e}")
        return pd.DataFrame()

def fetch_remoteok_jobs():
    """Fetches tech jobs from RemoteOK. Bypasses their legal notice header."""
    try:
        url = "https://remoteok.com/api"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        jobs = response.json()

        parsed_jobs = []
        for j in jobs:
            if "legal" in j:
                continue
            parsed_jobs.append({
                "title": j.get("position", ""),
                "company": j.get("company", ""),
                "location": j.get("location", "Remote"),
                "job_url": j.get("url", ""),
                "description": j.get("description", ""),
                "date_posted": j.get("date", "")
            })
        return pd.DataFrame(parsed_jobs)
    except Exception as e:
        print(f"Failed to fetch RemoteOK jobs: {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Wave 1 additions — Himalayas, The Muse, WeWorkRemotely
# ---------------------------------------------------------------------------

def fetch_himalayas_jobs(limit=100):
    """https://himalayas.app/jobs/api — 100% free, no auth, remote-tech focus.

    Response shape:
      {"comments": ..., "offset": 0, "limit": 100, "totalCount": N, "jobs": [
        {"title": "...", "companyName": "...", "locationRestrictions": ["EU", "Worldwide"],
         "applicationLink": "...", "pubDate": "ISO8601", "description": "...", ...}
      ]}
    """
    try:
        url = f"https://himalayas.app/jobs/api?limit={limit}"
        response = requests.get(url, timeout=HTTP_TIMEOUT, headers={"User-Agent": USER_AGENT})
        response.raise_for_status()
        payload = response.json()
        return parse_himalayas_payload(payload)
    except Exception as e:
        print(f"Failed to fetch Himalayas jobs: {e}")
        return pd.DataFrame()


def parse_himalayas_payload(payload):
    """Pure parser. Separate from network so QA can pin it down."""
    parsed = []
    for j in (payload or {}).get("jobs", []) or []:
        locations = j.get("locationRestrictions") or []
        if isinstance(locations, list):
            location = ", ".join(str(x) for x in locations) or "Remote"
        else:
            location = str(locations) or "Remote"
        parsed.append({
            "title": j.get("title", ""),
            "company": j.get("companyName", ""),
            "location": location,
            "job_url": j.get("applicationLink", "") or j.get("guid", ""),
            "description": j.get("description", "") or j.get("excerpt", ""),
            "date_posted": j.get("pubDate", ""),
        })
    return pd.DataFrame(parsed)


def fetch_themuse_jobs(page=0, category=None):
    """https://www.themuse.com/api/public/jobs — 500 req/hour without API key.

    Response shape:
      {"page": 0, "page_count": N, "items_per_page": 20, "total": N, "results": [
        {"name": "...", "company": {"name": "..."}, "locations": [{"name": "Remote"}],
         "refs": {"landing_page": "..."}, "publication_date": "ISO8601", ...}
      ]}
    """
    try:
        params = {"page": page}
        if category:
            params["category"] = category
        url = "https://www.themuse.com/api/public/jobs"
        response = requests.get(url, params=params, timeout=HTTP_TIMEOUT, headers={"User-Agent": USER_AGENT})
        response.raise_for_status()
        payload = response.json()
        return parse_themuse_payload(payload)
    except Exception as e:
        print(f"Failed to fetch The Muse jobs: {e}")
        return pd.DataFrame()


def parse_themuse_payload(payload):
    parsed = []
    for j in (payload or {}).get("results", []) or []:
        company = (j.get("company") or {}).get("name", "") or ""
        locs = j.get("locations") or []
        if isinstance(locs, list) and locs:
            location = ", ".join(str((x or {}).get("name", "")) for x in locs if isinstance(x, dict))
        else:
            location = "Remote/Unspecified"
        refs = j.get("refs") or {}
        parsed.append({
            "title": j.get("name", ""),
            "company": company,
            "location": location or "Remote/Unspecified",
            "job_url": refs.get("landing_page", "") if isinstance(refs, dict) else "",
            "description": j.get("contents", "") or "",
            "date_posted": j.get("publication_date", ""),
        })
    return pd.DataFrame(parsed)


def fetch_wwr_jobs():
    """WeWorkRemotely programming-jobs RSS feed.

    No REST API — they publish per-category RSS that we parse via the
    `feedparser` library. Largest remote-work board globally.

    Each entry encodes Company in the title (format "Company: Title") and
    metadata (region, headquarters) inside the description HTML.
    """
    try:
        import feedparser  # lazy import — keeps QA imports cheap
    except ImportError:
        print("Failed to fetch WWR jobs: feedparser not installed")
        return pd.DataFrame()
    try:
        url = "https://weworkremotely.com/categories/remote-programming-jobs.rss"
        feed = feedparser.parse(url)
        return parse_wwr_feed(feed)
    except Exception as e:
        print(f"Failed to fetch WWR jobs: {e}")
        return pd.DataFrame()


def parse_wwr_feed(feed):
    """Pure parser. Accepts either a feedparser FeedParserDict or a stub dict."""
    parsed = []
    entries = getattr(feed, "entries", None)
    if entries is None and isinstance(feed, dict):
        entries = feed.get("entries", []) or []
    for entry in (entries or []):
        # Title in WWR's RSS is often "CompanyName: Job Title" — split on first colon.
        raw_title = (entry.get("title", "") if isinstance(entry, dict) else getattr(entry, "title", "")) or ""
        company, title = _split_wwr_title(raw_title)
        link = (entry.get("link", "") if isinstance(entry, dict) else getattr(entry, "link", "")) or ""
        summary = (entry.get("summary", "") if isinstance(entry, dict) else getattr(entry, "summary", "")) or ""
        pub = (entry.get("published", "") if isinstance(entry, dict) else getattr(entry, "published", "")) or ""
        parsed.append({
            "title": title,
            "company": company,
            "location": "Remote",
            "job_url": link,
            "description": summary,
            "date_posted": pub,
        })
    return pd.DataFrame(parsed)


def _split_wwr_title(raw):
    """WWR's <title> usually formats as 'Company: Job Title'. Split conservatively."""
    if not raw or ":" not in raw:
        return ("", raw or "")
    company, _, title = raw.partition(":")
    return (company.strip(), title.strip())


# ---------------------------------------------------------------------------
# Wave 4 — Y Combinator's Work at a Startup
# ---------------------------------------------------------------------------

# Work at a Startup (YC's job board) doesn't expose a clean public REST API.
# Instead we use the same BS4-before-Jina tiered extraction we use for ATS-
# less Palestinian companies — fetch the listings page, let BeautifulSoup pull
# the visible text, then feed it to Gemini for structuring. This gives us
# fresh YC-backed remote roles without parsing brittle HTML directly.

YC_WAAS_BASE_URL = "https://www.workatastartup.com/jobs"


def fetch_yc_workatastartup_jobs(query="software engineer", remote=True, gemini_api_key=None):
    """Fetch YC-funded startup jobs via Work at a Startup.

    Uses the existing tiered (BS4 -> Jina -> Gemini) extractor from core_ats so
    we don't duplicate scraping logic. Returns an empty DataFrame on any
    failure — same contract as the other fetchers.

    Args:
        query: search term (default "software engineer"). YC's UI filters by
               role title; we pass it through to the URL.
        remote: when True, filters to remote-friendly listings.
        gemini_api_key: required for the Gemini extraction step. If absent,
               the function short-circuits to an empty DataFrame (so QA
               imports stay cheap and CI without a key still passes).
    """
    import os
    api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("YC Work at a Startup: GEMINI_API_KEY missing, skipping.")
        return pd.DataFrame()
    try:
        from urllib.parse import urlencode
        from pipeline.core_ats import extract_jobs_from_careers_page
        params = {"query": query}
        if remote:
            params["remote"] = "yes"
        url = f"{YC_WAAS_BASE_URL}?{urlencode(params)}"
        print(f"Fetching YC Work at a Startup: {url}")
        jobs = extract_jobs_from_careers_page(url, "YC Startup Network", api_key)
        if not jobs:
            print("YC Work at a Startup: no jobs extracted.")
            return pd.DataFrame()
        # Normalize the field set so the rest of the pipeline can concat
        # without surprises. The extractor returns dicts with title/company/
        # location/job_url/description/job_type/date_posted already.
        out = []
        for j in jobs:
            out.append({
                "title": j.get("title", ""),
                "company": j.get("company", "") or "YC Startup",
                "location": j.get("location", "") or "Remote",
                "job_url": j.get("job_url", ""),
                "description": j.get("description", ""),
                "date_posted": j.get("date_posted", ""),
            })
        return pd.DataFrame(out)
    except Exception as e:
        print(f"Failed to fetch YC Work at a Startup jobs: {str(e)[:200]}")
        return pd.DataFrame()
