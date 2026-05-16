"""
CORE ATS MODULE — boost local-jobs discovery by hitting structured hiring APIs.

Background:
Palestinian IT companies were producing ~0 real jobs via the old DDG-based
local pipeline. The problem isn't our code — it's that DDG/Bing don't reliably
index small-company career pages. The fix is to bypass DDG entirely and talk
to the actual hiring platforms these companies use.

How:
1. Most companies don't write their own careers page. They use a SaaS ATS:
   Greenhouse, Lever, Workable, BambooHR, SmartRecruiters, etc.
2. Each of these has a free public JSON API.
3. We fetch the company's /careers page ONCE, detect the ATS, cache the token.
4. From then on we hit the ATS API directly — clean structured JSON, no HTML.

Public API endpoints used:
  Greenhouse:  GET https://boards-api.greenhouse.io/v1/boards/{token}/jobs
  Lever:       GET https://api.lever.co/v0/postings/{token}?mode=json
  Workable:    GET https://apply.workable.com/api/v3/accounts/{token}/jobs

Cache file:  data/ats_cache.json  (gitignored, regenerated as needed)

Network calls are wrapped in try/except + short timeouts so a slow/down
ATS never blocks the rest of the pipeline.
"""
from __future__ import annotations
import json
import os
import re
import time
from datetime import datetime, timezone


ATS_CACHE_FILE = "data/ats_cache.json"
HTTP_TIMEOUT = 8                          # seconds per network call
CACHE_TTL_DAYS = 30                       # re-detect a company's ATS after this many days
USER_AGENT = "Mozilla/5.0 (compatible; JobAlertsBot/1.0)"


# ---------------------------------------------------------------------------
# 1. LinkedIn handle extraction (option C from our plan)
# ---------------------------------------------------------------------------

_LI_HANDLE_RE = re.compile(
    r"linkedin\.com/(?:company|in|school)/([a-zA-Z0-9\-_.]+?)(?:/|$)",
    re.IGNORECASE,
)


def extract_linkedin_handle(url):
    """Pull the company handle out of a LinkedIn URL.

    Accepts any of:
      https://www.linkedin.com/company/adham-inc./
      https://linkedin.com/company/adham-inc
      https://www.linkedin.com/company/alameen-technologies/about/
      linkedin.com/company/foo/posts
    Returns the bare handle (`adham-inc.`, `alameen-technologies`, `foo`),
    or None if no recognizable pattern is found.
    """
    if not url or not isinstance(url, str):
        return None
    match = _LI_HANDLE_RE.search(url)
    if not match:
        return None
    return match.group(1).strip()


# ---------------------------------------------------------------------------
# 2. ATS detection (option A — the smart version)
# ---------------------------------------------------------------------------

# Order matters: more specific patterns first. Each regex captures the ATS token.
_ATS_DETECTORS = [
    ("greenhouse", re.compile(r"boards\.greenhouse\.io/(?:embed/job_board\?for=)?([a-zA-Z0-9\-_]+)", re.IGNORECASE)),
    ("lever",      re.compile(r"jobs\.lever\.co/([a-zA-Z0-9\-_]+)",                                   re.IGNORECASE)),
    ("workable",   re.compile(r"apply\.workable\.com/([a-zA-Z0-9\-_]+)",                              re.IGNORECASE)),
    ("workable",   re.compile(r"workable\.com/(?:n/)?([a-zA-Z0-9\-_]+)/jobs",                          re.IGNORECASE)),
    ("bamboohr",   re.compile(r"([a-zA-Z0-9\-_]+)\.bamboohr\.com",                                     re.IGNORECASE)),
    ("smartrecruiters", re.compile(r"careers\.smartrecruiters\.com/([a-zA-Z0-9\-_]+)",                 re.IGNORECASE)),
]


def detect_ats_from_html(html):
    """Scan a careers-page HTML body for a known ATS signature.

    Returns (ats_name, token) on first match, (None, None) if no ATS detected.
    Pure-function — easy to unit test without network.
    """
    if not html:
        return (None, None)
    for ats_name, pattern in _ATS_DETECTORS:
        m = pattern.search(html)
        if m:
            return (ats_name, m.group(1))
    return (None, None)


def fetch_careers_page(url):
    """One-shot HTTP GET of the careers page. Returns body text or empty string."""
    if not url:
        return ""
    import requests  # lazy — keeps imports cheap for tests that don't need HTTP
    try:
        r = requests.get(url, timeout=HTTP_TIMEOUT, headers={"User-Agent": USER_AGENT})
        if r.status_code == 200:
            return r.text
    except Exception as e:
        print(f"  ATS detect: fetch failed for {url}: {str(e)[:120]}")
    return ""


# ---------------------------------------------------------------------------
# 3. Per-platform job fetchers — each returns a list of normalized job dicts
#    matching the shape the rest of the pipeline expects:
#      {title, company, location, job_url, description, job_type, date_posted}
# ---------------------------------------------------------------------------

def _normalize_job(title, company, location, job_url, description, date_posted, job_type="fulltime"):
    return {
        "title": title or "",
        "company": company or "",
        "location": location or "Remote/Unspecified",
        "job_url": job_url or "",
        "description": description or "",
        "job_type": job_type,
        "date_posted": date_posted or "",
    }


def fetch_greenhouse_jobs(token, company_name):
    """https://boards-api.greenhouse.io/v1/boards/{token}/jobs (public, no auth)."""
    import requests
    url = f"https://boards-api.greenhouse.io/v1/boards/{token}/jobs"
    try:
        r = requests.get(url, timeout=HTTP_TIMEOUT, headers={"User-Agent": USER_AGENT})
        r.raise_for_status()
        payload = r.json()
    except Exception as e:
        print(f"  ATS greenhouse: {company_name} fetch failed: {str(e)[:120]}")
        return []
    return parse_greenhouse_payload(payload, company_name)


def parse_greenhouse_payload(payload, company_name):
    """Pure parsing — separate from network so tests can pin payload shape."""
    out = []
    for j in (payload or {}).get("jobs", []) or []:
        loc = ""
        if isinstance(j.get("location"), dict):
            loc = j["location"].get("name", "") or ""
        out.append(_normalize_job(
            title=j.get("title", ""),
            company=company_name,
            location=loc,
            job_url=j.get("absolute_url", ""),
            description="",                      # detail endpoint required for full content; skip for now
            date_posted=j.get("updated_at", "") or j.get("first_published", ""),
        ))
    return out


def fetch_lever_jobs(token, company_name):
    """https://api.lever.co/v0/postings/{token}?mode=json (public, no auth)."""
    import requests
    url = f"https://api.lever.co/v0/postings/{token}?mode=json"
    try:
        r = requests.get(url, timeout=HTTP_TIMEOUT, headers={"User-Agent": USER_AGENT})
        r.raise_for_status()
        payload = r.json()
    except Exception as e:
        print(f"  ATS lever: {company_name} fetch failed: {str(e)[:120]}")
        return []
    return parse_lever_payload(payload, company_name)


def parse_lever_payload(payload, company_name):
    out = []
    for j in (payload or []):
        if not isinstance(j, dict):
            continue
        cats = j.get("categories", {}) or {}
        out.append(_normalize_job(
            title=j.get("text", ""),
            company=company_name,
            location=cats.get("location", "") or "",
            job_url=j.get("hostedUrl", ""),
            description=j.get("descriptionPlain", "") or j.get("description", "") or "",
            date_posted=_lever_ms_to_iso(j.get("createdAt")),
        ))
    return out


def _lever_ms_to_iso(ms):
    if not ms:
        return ""
    try:
        return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).isoformat()
    except (ValueError, TypeError, OverflowError, OSError):
        return ""


def fetch_workable_jobs(token, company_name):
    """https://apply.workable.com/api/v3/accounts/{token}/jobs (public, no auth)."""
    import requests
    url = f"https://apply.workable.com/api/v3/accounts/{token}/jobs"
    try:
        r = requests.get(url, timeout=HTTP_TIMEOUT, headers={"User-Agent": USER_AGENT})
        r.raise_for_status()
        payload = r.json()
    except Exception as e:
        print(f"  ATS workable: {company_name} fetch failed: {str(e)[:120]}")
        return []
    return parse_workable_payload(payload, company_name, token)


def parse_workable_payload(payload, company_name, token=""):
    out = []
    for j in (payload or {}).get("jobs", []) or []:
        loc_dict = j.get("location") or {}
        loc = ", ".join(filter(None, [loc_dict.get("city"), loc_dict.get("country")]))
        # Workable usually doesn't include a direct apply URL in the list endpoint —
        # construct one from the shortcode if available, else fall back to account-level URL.
        shortcode = j.get("shortcode") or j.get("id") or ""
        url = (
            f"https://apply.workable.com/{token}/j/{shortcode}/"
            if token and shortcode else
            j.get("url", "") or j.get("apply_url", "")
        )
        out.append(_normalize_job(
            title=j.get("title", ""),
            company=company_name,
            location=loc or "Remote/Unspecified",
            job_url=url,
            description=j.get("description", "") or "",
            date_posted=j.get("published_on", "") or j.get("created_at", ""),
        ))
    return out


# Dispatch table so callers don't need to switch/case.
_ATS_FETCHERS = {
    "greenhouse": fetch_greenhouse_jobs,
    "lever":      fetch_lever_jobs,
    "workable":   fetch_workable_jobs,
}


# ---------------------------------------------------------------------------
# 4. Cache layer — avoids re-detecting the ATS every run
# ---------------------------------------------------------------------------

class AtsCache:
    """{company_name -> {ats, token, detected_at}} persisted as JSON.

    Entries older than CACHE_TTL_DAYS are treated as misses so we re-detect
    if a company changes platform. Failed detections (ats=None) are also
    cached but with a shorter implicit TTL — caller decides whether to retry.
    """
    def __init__(self, filepath=ATS_CACHE_FILE):
        self.filepath = filepath
        self.data = {}
        self.load()

    def load(self):
        if not os.path.exists(self.filepath):
            return
        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                self.data = json.load(f) or {}
        except Exception as e:
            print(f"AtsCache load failed: {e}")
            self.data = {}

    def save(self):
        try:
            d = os.path.dirname(self.filepath)
            if d:
                os.makedirs(d, exist_ok=True)
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2)
        except Exception as e:
            print(f"AtsCache save failed: {e}")

    def get(self, company_name):
        entry = self.data.get(company_name)
        if not entry:
            return None
        # Expire stale detections so we re-check periodically.
        detected = entry.get("detected_at")
        if detected:
            try:
                dt = datetime.fromisoformat(detected.replace("Z", "+00:00"))
                age_days = (datetime.now(timezone.utc) - dt).days
                if age_days > CACHE_TTL_DAYS:
                    return None
            except Exception:
                pass
        return entry

    def set(self, company_name, ats, token):
        self.data[company_name] = {
            "ats": ats,
            "token": token,
            "detected_at": datetime.now(timezone.utc).isoformat(),
        }


# ---------------------------------------------------------------------------
# 5. Public entry point — used by local_companies.py
# ---------------------------------------------------------------------------

def get_jobs_for_company(company_name, careers_url, cache=None):
    """Detect (if needed) the ATS for one company and return its current jobs.

    Returns a list of normalized job dicts. Empty list if no ATS detected,
    no cached token, or every fetch failed.

    Side effect: when this triggers a fresh detection, the cache is updated
    in memory. Callers should `cache.save()` once at the end of the run.
    """
    cache = cache if cache is not None else AtsCache()
    cached = cache.get(company_name)

    if cached and cached.get("ats") and cached.get("token"):
        ats = cached["ats"]; token = cached["token"]
    elif cached and cached.get("ats") is None:
        # We tried before and found no ATS — don't re-fetch every run.
        return []
    else:
        html = fetch_careers_page(careers_url)
        ats, token = detect_ats_from_html(html)
        cache.set(company_name, ats, token)
        if not ats:
            return []
        print(f"  ATS detected for {company_name}: {ats}/{token}")

    fetcher = _ATS_FETCHERS.get(ats)
    if not fetcher:
        return []
    return fetcher(token, company_name)
