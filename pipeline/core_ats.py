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
  Greenhouse:  GET https://boards-api.greenhouse.io/v1/boards/{token}/jobs?content=true
  Lever:       GET https://api.lever.co/v0/postings/{token}?mode=json
  Workable:    GET https://apply.workable.com/api/v3/accounts/{token}/jobs
  Ashby:       GET https://api.ashbyhq.com/posting-api/job-board/{token}?includeCompensation=true
  Workday:     POST https://{tenant}.{cluster}.myworkdayjobs.com/wday/cxs/{tenant}/{site}/jobs
               (token is encoded as "tenant|cluster|site" — see _parse_workday_token)
  FactorialHR: GET https://api.factorialhr.com/resources/atsapi/v1/jobs?slug={token}
               (used by many EU SMBs incl. Palestinian Innotech)

Jina Reader fallback (Wave 2):
  When ATS detection fails on a careers page, the caller can route to Jina
  Reader (https://r.jina.ai/{url}) which extracts clean markdown from ANY
  webpage — including JS-rendered sites. The markdown is then fed to Gemini
  to produce structured job listings. Used as last resort for the Palestinian
  companies pipeline where most career pages aren't on a known ATS.

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
# Workday is special: its token is composite (tenant|cluster|site) — see _detect_workday.
_ATS_DETECTORS = [
    ("greenhouse", re.compile(r"boards\.greenhouse\.io/(?:embed/job_board\?for=)?([a-zA-Z0-9\-_]+)", re.IGNORECASE)),
    ("lever",      re.compile(r"jobs\.lever\.co/([a-zA-Z0-9\-_]+)",                                   re.IGNORECASE)),
    ("ashby",      re.compile(r"jobs\.ashbyhq\.com/([a-zA-Z0-9\-_]+)",                                re.IGNORECASE)),
    ("ashby",      re.compile(r"ashbyhq\.com/(?:embed/)?([a-zA-Z0-9\-_]+)",                            re.IGNORECASE)),
    ("workable",   re.compile(r"apply\.workable\.com/([a-zA-Z0-9\-_]+)",                              re.IGNORECASE)),
    ("workable",   re.compile(r"workable\.com/(?:n/)?([a-zA-Z0-9\-_]+)/jobs",                          re.IGNORECASE)),
    ("bamboohr",   re.compile(r"([a-zA-Z0-9\-_]+)\.bamboohr\.com",                                     re.IGNORECASE)),
    ("smartrecruiters", re.compile(r"careers\.smartrecruiters\.com/([a-zA-Z0-9\-_]+)",                 re.IGNORECASE)),
    # FactorialHR — common among EU SMBs. URL form: {slug}.factorialhr.com[/...]
    ("factorialhr", re.compile(r"([a-zA-Z0-9\-_]+)\.factorialhr\.com",                                  re.IGNORECASE)),
]

# Workday URLs look like `{tenant}.wd{N}.myworkdayjobs.com/{lang}/{site}` (optional
# lang segment). We need all three pieces — tenant, cluster, site — so we handle
# it outside the generic detector and pack them into a single token string.
_WORKDAY_RE = re.compile(
    r"([a-z0-9\-]+)\.(wd\d+)\.myworkdayjobs\.com/(?:[a-zA-Z\-]+/)?([a-zA-Z0-9_\-]+)",
    re.IGNORECASE,
)


def _detect_workday(html):
    """Return ('workday', 'tenant|cluster|site') or (None, None)."""
    if not html:
        return (None, None)
    m = _WORKDAY_RE.search(html)
    if not m:
        return (None, None)
    tenant, cluster, site = m.group(1), m.group(2), m.group(3)
    return ("workday", f"{tenant.lower()}|{cluster.lower()}|{site}")


def _parse_workday_token(token):
    """Unpack 'tenant|cluster|site' back into a tuple. Returns (None, None, None) on garbage."""
    if not token or "|" not in token:
        return (None, None, None)
    parts = token.split("|", 2)
    if len(parts) != 3:
        return (None, None, None)
    return (parts[0], parts[1], parts[2])


def detect_ats_from_html(html):
    """Scan a careers-page HTML body for a known ATS signature.

    Returns (ats_name, token) on first match, (None, None) if no ATS detected.
    Pure-function — easy to unit test without network.

    Workday is checked FIRST because its multi-piece URL is more specific than
    the generic detectors and we don't want a stray 'myworkdayjobs.com' string
    to leak through as a different platform.
    """
    if not html:
        return (None, None)
    wd_ats, wd_token = _detect_workday(html)
    if wd_ats:
        return (wd_ats, wd_token)
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
    """https://boards-api.greenhouse.io/v1/boards/{token}/jobs?content=true (public, no auth).

    The `content=true` query param makes Greenhouse inline each job's full
    description in the SAME response — no N+1 follow-up calls. Saves us from
    the anti-bot scraping fallback in core_ai.get_full_job_description when
    a Greenhouse job reaches the AI evaluator.
    """
    import requests
    url = f"https://boards-api.greenhouse.io/v1/boards/{token}/jobs?content=true"
    try:
        r = requests.get(url, timeout=HTTP_TIMEOUT, headers={"User-Agent": USER_AGENT})
        r.raise_for_status()
        payload = r.json()
    except Exception as e:
        print(f"  ATS greenhouse: {company_name} fetch failed: {str(e)[:120]}")
        return []
    return parse_greenhouse_payload(payload, company_name)


def parse_greenhouse_payload(payload, company_name):
    """Pure parsing — separate from network so tests can pin payload shape.

    With `?content=true`, each job carries an HTML-encoded `content` field.
    We strip the HTML tags so downstream consumers (embedding, AI, viability
    pre-screen) all see clean plain text.
    """
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
            description=_clean_greenhouse_content(j.get("content", "")),
            date_posted=j.get("updated_at", "") or j.get("first_published", ""),
        ))
    return out


def _clean_greenhouse_content(content):
    """Strip HTML tags + unescape entities from Greenhouse's `content` field.

    Greenhouse stores the description as HTML-escaped HTML (e.g.
    `&lt;p&gt;Build great products&lt;/p&gt;`). We unescape, strip tags, and
    collapse whitespace. BS4 is imported lazily because most pipelines never
    hit a Greenhouse job at all.
    """
    if not content:
        return ""
    import html as _html
    unescaped = _html.unescape(content)
    try:
        from bs4 import BeautifulSoup
        text = BeautifulSoup(unescaped, "html.parser").get_text(separator=" ")
    except Exception:
        text = re.sub(r"<[^>]+>", " ", unescaped)        # fallback if BS4 absent
    return re.sub(r"\s+", " ", text).strip()


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


def fetch_ashby_jobs(token, company_name):
    """https://api.ashbyhq.com/posting-api/job-board/{token}?includeCompensation=true (public, no auth)."""
    import requests
    url = f"https://api.ashbyhq.com/posting-api/job-board/{token}?includeCompensation=true"
    try:
        r = requests.get(url, timeout=HTTP_TIMEOUT, headers={"User-Agent": USER_AGENT})
        r.raise_for_status()
        payload = r.json()
    except Exception as e:
        print(f"  ATS ashby: {company_name} fetch failed: {str(e)[:120]}")
        return []
    return parse_ashby_payload(payload, company_name)


def parse_ashby_payload(payload, company_name):
    """Pure parser for Ashby's job board response.

    Shape:
      {"apiVersion": "...", "jobs": [
        {"id": "...", "title": "...", "location": "...", "department": "...",
         "employmentType": "FullTime", "jobUrl": "https://jobs.ashbyhq.com/.../...",
         "publishedAt": "ISO8601", "descriptionPlain": "...", ...}
      ]}
    """
    out = []
    for j in (payload or {}).get("jobs", []) or []:
        if not isinstance(j, dict):
            continue
        # Ashby usually exposes a plain string `location` but sometimes nests it.
        loc = j.get("location", "")
        if isinstance(loc, dict):
            loc = loc.get("name", "") or ""
        emp_type = (j.get("employmentType", "") or "").lower()
        job_type = "internship" if "intern" in emp_type else "fulltime"
        out.append(_normalize_job(
            title=j.get("title", ""),
            company=company_name,
            location=loc,
            job_url=j.get("jobUrl", "") or j.get("applyUrl", ""),
            description=j.get("descriptionPlain", "") or j.get("descriptionHtml", "") or "",
            date_posted=j.get("publishedAt", "") or j.get("updatedAt", ""),
            job_type=job_type,
        ))
    return out


def fetch_workday_jobs(token, company_name):
    """POST https://{tenant}.{cluster}.myworkdayjobs.com/wday/cxs/{tenant}/{site}/jobs.

    Token format: 'tenant|cluster|site' — see _detect_workday + _parse_workday_token.
    Workday's API is POST-only with a JSON body specifying paging + search text.
    """
    import requests
    tenant, cluster, site = _parse_workday_token(token)
    if not tenant or not cluster or not site:
        print(f"  ATS workday: {company_name} skipped: malformed token {token!r}")
        return []
    url = f"https://{tenant}.{cluster}.myworkdayjobs.com/wday/cxs/{tenant}/{site}/jobs"
    body = {"limit": 20, "offset": 0, "searchText": ""}
    try:
        r = requests.post(
            url, json=body, timeout=HTTP_TIMEOUT,
            headers={"User-Agent": USER_AGENT, "Accept": "application/json", "Content-Type": "application/json"},
        )
        r.raise_for_status()
        payload = r.json()
    except Exception as e:
        print(f"  ATS workday: {company_name} fetch failed: {str(e)[:120]}")
        return []
    return parse_workday_payload(payload, company_name, tenant=tenant, cluster=cluster, site=site)


def parse_workday_payload(payload, company_name, tenant="", cluster="", site=""):
    """Pure parser for Workday's CXS /jobs response.

    Shape:
      {"total": N, "jobPostings": [
        {"title": "...", "externalPath": "/job/.../...",
         "locationsText": "Remote, USA", "postedOn": "Posted Yesterday", "bulletFields": []}
      ]}

    `externalPath` is a relative URL — we glue the public origin back on so the
    resulting job_url opens directly in a browser.
    """
    out = []
    base = f"https://{tenant}.{cluster}.myworkdayjobs.com" if tenant and cluster else ""
    for j in (payload or {}).get("jobPostings", []) or []:
        if not isinstance(j, dict):
            continue
        path = j.get("externalPath", "") or ""
        if path and base and path.startswith("/"):
            job_url = f"{base}/{site}{path}" if site else f"{base}{path}"
        else:
            job_url = path
        out.append(_normalize_job(
            title=j.get("title", ""),
            company=company_name,
            location=j.get("locationsText", "") or "",
            job_url=job_url,
            description="",                       # detail endpoint required for body; skip
            date_posted=j.get("postedOn", "") or "",
        ))
    return out


def fetch_factorialhr_jobs(token, company_name):
    """https://api.factorialhr.com/resources/atsapi/v1/jobs?slug={token}.

    Public endpoint, no auth. Many EU SMBs use FactorialHR — including the
    Palestinian company Innotech we kept getting ghost-listings for. Hitting
    the live API instead of scraping the indexed HTML kills the stale-listing
    problem because the endpoint only returns currently-open positions.
    """
    import requests
    url = f"https://api.factorialhr.com/resources/atsapi/v1/jobs?slug={token}"
    try:
        r = requests.get(url, timeout=HTTP_TIMEOUT, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
        r.raise_for_status()
        payload = r.json()
    except Exception as e:
        print(f"  ATS factorialhr: {company_name} fetch failed: {str(e)[:120]}")
        return []
    return parse_factorialhr_payload(payload, company_name, token=token)


def parse_factorialhr_payload(payload, company_name, token=""):
    """Pure parser for FactorialHR's ATS API response.

    Tolerant to several shapes the endpoint returns across versions:
      - Bare list: [{...}, {...}]
      - Wrapped under "jobs" key: {"jobs": [{...}]}
      - Wrapped under "data" key: {"data": [{...}]}

    Each job entry typically has: id, title, description, locations,
    employment_type, url, slug. The public-facing apply URL pattern is
    `{token}.factorialhr.com/job_posting/{slug}` if `url` isn't explicit.
    """
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        items = payload.get("jobs") or payload.get("data") or []
    else:
        items = []

    out = []
    for j in items or []:
        if not isinstance(j, dict):
            continue
        # FactorialHR uses different keys depending on the org's setup;
        # try them in order and fall back to empty string.
        loc = ""
        loc_field = j.get("locations") or j.get("location") or j.get("city")
        if isinstance(loc_field, list):
            # Sometimes a list of dicts, sometimes a list of strings.
            parts = []
            for x in loc_field:
                if isinstance(x, dict):
                    parts.append(x.get("name", "") or x.get("city", "") or "")
                elif isinstance(x, str):
                    parts.append(x)
            loc = ", ".join(p for p in parts if p)
        elif isinstance(loc_field, dict):
            loc = loc_field.get("name", "") or loc_field.get("city", "") or ""
        elif isinstance(loc_field, str):
            loc = loc_field

        emp_type = (j.get("employment_type", "") or j.get("contract_type", "") or "").lower()
        job_type = "internship" if "intern" in emp_type else "fulltime"

        # Construct apply URL if the API didn't supply one.
        job_url = j.get("url") or j.get("apply_url") or ""
        if not job_url and token:
            slug = j.get("slug") or j.get("id") or ""
            if slug:
                job_url = f"https://{token}.factorialhr.com/job_posting/{slug}"

        # Description may be HTML; let downstream HTML-stripping in core_ai
        # handle that. We just pass the raw text/HTML through.
        description = j.get("description", "") or j.get("description_plain", "") or ""

        out.append(_normalize_job(
            title=j.get("title", "") or j.get("name", ""),
            company=company_name,
            location=loc,
            job_url=job_url,
            description=description,
            date_posted=j.get("published_at", "") or j.get("created_at", "") or "",
            job_type=job_type,
        ))
    return out


# Dispatch table so callers don't need to switch/case.
_ATS_FETCHERS = {
    "greenhouse":  fetch_greenhouse_jobs,
    "lever":       fetch_lever_jobs,
    "workable":    fetch_workable_jobs,
    "ashby":       fetch_ashby_jobs,
    "workday":     fetch_workday_jobs,
    "factorialhr": fetch_factorialhr_jobs,
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
# 5. Custom-careers-page fallback (Wave 2 + Wave 3) — for pages with no ATS
# ---------------------------------------------------------------------------
#
# Many Palestinian companies host custom HTML / JS-rendered careers pages with
# no SaaS ATS at all. Two-tier extraction:
#
#   Tier 1 — BeautifulSoup on the careers HTML we ALREADY fetched during ATS
#            detection. Free, instant, works for SSR sites (which is most
#            company careers pages).
#   Tier 2 — Jina Reader (https://r.jina.ai/) renders JS and returns markdown.
#            Only used when Tier 1 yields ~no text (true SPA / React shell).
#
# Both tiers feed the same Gemini prompt that structures the text into job
# listings. Tier 2 is roughly 5-10x slower and uses an external rate-limited
# service, so the tiered approach is a real latency / reliability win.

JINA_BASE_URL = "https://r.jina.ai/"
JINA_TIMEOUT = 20                                  # Jina renders the page; needs more headroom

# Hard ceilings to keep the Gemini call cheap when a careers page is enormous.
JINA_MAX_MARKDOWN_CHARS = 12000

# Tier-1 BS extraction needs at least this many chars of meaningful text to be
# trusted. Below this we assume the page is SPA / JS-rendered and fall through
# to Jina (Tier 2).
BS_MIN_USEFUL_CHARS = 500


_JINA_PROMPT_TEMPLATE = """You are extracting job listings from a company's careers-page content.

Company: {company_name}
Source URL: {careers_url}

The content below was fetched from the careers page. Find every distinct job
listing it advertises. For each listing, return a JSON object with these
exact keys:
  - title            (job title as written)
  - location         (city/country/remote — leave empty string if not stated)
  - job_url          (absolute URL to the job posting; empty string if not on page)
  - description      (one or two sentences summarising the role; empty if none)
  - date_posted      (date string if shown, else empty)

Return a JSON object: {{"jobs": [ ... ]}}

CRITICAL RULES:
- Only include real job postings. Skip generic "Send us your CV" / "Join our
  talent pool" sections that aren't tied to a specific role.
- If the page advertises NO open positions, return {{"jobs": []}}.
- Output VALID JSON only. No prose, no markdown fences.

--- CONTENT START ---
{markdown}
--- CONTENT END ---
"""


def extract_text_from_html(html):
    """Strip scripts/styles/nav from HTML and return clean visible text.

    Used as the Tier-1 extractor for ATS-less careers pages. Cheap because we
    already have the HTML in memory from the ATS-detection fetch.
    Returns "" if BS4 isn't installed or parsing fails.
    """
    if not html:
        return ""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "svg"]):
            tag.decompose()
        text = soup.get_text(separator=" ")
    except Exception as e:
        print(f"  BS4 extraction failed: {str(e)[:120]}")
        return ""
    return re.sub(r"\s+", " ", text).strip()


def fetch_jina_markdown(careers_url):
    """Hit Jina Reader and return the extracted markdown (or empty string)."""
    if not careers_url:
        return ""
    import requests
    url = JINA_BASE_URL + careers_url
    try:
        r = requests.get(url, timeout=JINA_TIMEOUT, headers={"User-Agent": USER_AGENT, "Accept": "text/plain"})
        if r.status_code == 200:
            return r.text or ""
        print(f"  Jina: {careers_url} returned HTTP {r.status_code}")
    except Exception as e:
        print(f"  Jina: fetch failed for {careers_url}: {str(e)[:120]}")
    return ""


def parse_jina_jobs_response(text, company_name):
    """Pure parser. Accepts whatever Gemini returns and yields normalized job dicts.

    Tolerates: bare JSON object, JSON inside ```json fences, leading/trailing prose.
    """
    if not text:
        return []
    cleaned = text.strip()
    # Strip markdown fences if the model wrapped its JSON despite our instructions.
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```\s*$", "", cleaned)
    # Carve out the first {...} block in case there's leading prose.
    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace == -1 or last_brace == -1 or last_brace <= first_brace:
        return []
    try:
        payload = json.loads(cleaned[first_brace : last_brace + 1])
    except Exception:
        return []
    out = []
    for j in (payload or {}).get("jobs", []) or []:
        if not isinstance(j, dict):
            continue
        title = (j.get("title") or "").strip()
        if not title:
            continue                                # skip placeholders
        out.append(_normalize_job(
            title=title,
            company=company_name,
            location=(j.get("location") or "").strip(),
            job_url=(j.get("job_url") or "").strip(),
            description=(j.get("description") or "").strip(),
            date_posted=(j.get("date_posted") or "").strip(),
        ))
    return out


def _gemini_structure_jobs(text_content, careers_url, company_name, gemini_api_key,
                           source_label, model="gemini-3.1-flash-lite"):
    """Send extracted text (either BS4 or Jina) to Gemini for job-list structuring.

    Returns a normalized job list, or [] if anything goes wrong. `source_label`
    is purely cosmetic (used in print statements so you can tell which tier
    contributed).
    """
    if not text_content.strip() or not gemini_api_key:
        return []
    truncated = text_content[:JINA_MAX_MARKDOWN_CHARS]
    prompt = _JINA_PROMPT_TEMPLATE.format(
        company_name=company_name,
        careers_url=careers_url,
        markdown=truncated,
    )
    try:
        from google import genai                   # lazy — keeps QA imports cheap
        client = genai.Client(api_key=gemini_api_key)
        response = client.models.generate_content(model=model, contents=prompt)
        text = getattr(response, "text", "") or ""
    except Exception as e:
        print(f"  {source_label}+Gemini extraction failed for {company_name}: {str(e)[:120]}")
        return []
    jobs = parse_jina_jobs_response(text, company_name)
    if jobs:
        print(f"  {source_label} fallback: extracted {len(jobs)} job(s) for {company_name}")
    return jobs


def extract_jobs_via_jina(careers_url, company_name, gemini_api_key,
                          model="gemini-3.1-flash-lite"):
    """Tier-2 only: Jina Reader -> Gemini structuring.

    Kept as a thin wrapper for backward compatibility and for callers that
    want to skip Tier 1 entirely (e.g. when they don't have the HTML cached).
    """
    if not careers_url or not gemini_api_key:
        return []
    markdown = fetch_jina_markdown(careers_url)
    return _gemini_structure_jobs(markdown, careers_url, company_name, gemini_api_key,
                                  source_label="Jina", model=model)


def extract_jobs_from_careers_page(careers_url, company_name, gemini_api_key,
                                   html=None, model="gemini-3.1-flash-lite"):
    """Two-tier fallback for careers pages with no detectable ATS.

    Tier 1: BeautifulSoup extract on the supplied `html` (or freshly fetched if
            None). If the extracted text passes BS_MIN_USEFUL_CHARS, structure
            it with Gemini and return.
    Tier 2: Jina Reader renders the page (handles SPAs). Send markdown to
            Gemini, return results.

    Returns [] only if both tiers fail. Side effect: prints which tier won.
    """
    if not careers_url or not gemini_api_key:
        return []
    if html is None:
        html = fetch_careers_page(careers_url)

    bs_text = extract_text_from_html(html)
    if len(bs_text) >= BS_MIN_USEFUL_CHARS:
        jobs = _gemini_structure_jobs(bs_text, careers_url, company_name, gemini_api_key,
                                      source_label="BS4", model=model)
        if jobs:
            return jobs
        # BS extraction yielded text but no jobs — could be the model missed them,
        # or the page legitimately has no openings. Fall through to Jina anyway
        # because Jina sometimes surfaces JS-injected job links the SSR markup hid.
        print(f"  BS4 extraction returned 0 jobs for {company_name}; trying Jina")

    return extract_jobs_via_jina(careers_url, company_name, gemini_api_key, model=model)


# ---------------------------------------------------------------------------
# 6. Public entry point — used by local_companies.py
# ---------------------------------------------------------------------------

def get_jobs_for_company(company_name, careers_url, cache=None,
                         gemini_api_key=None, jina_fallback=False):
    """Detect (if needed) the ATS for one company and return its current jobs.

    Returns a list of normalized job dicts. Empty list if no ATS detected,
    no cached token, or every fetch failed.

    When `jina_fallback=True` and `gemini_api_key` is set, companies with no
    detectable ATS are routed through Jina Reader + Gemini extraction as a
    last resort. Cached ATS misses are honored either way so we don't hammer
    Jina on every run for the same dead-end pages.

    Side effect: when this triggers a fresh detection, the cache is updated
    in memory. Callers should `cache.save()` once at the end of the run.
    """
    cache = cache if cache is not None else AtsCache()
    cached = cache.get(company_name)

    if cached and cached.get("ats") and cached.get("token"):
        ats = cached["ats"]; token = cached["token"]
    elif cached and cached.get("ats") is None:
        # We tried before and found no ATS. The custom-page fallback still runs
        # because pages legitimately gain new postings between runs. No cached
        # HTML, so the tiered extractor will fetch fresh.
        if jina_fallback and gemini_api_key:
            return extract_jobs_from_careers_page(careers_url, company_name, gemini_api_key)
        return []
    else:
        html = fetch_careers_page(careers_url)
        ats, token = detect_ats_from_html(html)
        cache.set(company_name, ats, token)
        if not ats:
            print(f"  ATS not detected for {company_name}; trying tiered fallback" if jina_fallback else f"  ATS not detected for {company_name}")
            # Pass the already-fetched HTML so Tier 1 (BS4) is free of an
            # extra HTTP call. Tier 2 (Jina) only fires for true SPAs.
            if jina_fallback and gemini_api_key:
                return extract_jobs_from_careers_page(careers_url, company_name, gemini_api_key, html=html)
            return []
        print(f"  ATS detected for {company_name}: {ats}/{token}")

    fetcher = _ATS_FETCHERS.get(ats)
    if not fetcher:
        return []
    return fetcher(token, company_name)
