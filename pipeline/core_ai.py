import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import re
import json

from pipeline.logging_setup import get_logger

logger = get_logger(__name__)
# Heavy deps (`google.genai`, `ddgs`) are lazy-imported inside the functions that
# actually need them so the parser/renderer helpers can be unit-tested without
# requiring the full runtime stack to be installed.

"""
CORE AI MODULE
--------------
The brain of the operation. Uses DuckDuckGo to deep-search the web for remote policies
and leverages Google's Gemini AI to mathematically evaluate the candidate's CV against
the specific job requirements, yielding a rich verdict with sub-scores and metadata.

Output schema (see DEFAULT_AI_RESULT) includes:
- is_valid:         bool — true if candidate should consider applying
- verdict:          str  — 1-2 sentences citing the SPECIFIC CV asset driving the match
- tech_fit:         int  — 0-100, overlap with candidate's STRONGEST stack
- experience_fit:   int  — 0-100, how the years-required gap compares to candidate's 0 prof years
- logistics_fit:    int  — 0-100, geographic + timezone overlap with UTC+2 Palestine
- match_percentage: int  — 0-100, weighted composite (cap 60 if suspicious)
- compensation:     str  — extracted pay info, "Not stated" if nothing
- effort:           str  — "low" | "medium" | "high" application effort
- suspicious:       bool — true for job-mill / training-program / lazy-repost patterns
"""

# ---------------------------------------------------------------------------
# Result schema + safe coercion helpers (separated so they can be unit-tested
# without making real Gemini calls).
# ---------------------------------------------------------------------------

DEFAULT_AI_RESULT = {
    "is_valid": False,
    "verdict": "No verdict",
    "tech_fit": 0,
    "experience_fit": 0,
    "logistics_fit": 0,
    "match_percentage": 0,
    "compensation": "Not stated",
    "effort": "unknown",
    "suspicious": False,
    "scam": False,                          # set True only after web-based scam confirmation
}

# Phrases that, in a company name or job location, strongly suggest an India-based
# employer. Used by the scam-check to limit Reddit/review lookups to that segment.
_INDIA_SIGNALS = (
    "india", "pvt ltd", "pvt. ltd", "private limited", "(p) ltd", "pvt limited",
    "bangalore", "bengaluru", "hyderabad", "mumbai", "delhi", "noida", "gurgaon",
    "chennai", "pune", "kolkata", "ahmedabad",
)
_SCAM_KEYWORDS = (
    "scam", "fraud", "fake job", "fraudulent", "did not pay", "didn't pay",
    "ghost job", "ponzi", "pyramid scheme",
)


def looks_like_india_employer(location, company):
    """Cheap text check: is this posting plausibly an India-based employer?"""
    text = f"{location or ''} {company or ''}".lower()
    return any(sig in text for sig in _INDIA_SIGNALS)


def scan_for_scam_signals(text, min_matches=2):
    """Count how many scam keywords appear in a body of text. >=min_matches -> True."""
    t = (text or "").lower()
    hits = sum(1 for kw in _SCAM_KEYWORDS if kw in t)
    return hits >= min_matches


def detect_company_scam(company_name):
    """For an India-flagged suspicious company, search the open web for scam reports.

    Runs three short DDG queries and returns True iff at least 2 distinct scam keywords
    appear across the combined result bodies. Conservative by design — false positives
    here would incorrectly tag legitimate Indian companies.
    """
    if not company_name:
        return False
    queries = [
        f'"{company_name}" scam',
        f'"{company_name}" fake job complaints',
        f'"{company_name}" reddit review',
    ]
    snippets = []
    for q in queries:
        snippets.extend(r.get('body', '') for r in _ddg_text(q, max_results=2))
    return scan_for_scam_signals(" ".join(snippets))

_NUMBER_RE = re.compile(r'^\s*(-?\d+(?:\.\d+)?)')

def _safe_int(val, default=0):
    """Best-effort integer in [0, 100]. Handles ints, floats, '82', '82%', '82.5', None."""
    if isinstance(val, bool):
        return min(100, max(0, int(val)))
    if isinstance(val, (int, float)):
        return min(100, max(0, int(val)))
    if isinstance(val, str):
        m = _NUMBER_RE.match(val)
        if m:
            return min(100, max(0, int(float(m.group(1)))))
    return default

def _safe_bool(val, default=False):
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in ('true', 'yes', '1')
    if isinstance(val, (int, float)):
        return bool(val)
    return default

def _safe_str(val, default=""):
    if val is None:
        return default
    return str(val).strip() or default

def _normalize_effort(val):
    """Constrain effort to one of: low, medium, high, unknown."""
    s = _safe_str(val, "unknown").lower()
    if s in ("low", "medium", "high"):
        return s
    return "unknown"

def _normalize_result(raw):
    """Coerce a raw AI response dict into the canonical schema with safe types."""
    if not isinstance(raw, dict):
        return dict(DEFAULT_AI_RESULT)
    return {
        "is_valid":         _safe_bool(raw.get("is_valid"), False),
        "verdict":          _safe_str(raw.get("verdict"), "No verdict"),
        "tech_fit":         _safe_int(raw.get("tech_fit"), 0),
        "experience_fit":   _safe_int(raw.get("experience_fit"), 0),
        "logistics_fit":    _safe_int(raw.get("logistics_fit"), 0),
        "match_percentage": _safe_int(raw.get("match_percentage"), 0),
        "compensation":     _safe_str(raw.get("compensation"), "Not stated"),
        "effort":           _normalize_effort(raw.get("effort")),
        "suspicious":       _safe_bool(raw.get("suspicious"), False),
        "scam":             _safe_bool(raw.get("scam"), False),
    }

def apply_post_ai_caps(result, row):
    """Apply deterministic match-percentage caps to an AI verdict result.

    Two caps, both at 55%, applied in priority order:

      1. Reputation cap (A1): companies on `data/reputation.json`'s blacklist
         (carried into the row as `pre_flagged_low_quality=True` by the
         filter chain) cannot score above 55 no matter what the AI thinks.
         Verdict gets a `[BLACKLISTED]` prefix.

      2. AI-suspicious self-cap (new 2026-05-17 after India-sus internships
         leaked into the daily email at match=80%). Even when the reputation
         list hasn't caught a company yet, if the AI itself sets
         `suspicious=True` and gives a score above 55, we clamp to 55 and
         tag the verdict `[AI-SUSPICIOUS]`. The reputation cap is checked
         first because it's a stronger signal (curated human judgment beats
         the AI's per-row guess).

    Pure function — mutates and returns the result dict. Reading-only against
    the row. Tested in QA/unit/test_post_ai_caps.py.
    """
    if not isinstance(result, dict):
        return result

    is_blacklisted = bool(row.get("pre_flagged_low_quality", False))

    # 1. Reputation cap (A1).
    if is_blacklisted:
        if result.get("match_percentage", 0) > 55:
            result["match_percentage"] = 55
        verdict = result.get("verdict", "") or ""
        if not verdict.startswith("[BLACKLISTED]"):
            result["verdict"] = "[BLACKLISTED] " + verdict
        return result

    # 2. AI-suspicious self-cap (Fix #5).
    if bool(result.get("suspicious")) and result.get("match_percentage", 0) > 55:
        result["match_percentage"] = 55
        verdict = result.get("verdict", "") or ""
        # Don't double-tag if the AI's own verdict already mentions suspicion;
        # only add our explicit marker. We never collide with [BLACKLISTED]
        # because that branch returned above.
        if not verdict.startswith(("[BLACKLISTED]", "[SCAM]", "[AI-SUSPICIOUS]")):
            result["verdict"] = "[AI-SUSPICIOUS] " + verdict

    return result


def _extract_json_object(text):
    """Return the first balanced top-level {...} JSON object in `text`, or None.

    Reasoning models (e.g. gpt-oss-120b) can prepend chain-of-thought prose or
    append commentary around the JSON verdict. We scan for the first '{' and
    brace-match to its close, respecting quoted strings and escapes, so a stray
    preamble/suffix doesn't break json.loads.
    """
    start = text.find('{')
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == '\\':
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _parse_ai_response(text):
    """Parse a raw model output string into the canonical result dict.

    Tolerates: markdown ```json fences, surrounding whitespace, missing fields,
    string-typed numbers, percent signs in numeric fields, and a reasoning-model
    preamble/suffix wrapped around the JSON object.
    """
    if not text:
        raise ValueError("Empty AI response")
    t = text.strip()
    if t.startswith('```json'):
        t = t[7:]
    elif t.startswith('```'):
        t = t[3:]
    if t.endswith('```'):
        t = t[:-3]
    t = t.strip()
    try:
        raw = json.loads(t)
    except json.JSONDecodeError:
        # The model wrapped the JSON in prose (common with reasoning models).
        # Extract the first balanced object and parse that instead.
        block = _extract_json_object(t)
        if block is None:
            raise
        raw = json.loads(block)
    return _normalize_result(raw)

# ---------------------------------------------------------------------------
# DDG search helper with exponential backoff
# ---------------------------------------------------------------------------

_DDG_BACKOFF_SECONDS = [2, 5]          # wait 2s after attempt 1, 5s after attempt 2

def _ddg_text(query, max_results=2):
    """Execute one DDG text query with exponential backoff on rate-limit / network errors.

    GitHub Actions runners share IP space so DDG's rate limiter fires more often
    than it would from a home IP. Three attempts (2s → 5s backoff) handle the
    brief rate-limit windows DDG applies without waiting so long that the runner
    times out.

    Returns a list of result dicts on success, [] if all retries are exhausted.
    """
    from ddgs import DDGS  # lazy — only loaded when an AI eval actually fires
    max_retries = len(_DDG_BACKOFF_SECONDS) + 1
    for attempt in range(max_retries):
        try:
            return list(DDGS().text(query, max_results=max_results))
        except Exception as e:
            err = str(e)[:120]
            if attempt < len(_DDG_BACKOFF_SECONDS):
                wait = _DDG_BACKOFF_SECONDS[attempt]
                logger.warning("DDG attempt %d/%d failed (%s), retrying in %ds...", attempt + 1, max_retries, err, wait)
                time.sleep(wait)
            else:
                logger.warning("DDG exhausted retries for query: %r (%s)", query[:80], err)
    return []


# ---------------------------------------------------------------------------
# Description fetch + web search helpers
# ---------------------------------------------------------------------------

def get_full_job_description(url):
    """Fallback scraper for job URLs when the API only returns a truncated description."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64 AppleWebKit/537.36)'}
        res = requests.get(url, headers=headers, timeout=5)
        if res.status_code == 200:
            if "authwall" in res.url.lower() or "sign in to linkedin" in res.text.lower():
                return "[DESCRIPTION TRUNCATED BY LINKEDIN LOGIN WALL]"
            soup = BeautifulSoup(res.text, 'html.parser')
            for script in soup(["script", "style"]):
                script.extract()
            text = soup.get_text(separator=' ')
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            return '\n'.join(chunk for chunk in chunks if chunk)
    except:
        pass
    return ""

def search_company_remote_policy(company_name, job_title):
    """Dual DuckDuckGo search for specific geographic restrictions tied to this role."""
    logger.info("Deep web search triggered for %s (%s) remote policy...", company_name, job_title)
    snippets = []
    q1 = f"{company_name} \"{job_title}\" remote eligible countries"
    snippets.extend(r.get('body', '') for r in _ddg_text(q1, max_results=2))
    q2 = f"{company_name} hire remote Middle East Palestine EMEA"
    snippets.extend(r.get('body', '') for r in _ddg_text(q2, max_results=2))
    return " ".join(snippets)

# ---------------------------------------------------------------------------
# The main evaluation entry point
# ---------------------------------------------------------------------------

def _error_result(message):
    """Build an AI-error result dict (everything zeroed, is_valid False)."""
    r = dict(DEFAULT_AI_RESULT)
    r["verdict"] = message
    return r


def skipped_result(reason):
    """Build a result dict for a job the pre-screen rejected (A2 / Fix 9).

    The skip is deterministic, so callers should still mark the URL as seen —
    re-running the heuristic on the same row will produce the same outcome,
    no point spending tokens on it again.
    """
    r = dict(DEFAULT_AI_RESULT)
    r["verdict"] = f"Pre-screen skipped: {reason}"
    r["is_valid"] = False
    return r


# Heuristic pre-screen patterns. Kept as module constants so tests can import them.
_NON_TECH_TITLE_SIGNALS = (
    "sales ", "marketing ", " hr ", " hr-", "recruiter", "talent acquisition",
    "customer success", "account executive", "account manager",
    " operations ", "social media", "copywriter", "growth ",
    "business development",
)
_SENIOR_REGEX_PATTERNS = (
    r"\b[5-9]\+?\s*years?\s+(?:of\s+)?(?:experience|exp)\b",
    r"\b1[0-9]\+?\s*years?\s+(?:of\s+)?(?:experience|exp)\b",
    r"\bminimum\s+(?:of\s+)?[5-9]\s*years?\b",
    r"\bsenior-level\s+(?:engineer|developer|role)\b",
    r"\bstaff\s+engineer\b",
    r"\bprincipal\s+engineer\b",
)

# Hard work-auth / clearance disqualifiers for a Palestine-based candidate.
# Match conservatively — we want zero false positives, since these rules drop
# the job entirely before any AI scoring. Each pattern is a real phrase seen in
# US-locked postings; generic mentions of "US" or "United States" alone are
# NOT enough to disqualify (a global remote company can mention HQ location).
_HARD_DISQUALIFIER_PATTERNS = (
    # Explicit citizenship / residency requirements
    (r"\bu\.?s\.?\s+citizens?\s+only\b",                          "US citizens only"),
    (r"\bmust\s+be\s+(?:a\s+)?u\.?s\.?\s+citizen\b",              "must be a US citizen"),
    (r"\bmust\s+reside\s+in\s+the\s+(?:u\.?s\.?|united\s+states)\b", "must reside in the US"),
    (r"\bu\.?s\.?[\-\s]based\s+(?:candidates?|applicants?)\s+only\b", "US-based candidates only"),
    (r"\bus\s+only\s+remote\b",                                   "US-only remote"),
    (r"\bremote\s*[-(]\s*us\s*(?:only)?\s*[)\-]?\s*$",            "remote (US only) — anchored"),
    # Work authorization that excludes non-residents (Palestine isn't covered)
    (r"\bmust\s+be\s+authorized\s+to\s+work\s+in\s+the\s+(?:u\.?s\.?|united\s+states)\b", "must be authorized to work in the US"),
    (r"\b(?:u\.?s\.?|united\s+states)\s+work\s+authorization\s+required\b", "US work authorization required"),
    (r"\bno\s+(?:visa\s+)?sponsorship\s+(?:will\s+be\s+)?(?:provided|offered|available)\b", "no visa sponsorship"),
    (r"\bunable\s+to\s+sponsor\s+(?:work\s+)?visas?\b",           "unable to sponsor visas"),
    # Security clearances (Palestinian candidate categorically cannot hold these)
    (r"\bsecurity\s+clearance\s+required\b",                      "security clearance required"),
    (r"\bactive\s+(?:security\s+)?clearance\b",                   "active clearance required"),
    (r"\b(?:ts\s*/\s*sci|top\s+secret\s*/\s*sci)\b",              "TS/SCI clearance"),
    (r"\bsecret\s+clearance\b",                                   "secret clearance"),
    (r"\bpublic\s+trust\s+clearance\b",                           "public trust clearance"),
)

# Specific country / region names that, when a job is exclusively locked to
# them, mean a Palestine-based candidate can't apply. Deliberately omits MENA
# / Middle East / Worldwide / Global since those would include Palestine.
# US has its own dedicated patterns in _HARD_DISQUALIFIER_PATTERNS above.
_GEO_LOCK_LOCATIONS = (
    # Anglosphere
    "canada", "canadian",
    "australia", "australian", "australasia",
    "new zealand",
    "united kingdom", "uk", "britain", "british", "england", "scotland", "wales", "ireland",
    # Europe / EU
    "germany", "german", "austria", "austrian", "switzerland", "swiss",
    "france", "french", "italy", "italian", "spain", "spanish",
    "portugal", "portuguese", "greece", "greek",
    "netherlands", "dutch", "belgium", "belgian",
    "sweden", "swedish", "norway", "norwegian", "denmark", "danish",
    "finland", "finnish", "poland", "polish",
    "romania", "romanian", "hungary", "hungarian",
    "bulgaria", "bulgarian", "ukraine", "ukrainian",
    # Latin America
    "brazil", "brazilian", "argentina", "argentine",
    "mexico", "mexican", "chile", "chilean",
    "colombia", "colombian", "peru", "peruvian",
    # Asia
    "india", "indian", "china", "chinese",
    "japan", "japanese", "south korea", "korean",
    "singapore", "thailand", "thai", "vietnam", "vietnamese",
    "philippines", "filipino", "indonesia", "indonesian",
    "malaysia", "malaysian",
    # Africa
    "south africa", "south african",
    "nigeria", "nigerian", "kenya", "kenyan",
    # Russia / Turkey / Israel
    "russia", "russian",
    "turkey", "türkiye", "turkish",
    "israel", "israeli",
    # Gulf (Palestine doesn't share Gulf work-permit reciprocity)
    "saudi arabia", "uae", "united arab emirates",
    "qatar", "kuwait", "bahrain", "oman",
    # Regional collective locks that exclude Palestine
    "eu", "european union", "eea",
    "latam", "latin america", "apac", "asia pacific", "asia-pacific",
    "anz",
)


_GEO_LOCK_ALTERNATION = "|".join(re.escape(g) for g in _GEO_LOCK_LOCATIONS)


def _build_geo_lock_regex():
    """Compile a single regex catching '<non-Palestine geo>-only' restrictions.

    One compiled regex is ~80x faster than iterating per-country patterns when
    quick_viability_check() runs on every filtered job. The location list
    omits MENA / Middle East / Worldwide / Global since those include Palestine
    and shouldn't disqualify.
    """
    geos = _GEO_LOCK_ALTERNATION
    templates = [
        # "Canadian residents only", "Brazil citizens only"
        rf"\b(?:{geos})\s+(?:citizens?|residents?|nationals?)\s+only\b",
        # "must be based in Germany", "must reside in Canada"
        rf"\bmust\s+(?:be\s+)?(?:a\s+|legal\s+)?(?:located|based|reside|residing|resident|citizen)\s+(?:in|of)\s+(?:the\s+)?(?:{geos})\b",
        # "only open to candidates in India", "only available for applicants based in UK"
        rf"\bonly\s+(?:open|available|hiring)\s+(?:to|for)\s+(?:candidates?|applicants?)\s+(?:in|from|based\s+in|located\s+in|residing\s+in)\s+(?:the\s+)?(?:{geos})\b",
        # "role is only available to candidates in Mexico"
        rf"\b(?:role|position|opportunity|opening|job)\s+(?:is\s+)?(?:only\s+)?(?:open\s+to|available\s+(?:to|in|for))\s+(?:candidates?\s+in\s+)?(?:the\s+)?(?:{geos})\b",
        # "must have the right to work in Australia"
        rf"\bmust\s+(?:have|hold|possess|be\s+able)\s+(?:the\s+)?(?:right|legal\s+right|authorization|work\s+permit)\s+to\s+work\s+in\s+(?:the\s+)?(?:{geos})\b",
        # "authorized to work in <country>" / "<country> work authorization required"
        rf"\bmust\s+be\s+authorized\s+to\s+work\s+in\s+(?:the\s+)?(?:{geos})\b",
        rf"\b(?:{geos})\s+work\s+authorization\s+required\b",
        # "remote within Canada only", "remote in EU only"
        rf"\bremote(?:\s+(?:work|role|position))?\s+(?:in|within)\s+(?:the\s+)?(?:{geos})\s+only\b",
        # "candidates must be located in Brazil only"
        rf"\bcandidates?\s+(?:must\s+)?(?:be\s+)?(?:from|based\s+in|located\s+in|residing\s+in)\s+(?:the\s+)?(?:{geos})\s+only\b",
        # "Canada-based candidates only" / "Canada based hires only"
        rf"\b(?:{geos})[\-\s]based\s+(?:candidates?|applicants?|hires?|employees?)\s+only\b",
        # "APAC candidates only" / "Brazil applicants only" (no "-based" between)
        rf"\b(?:{geos})\s+(?:candidates?|applicants?|workers?|hires?|employees?)\s+only\b",
        # "hiring APAC candidates only" / "hiring Brazil workers only"
        rf"\bhiring\s+(?:{geos})\s+(?:candidates?|applicants?|workers?|hires?|employees?)\s+only\b",
        # "<country>-only role" / "<country>-only remote"
        rf"\b(?:{geos})[\-\s]only\s+(?:role|position|remote|hiring)\b",
        # "UK (Remote)" / "UK or US (Remote)" — location-label style restriction
        # appearing in descriptions copied from job boards. Parenthesised "(Remote)"
        # after a geo-locked country is unambiguous: the role is remote-eligible only
        # for residents of that country.
        rf"\b(?:{geos})\b(?:\s*(?:or|/)\s*\b(?:{geos})\b)*\s*\(remote\)",
        # "Remote — UK" / "Remote - US" — dash-separated location label
        rf"\bremote\s*[-–—]\s*(?:the\s+)?(?:{geos})\b",
    ]
    return re.compile("|".join(templates))


_GEO_LOCK_REGEX = _build_geo_lock_regex()

_LOCATION_GEO_LOCK_RE = re.compile(
    # "Remote in Netherlands" / "Remote from UK" / "Remote — Germany"
    rf"\bremote\s+(?:in|from|[-–—])\s+(?:the\s+)?(?:{_GEO_LOCK_ALTERNATION})\b",
    re.IGNORECASE,
)

_LOCATION_COUNTRY_PAREN_REMOTE_RE = re.compile(
    # "UK (Remote)" / "UK or US (Remote)" / "UK/US (Remote)" / "United Kingdom (Remote)"
    # The location field is short structured data so .{0,30} is safe — it allows an
    # optional second country like "or US" between the geo-locked country and "(Remote)".
    rf"\b(?:{_GEO_LOCK_ALTERNATION})\b.{{0,30}}\(remote\)",
    re.IGNORECASE,
)

_MIN_DESCRIPTION_CHARS = 150


def quick_viability_check(row):
    """Cheap heuristic that decides whether to spend a Gemini call on this job (A2).

    Runs AFTER apply_pipeline_filters (so seniority titles, non-tech keywords,
    non-English titles, etc. are already gone) but BEFORE the expensive AI eval.

    Returns (is_viable: bool, reason: str). reason is a short tag suitable for log lines.
    """
    title = str(row.get("title", "")).lower()
    description = str(row.get("description", "")).lower()
    description_clean = description.strip()

    # 1. Reputation list already flagged this. Don't waste a call.
    if bool(row.get("pre_flagged_low_quality", False)):
        return False, "blacklisted by reputation list"

    # 2. Title looks tech-keyworded but the role is actually sales/marketing/HR/etc.
    for signal in _NON_TECH_TITLE_SIGNALS:
        if signal in title:
            return False, f"non-tech role signal in title ({signal.strip()})"

    # 3. Description-length sanity check, with several bypasses for cases where a
    # "short" description doesn't actually mean "lazy repost":
    #   - Literally missing values from APIs (`nan` / `None` / `null` / empty / <20 chars)
    #     should pass through so `evaluate_job_with_ai`'s URL-fetch fallback can run.
    #   - DDG-sourced snippets (from the local pipeline) are legitimately terse —
    #     a LinkedIn/website hit's body is often just a hashtag teaser. These are
    #     tagged source="ddg_*". (Historically detected via a "LinkedIn Post:"
    #     title prefix, removed when local titles became real — the source tag is
    #     the durable signal now.)
    #   - Truncated-by-authwall placeholders trigger the AI's Limited Info Protocol.
    is_missing = description_clean in ("", "nan", "none", "null") or len(description_clean) < 20
    source = str(row.get("source", "")).strip().lower()
    is_ddg_snippet = source.startswith("ddg_") or title.startswith("linkedin post:")
    is_truncated_placeholder = (
        "[no description" in description or "[description truncated" in description
    )

    if not (is_missing or is_ddg_snippet or is_truncated_placeholder):
        if len(description_clean) < _MIN_DESCRIPTION_CHARS:
            return False, f"description too short ({len(description_clean)} chars)"

    # 4. Explicit senior-experience requirement that we have no chance of meeting.
    for pat in _SENIOR_REGEX_PATTERNS:
        if re.search(pat, description):
            return False, "senior experience requirement in description"

    # 5. Hard work-auth / clearance disqualifiers. Candidate is in Palestine —
    # US-citizen / US-resident / clearance-required roles are non-starters and
    # there's no point burning a Gemini call to reach that same conclusion.
    for pat, reason in _HARD_DISQUALIFIER_PATTERNS:
        if re.search(pat, description):
            return False, f"hard disqualifier: {reason}"

    # 6. Generalized geo-lock to a non-Palestine country/region (added 2026-05-19).
    # Catches phrases like "Canadian residents only", "must reside in Australia",
    # "EU candidates only", "Brazil-based hires only". One compiled regex covers
    # ~80 countries × 10 phrase patterns. MENA / Middle East / Worldwide / Global
    # are intentionally absent from the geo list since they include Palestine.
    m = _GEO_LOCK_REGEX.search(description)
    if m:
        return False, f"hard disqualifier: geo-locked ({m.group(0)[:60].strip()})"

    # 7. Location-field geo-lock (added 2026-05-22). Job boards often encode
    # geo-restrictions directly in the location string: "Remote in Netherlands",
    # "Remote - Germany". The description regex (step 6) requires an "only"
    # suffix to avoid false positives in longer text, but a short location
    # field like "Remote in Netherlands" is unambiguous.
    location = str(row.get("location", "")).strip()
    if location:
        m = _LOCATION_GEO_LOCK_RE.search(location)
        if m:
            return False, f"location geo-locked ({location[:60].strip()})"
        m2 = _LOCATION_COUNTRY_PAREN_REMOTE_RE.search(location)
        if m2:
            return False, f"location geo-locked ({location[:60].strip()})"

    return True, "viable"

_WEB_SEARCH_TRIGGER_PHRASES = (
    "eligible countries", "selected countries", "certain countries",
    "must be based", "candidates based", "residents of",
    "remote in the", "must be located", "work authorization",
    "within the united states", "us only", "us-based", "uk only", "eu only",
)
_EXPLICIT_GLOBAL_PHRASES = (
    "worldwide", "globally remote", "anywhere in the world",
    "no location restriction", "no geographic restriction",
    "open to all locations", "global candidates", "emea welcome",
    "middle east", "fully remote globally", "all countries", "global remote",
)
_AMBIGUITY_CONTEXT_PHRASES = (
    "timezone", "country", "region", "office", "headquartered",
    "headquarters", "based ", "located in",
)


def _maybe_get_full_description(row):
    """Pull a fuller description via the URL scraper when the API description is truncated."""
    description = str(row.get("description", ""))
    if pd.isna(description) or len(description) < 100:
        description = get_full_job_description(str(row.get("job_url", "")))
        if not description:
            description = "[NO DESCRIPTION AVAILABLE - SCRAPING BLOCKED]"
    return description


def _build_verdict_prompt(row, cv_text, description, web_search_context="", learned_preferences=""):
    """Construct the canonical recruiter-screen prompt shared by Cerebras+Groq and Gemini paths.

    `learned_preferences` is the AI-summarized profile derived from the user's
    historical feedback (Applied / Bookmarked / Not relevant / etc.). When
    non-empty it appears as its own section and is treated as candidate
    context, not as a job-specific instruction.
    """
    title = str(row.get("title", ""))
    company = str(row.get("company", ""))
    job_type = str(row.get("job_type", "")).lower()
    is_internship = 'intern' in title.lower() or 'internship' in job_type

    preferences_block = ""
    if learned_preferences and learned_preferences.strip():
        preferences_block = (
            "\nCANDIDATE LEARNED PREFERENCES (inferred from prior user feedback — "
            "weight your scoring toward roles aligned with these, and away from the rejected patterns):\n"
            f"{learned_preferences.strip()}\n"
        )

    return f"""You are a SKEPTICAL technical recruiter screening a candidate. Your job is to find
DISQUALIFYING reasons. Default to skepticism. A 90+ score is reserved for cases where
you cannot reasonably imagine why the candidate would NOT be considered.

CANDIDATE CV (read it fully — everything you score MUST come from this CV, not assumptions):
{cv_text[:3000]}

HOW TO READ THE CV (derive these from the CV above before scoring — do NOT invent):
- LOCATION: the candidate is based in Palestine (UTC+2). This is fixed for every
  candidate on this platform; use it for LOGISTICS_FIT below.
- PROFESSIONAL EXPERIENCE: count real work experience from the CV — jobs and
  internships. If the CV states years of experience, use that. If it does not,
  INFER from internships and substantial projects: no internship and no related
  work experience → treat as 0 professional years (a student/new-grad profile);
  internships and strong projects raise the effective level. Projects DO count as
  evidence of ability, but they are not the same as professional years — weigh
  them as project experience, not employment.
- STRONGEST ASSETS: identify the candidate's actual strongest skills, tools, and
  projects FROM THE CV (languages, frameworks, domains, named projects). Use
  THESE — not any assumed stack — when judging TECH_FIT and when naming matches.
{preferences_block}
JOB:
- Title: {title}
- Company: {company}
- Is Internship?: {is_internship}
- Description:
{description[:5000]}
{web_search_context}

EVALUATION RULES (apply rigorously, default to deducting points):

1. LOGISTICS_FIT (0-100) — purely about whether the candidate is GEOGRAPHICALLY eligible.
   Timezone differences DO NOT matter — the candidate is happy working any shift as
   long as the role is fully remote and open to their location.

   ELIGIBILITY BANDS:
   - "Worldwide" / "Global" / "EMEA welcome" / "anywhere" / no geographic restriction → 90-100
   - Description doesn't mention restriction but doesn't explicitly welcome remote-global either → 70-85
   - WORK AUTHORIZATION RESTRICTIONS — these are HARD disqualifiers. Scan for ANY of:
       * "must be authorized to work in the US/UK/EU/[country]"
       * "must be legally able to work in [country]"
       * "visa sponsorship not available" / "no visa sponsorship"
       * "must be eligible to work in [country] without sponsorship"
       * "US citizens / green card holders only"
       * "must reside in [non-MENA country]"
       * "UK or US remote" / "remote (UK only)" / "[country] (Remote)" — remote-eligible
         ONLY for residents of named countries. This IS a geographic restriction.
         ⚠ Do NOT write "no country restrictions" for a "UK or US Remote" role.
         A role that is remote for UK and US residents only EXCLUDES the candidate in
         Palestine. Set is_valid=false AND logistics_fit <= 15.
     If ANY such phrase appears (or the web search reveals one), set is_valid=false AND
     logistics_fit <= 15. Note it explicitly in the verdict.
   - Explicit exclusion of Palestine / Middle East → same: is_valid=false, logistics_fit <= 15.

2. EXPERIENCE_FIT (0-100) — the role's REQUIRED experience vs the candidate's
   experience as derived from the CV (professional years inferred per the rules
   above; projects count as project experience, not employment years):
   - Role wants "Internship" / "0-1 year" / "entry-level" → 85-100 for a
     student / new-grad profile; higher if the CV shows relevant internships.
   - Role wants "1-2 years" → score by what the CV actually shows: a candidate
     with relevant internships/projects lands mid-high (60-80); one with nothing
     relevant lands lower (40-60).
   - Role wants "3+ years" → 20-40 unless the CV shows comparable real experience
     (then score it on the evidence); likely is_valid=false for a pure-projects
     profile unless the role explicitly accepts project-equivalents.
   - Role wants "5+ years" / "Senior" → unless the CV genuinely shows senior-level
     experience, set is_valid=false AND experience_fit <= 25.

3. TECH_FIT (0-100) — overlap between the candidate's STRONGEST assets (as found in
   the CV) and the job's REQUIRED stack. Judge against what THIS candidate's CV
   actually shows, not any assumed stack:
   - Direct overlap (the role explicitly wants skills/tools/domains the CV clearly
     demonstrates) → 85-100, and the verdict MUST name the specific CV skill or
     project that matches.
   - Adjacent overlap (same broad area as the CV, but no specific named match) → 60-80.
   - The role's core stack is a domain the CV does NOT demonstrate (e.g. a heavy
     frontend role for a backend-only CV, or vice versa) → max 60.
   - Fundamental mismatch (the role's core discipline is absent from the CV, e.g.
     DevOps/SRE/Quant for a CV with none of it) → max 40.
   - A single shared language or buzzword alone is NOT 90+. Require specific,
     demonstrated overlap to clear 85.

4. SUSPICIOUS POSTING — set suspicious=true if ANY of these hold:
   - Company name fits the pattern "[Random Word] IT Solution/Service/Solutions/Services/Jobs/Mentor"
   - Description mentions: "certification provided", "earn while you learn", "$0 stipend",
     "course-based internship", "training program with placement", "bootcamp with placement"
   - Description is under 150 characters (lazy reposting)
   - "Stipend" mentioned with no concrete dollar amount
   When suspicious=true, CAP match_percentage at 60 regardless of sub-scores.

5. COMPENSATION — extract pay info verbatim:
   - "$X/hour" or "$X-Y/year" if explicit
   - "Unpaid" if stated
   - "Stipend (amount unclear)" if mentioned without a number
   - "Not stated" if no info

6. EFFORT — estimated application effort:
   - "low": one-click apply or just submit resume
   - "medium": cover letter / short essay / portfolio link required
   - "high": take-home assignment, 4+ interview rounds, multi-page essays

7. MATCH_PERCENTAGE COMPUTATION (do the math explicitly):
   - base = 0.5 * tech_fit + 0.3 * experience_fit + 0.2 * logistics_fit
   - if suspicious=true: match_percentage = min(60, base)
   - else: match_percentage = base (clamped to 0-100)
   - Anchored interpretation:
       90-100: Near-ideal pairing; apply immediately.
       75-89:  Strong fit with minor concern.
       60-74:  Stretch fit; significant concern noted in verdict.
       40-59:  Poor fit; would not recommend applying.
       0-39:   Should not apply.

8. VERDICT — write as a senior recruiter would: structured, specific, no fluff.
   Required structure (3-5 sentences, in this order):
   a) MATCH: name 1-2 SPECIFIC assets FROM THIS CANDIDATE'S CV (projects or
      technologies, by their real name as written in the CV) that directly address
      what the job asks for. Pull the names from the CV above — never invent a
      project the CV doesn't mention. E.g. (illustrative format only — use the
      candidate's ACTUAL projects/skills):
      "Their [named CV project] using [CV technologies] directly matches the role's
       stated need for [requirement from the job]."
   b) REMOTE: in ONE sentence, state WHY this role is geographically accessible to
      the candidate in Palestine. This is MANDATORY for every verdict. Examples:
      "Listed as worldwide remote with no country restrictions."
      "Explicitly welcomes EMEA candidates."
      "No geographic restriction mentioned — assumed open."
      ⚠ "UK or US Remote" is NOT "no country restrictions." If the role is remote
      only for specific countries that do not include Palestine, write that it is
      geo-restricted and set is_valid=false. E.g., "Remote is UK/US-only — candidate
      in Palestine is excluded."
   c) GAP: name the SPECIFIC missing requirement the candidate doesn't have, judged
      against THIS CV. E.g., "The role also requires [requirement] — the CV shows
      no evidence of it."
   d) (Optional) SECOND MATCH/GAP: a secondary positive or concern.
   e) (Required only if is_valid=false) CLOSING REASON: explicitly state the
      disqualifier (work auth, geo exclusion, senior-only, scam suspicion).

   STRICT VOCABULARY RULES:
   - Generic phrases are FORBIDDEN: "strong technical match", "strong Python skills",
     "good fit", "well-aligned". Be specific or don't say it.
   - Cite the candidate's projects and tech BY NAME, using the real names from THIS
     CV (whatever they are) — never a project the CV doesn't contain.
   - When citing a gap, name the missing tech/experience SPECIFICALLY ("no AWS production
     experience", "no React frontend background"), not vaguely ("limited experience").

9. LIMITED INFO PROTOCOL:
   - If description contains [DESCRIPTION TRUNCATED] or [NO DESCRIPTION], deduct 10 from
     each sub-score. Do NOT set suspicious=true purely because of missing info.
   - Note the missing description in the verdict.

Reply with VALID JSON ONLY (no markdown, no comments):
{{"is_valid": true|false, "verdict": "...", "tech_fit": 0-100, "experience_fit": 0-100, "logistics_fit": 0-100, "match_percentage": 0-100, "compensation": "...", "effort": "low|medium|high", "suspicious": true|false}}
"""


def _maybe_web_search_context(company, title, description):
    """Build the optional `[LIVE WEB SEARCH RESULTS...]` block when the description
    looks restriction-ambiguous. Returns empty string if no triggers fire (or if
    the description has explicit-global signals)."""
    desc_lower = description.lower()
    has_restriction = any(t in desc_lower for t in _WEB_SEARCH_TRIGGER_PHRASES)
    has_global_signal = any(p in desc_lower for p in _EXPLICIT_GLOBAL_PHRASES)
    has_ambiguity_context = any(p in desc_lower for p in _AMBIGUITY_CONTEXT_PHRASES)

    if has_restriction or (has_ambiguity_context and not has_global_signal):
        search_data = search_company_remote_policy(company, title)
        if search_data:
            return (
                f"\n\n[LIVE WEB SEARCH RESULTS FOR '{company}' REMOTE POLICY]:\n"
                f"{search_data}\n\n"
                f"Use this live web data to determine if Palestine/Middle East is "
                f"explicitly excluded from their remote eligible countries."
            )
    return ""


def _apply_india_scam_check(result, row, company):
    """Run the DDG-based scam confirmation for India-flagged suspicious companies.

    Mutates `result` in place when a scam is confirmed (sets scam=True, is_valid=False,
    caps match_percentage at 30, prefixes verdict with [SCAM]). Pure no-op otherwise.
    """
    if not result.get("suspicious"):
        return result
    location_text = str(row.get("location", ""))
    if not looks_like_india_employer(location_text, company):
        return result
    if detect_company_scam(company):
        result["scam"] = True
        result["is_valid"] = False
        if result["match_percentage"] > 30:
            result["match_percentage"] = 30
        if not result["verdict"].startswith("[SCAM]"):
            result["verdict"] = "[SCAM] " + result["verdict"]
    return result


def evaluate_job_with_ai(row, cv_text, cerebras_key, groq_key, learned_preferences=""):
    """
    Evaluate a single job posting against the candidate's CV using qwen-3-235b
    via Cerebras (primary) with llama-3.3-70b on Groq as fallback.

    Returns a 2-tuple: (result_dict, evaluated_bool).
    - result_dict follows DEFAULT_AI_RESULT schema.
    - evaluated_bool is True ONLY when the LLM returned a real verdict; callers should
      use this to decide whether to mark the URL as "seen" (so transient API errors
      don't lose jobs forever — see core_filter.JobTracker).

    `learned_preferences` is an optional AI-summarized profile from prior user
    feedback; when present it gets injected into the prompt's candidate
    context and steers scoring toward (or away from) historical patterns.

    Fallback behavior lives in pipeline.core_llm.call_llm_with_fallback:
    Cerebras -> Groq -> Cerebras -> Groq (min 4 attempts on transient errors).
    """
    if not cerebras_key and not groq_key:
        return _error_result("No LLM API Key provided (CEREBRAS_API_KEY or GROQ_API_KEY)"), False

    from pipeline.core_llm import call_llm_with_fallback  # lazy import — SDK only loaded when needed

    title = str(row.get("title", ""))
    company = str(row.get("company", ""))
    description = _maybe_get_full_description(row)
    web_search_context = _maybe_web_search_context(company, title, description)
    prompt = _build_verdict_prompt(
        row, cv_text, description,
        web_search_context=web_search_context,
        learned_preferences=learned_preferences,
    )

    # Pacing — Cerebras free tier caps at 5 RPM (12s/call minimum). 13s gives
    # a 1s buffer. With 60 AI evals/run this adds ~13 min of pacing, fine
    # for a daily cron.
    time.sleep(13)

    try:
        response_text = call_llm_with_fallback(
            prompt,
            cerebras_key=cerebras_key,
            groq_key=groq_key,
            max_attempts=4,
            label=title,
        )
        result = _parse_ai_response(response_text)
        result = apply_post_ai_caps(result, row)
        result = _apply_india_scam_check(result, row, company)

        badge = " SCAM" if result["scam"] else (" SUSPICIOUS" if result["suspicious"] else "")
        badge += " BLACKLISTED" if bool(row.get("pre_flagged_low_quality", False)) else ""
        logger.info(
            "[AI] %-55s -> match=%d%% (T:%d E:%d L:%d)%s",
            title[:55], result['match_percentage'],
            result['tech_fit'], result['experience_fit'], result['logistics_fit'],
            badge,
        )
        return result, True

    except Exception as e:
        logger.error("[AI ERROR] %s: %s", title[:55], str(e)[:300])
        error_msg = str(e).replace('"', "'")
        return _error_result(f"AI Error: {error_msg[:100]}..."), False


def evaluate_job_with_gemini(row, cv_text, gemini_key, learned_preferences=""):
    """Cheap second-pass verdict for lower-ranked jobs via Gemini 3.1 Flash Lite.

    Same prompt and post-processing as evaluate_job_with_ai, with two cost-saving
    differences for the lower-ranked path:
      - Skips the DDG web-search context (lower-ranked jobs already trail the
        top set in similarity; we don't burn DDG quota verifying them).
      - Skips the open-web scam check (reputation blacklist already handles
        known offenders upstream; 25 extra runs of detect_company_scam would
        cost ~75 DDG calls per pipeline tick).

    `learned_preferences` is passed through to the prompt builder so the
    lower-ranked path also benefits from the user's historical signals.

    Returns the same (result, evaluated_bool) shape so callers can treat both
    paths uniformly. Pacing is 4s per call (15 RPM Gemini Flash Lite free tier).
    """
    if not gemini_key:
        return _error_result("No GEMINI_API_KEY provided"), False

    from pipeline.core_llm import call_gemini_verdict  # lazy import

    title = str(row.get("title", ""))
    description = _maybe_get_full_description(row)
    prompt = _build_verdict_prompt(
        row, cv_text, description,
        web_search_context="",
        learned_preferences=learned_preferences,
    )

    # Pacing: Gemini Flash Lite free tier is 15 RPM = 4s/call minimum.
    time.sleep(4)

    try:
        response_text = call_gemini_verdict(prompt, gemini_key, max_attempts=3, label=title)
        result = _parse_ai_response(response_text)
        result = apply_post_ai_caps(result, row)
        # Intentionally NO india scam check here — keeps the cheap path cheap.

        badge = " SUSPICIOUS" if result["suspicious"] else ""
        badge += " BLACKLISTED" if bool(row.get("pre_flagged_low_quality", False)) else ""
        logger.info(
            "[AI-G] %-55s -> match=%d%% (T:%d E:%d L:%d)%s",
            title[:55], result['match_percentage'],
            result['tech_fit'], result['experience_fit'], result['logistics_fit'],
            badge,
        )
        return result, True

    except Exception as e:
        logger.error("[AI-G ERROR] %s: %s", title[:55], str(e)[:300])
        error_msg = str(e).replace('"', "'")
        return _error_result(f"AI Error: {error_msg[:100]}..."), False
