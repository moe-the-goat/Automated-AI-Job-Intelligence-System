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


def _parse_ai_response(text):
    """Parse a raw model output string into the canonical result dict.

    Tolerates: markdown ```json fences, surrounding whitespace, missing fields,
    string-typed numbers, percent signs in numeric fields.
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
    raw = json.loads(t)
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
    #   - LinkedIn post snippets (from the local pipeline) are legitimately terse —
    #     the title is prefixed with "LinkedIn Post:" and the body is a hashtag teaser.
    #   - Truncated-by-authwall placeholders trigger the AI's Limited Info Protocol.
    is_missing = description_clean in ("", "nan", "none", "null") or len(description_clean) < 20
    is_linkedin_post = title.startswith("linkedin post:")
    is_truncated_placeholder = (
        "[no description" in description or "[description truncated" in description
    )

    if not (is_missing or is_linkedin_post or is_truncated_placeholder):
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

    return True, "viable"

def evaluate_job_with_ai(row, cv_text, api_key):
    """
    Evaluate a single job posting against the candidate's CV using Gemini 3.1 Flash Lite.

    Returns a 2-tuple: (result_dict, evaluated_bool).
    - result_dict follows DEFAULT_AI_RESULT schema.
    - evaluated_bool is True ONLY when the AI returned a real verdict; callers should
      use this to decide whether to mark the URL as "seen" (so transient API errors
      don't lose jobs forever — see core_filter.JobTracker).
    """
    if not api_key:
        return _error_result("No API Key provided"), False

    from google import genai  # lazy import — see comment near top of file
    client = genai.Client(api_key=api_key)

    title = str(row.get("title", ""))
    company = str(row.get("company", ""))
    job_type = str(row.get("job_type", "")).lower()
    description = str(row.get("description", ""))

    if pd.isna(description) or len(description) < 100:
        description = get_full_job_description(str(row.get("job_url", "")))
        if not description:
            description = "[NO DESCRIPTION AVAILABLE - SCRAPING BLOCKED]"

    is_internship = 'intern' in title.lower() or 'internship' in job_type

    # --- Web search trigger logic ---
    web_search_triggers = [
        "eligible countries", "selected countries", "certain countries",
        "must be based", "candidates based", "residents of",
        "remote in the", "must be located", "work authorization",
        "within the united states", "us only", "us-based", "uk only", "eu only"
    ]
    explicit_global_phrases = [
        "worldwide", "globally remote", "anywhere in the world",
        "no location restriction", "no geographic restriction",
        "open to all locations", "global candidates", "emea welcome",
        "middle east", "fully remote globally", "all countries", "global remote",
    ]
    ambiguity_context_phrases = [
        "timezone", "country", "region", "office", "headquartered",
        "headquarters", "based ", "located in",
    ]

    desc_lower = description.lower()
    has_restriction = any(t in desc_lower for t in web_search_triggers)
    has_global_signal = any(p in desc_lower for p in explicit_global_phrases)
    has_ambiguity_context = any(p in desc_lower for p in ambiguity_context_phrases)

    web_search_context = ""
    if has_restriction or (has_ambiguity_context and not has_global_signal):
        search_data = search_company_remote_policy(company, title)
        if search_data:
            web_search_context = f"\n\n[LIVE WEB SEARCH RESULTS FOR '{company}' REMOTE POLICY]:\n{search_data}\n\nUse this live web data to determine if Palestine/Middle East is explicitly excluded from their remote eligible countries."

    prompt = f"""You are a SKEPTICAL technical recruiter screening a candidate. Your job is to find
DISQUALIFYING reasons. Default to skepticism. A 90+ score is reserved for cases where
you cannot reasonably imagine why the candidate would NOT be considered.

CANDIDATE CV SUMMARY:
{cv_text[:3000]}

CANDIDATE FACTS (use directly when scoring):
- Location: Birzeit, Palestine (UTC+2)
- Professional employment: 0 years (academic projects, coursework, Udacity nanodegrees only)
- Status: 4th-year Computer Engineering student, graduating Feb 2027
- Strongest specific assets: RAG (LangChain/FAISS/Ollama), FastAPI deployment,
  PyTorch computer vision, MSR-VTT text-to-video retrieval (Recall@10 60.5%),
  Arabic CNN (98.86% test accuracy)

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
     If ANY such phrase appears (or the web search reveals one), set is_valid=false AND
     logistics_fit <= 15. Note it explicitly in the verdict.
   - Explicit exclusion of Palestine / Middle East → same: is_valid=false, logistics_fit <= 15.

2. EXPERIENCE_FIT (0-100) — required experience vs candidate's 0 professional years:
   - "Internship" / "0-1 year" / "entry-level" → 85-100
   - "1-2 years required" → 50-70 (candidate has projects but no professional years)
   - "3+ years" → 20-40 (likely is_valid=false unless project-equivalents accepted)
   - "5+ years" / "Senior" → set is_valid=false AND experience_fit <= 25

3. TECH_FIT (0-100) — overlap between candidate's STRONGEST assets and job's REQUIRED stack:
   - Direct overlap (role explicitly wants RAG / LLM apps / FastAPI / PyTorch CV / video retrieval) → 85-100
     and the verdict MUST name the specific CV project.
   - Adjacent overlap (generic Python ML/data role, no specific match) → 60-80
   - Pure Web/Frontend role (React/HTML/CSS heavy) → max 60 (candidate is backend-heavy)
   - Cloud-ML / heavy AWS/GCP role → max 60 (candidate is local-Ollama-centric)
   - DevOps / SRE / Quant Finance → max 40 (fundamental mismatch)
   - Generic Python presence alone is NOT 90+. Require specific overlap to clear 85.

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
   Required structure (2-4 sentences, in this order):
   a) MATCH: name 1-2 SPECIFIC CV assets (projects or technologies, by name) that
      directly address what the job asks for. E.g.,
      "Your RAG project (LangChain + FAISS + Ollama) directly matches their stated
       need for LLM-integrated app development."
   b) GAP: name the SPECIFIC missing requirement that the candidate doesn't have.
      E.g., "Their stack also requires React/TypeScript frontend — your CV shows
      backend-only experience."
   c) (Optional) SECOND MATCH/GAP: a secondary positive or concern.
   d) (Required only if is_valid=false) CLOSING REASON: explicitly state the
      disqualifier (work auth, geo exclusion, senior-only, scam suspicion).

   STRICT VOCABULARY RULES:
   - Generic phrases are FORBIDDEN: "strong technical match", "strong Python skills",
     "good fit", "well-aligned". Be specific or don't say it.
   - Cite projects and tech BY NAME (MSR-VTT, FAISS, FastAPI, Ollama, etc.).
   - When citing a gap, name the missing tech/experience SPECIFICALLY ("no AWS production
     experience", "no React frontend background"), not vaguely ("limited experience").

9. LIMITED INFO PROTOCOL:
   - If description contains [DESCRIPTION TRUNCATED] or [NO DESCRIPTION], deduct 10 from
     each sub-score. Do NOT set suspicious=true purely because of missing info.
   - Note the missing description in the verdict.

Reply with VALID JSON ONLY (no markdown, no comments):
{{"is_valid": true|false, "verdict": "...", "tech_fit": 0-100, "experience_fit": 0-100, "logistics_fit": 0-100, "match_percentage": 0-100, "compensation": "...", "effort": "low|medium|high", "suspicious": true|false}}
"""

    # Retry up to 3 attempts on transient 5xx errors (Gemini gets demand spikes).
    last_exception = None
    for attempt in range(3):
        try:
            if attempt == 0:
                time.sleep(4)  # standard throttle to stay under 15 RPM
            else:
                backoff = 10 * (2 ** (attempt - 1))  # 10s, 20s
                logger.warning("[AI RETRY %d/2] backing off %ds for %s", attempt, backoff, title[:55])
                time.sleep(backoff)

            response = client.models.generate_content(
                model='gemini-3.1-flash-lite',
                contents=prompt
            )
            result = _parse_ai_response(response.text)

            # Apply deterministic caps before the (optional) network-based scam
            # check. Both caps are pure functions of the AI's verdict + the row's
            # pre-screening flags, so we extract them so tests can lock them down
            # without mocking the Gemini client.
            result = apply_post_ai_caps(result, row)

            # India-suspicious -> open-web scam check. Only fires when both signals
            # hold, keeping DDG calls cheap. Confirmed scams get a hard cap + tag.
            location_text = str(row.get("location", ""))
            if result["suspicious"] and looks_like_india_employer(location_text, company):
                if detect_company_scam(company):
                    result["scam"] = True
                    result["is_valid"] = False
                    if result["match_percentage"] > 30:
                        result["match_percentage"] = 30
                    if not result["verdict"].startswith("[SCAM]"):
                        result["verdict"] = "[SCAM] " + result["verdict"]

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
            last_exception = e
            msg = str(e)
            if not any(t in msg for t in ("503", "500", "UNAVAILABLE", "INTERNAL")):
                break

    # All attempts exhausted (or non-retryable error).
    logger.error("[AI ERROR] %s: %s", title[:55], str(last_exception)[:300])
    error_msg = str(last_exception).replace('"', "'")
    return _error_result(f"AI Error: {error_msg[:100]}..."), False
