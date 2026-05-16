import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import re
import json
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
}

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
    }

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
# Description fetch + web search helpers (unchanged behaviour)
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
    from ddgs import DDGS  # lazy import — only loaded when AI eval actually fires
    print(f"Deep web search triggered for {company_name} ({job_title}) remote policy...")
    snippets = []
    try:
        q1 = f"{company_name} \"{job_title}\" remote eligible countries"
        res1 = DDGS().text(q1, max_results=2)
        snippets.extend([r.get('body', '') for r in res1])

        q2 = f"{company_name} hire remote Middle East Palestine EMEA"
        res2 = DDGS().text(q2, max_results=2)
        snippets.extend([r.get('body', '') for r in res2])

        return " ".join(snippets)
    except Exception as e:
        print(f"Web search failed: {e}")
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

1. LOGISTICS_FIT (0-100) — geography + timezone overlap with UTC+2:
   - "Worldwide" / "EMEA" / no geographic restriction → 85-100
   - Required sync TZ overlaps UTC+2 by >=4 working hours (EU, MENA, UK) → 75-95
   - Required sync TZ overlaps UTC+2 by <4 working hours (US Pacific, Australia, East Asia) → 20-50
   - Explicit exclusion of Palestine/MENA → set is_valid=false AND logistics_fit <= 20

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

8. VERDICT (1-2 sentences) MUST:
   - Cite the specific CV project or technology by name driving the match (e.g.,
     "your MSR-VTT text-to-video retrieval directly maps to their video search feature",
     "your FAISS + Ollama RAG project aligns with their on-prem LLM stack").
   - Generic phrases like "strong Python skills" or "strong technical match" are FORBIDDEN.
   - State the SINGLE biggest concern if any (TZ gap, exp gap, frontend gap, suspicious posting).

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
                print(f"  [AI RETRY {attempt}/2] backing off {backoff}s for {title[:55]}")
                time.sleep(backoff)

            response = client.models.generate_content(
                model='gemini-3.1-flash-lite',
                contents=prompt
            )
            result = _parse_ai_response(response.text)

            # Apply reputation-based cap (A1): pre-flagged low-quality companies
            # cannot score above 55 regardless of what the AI says.
            if bool(row.get("pre_flagged_low_quality", False)):
                if result["match_percentage"] > 55:
                    result["match_percentage"] = 55
                if not result["verdict"].startswith("[BLACKLISTED]"):
                    result["verdict"] = "[BLACKLISTED] " + result["verdict"]

            badge = " SUSPICIOUS" if result["suspicious"] else ""
            badge += " BLACKLISTED" if bool(row.get("pre_flagged_low_quality", False)) else ""
            print(
                f"  [AI] {title[:55]:<55} -> match={result['match_percentage']}% "
                f"(T:{result['tech_fit']} E:{result['experience_fit']} L:{result['logistics_fit']}){badge}"
            )
            return result, True
        except Exception as e:
            last_exception = e
            msg = str(e)
            if not any(t in msg for t in ("503", "500", "UNAVAILABLE", "INTERNAL")):
                break

    # All attempts exhausted (or non-retryable error).
    print(f"  [AI ERROR] {title[:55]}: {str(last_exception)[:300]}")
    error_msg = str(last_exception).replace('"', "'")
    return _error_result(f"AI Error: {error_msg[:100]}..."), False
