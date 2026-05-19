"""
CORE GEO MODULE (Layer 3)
-------------------------
Live remote-eligibility verification via Gemini with Google Search grounding.

This is the AI-driven "is this job actually open to a Palestine-based candidate"
check that fires AFTER the deterministic filters in core_ai.quick_viability_check
have done their work. It's the closest automated equivalent to manually pasting
a job URL into Google AI and asking "can I apply from Palestine?".

Cost model: ~100 Gemini calls/day on the global scraper.py (top-60 + lower-ranked).
Well under the Gemini free tier (500/day). Not used in local_companies.py since
Palestinian companies don't have geo-restriction concerns.

Outcomes feed into apply_geo_check_result():
  - "open"       -> no change to the verdict
  - "uncertain"  -> match_percentage capped at 50, verdict tagged [GEO-UNVERIFIED]
  - "restricted" -> is_valid=False, logistics_fit<=15, verdict tagged [GEO-RESTRICTED]
"""
import json
import re
import time

from pipeline.logging_setup import get_logger

logger = get_logger(__name__)


# Title patterns that signal a country-tagged remote (e.g. "Remote, Mexico").
# This is one of the trigger signals for should_geo_check().
_TITLE_GEO_PATTERNS = (
    # "Remote, Country" / "Remote – Country" / "Remote — Country"
    re.compile(r"\bremote\s*[,\-–—]\s*(?:[a-z]+\s+)?(?:"
               r"canada|canadian|australia|brazil|argentina|mexico|chile|colombia|peru|"
               r"india|china|japan|korea|singapore|thailand|philippines|indonesia|malaysia|"
               r"germany|france|italy|spain|portugal|netherlands|belgium|"
               r"sweden|norway|denmark|finland|poland|romania|hungary|ukraine|"
               r"russia|turkey|türkiye|israel|south\s+africa|nigeria|kenya|"
               r"uk|britain|ireland|scotland|wales|england|"
               r"new\s+zealand|saudi|uae|qatar|egypt|"
               r"eu|eea|latam|apac|anz|europe|asia"
               r")\b", re.IGNORECASE),
    # "(EU)", "(India)", "(LATAM)" parentheses in title
    re.compile(r"\((?:"
               r"eu|eea|latam|apac|anz|"
               r"india|china|brazil|argentina|mexico|canada|australia|"
               r"germany|france|italy|spain|uk|britain|israel|turkey|"
               r"japan|korea|singapore"
               r")\)", re.IGNORECASE),
)


# If ANY of these phrases appear in the description, we trust the job is
# explicitly worldwide-friendly and skip the geo check.
_GLOBAL_CONFIRMERS = (
    "worldwide", "globally remote", "anywhere in the world",
    "no location restriction", "no geographic restriction",
    "open to all locations", "global candidates",
    "fully remote globally", "remote worldwide", "remote globally",
    "global remote", "all countries", "remote from anywhere",
    "emea welcome", "middle east", "mena", "global hire",
)


def should_geo_check(row):
    """Decide if a job needs the Gemini eligibility verification.

    User asked us to be generous with this check, so we trigger broadly:
      - Title matches "Remote, <country>" / "(EU)" / similar
      - Location field has a specific country/city (anything beyond "Remote")
      - Description lacks any explicit worldwide-confirming phrase

    Returns False only when the job is unambiguously global (or has no useful
    text to check at all).
    """
    title = str(row.get("title", "")).strip()
    location = str(row.get("location", "")).strip()
    description = str(row.get("description", "")).lower()

    # Trigger 1: title carries a country-tagged remote pattern.
    for pattern in _TITLE_GEO_PATTERNS:
        if pattern.search(title):
            return True

    # Trigger 2: location field has a specific value beyond pure remote markers.
    loc_lower = location.lower().strip()
    if loc_lower and loc_lower not in ("", "remote", "worldwide", "global", "anywhere", "remote - worldwide", "any"):
        return True

    # Trigger 3: description lacks a worldwide confirmer.
    if not any(phrase in description for phrase in _GLOBAL_CONFIRMERS):
        return True

    return False


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

_DEFAULT_GEO_RESULT = {"eligibility": "uncertain", "confidence": 0, "evidence": ""}


def _parse_geo_response(text):
    """Parse a Gemini geo-check response into the canonical schema.

    Tolerates markdown fences, surrounding prose, and missing fields. On any
    parse failure returns the "uncertain / 0 confidence" fallback — we
    deliberately default to uncertain rather than open, because letting a
    malformed response slip through as open would defeat the whole point.
    """
    if not text:
        return dict(_DEFAULT_GEO_RESULT, evidence="empty response")

    t = text.strip()
    if t.startswith('```json'):
        t = t[7:]
    elif t.startswith('```'):
        t = t[3:]
    if t.endswith('```'):
        t = t[:-3]
    t = t.strip()

    raw = None
    try:
        raw = json.loads(t)
    except Exception:
        # Look for a JSON object embedded in surrounding text (Gemini sometimes
        # writes a sentence before the JSON despite the prompt).
        match = re.search(r'\{[^{}]*?"eligibility"[^{}]*?\}', t, re.DOTALL)
        if match:
            try:
                raw = json.loads(match.group(0))
            except Exception:
                return dict(_DEFAULT_GEO_RESULT, evidence="JSON parse error")
        else:
            return dict(_DEFAULT_GEO_RESULT, evidence="no JSON in response")

    if not isinstance(raw, dict):
        return dict(_DEFAULT_GEO_RESULT, evidence="non-dict response")

    eligibility = str(raw.get("eligibility", "uncertain")).lower().strip()
    if eligibility not in ("open", "uncertain", "restricted"):
        eligibility = "uncertain"

    confidence = raw.get("confidence", 0)
    if isinstance(confidence, str):
        digits = re.sub(r"[^\d]", "", confidence)
        confidence = int(digits) if digits else 0
    elif isinstance(confidence, bool):
        confidence = 100 if confidence else 0
    elif isinstance(confidence, (int, float)):
        confidence = int(confidence)
    else:
        confidence = 0
    confidence = max(0, min(100, confidence))

    evidence = str(raw.get("evidence", "")).strip()[:300]

    return {"eligibility": eligibility, "confidence": confidence, "evidence": evidence}


# ---------------------------------------------------------------------------
# Main Gemini call
# ---------------------------------------------------------------------------

_GEO_PROMPT_TEMPLATE = """You are verifying whether a candidate based in PALESTINE (West Bank, UTC+2 timezone)
can apply to a specific remote job. You MUST use Google Search to find live
evidence about the company's hiring policy before deciding.

JOB:
- Title: {title}
- Company: {company}
- URL: {job_url}
- Description excerpt:
{description}

INVESTIGATION CHECKLIST (use Google Search to answer each):
1. Does {company} hire remote workers from Palestine, MENA, or globally?
2. Does the job listing or company policy explicitly restrict to a specific country/region?
3. Is there public evidence (LinkedIn, Glassdoor, careers blog, job boards) of geographic eligibility?

DECISION RULES (be strict):
- "open"       = You found CONCRETE evidence the company hires from MENA / globally / no geo restrictions.
- "uncertain"  = Public info is silent or ambiguous; can't confirm either way.
- "restricted" = You found evidence the role is locked to a specific non-Palestine country/region
                 (e.g. "Remote, Mexico" actually meaning Mexico-residents-only).

DO NOT default to "open" without concrete evidence. Default to "uncertain" if unsure.

Return ONLY this exact JSON object (no markdown fences, no commentary):
{{"eligibility": "open" or "uncertain" or "restricted", "confidence": 0-100, "evidence": "one short sentence summarizing what Google Search told you"}}"""


def check_remote_eligibility(title, company, job_url, description, api_key):
    """Verify remote eligibility via Gemini + Google Search grounding.

    Same retry-on-5xx pattern as the original Gemini verdict code (3 attempts
    with exponential backoff). On total failure returns "uncertain" so the
    caller applies the safe-cap behavior rather than treating it as open.

    Args:
        title:       job title string
        company:     company name string
        job_url:     direct link to the posting (Gemini may visit it via search)
        description: full description text (truncated to 1500 chars in prompt)
        api_key:     GEMINI_API_KEY

    Returns:
        dict with keys: eligibility (open|uncertain|restricted), confidence (0-100), evidence (str)
    """
    if not api_key:
        return dict(_DEFAULT_GEO_RESULT, evidence="no Gemini API key")

    # Lazy import — same pattern as core_ai.py.
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    prompt = _GEO_PROMPT_TEMPLATE.format(
        title=title or "(no title)",
        company=company or "(no company)",
        job_url=job_url or "(no URL)",
        description=(description or "")[:1500],
    )

    last_exception = None
    for attempt in range(3):
        try:
            if attempt == 0:
                time.sleep(4)  # Gemini free tier is 15 RPM
            else:
                backoff = 5 * (2 ** (attempt - 1))  # 5s, 10s
                logger.warning("[GEO RETRY %d/2] backing off %ds for %s",
                               attempt, backoff, title[:55])
                time.sleep(backoff)

            response = client.models.generate_content(
                model='gemini-3.1-flash-lite',
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(google_search=types.GoogleSearch())]
                ),
            )
            result = _parse_geo_response(response.text)
            logger.info(
                "[GEO] %-55s -> %s (conf=%d) %s",
                title[:55], result['eligibility'], result['confidence'],
                result['evidence'][:80],
            )
            return result

        except Exception as e:
            last_exception = e
            msg = str(e)
            if not any(t in msg for t in ("503", "500", "UNAVAILABLE", "INTERNAL")):
                break

    # All attempts exhausted (or non-retryable). Safe default = uncertain.
    logger.warning("[GEO ERROR] %s: %s", title[:55], str(last_exception)[:200])
    return dict(_DEFAULT_GEO_RESULT,
                evidence=f"check failed: {str(last_exception)[:80]}")


# ---------------------------------------------------------------------------
# Applying geo-check results to a verdict
# ---------------------------------------------------------------------------

def apply_geo_check_result(verdict_result, geo_result):
    """Merge a geo-check outcome into a main AI verdict dict.

    Mutates and returns the verdict result. Cap policy:
      open        -> no changes
      uncertain   -> match_percentage capped at 50, [GEO-UNVERIFIED] tag on verdict
      restricted  -> is_valid=False, logistics_fit<=15, match_percentage<=30,
                     [GEO-RESTRICTED] tag on verdict

    Existing precedence tags ([BLACKLISTED], [SCAM]) are preserved — we don't
    add the geo tag if a stronger negative tag is already present.
    """
    if not isinstance(verdict_result, dict) or not isinstance(geo_result, dict):
        return verdict_result

    eligibility = geo_result.get("eligibility", "uncertain")
    evidence = (geo_result.get("evidence") or "").strip()

    existing_verdict = verdict_result.get("verdict", "") or ""
    has_stronger_tag = existing_verdict.startswith(("[BLACKLISTED]", "[SCAM]"))

    if eligibility == "restricted":
        verdict_result["is_valid"] = False
        if verdict_result.get("logistics_fit", 0) > 15:
            verdict_result["logistics_fit"] = 15
        if verdict_result.get("match_percentage", 0) > 30:
            verdict_result["match_percentage"] = 30
        if not has_stronger_tag and not existing_verdict.startswith("[GEO-RESTRICTED]"):
            verdict_result["verdict"] = (
                f"[GEO-RESTRICTED] {evidence} " + existing_verdict
            ).strip()

    elif eligibility == "uncertain":
        if verdict_result.get("match_percentage", 0) > 50:
            verdict_result["match_percentage"] = 50
        if not has_stronger_tag and not existing_verdict.startswith(
            ("[GEO-UNVERIFIED]", "[GEO-RESTRICTED]")
        ):
            verdict_result["verdict"] = (
                f"[GEO-UNVERIFIED] {evidence} " + existing_verdict
            ).strip()

    return verdict_result
