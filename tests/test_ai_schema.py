"""Standalone tests for the AI verdict schema + parser + rendering helpers.

Run:  python tests/test_ai_schema.py

No network calls; no Gemini key required. Exits 0 on success, non-zero on first failure.
"""
import os
import sys

# Make the project root importable when running this file directly.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import pandas as pd

from core_ai import (
    DEFAULT_AI_RESULT,
    _safe_int,
    _safe_bool,
    _safe_str,
    _normalize_effort,
    _normalize_result,
    _parse_ai_response,
    quick_viability_check,
    skipped_result,
    looks_like_india_employer,
    scan_for_scam_signals,
)
from local_companies import linkedin_post_date, LOCAL_LOOKBACK_DAYS
from core_notify import (
    sort_by_match_percentage,
    _fmt_match_cell_html,
    _fmt_match_cell_md,
    _suspicious_title,
    format_email_html,
    format_github_markdown,
    _normalize_repo,
    _render_lower_ranked_html,
)
from core_filter import _pre_flag_reputation, _load_reputation
from core_embedding import (
    _cv_hash,
    _read_cached_embedding,
    _write_cached_embedding,
    cosine_similarity,
    rank_by_similarity,
    CV_EMBEDDING_CACHE,
)


def _check(name, got, expected):
    if got != expected:
        raise AssertionError(f"[{name}] expected {expected!r}, got {got!r}")
    print(f"  ok  {name}")


# ---------------------------------------------------------------------------
# Coercion helpers
# ---------------------------------------------------------------------------

def test_safe_int():
    _check("int passthrough",    _safe_int(82),      82)
    _check("int clamp high",     _safe_int(150),     100)
    _check("int clamp low",      _safe_int(-5),      0)
    _check("float truncates",    _safe_int(82.7),    82)
    _check("str clean",          _safe_int("82"),    82)
    _check("str with percent",   _safe_int("82%"),   82)
    _check("str decimal",        _safe_int("82.5"),  82)
    _check("str decimal+pct",    _safe_int("82.5%"), 82)
    _check("str garbage",        _safe_int("abc"),   0)
    _check("str empty",          _safe_int(""),      0)
    _check("none default",       _safe_int(None),    0)
    _check("none custom default",_safe_int(None, 50),50)
    _check("bool true",          _safe_int(True),    1)


def test_safe_bool():
    _check("bool true",      _safe_bool(True),    True)
    _check("bool false",     _safe_bool(False),   False)
    _check("str true",       _safe_bool("true"),  True)
    _check("str True",       _safe_bool("True"),  True)
    _check("str yes",        _safe_bool("yes"),   True)
    _check("str 1",          _safe_bool("1"),     True)
    _check("str no",         _safe_bool("no"),    False)
    _check("str empty",      _safe_bool(""),      False)
    _check("none default",   _safe_bool(None),    False)
    _check("int 0",          _safe_bool(0),       False)
    _check("int nonzero",    _safe_bool(7),       True)


def test_normalize_effort():
    _check("low",     _normalize_effort("low"),    "low")
    _check("medium",  _normalize_effort("Medium"), "medium")
    _check("high",    _normalize_effort("HIGH"),   "high")
    _check("garbage", _normalize_effort("extreme"),"unknown")
    _check("none",    _normalize_effort(None),     "unknown")


# ---------------------------------------------------------------------------
# Normalize / parse
# ---------------------------------------------------------------------------

def test_normalize_full():
    raw = {
        "is_valid": True, "verdict": "Good fit",
        "tech_fit": 90, "experience_fit": 70, "logistics_fit": 85,
        "match_percentage": 82, "compensation": "$25/hr",
        "effort": "low", "suspicious": False,
    }
    r = _normalize_result(raw)
    _check("verdict",          r["verdict"],          "Good fit")
    _check("is_valid",         r["is_valid"],         True)
    _check("tech_fit",         r["tech_fit"],         90)
    _check("experience_fit",   r["experience_fit"],   70)
    _check("logistics_fit",    r["logistics_fit"],    85)
    _check("match_percentage", r["match_percentage"], 82)
    _check("compensation",     r["compensation"],     "$25/hr")
    _check("effort",           r["effort"],           "low")
    _check("suspicious",       r["suspicious"],       False)


def test_normalize_missing_fields():
    r = _normalize_result({})
    _check("default is_valid",         r["is_valid"],         False)
    _check("default verdict",          r["verdict"],          "No verdict")
    _check("default tech_fit",         r["tech_fit"],         0)
    _check("default experience_fit",   r["experience_fit"],   0)
    _check("default logistics_fit",    r["logistics_fit"],    0)
    _check("default match_percentage", r["match_percentage"], 0)
    _check("default compensation",     r["compensation"],     "Not stated")
    _check("default effort",           r["effort"],           "unknown")
    _check("default suspicious",       r["suspicious"],       False)


def test_normalize_non_dict():
    _check("non-dict passes through to defaults",
           _normalize_result("nope")["match_percentage"], 0)
    _check("None passes through to defaults",
           _normalize_result(None)["match_percentage"], 0)


def test_normalize_string_numbers():
    raw = {"tech_fit": "92%", "experience_fit": "65", "logistics_fit": "90.0",
           "match_percentage": "75.4%", "suspicious": "true", "is_valid": "yes"}
    r = _normalize_result(raw)
    _check("string % tech_fit",         r["tech_fit"],         92)
    _check("string experience_fit",     r["experience_fit"],   65)
    _check("string decimal log",        r["logistics_fit"],    90)
    _check("string decimal+% match",    r["match_percentage"], 75)
    _check("string suspicious",         r["suspicious"],       True)
    _check("string is_valid yes",       r["is_valid"],         True)


def test_normalize_extra_fields_ignored():
    raw = {"match_percentage": 80, "extra": "should be dropped", "score": 999}
    r = _normalize_result(raw)
    _check("extra key not in result", "extra" in r, False)
    _check("score not used",          r["match_percentage"], 80)


def test_parse_clean_json():
    s = '{"is_valid": true, "verdict": "Strong RAG match", "tech_fit": 95, "experience_fit": 80, "logistics_fit": 90, "match_percentage": 90, "compensation": "$30/hr", "effort": "low", "suspicious": false}'
    r = _parse_ai_response(s)
    _check("clean json match",   r["match_percentage"], 90)
    _check("clean json verdict", r["verdict"],          "Strong RAG match")


def test_parse_with_markdown_fence():
    s = '```json\n{"is_valid": true, "verdict": "ok", "tech_fit": 50, "experience_fit": 50, "logistics_fit": 50, "match_percentage": 50, "compensation": "Unpaid", "effort": "medium", "suspicious": false}\n```'
    r = _parse_ai_response(s)
    _check("fenced compensation", r["compensation"], "Unpaid")
    _check("fenced effort",       r["effort"],       "medium")


def test_parse_with_loose_fence():
    s = '```\n{"is_valid": false, "verdict": "nope", "tech_fit": 10, "experience_fit": 10, "logistics_fit": 10, "match_percentage": 10, "compensation": "Not stated", "effort": "low", "suspicious": true}\n```'
    r = _parse_ai_response(s)
    _check("loose-fence suspicious", r["suspicious"], True)
    _check("loose-fence is_valid",   r["is_valid"],   False)


def test_parse_empty_raises():
    raised = False
    try:
        _parse_ai_response("")
    except ValueError:
        raised = True
    _check("empty input raises", raised, True)


def test_default_schema_keys():
    expected_keys = {"is_valid", "verdict", "tech_fit", "experience_fit",
                     "logistics_fit", "match_percentage", "compensation",
                     "effort", "suspicious", "scam"}
    _check("default keys", set(DEFAULT_AI_RESULT.keys()), expected_keys)


# ---------------------------------------------------------------------------
# LinkedIn post date decoder (formerly buggy: treated ms as seconds -> always None)
# ---------------------------------------------------------------------------

def test_linkedin_post_date_iconnect_5mo():
    """IConnect Technologies post the user flagged as 5 months stale -> Nov 22, 2025."""
    url = "https://www.linkedin.com/posts/iconnect-tech_hiring-fullstackdeveloper-ai-activity-7397959444342575104-p6DK"
    d = linkedin_post_date(url)
    _check("iconnect decodes to a real datetime", d is not None, True)
    _check("iconnect year is 2025",  d.year,  2025)
    _check("iconnect month is November", d.month, 11)


def test_linkedin_post_date_smartweb_1yr():
    """SmartWeb post the user flagged as ~1 year stale -> Jan 13, 2025."""
    url = "https://www.linkedin.com/posts/smartweb-labs_hiring-python-python-activity-7284632728031944704-93L4"
    d = linkedin_post_date(url)
    _check("smartweb decodes to a real datetime", d is not None, True)
    _check("smartweb year is 2025", d.year, 2025)
    _check("smartweb month is January", d.month, 1)


def test_linkedin_post_date_no_activity_id():
    """URLs without an activity-XYZ segment should return None."""
    _check("missing activity returns None",
           linkedin_post_date("https://www.linkedin.com/posts/someone_a-cool-post-xxxx"),
           None)
    _check("empty URL returns None", linkedin_post_date(""), None)
    _check("None URL returns None", linkedin_post_date(None), None)


def test_linkedin_post_date_filter_logic():
    """The post-date filter should reject stale posts and accept fresh ones."""
    from datetime import datetime, timezone, timedelta
    cutoff = datetime.now(timezone.utc) - timedelta(days=LOCAL_LOOKBACK_DAYS)
    stale = linkedin_post_date(
        "https://www.linkedin.com/posts/x_y-activity-7397959444342575104-z")
    _check("stale post is before cutoff", stale < cutoff, True)


# ---------------------------------------------------------------------------
# Scam detection helpers (India signal + keyword scan)
# ---------------------------------------------------------------------------

def test_india_employer_by_location():
    _check("location India flagged",     looks_like_india_employer("Bangalore, India", "Foo Corp"), True)
    _check("location Bengaluru flagged", looks_like_india_employer("Bengaluru", "X"),               True)
    _check("location Mumbai flagged",    looks_like_india_employer("Mumbai, IN", "X"),              True)


def test_india_employer_by_company_suffix():
    _check("Private Limited flagged", looks_like_india_employer("Remote", "Zetheta Algorithms Private Limited"), True)
    _check("Pvt Ltd flagged",         looks_like_india_employer("Remote", "Acme Tech Pvt Ltd"),                  True)


def test_india_employer_no_signal():
    _check("US company not flagged",     looks_like_india_employer("San Francisco, CA", "Anthropic"),       False)
    _check("EU company not flagged",     looks_like_india_employer("Berlin, Germany", "Hugging Face GmbH"), False)
    _check("empty location/company",     looks_like_india_employer("", ""),                                  False)


def test_scam_keyword_scan_positive():
    text = (
        "I worked there for two months and it turned out to be a scam, they never paid me. "
        "Lots of reddit threads about this being a fake job."
    )
    _check("two+ scam signals triggers True", scan_for_scam_signals(text), True)


def test_scam_keyword_scan_below_threshold():
    # Single mention isn't enough — many real companies appear in articles about scams generally.
    text = "Top 10 ways to spot a job scam in 2026: tip 1, tip 2..."
    _check("single 'scam' mention does NOT trigger", scan_for_scam_signals(text), False)


def test_scam_keyword_scan_empty():
    _check("empty text returns False", scan_for_scam_signals(""), False)
    _check("None text returns False",  scan_for_scam_signals(None), False)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def test_match_cell_html_rich():
    row = {"match_percentage": 75, "tech_fit": 92, "experience_fit": 65, "logistics_fit": 90}
    cell = _fmt_match_cell_html(row)
    _check("html has bold pct",   "<b>75%</b>" in cell,             True)
    _check("html has tech sub",   "T:92" in cell,                   True)
    _check("html has exp sub",    "E:65" in cell,                   True)
    _check("html has log sub",    "L:90" in cell,                   True)


def test_match_cell_html_fallback():
    row = {"match_percentage": "N/A"}
    cell = _fmt_match_cell_html(row)
    _check("html N/A fallback", cell, "<b>N/A</b>")


def test_match_cell_md_rich():
    row = {"match_percentage": 75, "tech_fit": 92, "experience_fit": 65, "logistics_fit": 90}
    cell = _fmt_match_cell_md(row)
    _check("md has bold pct",   "**75%**" in cell, True)
    _check("md has sub-scores", "T:92 E:65 L:90" in cell, True)


def test_suspicious_title_marker():
    _check("suspicious adds warning",  _suspicious_title("Data Intern", True),  "⚠️ Data Intern")
    _check("clean title unchanged",    _suspicious_title("Data Intern", False), "Data Intern")
    _check("blacklisted shows blacklist badge",     _suspicious_title("Data Intern", False, True), "🚫 Data Intern")
    _check("blacklisted overrides suspicious flag", _suspicious_title("Data Intern", True, True),  "🚫 Data Intern")
    _check("scam shows scam badge",             _suspicious_title("Data Intern", False, False, True), "🚨 Data Intern")
    _check("scam overrides blacklist",          _suspicious_title("Data Intern", False, True, True),  "🚨 Data Intern")
    _check("scam overrides suspicious",         _suspicious_title("Data Intern", True, False, True),  "🚨 Data Intern")
    _check("scam overrides everything",         _suspicious_title("Data Intern", True, True, True),   "🚨 Data Intern")


# ---------------------------------------------------------------------------
# A2 — Heuristic pre-screen (quick_viability_check + skipped_result)
# ---------------------------------------------------------------------------

_GOOD_DESC = (
    "Build production RAG systems with LangChain and FastAPI. We deploy ML "
    "models on AWS and our team is fully distributed across EMEA timezones. "
    "Looking for candidates passionate about LLM applications."
)


def test_viability_good_row():
    row = {"title": "AI Engineering Intern", "description": _GOOD_DESC}
    ok, reason = quick_viability_check(row)
    _check("good row is viable", ok, True)
    _check("good row reason tag", reason, "viable")


def test_viability_rejects_blacklisted_first():
    """Blacklisted should short-circuit before any other check."""
    row = {"title": "AI Engineering Intern", "description": _GOOD_DESC,
           "pre_flagged_low_quality": True}
    ok, reason = quick_viability_check(row)
    _check("blacklisted -> not viable", ok, False)
    _check("blacklist reason mentions reputation", "reputation" in reason, True)


def test_viability_rejects_sales_title():
    row = {"title": "Sales Engineer", "description": _GOOD_DESC}
    ok, reason = quick_viability_check(row)
    _check("sales role rejected", ok, False)
    _check("sales reason tag", "non-tech" in reason, True)


def test_viability_rejects_recruiter_title():
    row = {"title": "Technical Recruiter Intern", "description": _GOOD_DESC}
    ok, reason = quick_viability_check(row)
    _check("recruiter role rejected", ok, False)


def test_viability_rejects_short_description():
    # 47 chars — in the "substantive but lazy repost" 20-149 range that still gets skipped.
    row = {"title": "AI Intern",
           "description": "Apply now for this great role at our company."}
    ok, reason = quick_viability_check(row)
    _check("substantive short desc rejected", ok, False)
    _check("short desc reason mentions chars", "chars" in reason, True)


def test_viability_keeps_jobs_with_limited_info_protocol():
    """When the description is the explicit '[NO DESCRIPTION]' placeholder,
    the AI handles it via its Limited Info Protocol — we don't pre-skip."""
    row = {"title": "AI Engineering Intern",
           "description": "[NO DESCRIPTION AVAILABLE - SCRAPING BLOCKED]"}
    ok, reason = quick_viability_check(row)
    _check("no-description placeholder allowed through", ok, True)


def test_viability_keeps_nan_description():
    """API-sourced rows with description='nan' must pass to the AI URL-fetch fallback."""
    row = {"title": "AI Engineering Intern", "description": "nan"}
    ok, reason = quick_viability_check(row)
    _check("nan desc passes pre-screen", ok, True)


def test_viability_keeps_none_string_description():
    row = {"title": "AI Engineering Intern", "description": "None"}
    ok, reason = quick_viability_check(row)
    _check("'None' desc passes pre-screen", ok, True)


def test_viability_keeps_empty_description():
    row = {"title": "AI Engineering Intern", "description": ""}
    ok, reason = quick_viability_check(row)
    _check("empty desc passes pre-screen", ok, True)


def test_viability_keeps_linkedin_post_with_short_body():
    """LinkedIn posts have legitimately short hashtag-teaser bodies — allow."""
    row = {"title": "LinkedIn Post: #hiring #python #react ...",
           "description": "#hiring #python #python #react #developer #remote"}
    ok, reason = quick_viability_check(row)
    _check("LinkedIn post with short body passes", ok, True)


def test_viability_still_rejects_substantive_short_description():
    """A real but lazy-repost description in the 30-149 range is still rejected."""
    row = {"title": "AI Intern",
           "description": "Apply now. We're hiring an AI intern for a paid role this summer."}
    ok, reason = quick_viability_check(row)
    _check("substantive 30-150 char desc still rejected", ok, False)
    _check("substantive-short reason tag", "chars" in reason, True)


def test_viability_rejects_5_plus_years():
    row = {"title": "AI Intern",
           "description": "We require 5+ years of experience in production ML. " * 5}
    ok, reason = quick_viability_check(row)
    _check("5+ years rejected", ok, False)
    _check("years reason tag", "senior experience" in reason, True)


def test_viability_rejects_minimum_7_years():
    row = {"title": "AI Intern",
           "description": "Minimum of 7 years experience required in this field. " * 5}
    ok, reason = quick_viability_check(row)
    _check("minimum 7y rejected", ok, False)


def test_viability_rejects_principal_engineer():
    row = {"title": "AI Intern",
           "description": "You'll be a principal engineer leading the AI team. " * 5}
    ok, reason = quick_viability_check(row)
    _check("principal engineer rejected", ok, False)


def test_skipped_result_schema():
    r = skipped_result("test reason")
    _check("skipped result has all keys", set(r.keys()), set(DEFAULT_AI_RESULT.keys()))
    _check("skipped result is_valid False", r["is_valid"], False)
    _check("skipped result verdict tag", r["verdict"].startswith("Pre-screen skipped:"), True)
    _check("skipped result match 0", r["match_percentage"], 0)


# ---------------------------------------------------------------------------
# A1 — Reputation pre-filter
# ---------------------------------------------------------------------------

def test_reputation_load_returns_dict():
    rep = _load_reputation()
    _check("load returns dict",          isinstance(rep, dict),                  True)
    _check("has blacklist_name key",     "blacklist_name" in rep,                True)
    _check("has blacklist_handle key",   "blacklist_handle" in rep,              True)
    _check("has trust_boost key",        "trust_boost" in rep,                   True)


def test_pre_flag_reputation_blacklist_by_name():
    df = pd.DataFrame([
        {"title": "A", "company": "Skillfied Mentor",   "job_url": "http://ex.com/1"},
        {"title": "B", "company": "Anthropic",          "job_url": "http://ex.com/2"},
        {"title": "C", "company": "Random Real Corp",   "job_url": "http://ex.com/3"},
    ])
    flagged = _pre_flag_reputation(df).reset_index(drop=True)
    _check("blacklisted company flagged",    bool(flagged.iloc[0]["pre_flagged_low_quality"]), True)
    _check("trusted company not low-q",      bool(flagged.iloc[1]["pre_flagged_low_quality"]), False)
    _check("trusted company gets trust flag",bool(flagged.iloc[1]["pre_flagged_trusted"]),     True)
    _check("random company not flagged",     bool(flagged.iloc[2]["pre_flagged_low_quality"]), False)
    _check("random company not trusted",     bool(flagged.iloc[2]["pre_flagged_trusted"]),     False)


def test_pre_flag_reputation_blacklist_by_handle():
    df = pd.DataFrame([
        {"title": "A", "company": "Some Random Name",
         "job_url": "https://linkedin.com/posts/pankh-workforce-solution_x"},
    ])
    flagged = _pre_flag_reputation(df).reset_index(drop=True)
    _check("blacklisted by URL handle", bool(flagged.iloc[0]["pre_flagged_low_quality"]), True)


def test_pre_flag_reputation_case_insensitive():
    df = pd.DataFrame([
        {"title": "A", "company": "SKILLFIED MENTOR LLP", "job_url": "x"},
        {"title": "B", "company": "skillfied mentor",     "job_url": "x"},
    ])
    flagged = _pre_flag_reputation(df).reset_index(drop=True)
    _check("uppercase matches", bool(flagged.iloc[0]["pre_flagged_low_quality"]), True)
    _check("lowercase matches", bool(flagged.iloc[1]["pre_flagged_low_quality"]), True)


def test_pre_flag_reputation_empty_df():
    out = _pre_flag_reputation(pd.DataFrame())
    _check("empty df survives", out.empty, True)


def test_pre_flag_reputation_renders_in_email():
    """Integration: a blacklisted job should produce a 🚫 in the HTML."""
    df = pd.DataFrame([{
        "title": "AI Engineering Intern", "company": "Webboost Solutions",
        "location": "Remote", "ai_verdict": "[BLACKLISTED] some text",
        "match_percentage": 55, "tech_fit": 80, "experience_fit": 80,
        "logistics_fit": 80, "compensation": "Not stated", "effort": "low",
        "suspicious": False, "pre_flagged_low_quality": True,
        "job_url": "https://example.com/x",
    }])
    html = format_email_html(df, pd.DataFrame(), {"scraped": 1, "filtered": 1, "approved": 1})
    _check("blacklisted shows badge in html",     "🚫 AI Engineering Intern" in html, True)
    _check("blacklisted shows BLACKLISTED tag",   "[BLACKLISTED]" in html,            True)


# ---------------------------------------------------------------------------
# A3 — Embedding pre-rank helpers (no network)
# ---------------------------------------------------------------------------

def test_cv_hash_stability():
    _check("same text -> same hash", _cv_hash("hello world"), _cv_hash("hello world"))
    _check("different text -> different hash", _cv_hash("a") != _cv_hash("b"), True)
    _check("hash length is 16",      len(_cv_hash("anything")),                16)


def test_cv_hash_handles_empty_and_none():
    _check("empty hash exists",      isinstance(_cv_hash(""), str),  True)
    _check("none hash exists",       isinstance(_cv_hash(None), str), True)


def test_cosine_self_equals_one():
    v = [1.0, 2.0, 3.0]
    sim = cosine_similarity(v, v)
    _check("self cosine ~ 1.0", abs(sim - 1.0) < 1e-9, True)


def test_cosine_orthogonal_equals_zero():
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    _check("orthogonal cosine ~ 0.0", abs(cosine_similarity(a, b)) < 1e-9, True)


def test_cosine_handles_none_and_zero_vec():
    _check("None vec returns 0",         cosine_similarity(None, [1, 2]), 0.0)
    _check("zero-norm vec returns 0",    cosine_similarity([0, 0, 0], [1, 2, 3]), 0.0)
    _check("empty vec returns 0",        cosine_similarity([], [1, 2, 3]), 0.0)


def test_rank_by_similarity_ordering():
    cv = [1.0, 0.0, 0.0]
    jobs = [
        [1.0, 0.0, 0.0],   # identical to CV
        [0.7, 0.7, 0.0],   # half-aligned
        [0.0, 1.0, 0.0],   # orthogonal
    ]
    sims = rank_by_similarity(cv, jobs)
    _check("identical first",     sims[0] > sims[1], True)
    _check("half before orthog",  sims[1] > sims[2], True)
    _check("orthog ~ 0",          abs(sims[2]) < 1e-9, True)


def test_cv_embedding_cache_roundtrip(tmp_dir=".test_tmp_cache"):
    """Write then read back; hash must match for the read to succeed."""
    import shutil
    test_cache_path = ".test_cv_embedding.json"
    # monkeypatch the cache path
    import core_embedding as _ce
    original_path = _ce.CV_EMBEDDING_CACHE
    _ce.CV_EMBEDDING_CACHE = test_cache_path
    try:
        text = "the candidate has strong python skills"
        vec = [0.1, 0.2, 0.3]
        _write_cached_embedding(text, vec)
        loaded = _read_cached_embedding(text)
        _check("cache roundtrip preserves vec", loaded, vec)
        _check("wrong-text returns None",       _read_cached_embedding(text + " mutated"), None)
    finally:
        _ce.CV_EMBEDDING_CACHE = original_path
        if os.path.exists(test_cache_path):
            os.remove(test_cache_path)


def test_lower_ranked_html_rendering():
    """Integration: format_email_html should embed the lower-ranked section."""
    main_df = pd.DataFrame([{
        "title": "AI Engineering Intern", "company": "iion", "location": "Remote",
        "ai_verdict": "Strong RAG match", "match_percentage": 92,
        "tech_fit": 95, "experience_fit": 90, "logistics_fit": 90,
        "compensation": "$25/hr", "effort": "low", "suspicious": False,
        "pre_flagged_low_quality": False, "job_url": "https://example.com/1",
    }])
    lower_df = pd.DataFrame([
        {"title": "Low-Sim Job 1", "company": "X", "location": "R",
         "similarity": 0.45, "job_url": "https://example.com/2"},
        {"title": "Low-Sim Job 2", "company": "Y", "location": "R",
         "similarity": 0.32, "job_url": "https://example.com/3"},
    ])
    html = format_email_html(
        main_df, pd.DataFrame(),
        {"scraped": 100, "filtered": 50, "approved": 1},
        lower_ranked_df=lower_df,
    )
    _check("lower-ranked section present",   "Lower-Ranked Matches" in html, True)
    _check("lower-ranked count in heading",  "2 jobs" in html,               True)
    _check("lower-ranked job 1 listed",      "Low-Sim Job 1" in html,        True)
    _check("lower-ranked similarity shown",  "0.45" in html,                 True)


def test_lower_ranked_section_omitted_when_empty():
    """An empty / None lower-ranked df should NOT render a section header."""
    main_df = pd.DataFrame([{
        "title": "X", "company": "Y", "location": "R",
        "ai_verdict": "v", "match_percentage": 80,
        "tech_fit": 80, "experience_fit": 80, "logistics_fit": 80,
        "compensation": "Not stated", "effort": "low", "suspicious": False,
        "pre_flagged_low_quality": False, "job_url": "#",
    }])
    html_none = format_email_html(main_df, pd.DataFrame(), {"scraped": 1, "filtered": 1, "approved": 1})
    html_empty = format_email_html(main_df, pd.DataFrame(), {"scraped": 1, "filtered": 1, "approved": 1}, lower_ranked_df=pd.DataFrame())
    _check("no section when lower_ranked_df is None",  "Lower-Ranked Matches" in html_none,  False)
    _check("no section when lower_ranked_df is empty", "Lower-Ranked Matches" in html_empty, False)


def test_normalize_repo_passthrough():
    _check("owner/name passthrough", _normalize_repo("moe-the-goat/job-scrapper-logs"),
           "moe-the-goat/job-scrapper-logs")


def test_normalize_repo_strips_full_url():
    _check("https URL stripped",
           _normalize_repo("https://github.com/moe-the-goat/job-scrapper-logs"),
           "moe-the-goat/job-scrapper-logs")


def test_normalize_repo_strips_trailing_slash():
    _check("trailing slash stripped",
           _normalize_repo("https://github.com/moe-the-goat/job-scrapper-logs/"),
           "moe-the-goat/job-scrapper-logs")


def test_normalize_repo_strips_git_suffix():
    _check("trailing .git stripped",
           _normalize_repo("https://github.com/moe-the-goat/job-scrapper-logs.git"),
           "moe-the-goat/job-scrapper-logs")


def test_normalize_repo_handles_none_and_empty():
    _check("None passthrough",  _normalize_repo(None), None)
    _check("empty passthrough", _normalize_repo(""),    None)
    _check("whitespace -> None", _normalize_repo("   "), None)


def test_lower_ranked_zero_similarity_renders_as_number():
    """Regression: a similarity of 0.0 used to render as '—'. Should be '0.00'."""
    df = pd.DataFrame([{"title": "X", "company": "Y", "location": "R",
                        "similarity": 0.0, "job_url": "#"}])
    html = _render_lower_ranked_html(df)
    _check("zero similarity renders as 0.00", "0.00" in html, True)
    _check("zero similarity does NOT render as em-dash", "—" not in html, True)


def test_sort_with_tiebreaker():
    df = pd.DataFrame([
        {"title": "A", "match_percentage": 80, "tech_fit": 60},
        {"title": "B", "match_percentage": 80, "tech_fit": 95},
        {"title": "C", "match_percentage": 90, "tech_fit": 50},
    ])
    sorted_df = sort_by_match_percentage(df)
    order = sorted_df["title"].tolist()
    _check("sort order: high match wins", order[0], "C")
    _check("tiebreak by tech_fit",        order[1], "B")
    _check("lowest tech_fit last in tie", order[2], "A")


def test_sort_empty_df():
    out = sort_by_match_percentage(pd.DataFrame())
    _check("empty df survives sort", out.empty, True)


# ---------------------------------------------------------------------------
# Integration: rendering a real dataframe doesn't crash + contains key signals
# ---------------------------------------------------------------------------

def test_format_email_html_integration():
    df = pd.DataFrame([
        {
            "title": "AI Engineering Intern",
            "company": "iion",
            "location": "Remote",
            "ai_verdict": "Your MSR-VTT retrieval project maps directly to their video search.",
            "match_percentage": 92,
            "tech_fit": 95,
            "experience_fit": 90,
            "logistics_fit": 90,
            "compensation": "$25/hr",
            "effort": "low",
            "suspicious": False,
            "job_url": "https://example.com/1",
        },
        {
            "title": "Data Science Intern",
            "company": "Webs IT Solution",
            "location": "Remote",
            "ai_verdict": "Generic skill overlap; suspicious training-program pattern.",
            "match_percentage": 55,
            "tech_fit": 70,
            "experience_fit": 60,
            "logistics_fit": 80,
            "compensation": "Stipend (amount unclear)",
            "effort": "medium",
            "suspicious": True,
            "job_url": "https://example.com/2",
        },
    ])
    stats = {"scraped": 100, "filtered": 50, "approved": 2}
    html = format_email_html(df, pd.DataFrame(), stats)
    _check("html opens table",            "<table" in html,                              True)
    _check("html shows iion title",       "AI Engineering Intern" in html,               True)
    _check("html shows MSR-VTT verdict",  "MSR-VTT" in html,                             True)
    _check("html shows compensation",     "$25/hr" in html,                              True)
    _check("html shows effort",           ">low<" in html or ">medium<" in html,         True)
    _check("html shows suspicious badge", "⚠️ Data Science Intern" in html,             True)
    _check("html shows sub-scores",       "T:95" in html,                                True)
    _check("html shows stats",            "Scraped: 100" in html,                        True)


def test_format_github_markdown_integration():
    df = pd.DataFrame([
        {
            "title": "AI Engineering Intern",
            "company": "iion",
            "location": "Remote",
            "ai_verdict": "Strong RAG match.",
            "match_percentage": 92,
            "tech_fit": 95,
            "experience_fit": 90,
            "logistics_fit": 90,
            "compensation": "$25/hr",
            "effort": "low",
            "suspicious": False,
            "job_url": "https://example.com/1",
        },
    ])
    stats = {"scraped": 100, "filtered": 50, "approved": 1}
    md = format_github_markdown(df, pd.DataFrame(), stats)
    _check("md has header row", "| Title | Company |" in md,         True)
    _check("md has match cell", "**92%** (T:95 E:90 L:90)" in md,    True)
    _check("md has compensation", "$25/hr" in md,                    True)


def test_verdict_pipe_escaping_in_md():
    df = pd.DataFrame([
        {
            "title": "T", "company": "C", "location": "L",
            "ai_verdict": "uses pipe | inside text",
            "match_percentage": 80, "tech_fit": 80,
            "experience_fit": 80, "logistics_fit": 80,
            "compensation": "Not stated", "effort": "low",
            "suspicious": False, "job_url": "#",
        },
    ])
    md = format_github_markdown(df, pd.DataFrame(), {"scraped": 1, "filtered": 1, "approved": 1})
    _check("pipe is escaped in md verdict", "uses pipe \\| inside text" in md, True)


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    print(f"Running {len(tests)} tests...\n")
    for t in tests:
        print(f"--- {t.__name__} ---")
        t()
    print(f"\nAll {len(tests)} tests passed.")
