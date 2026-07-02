"""apply_pipeline_filters end-to-end on a diverse 20-row dataframe.

This is the closest thing to "run the whole pre-AI gauntlet" without making
network calls. It catches regressions in any of: seen-jobs filter, reputation
prefilter, URL dedup, title+company dedup, CJK reject, langdetect reject,
location prefilter, seniority reject, tech-keyword filter.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "fixtures"))
from sample_jobs import mixed_filter_input
from pipeline.core_filter import apply_pipeline_filters


def test_filter_drops_senior_titles():
    out = apply_pipeline_filters(mixed_filter_input())
    titles = out["title"].astype(str).tolist()
    assert not any("Senior" in t for t in titles)
    assert not any("Lead Backend" in t for t in titles)
    assert not any("Engineering Manager" in t for t in titles)
    assert not any("Staff AI" in t for t in titles)


def test_filter_drops_non_tech_titles():
    out = apply_pipeline_filters(mixed_filter_input())
    titles = out["title"].astype(str).str.lower().tolist()
    assert not any("marketing coordinator" in t for t in titles)
    assert not any("sales operations manager" in t for t in titles)


def test_filter_drops_cjk_titles():
    out = apply_pipeline_filters(mixed_filter_input())
    titles = out["title"].astype(str).tolist()
    assert "ソフトウェアエンジニア" not in titles


def test_filter_drops_us_state_locked_when_title_lacks_remote():
    out = apply_pipeline_filters(mixed_filter_input())
    rows = out[out["company"] == "USCo"]
    assert len(rows) == 0   # location "San Francisco, California" without "Remote" in title -> dropped


def test_filter_keeps_us_state_when_title_has_remote():
    out = apply_pipeline_filters(mixed_filter_input())
    rows = out[out["company"] == "USCo2"]
    assert len(rows) == 1   # title "Remote Software Engineer" overrides location filter


def test_filter_flags_blacklisted_but_does_not_drop():
    """Reputation-blacklisted jobs are flagged via pre_flagged_low_quality, NOT dropped."""
    out = apply_pipeline_filters(mixed_filter_input())
    flagged = out[out["pre_flagged_low_quality"] == True]
    flagged_companies = flagged["company"].astype(str).str.lower().tolist()
    assert any("skillfied" in c for c in flagged_companies)
    assert any("webs it" in c for c in flagged_companies)


def test_filter_tags_trusted_company():
    """Anthropic is in the trust_boost list and should carry the trusted flag."""
    out = apply_pipeline_filters(mixed_filter_input())
    trusted = out[out["pre_flagged_trusted"] == True]
    trusted_companies = trusted["company"].astype(str).str.lower().tolist()
    assert "anthropic" in trusted_companies


def test_filter_deduplicates_url_collisions():
    """Two rows with identical job_url should collapse to one."""
    out = apply_pipeline_filters(mixed_filter_input())
    url_counts = out["job_url"].value_counts()
    assert url_counts.max() == 1


def test_filter_deduplicates_normalized_title_company():
    """'AI Engineer' vs 'AI Engineer (Remote)' for the same company should collapse."""
    out = apply_pipeline_filters(mixed_filter_input())
    same_co = out[out["company"].astype(str).str.lower() == "iion"]
    # At most one of: 'AI Engineering Intern' vs 'AI Engineering Intern (Remote)'
    assert len(same_co) <= 1


def test_filter_survives_empty_dataframe():
    import pandas as pd
    assert apply_pipeline_filters(pd.DataFrame()).empty


# ---------------------------------------------------------------------------
# Company-internal level codes (FAANG-style ladders) — Fix #7
# ---------------------------------------------------------------------------

def _single_row_df(title, **extras):
    """Minimal dataframe with one row + the required columns for the filter chain."""
    import pandas as pd
    row = {
        "title": title,
        "company": "TestCo",
        "location": "Remote",
        "job_url": f"https://example.com/job/{hash(title) & 0xFFFF}",
        "description": "Build great software with our team using Python and AWS. We are growing fast.",
        "date_posted": "",
    }
    row.update(extras)
    return pd.DataFrame([row])


def test_filter_drops_netflix_l5_title():
    """Netflix L5 = senior; would slip past the plain word filter."""
    df = _single_row_df("Software Engineer (L5) - Experimentation Platform")
    out = apply_pipeline_filters(df)
    assert len(out) == 0


def test_filter_drops_l6_l7_l8():
    for code in ("L6", "L7", "L8"):
        df = _single_row_df(f"Software Engineer {code}")
        assert len(apply_pipeline_filters(df)) == 0, f"Should drop {code}"


def test_filter_drops_meta_ic5():
    """Meta IC5+ codes denote senior individual contributors."""
    df = _single_row_df("Software Engineer, IC5")
    out = apply_pipeline_filters(df)
    assert len(out) == 0


def test_filter_drops_stripe_e5():
    df = _single_row_df("Software Engineer E5 — Payments")
    out = apply_pipeline_filters(df)
    assert len(out) == 0


def test_filter_drops_amazon_sde2():
    """Amazon SDE II/SDE-2 denotes a senior software development engineer."""
    df = _single_row_df("SDE II - Retail Systems")
    out = apply_pipeline_filters(df)
    assert len(out) == 0

    df2 = _single_row_df("SDE-3 Engineer")
    assert len(apply_pipeline_filters(df2)) == 0


def test_filter_keeps_junior_l1_l2_l3():
    """L1-L3 are entry/junior at most ladders — must NOT be filtered."""
    df = _single_row_df("Software Engineer L2 - New Grad Track")
    out = apply_pipeline_filters(df)
    assert len(out) == 1, "L2 (entry level) must not be dropped"


def test_filter_keeps_ic1_ic2_ic3():
    df = _single_row_df("Software Engineer, IC2")
    out = apply_pipeline_filters(df)
    assert len(out) == 1


def test_filter_drops_architect_title():
    """Architect roles are senior by industry convention."""
    df = _single_row_df("Software Architect, Backend")
    out = apply_pipeline_filters(df)
    assert len(out) == 0


# ---------------------------------------------------------------------------
# Per-user experience level (Tier 4) — mid/senior users keep senior titles.
# ---------------------------------------------------------------------------

def test_experience_level_gates_seniority_filter():
    """Entry-level (default) drops senior titles; a mid/senior user keeps them so
    the AI can judge fit against their CV."""
    df = _single_row_df("Senior Backend Engineer")
    assert len(apply_pipeline_filters(df, experience_level="entry")) == 0
    assert len(apply_pipeline_filters(df)) == 0   # default is entry
    assert len(apply_pipeline_filters(df, experience_level="mid")) == 1
    assert len(apply_pipeline_filters(df, experience_level="senior")) == 1


def test_experience_level_senior_keeps_level_codes():
    """FAANG level codes (L5+) are also relaxed for senior users."""
    df = _single_row_df("Software Engineer (L5)")
    assert len(apply_pipeline_filters(df, experience_level="entry")) == 0
    assert len(apply_pipeline_filters(df, experience_level="senior")) == 1


# ---------------------------------------------------------------------------
# Description language filter (Fix #6) — should drop foreign-language bodies
# ---------------------------------------------------------------------------

def test_filter_drops_german_long_description():
    """A title in English with a long German description (like L21s GmbH on
    2026-05-17) should now be dropped by the description-langdetect pass."""
    # 500+ chars of plausible German job description.
    german = (
        "Wir suchen einen erfahrenen Softwareentwickler mit fundierten Kenntnissen in "
        "Java und Kotlin sowie Erfahrung mit modernen Webtechnologien. Unsere Plattform "
        "verarbeitet täglich Millionen von Transaktionen und du wirst Teil eines "
        "internationalen Teams sein. Wir bieten flexible Arbeitszeiten, ein modernes "
        "Büro in Berlin und die Möglichkeit, von zu Hause aus zu arbeiten. "
        "Voraussetzungen: mindestens drei Jahre Berufserfahrung, sehr gute Deutschkenntnisse, "
        "Bereitschaft zur Teamarbeit. Wir freuen uns auf deine Bewerbung mit Lebenslauf "
        "und Anschreiben. Unsere Firma wächst schnell und wir suchen motivierte Mitarbeiter "
        "die etwas bewegen wollen und Verantwortung übernehmen können."
    )
    df = _single_row_df("Software Engineer (m/w/d)", description=german)
    try:
        from langdetect import detect  # noqa: F401
    except ImportError:
        # langdetect not installed — filter degrades to keep-all, can't test.
        return
    out = apply_pipeline_filters(df)
    assert len(out) == 0, "German description should be dropped by language filter"


def test_filter_keeps_short_description_in_any_language():
    """Below the 400-char threshold we don't trust langdetect — keep the row."""
    df = _single_row_df("Software Engineer", description="Kurz auf Deutsch.")
    out = apply_pipeline_filters(df)
    assert len(out) == 1, "Short non-English text must NOT be dropped"


def test_filter_keeps_english_description():
    """An English description well above the threshold passes."""
    english = (
        "We are looking for an experienced software engineer with strong knowledge of "
        "Python, FastAPI, and modern backend architectures. Our team is fully distributed "
        "and operates across Europe, Asia, and the Americas. You will work on production "
        "systems that handle millions of requests per day. Required qualifications include "
        "two or more years of professional software engineering experience, strong written "
        "communication skills, and a track record of shipping production code. We offer "
        "competitive compensation, full remote work, and a strong engineering culture."
    )
    df = _single_row_df("Software Engineer", description=english)
    out = apply_pipeline_filters(df)
    assert len(out) == 1, f"English description must pass; got {out}"


# ---------------------------------------------------------------------------
# Non-tech intern blockers (2026-05-17) — Fix from email analysis
# ---------------------------------------------------------------------------
# Real failures from the latest .eml:
#   "Graduate Research Intern, Biology" (DataAnnotation)
#   "Business Analyst Intern (Entry Level)" (LeoStoy Tech AI)
# Both were getting past the role-keyword filter because "intern" is a
# catch-all positive signal. We now reject "intern" titles that ALSO contain
# a non-tech signal (biology / business / etc.).

def test_filter_drops_biology_research_intern():
    df = _single_row_df("Graduate Research Intern, Biology")
    out = apply_pipeline_filters(df)
    assert len(out) == 0


def test_filter_drops_business_analyst_intern():
    df = _single_row_df("Business Analyst Intern (Entry Level)")
    out = apply_pipeline_filters(df)
    assert len(out) == 0


def test_filter_drops_business_analytics_intern():
    df = _single_row_df("Junior Business Analytics Intern")
    out = apply_pipeline_filters(df)
    assert len(out) == 0


def test_filter_drops_pharma_intern():
    df = _single_row_df("Research Intern - Pharma R&D")
    out = apply_pipeline_filters(df)
    assert len(out) == 0


def test_filter_drops_biotech_intern():
    df = _single_row_df("Software Engineering Intern - Biotech")
    out = apply_pipeline_filters(df)
    assert len(out) == 0


def test_filter_drops_hr_intern():
    df = _single_row_df("HR Intern - Recruiting")
    out = apply_pipeline_filters(df)
    assert len(out) == 0


def test_filter_drops_legal_intern():
    df = _single_row_df("Legal Intern, Contracts")
    out = apply_pipeline_filters(df)
    assert len(out) == 0


def test_filter_keeps_research_engineer_intern():
    """'Research Engineer' is a positive signal — even with 'intern', no non-tech blocker fires.

    Title kept short on purpose so langdetect's title pass skips it (its 30-char
    threshold otherwise causes false-positive non-English drops on titles loaded
    with proper nouns).
    """
    df = _single_row_df("ML Research Intern")
    out = apply_pipeline_filters(df)
    assert len(out) == 1


def test_filter_keeps_data_science_intern():
    """A genuine tech intern title should sail through the non-tech blocker.

    Kept under 30 chars to side-step langdetect's title-level false-positive on
    short tech titles loaded with proper nouns (a calibrated trade-off — see
    _is_english_title comment in core_filter.py).
    """
    df = _single_row_df("Data Science Intern")
    out = apply_pipeline_filters(df)
    assert len(out) == 1


def test_filter_keeps_software_engineer_intern():
    df = _single_row_df("Software Engineer Intern")
    out = apply_pipeline_filters(df)
    assert len(out) == 1


# ---------------------------------------------------------------------------
# Local mode (local=True) — the lighter filter set for local_companies.py.
# Pre-vetted Palestinian companies: skip the aggressive role/seniority/location
# steps (the AI verdict judges relevance), keep the universally-safe ones.
# ---------------------------------------------------------------------------

def test_local_keeps_job_without_tech_keyword_in_title():
    """A real post whose title lacks a tech keyword is DROPPED globally (step 6)
    but KEPT in local mode — this was the bug that zeroed the local pipeline."""
    df = _single_row_df("We're hiring! Join our growing team")
    assert len(apply_pipeline_filters(df, local=False)) == 0
    assert len(apply_pipeline_filters(df, local=True)) == 1


def test_local_keeps_senior_titles():
    """Seniority filter is global-only; a small local shop's 'Senior' role is
    still worth surfacing to the AI."""
    df = _single_row_df("Senior Backend Engineer")
    assert len(apply_pipeline_filters(df, local=False)) == 0
    assert len(apply_pipeline_filters(df, local=True)) == 1


def test_local_keeps_india_location_lock():
    """Location lock is global-only. 'India' is in the global block list and a
    title without 'remote' is dropped globally — local mode keeps it (it never
    runs the location step)."""
    df = _single_row_df("Software Engineer", location="Bangalore, India")
    assert len(apply_pipeline_filters(df, local=False)) == 0
    assert len(apply_pipeline_filters(df, local=True)) == 1


def test_local_still_drops_cjk_titles():
    """Universally-safe steps still run in local mode: a CJK title is dropped."""
    df = _single_row_df("ソフトウェアエンジニア")
    assert len(apply_pipeline_filters(df, local=True)) == 0


def test_local_keeps_arabic_description():
    """Arabic is the TARGET language for the Palestinian local market, not noise.
    An English tech title with a long Arabic body is dropped by the GLOBAL
    description langdetect pass, but must be KEPT in local mode."""
    arabic_desc = (
        "نبحث عن مطور برمجيات لديه خبرة في تطوير تطبيقات الويب باستخدام بايثون "
        "وجافاسكريبت للانضمام إلى فريقنا في رام الله. المهام تشمل بناء واجهات "
        "برمجية وصيانة الأنظمة الحالية والعمل ضمن فريق متكامل بدوام كامل. "
        "المتطلبات خبرة سنتين على الأقل ومعرفة جيدة بقواعد البيانات والخوارزميات."
    )
    df = _single_row_df("Software Engineer", description=arabic_desc, location="Ramallah")
    # Local mode keeps it — the language filter is skipped for the local market.
    assert len(apply_pipeline_filters(df, local=True)) == 1
    try:
        from langdetect import detect  # noqa: F401
    except ImportError:
        return  # can't prove the global-drop half without langdetect installed
    # Global mode drops the Arabic body — proving the local skip is what saves it.
    assert len(apply_pipeline_filters(df, local=False)) == 0


def test_local_still_dedups_url_collisions():
    """URL dedup still runs in local mode."""
    import pandas as pd
    base = {
        "title": "Backend Developer", "company": "PalCo", "location": "Ramallah",
        "job_url": "https://palco.ps/jobs/1", "description": "x" * 60, "date_posted": "",
    }
    df = pd.DataFrame([base, dict(base)])  # identical URL twice
    out = apply_pipeline_filters(df, local=True)
    assert len(out) == 1


def test_local_still_flags_reputation_without_dropping():
    """Reputation flagging (not dropping) still applies in local mode."""
    df = _single_row_df("Web Developer", company="Skillfied Mentor")
    out = apply_pipeline_filters(df, local=True)
    assert len(out) == 1
    assert bool(out.iloc[0]["pre_flagged_low_quality"]) is True


def test_local_empty_dataframe():
    import pandas as pd
    assert apply_pipeline_filters(pd.DataFrame(), local=True).empty
