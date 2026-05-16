"""Reusable test fixtures: realistic job dataframes for integration tests."""
import pandas as pd


def baseline_job_row(**overrides):
    """Build one realistic AI-evaluated job row with all expected columns.

    Override any field via kwargs. Used by integration tests to build small
    dataframes without repeating 12-column dict literals.
    """
    base = {
        "title": "AI Engineering Intern",
        "company": "ExampleCo",
        "location": "Remote",
        "ai_verdict": "Your RAG project (LangChain + FAISS) directly matches their LLM-app requirement.",
        "match_percentage": 80,
        "tech_fit": 85,
        "experience_fit": 75,
        "logistics_fit": 90,
        "compensation": "$25/hr",
        "effort": "low",
        "suspicious": False,
        "scam": False,
        "pre_flagged_low_quality": False,
        "job_url": "https://example.com/jobs/1",
    }
    base.update(overrides)
    return base


def mixed_filter_input():
    """A 20-row dataframe with diverse signals for testing apply_pipeline_filters."""
    return pd.DataFrame([
        # ↓ Tech roles that SHOULD survive
        {"title": "AI Engineering Intern", "company": "iion", "location": "Remote", "job_url": "https://example.com/1"},
        {"title": "Junior Software Engineer", "company": "RealCorp", "location": "Worldwide", "job_url": "https://example.com/2"},
        {"title": "Backend Developer (Junior)", "company": "Anthropic", "location": "Remote", "job_url": "https://example.com/3"},
        {"title": "Data Scientist Intern", "company": "OpenAI", "location": "Remote, EMEA", "job_url": "https://example.com/4"},
        # ↓ Senior — should be dropped
        {"title": "Senior Software Engineer", "company": "BigCo", "location": "Remote", "job_url": "https://example.com/5"},
        {"title": "Lead Backend Engineer", "company": "BigCo", "location": "Remote", "job_url": "https://example.com/6"},
        {"title": "Engineering Manager", "company": "BigCo", "location": "Remote", "job_url": "https://example.com/7"},
        {"title": "Staff AI Engineer", "company": "DeepMind", "location": "Remote", "job_url": "https://example.com/8"},
        # ↓ Non-tech keyword — should be dropped (no tech keyword)
        {"title": "Marketing Coordinator", "company": "AdCo", "location": "Remote", "job_url": "https://example.com/9"},
        {"title": "Sales Operations Manager", "company": "SalesCo", "location": "Remote", "job_url": "https://example.com/10"},
        # ↓ Non-English title — should be dropped by language filter
        {"title": "ソフトウェアエンジニア", "company": "JapanCo", "location": "Remote", "job_url": "https://example.com/11"},
        # ↓ Location-locked without "remote" — should be dropped
        {"title": "Software Engineer", "company": "USCo", "location": "San Francisco, California", "job_url": "https://example.com/12"},
        # ↓ Location-locked but title says "Remote" — kept
        {"title": "Remote Software Engineer", "company": "USCo2", "location": "New York", "job_url": "https://example.com/13"},
        # ↓ Blacklisted by reputation — flagged but kept
        {"title": "Data Science Intern", "company": "Skillfied Mentor", "location": "Remote", "job_url": "https://example.com/14"},
        {"title": "Web Developer Intern", "company": "Webs IT Solution", "location": "Remote", "job_url": "https://example.com/15"},
        # ↓ Trusted boost — kept and tagged
        {"title": "AI Engineer", "company": "Anthropic", "location": "Remote", "job_url": "https://example.com/16"},
        # ↓ Duplicate URL — second instance dropped by URL dedup
        {"title": "AI Engineer", "company": "Anthropic", "location": "Remote", "job_url": "https://example.com/16"},
        # ↓ Normalized-title-+-company dedup (same role, different formatting)
        {"title": "AI Engineering Intern (Remote)", "company": "iion", "location": "Worldwide", "job_url": "https://example.com/17"},
        # ↓ Empty title — depending on filter, might drop
        {"title": "", "company": "EmptyCo", "location": "Remote", "job_url": "https://example.com/18"},
        # ↓ NaN title — dropped by keyword filter (na=False)
        {"title": None, "company": "NullCo", "location": "Remote", "job_url": "https://example.com/19"},
    ])
