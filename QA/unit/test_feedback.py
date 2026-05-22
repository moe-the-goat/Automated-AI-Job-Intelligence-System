"""Feedback ingestion + page rendering tests. Exercises the deterministic
pieces (entry sanitizer, block-company → reputation update, HTML render).
GitHub API calls are not tested here — they are thin wrappers around
`requests` and would only verify mocks."""

import json
import os
import tempfile

import pandas as pd

import pipeline.core_feedback as cf
from pipeline.core_feedback import (
    _apply_block_companies,
    _normalize_repo,
    _sanitize_entries,
    VALID_FEEDBACK_TYPES,
)
from pipeline.core_feedback_page import (
    FEEDBACK_OPTIONS,
    render_feedback_page,
)


# ---------------------------------------------------------------------------
# Entry sanitization
# ---------------------------------------------------------------------------

def test_sanitize_accepts_valid_entry():
    entries = _sanitize_entries([{
        "job_url": "https://example.com/job/1",
        "company": "Stripe",
        "title": "Software Engineer",
        "location": "Remote",
        "feedback": "applied",
        "note": "Applied this morning",
        "date": "2026-05-22",
    }])
    assert len(entries) == 1
    assert entries[0]["feedback"] == "applied"
    assert entries[0]["company"] == "Stripe"


def test_sanitize_drops_unknown_feedback_type():
    entries = _sanitize_entries([{
        "job_url": "https://example.com/job/1",
        "feedback": "deleted_my_database",
    }])
    assert entries == []


def test_sanitize_drops_entry_without_url():
    entries = _sanitize_entries([{
        "job_url": "",
        "feedback": "applied",
    }])
    assert entries == []


def test_sanitize_normalizes_feedback_case():
    entries = _sanitize_entries([{
        "job_url": "https://example.com/x",
        "feedback": "APPLIED",
    }])
    assert entries[0]["feedback"] == "applied"


def test_sanitize_clamps_long_fields():
    long_note = "x" * 5000
    entries = _sanitize_entries([{
        "job_url": "https://example.com/x",
        "feedback": "other",
        "note": long_note,
    }])
    assert len(entries[0]["note"]) == 1000


def test_sanitize_skips_non_dict_entries():
    entries = _sanitize_entries([
        "not a dict",
        {"job_url": "https://example.com/x", "feedback": "applied"},
        None,
    ])
    assert len(entries) == 1


def test_sanitize_handles_non_list_input():
    assert _sanitize_entries(None) == []
    assert _sanitize_entries("garbage") == []
    assert _sanitize_entries({}) == []


def test_all_valid_feedback_types_accepted():
    for fb in VALID_FEEDBACK_TYPES:
        entries = _sanitize_entries([{
            "job_url": "https://example.com/x",
            "feedback": fb,
        }])
        assert len(entries) == 1, f"Feedback type {fb!r} should be accepted"


# ---------------------------------------------------------------------------
# Hard signals: block_company → reputation.json
# ---------------------------------------------------------------------------

class _RepPath:
    """Context manager that creates a temp reputation.json, points the
    module at it, and restores the original path + file content on exit.
    Stdlib-only replacement for pytest's tmp_path + monkeypatch."""

    def __init__(self, seed=None):
        self.seed = seed or {
            "blacklist_name_patterns": ["pre-existing-bad-corp"],
            "blacklist_handle_patterns": [],
            "trust_boost": [],
        }

    def __enter__(self):
        self.tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8",
        )
        json.dump(self.seed, self.tmp)
        self.tmp.close()
        self.original_path = cf.LOCAL_REPUTATION_PATH
        cf.LOCAL_REPUTATION_PATH = self.tmp.name
        return self.tmp.name

    def __exit__(self, *exc):
        cf.LOCAL_REPUTATION_PATH = self.original_path
        try:
            os.unlink(self.tmp.name)
        except OSError:
            pass


def test_block_company_appends_to_reputation():
    with _RepPath() as path:
        entries = _sanitize_entries([{
            "job_url": "https://example.com/x",
            "company": "Sketchy Co",
            "feedback": "block_company",
        }])
        added = _apply_block_companies(entries)
        assert added == 1
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "sketchy co" in data["blacklist_name_patterns"]


def test_block_company_is_idempotent():
    with _RepPath():
        entries = _sanitize_entries([
            {"job_url": "https://x/1", "company": "Twice Corp", "feedback": "block_company"},
            {"job_url": "https://x/2", "company": "Twice Corp", "feedback": "block_company"},
        ])
        added = _apply_block_companies(entries)
        assert added == 1


def test_block_existing_company_is_noop():
    with _RepPath():
        entries = _sanitize_entries([{
            "job_url": "https://x/1",
            "company": "Pre-Existing-Bad-Corp",
            "feedback": "block_company",
        }])
        added = _apply_block_companies(entries)
        assert added == 0


def test_non_block_feedback_does_not_touch_reputation():
    with _RepPath() as path:
        with open(path, "r", encoding="utf-8") as f:
            before = f.read()
        entries = _sanitize_entries([{
            "job_url": "https://x/1",
            "company": "Some Co",
            "feedback": "applied",
        }])
        added = _apply_block_companies(entries)
        assert added == 0
        with open(path, "r", encoding="utf-8") as f:
            assert f.read() == before


def test_block_company_without_company_name_skipped():
    with _RepPath():
        entries = _sanitize_entries([{
            "job_url": "https://x/1",
            "company": "",
            "feedback": "block_company",
        }])
        added = _apply_block_companies(entries)
        assert added == 0


# ---------------------------------------------------------------------------
# Repo URL normalization
# ---------------------------------------------------------------------------

def test_normalize_repo_owner_name():
    assert _normalize_repo("owner/repo") == "owner/repo"


def test_normalize_repo_full_url():
    assert _normalize_repo("https://github.com/owner/repo") == "owner/repo"


def test_normalize_repo_strips_git_suffix():
    assert _normalize_repo("https://github.com/owner/repo.git") == "owner/repo"


def test_normalize_repo_trailing_slash():
    assert _normalize_repo("github.com/owner/repo/") == "owner/repo"


def test_normalize_repo_empty():
    assert _normalize_repo("") is None
    assert _normalize_repo(None) is None


# ---------------------------------------------------------------------------
# Feedback page HTML rendering
# ---------------------------------------------------------------------------

def _sample_df():
    return pd.DataFrame([
        {
            "title": "AI Engineer",
            "company": "Anthropic",
            "location": "Remote",
            "ai_verdict": "MATCH: RAG project matches. REMOTE: Worldwide remote.",
            "job_url": "https://example.com/job/1",
            "match_percentage": 92,
        },
        {
            "title": "Software Engineer",
            "company": "Stripe",
            "location": "Remote (US)",
            "ai_verdict": "MATCH: FastAPI background. GAP: US-only.",
            "job_url": "https://example.com/job/2",
            "match_percentage": 68,
        },
    ])


def test_render_page_contains_all_jobs():
    html = render_feedback_page(
        dfs=[_sample_df()],
        logs_repo="owner/logs",
        write_token="ghp_fake_test_token",
    )
    assert "AI Engineer" in html
    assert "Software Engineer" in html
    assert "Anthropic" in html
    assert "Stripe" in html


def test_render_page_embeds_options():
    html = render_feedback_page(
        dfs=[_sample_df()],
        logs_repo="owner/logs",
        write_token="t",
    )
    for value, label in FEEDBACK_OPTIONS:
        assert label in html


def test_render_page_escapes_html_in_job_fields():
    df = pd.DataFrame([{
        "title": "<script>alert('xss')</script>",
        "company": "Evil & Co",
        "location": "Remote",
        "ai_verdict": "verdict with <b>html</b>",
        "job_url": "https://example.com/job/x",
        "match_percentage": 50,
    }])
    html = render_feedback_page(dfs=[df], logs_repo="o/r", write_token="t")
    assert "<script>alert" not in html
    assert "&lt;script&gt;" in html
    assert "&amp;" in html


def test_render_page_handles_empty_dfs():
    html = render_feedback_page(
        dfs=[pd.DataFrame(), None],
        logs_repo="owner/logs",
        write_token="t",
    )
    assert "No jobs" in html


def test_render_page_embeds_token_and_repo_as_json_literals():
    """Token and repo are injected as JSON string literals so quote characters
    can't break the surrounding JavaScript."""
    html = render_feedback_page(
        dfs=[_sample_df()],
        logs_repo="owner/repo-name",
        write_token="ghp_TESTTOKEN",
    )
    assert '"ghp_TESTTOKEN"' in html
    assert '"owner/repo-name"' in html


def test_render_page_badge_color_classes():
    html = render_feedback_page(
        dfs=[_sample_df()],
        logs_repo="o/r",
        write_token="t",
    )
    assert "match-green" in html  # 92
    assert "match-red" in html  # 68


# ---------------------------------------------------------------------------
# Verdict prompt with learned_preferences
# ---------------------------------------------------------------------------

def test_verdict_prompt_omits_preferences_when_empty():
    from pipeline.core_ai import _build_verdict_prompt
    prompt = _build_verdict_prompt(
        {"title": "Engineer", "company": "X", "job_type": "fulltime"},
        cv_text="CV",
        description="desc",
        learned_preferences="",
    )
    assert "LEARNED PREFERENCES" not in prompt


def test_verdict_prompt_includes_preferences_when_present():
    from pipeline.core_ai import _build_verdict_prompt
    prefs = "Consistently applies to RAG/LLM roles. Rejects pure frontend positions."
    prompt = _build_verdict_prompt(
        {"title": "Engineer", "company": "X", "job_type": "fulltime"},
        cv_text="CV",
        description="desc",
        learned_preferences=prefs,
    )
    assert "LEARNED PREFERENCES" in prompt
    assert prefs in prompt


def test_verdict_prompt_strips_whitespace_only_preferences():
    from pipeline.core_ai import _build_verdict_prompt
    prompt = _build_verdict_prompt(
        {"title": "Engineer", "company": "X", "job_type": "fulltime"},
        cv_text="CV",
        description="desc",
        learned_preferences="   \n  \t  ",
    )
    assert "LEARNED PREFERENCES" not in prompt
