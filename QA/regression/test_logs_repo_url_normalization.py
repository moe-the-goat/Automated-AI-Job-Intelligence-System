"""REGRESSION: GitHub repo identifier normalization.

Symptom on 2026-05-16: user added the GitHub Actions variable `LOGS_REPO`
with value `https://github.com/moe-the-goat/job-scrapper-logs`. Our code
naively used it as `owner/repo` and built `https://api.github.com/repos/https://github.com/.../issues`
— 404 every call, no issues ever landed in the private logs repo.

Fix: `_normalize_repo` strips scheme, github.com prefix, www, trailing slash,
and `.git` suffix before use.

This test locks the behaviour in so anyone who paste a full URL in the
GH variable doesn't break the pipeline.
"""
from pipeline.core_notify import _normalize_repo


def test_full_https_url_is_normalized():
    assert _normalize_repo(
        "https://github.com/moe-the-goat/job-scrapper-logs"
    ) == "moe-the-goat/job-scrapper-logs"


def test_full_url_with_trailing_slash_is_normalized():
    assert _normalize_repo(
        "https://github.com/moe-the-goat/job-scrapper-logs/"
    ) == "moe-the-goat/job-scrapper-logs"


def test_url_with_git_suffix_is_normalized():
    assert _normalize_repo(
        "https://github.com/moe-the-goat/job-scrapper-logs.git"
    ) == "moe-the-goat/job-scrapper-logs"


def test_url_with_www_prefix_is_normalized():
    assert _normalize_repo(
        "https://www.github.com/moe-the-goat/job-scrapper-logs"
    ) == "moe-the-goat/job-scrapper-logs"


def test_plain_owner_repo_passes_through_unchanged():
    assert _normalize_repo("owner/repo") == "owner/repo"


def test_empty_inputs_return_none():
    assert _normalize_repo("") is None
    assert _normalize_repo(None) is None
    assert _normalize_repo("   ") is None
