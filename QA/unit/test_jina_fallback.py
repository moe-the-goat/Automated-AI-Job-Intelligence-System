"""Jina Reader + Gemini extraction fallback for ATS-less careers pages.

Most of the logic is pure: we receive a string from Gemini and produce a list
of normalized job dicts. The HTTP side (Jina Reader, Gemini client) is mocked
via the bypass argument or by isolating the parser from the orchestrator.
"""
from core_ats import parse_jina_jobs_response, extract_jobs_via_jina


# ---------------------------------------------------------------------------
# parse_jina_jobs_response — pure parser
# ---------------------------------------------------------------------------

def test_parses_canonical_gemini_response():
    text = """{
      "jobs": [
        {"title": "Backend Engineer", "location": "Ramallah, Palestine",
         "job_url": "https://acme.ps/careers/backend",
         "description": "Build microservices.", "date_posted": "2026-05-10"},
        {"title": "Frontend Intern", "location": "Remote",
         "job_url": "https://acme.ps/careers/frontend",
         "description": "Ship features.", "date_posted": ""}
      ]
    }"""
    out = parse_jina_jobs_response(text, "Acme Inc")
    assert len(out) == 2
    assert out[0]["title"] == "Backend Engineer"
    assert out[0]["company"] == "Acme Inc"
    assert out[0]["location"] == "Ramallah, Palestine"
    assert out[0]["job_url"] == "https://acme.ps/careers/backend"
    assert out[1]["title"] == "Frontend Intern"


def test_parses_response_wrapped_in_json_fences():
    """Gemini sometimes wraps its output in ```json``` despite instructions otherwise."""
    text = """```json
{"jobs": [{"title": "X", "location": "L", "job_url": "u",
           "description": "d", "date_posted": "p"}]}
```"""
    out = parse_jina_jobs_response(text, "Co")
    assert len(out) == 1
    assert out[0]["title"] == "X"


def test_parses_response_with_leading_prose():
    """Carve out the JSON block even if the model prepends commentary."""
    text = 'Sure, here are the jobs:\n{"jobs": [{"title": "Y", "job_url": "u"}]}'
    out = parse_jina_jobs_response(text, "Co")
    assert len(out) == 1
    assert out[0]["title"] == "Y"


def test_returns_empty_on_no_jobs():
    out = parse_jina_jobs_response('{"jobs": []}', "Co")
    assert out == []


def test_returns_empty_on_malformed_json():
    assert parse_jina_jobs_response("this isn't JSON at all", "Co") == []
    assert parse_jina_jobs_response("", "Co") == []
    assert parse_jina_jobs_response(None, "Co") == []


def test_skips_entries_with_no_title():
    """If Gemini returns a half-baked entry without a title, drop it."""
    text = '{"jobs": [{"title": "", "job_url": "u"}, {"title": "Good Job", "job_url": "v"}]}'
    out = parse_jina_jobs_response(text, "Co")
    assert len(out) == 1
    assert out[0]["title"] == "Good Job"


def test_skips_non_dict_entries():
    text = '{"jobs": [null, "garbage", {"title": "Real"}]}'
    out = parse_jina_jobs_response(text, "Co")
    assert len(out) == 1
    assert out[0]["title"] == "Real"


def test_handles_missing_optional_fields():
    """A job with only a title is still acceptable — empty strings fill the rest."""
    text = '{"jobs": [{"title": "Minimal Job"}]}'
    out = parse_jina_jobs_response(text, "Co")
    assert len(out) == 1
    assert out[0]["title"] == "Minimal Job"
    assert out[0]["location"] == "Remote/Unspecified"        # _normalize_job default
    assert out[0]["description"] == ""


# ---------------------------------------------------------------------------
# extract_jobs_via_jina — orchestrator. Guard rails only (no network).
# ---------------------------------------------------------------------------

def test_extract_returns_empty_when_no_api_key():
    """If the caller didn't pass a Gemini key, we short-circuit instead of crashing."""
    assert extract_jobs_via_jina("https://example.com/careers", "Co", gemini_api_key="") == []
    assert extract_jobs_via_jina("https://example.com/careers", "Co", gemini_api_key=None) == []


def test_extract_returns_empty_when_no_careers_url():
    assert extract_jobs_via_jina("", "Co", gemini_api_key="fake") == []
    assert extract_jobs_via_jina(None, "Co", gemini_api_key="fake") == []
