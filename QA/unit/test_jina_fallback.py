"""Jina Reader + Gemini extraction fallback for ATS-less careers pages.

Most of the logic is pure: we receive a string from Gemini and produce a list
of normalized job dicts. The HTTP side (Jina Reader, Gemini client) is mocked
via the bypass argument or by isolating the parser from the orchestrator.

Wave-3 additions cover the BS4-before-Jina tiered fallback:
  extract_text_from_html — pure BS4 wrapper (no network)
  extract_jobs_from_careers_page — orchestrator (guard-rail tests only)
"""
from pipeline.core_ats import (
    parse_jina_jobs_response,
    extract_jobs_via_jina,
    extract_text_from_html,
    extract_jobs_from_careers_page,
    BS_MIN_USEFUL_CHARS,
)


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


# ---------------------------------------------------------------------------
# extract_text_from_html — pure Tier-1 extractor
# ---------------------------------------------------------------------------

def test_bs_extracts_visible_text_from_ssr_page():
    """A typical server-rendered careers page yields substantial text."""
    html = """
    <html><head><title>Careers</title></head>
      <body>
        <header>Our Company</header>
        <main>
          <h1>Open Positions</h1>
          <ul>
            <li>Senior Backend Engineer — Ramallah / Remote</li>
            <li>ML Engineering Intern — Worldwide</li>
            <li>Frontend Developer — Berlin Hybrid</li>
          </ul>
          <p>Apply at careers@example.com</p>
        </main>
        <footer>(c) 2026</footer>
      </body>
    </html>
    """
    text = extract_text_from_html(html)
    assert "Senior Backend Engineer" in text
    assert "ML Engineering Intern" in text
    assert "Frontend Developer" in text
    assert len(text) >= BS_MIN_USEFUL_CHARS or len(text) > 100  # plenty of real text


def test_bs_strips_scripts_and_styles():
    """Inline <script> and <style> blocks shouldn't pollute the extracted text."""
    html = """
    <html>
      <head>
        <style>body { color: red; } .nav { display: flex; }</style>
        <script>console.log('tracker'); var X = 'should not appear';</script>
      </head>
      <body>
        <p>Real visible content here.</p>
        <script>analytics('track');</script>
      </body>
    </html>
    """
    text = extract_text_from_html(html)
    assert "Real visible content here" in text
    assert "console.log" not in text
    assert "should not appear" not in text
    assert "color: red" not in text


def test_bs_returns_short_text_for_spa_shell():
    """A typical React/Vue SPA shell has near-zero visible text. Must fall through to Jina."""
    html = """
    <html>
      <head><title>App</title></head>
      <body>
        <div id="root"></div>
        <script src="/static/js/main.abc123.js"></script>
      </body>
    </html>
    """
    text = extract_text_from_html(html)
    # Body has effectively no real content — should be well below the threshold.
    assert len(text) < BS_MIN_USEFUL_CHARS


def test_bs_handles_empty_or_none_html():
    assert extract_text_from_html("") == ""
    assert extract_text_from_html(None) == ""


def test_bs_min_useful_chars_constant_is_reasonable():
    """Sanity check: the threshold should be in a sensible range so a brief
    careers page still passes Tier 1 but a SPA shell doesn't."""
    assert 100 <= BS_MIN_USEFUL_CHARS <= 2000


# ---------------------------------------------------------------------------
# extract_jobs_from_careers_page — orchestrator guard rails (no network)
# ---------------------------------------------------------------------------

def test_tiered_returns_empty_when_no_api_key():
    assert extract_jobs_from_careers_page("https://example.com/careers", "Co", gemini_api_key="") == []
    assert extract_jobs_from_careers_page("https://example.com/careers", "Co", gemini_api_key=None) == []


def test_tiered_returns_empty_when_no_careers_url():
    assert extract_jobs_from_careers_page("", "Co", gemini_api_key="fake", html="<p>x</p>") == []
    assert extract_jobs_from_careers_page(None, "Co", gemini_api_key="fake", html="<p>x</p>") == []
