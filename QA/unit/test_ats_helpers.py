"""LinkedIn handle extraction + ATS detection from HTML.

Pure functions, no network. The whole point of separating these from the HTTP
calls in core_ats is so we can lock down the patterns with deterministic tests.
"""
from core_ats import extract_linkedin_handle, detect_ats_from_html


# --- LinkedIn handle extraction ---

def test_handle_from_canonical_url():
    assert extract_linkedin_handle(
        "https://www.linkedin.com/company/adham-inc./") == "adham-inc."


def test_handle_from_url_with_about_suffix():
    assert extract_linkedin_handle(
        "https://www.linkedin.com/company/alameen-technologies/about/"
    ) == "alameen-technologies"


def test_handle_from_url_with_posts_suffix():
    assert extract_linkedin_handle(
        "https://www.linkedin.com/company/aipilot-software-inc/posts/"
    ) == "aipilot-software-inc"


def test_handle_from_url_without_www():
    assert extract_linkedin_handle(
        "https://linkedin.com/company/foo-corp"
    ) == "foo-corp"


def test_handle_from_url_without_scheme():
    assert extract_linkedin_handle("linkedin.com/company/bar/") == "bar"


def test_handle_from_personal_in_url_also_supported():
    # /in/ profile URLs work the same — handy when a company links to a founder.
    assert extract_linkedin_handle(
        "https://www.linkedin.com/in/jane-doe/") == "jane-doe"


def test_handle_returns_none_on_unrelated_url():
    assert extract_linkedin_handle("https://example.com/about") is None


def test_handle_returns_none_on_empty_or_nan():
    assert extract_linkedin_handle("") is None
    assert extract_linkedin_handle(None) is None
    assert extract_linkedin_handle("nan") is None


# --- ATS detection from HTML ---

def test_detect_greenhouse_from_iframe():
    html = """
    <html>
      <body>
        <iframe src="https://boards.greenhouse.io/asaltech"></iframe>
      </body>
    </html>
    """
    ats, token = detect_ats_from_html(html)
    assert ats == "greenhouse"
    assert token == "asaltech"


def test_detect_greenhouse_from_embed_url():
    html = '<a href="https://boards.greenhouse.io/embed/job_board?for=stripe">Apply</a>'
    ats, token = detect_ats_from_html(html)
    assert ats == "greenhouse"
    assert token == "stripe"


def test_detect_lever_from_link():
    html = '<a href="https://jobs.lever.co/anthropic/abc123">Open Roles</a>'
    ats, token = detect_ats_from_html(html)
    assert ats == "lever"
    assert token == "anthropic"


def test_detect_workable_from_link():
    html = '<a href="https://apply.workable.com/example-co/">Apply</a>'
    ats, token = detect_ats_from_html(html)
    assert ats == "workable"
    assert token == "example-co"


def test_detect_bamboohr_from_subdomain():
    html = '<a href="https://acmecorp.bamboohr.com/jobs/">See Jobs</a>'
    ats, token = detect_ats_from_html(html)
    assert ats == "bamboohr"
    assert token == "acmecorp"


def test_detect_smartrecruiters_from_link():
    html = '<a href="https://careers.smartrecruiters.com/Bosch">Careers</a>'
    ats, token = detect_ats_from_html(html)
    assert ats == "smartrecruiters"
    assert token == "Bosch"


def test_detect_returns_none_when_no_ats_present():
    html = "<html><body>Email us at careers@example.com</body></html>"
    ats, token = detect_ats_from_html(html)
    assert ats is None
    assert token is None


def test_detect_handles_empty_html():
    ats, token = detect_ats_from_html("")
    assert ats is None
    ats, token = detect_ats_from_html(None)
    assert ats is None


def test_detect_uses_first_match_on_multiple_ats_signals():
    # If a page lists multiple ATSes (rare, but happens for parent companies),
    # we take the first match in detector order.
    html = """
    <a href="https://boards.greenhouse.io/companyA">A</a>
    <a href="https://jobs.lever.co/companyB">B</a>
    """
    ats, token = detect_ats_from_html(html)
    assert ats == "greenhouse"
    assert token == "companyA"
