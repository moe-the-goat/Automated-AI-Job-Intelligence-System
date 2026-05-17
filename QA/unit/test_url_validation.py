"""URL validation layers — path-pattern check + HEAD-probe ghost-detection.

Both layers protect the local pipeline from accepting non-job URLs and stale
listings that DDG/Bing index for weeks after a company removes a posting.
"""
from pipeline.url_validation import (
    is_job_url_like,
    probe_url_alive,
    probe_urls_alive_batch,
)


# ---------------------------------------------------------------------------
# Path-pattern check (is_job_url_like) — pure, no network
# ---------------------------------------------------------------------------

def test_accepts_canonical_job_paths():
    assert is_job_url_like("https://innotech.factorialhr.com/job_posting/devops-engineer-20937")
    assert is_job_url_like("https://boards.greenhouse.io/asaltech/jobs/12345")
    assert is_job_url_like("https://acme.com/careers/senior-backend-engineer")
    assert is_job_url_like("https://acme.com/career/data-scientist")
    assert is_job_url_like("https://acme.com/positions/ml-intern")
    assert is_job_url_like("https://acme.com/vacancy/devops-2026")
    assert is_job_url_like("https://acme.com/apply/role-1234")
    assert is_job_url_like("https://acme.com/role/intern-software")
    assert is_job_url_like("https://acme.com/openings/intern")
    assert is_job_url_like("https://acme.com/we-are-hiring/junior-dev")


def test_rejects_freightos_market_update_blog():
    """The exact URL that slipped through on 2026-05-17 — a market-update blog post."""
    url = ("https://www.freightos.com/freight-industry-updates/market-updates/"
           "the-data-behind-amazons-logistics-and-fulfillment-play/")
    assert is_job_url_like(url) is False


def test_rejects_generic_blog_post():
    assert is_job_url_like("https://acme.com/blog/our-engineering-culture") is False
    assert is_job_url_like("https://acme.com/blogs/team-update-2024") is False


def test_rejects_news_and_press_paths():
    assert is_job_url_like("https://acme.com/news/funding-announcement") is False
    assert is_job_url_like("https://acme.com/press/series-b") is False
    assert is_job_url_like("https://acme.com/press-release/q1-2026") is False


def test_rejects_article_paths():
    assert is_job_url_like("https://acme.com/article/why-we-use-rust") is False
    assert is_job_url_like("https://acme.com/articles/intern-success-story") is False


def test_rejects_year_archive_paths():
    """Stale-content guard: paths with a year segment are almost always archives."""
    assert is_job_url_like("https://acme.com/2017/03/old-thing") is False
    assert is_job_url_like("https://acme.com/2024/12/something") is False
    assert is_job_url_like("https://acme.com/category/2020/data") is False


def test_rejects_market_update_path():
    assert is_job_url_like("https://acme.com/market-updates/q1-2026") is False
    assert is_job_url_like("https://acme.com/market_update/anything") is False


def test_rejects_case_study_and_whitepaper():
    assert is_job_url_like("https://acme.com/case-study/customer-x") is False
    assert is_job_url_like("https://acme.com/case-studies/all") is False
    assert is_job_url_like("https://acme.com/whitepaper/2025") is False
    assert is_job_url_like("https://acme.com/whitepapers/list") is False


def test_rejects_about_contact_team_paths():
    assert is_job_url_like("https://acme.com/about") is False
    assert is_job_url_like("https://acme.com/contact/sales") is False
    assert is_job_url_like("https://acme.com/team/leadership") is False
    assert is_job_url_like("https://acme.com/leadership/cto") is False


def test_rejects_event_and_webinar_paths():
    assert is_job_url_like("https://acme.com/events/2026-meetup") is False
    assert is_job_url_like("https://acme.com/webinar/ai-trends") is False
    assert is_job_url_like("https://acme.com/podcast/episode-12") is False


def test_rejects_url_with_no_path_signals():
    """A bare domain or root path has no positive signal so it's rejected."""
    assert is_job_url_like("https://acme.com/") is False
    assert is_job_url_like("https://acme.com") is False
    assert is_job_url_like("https://acme.com/random-page") is False


def test_rejects_garbage_input():
    assert is_job_url_like("") is False
    assert is_job_url_like(None) is False
    assert is_job_url_like(123) is False


def test_linkedin_posts_get_passed_through():
    """LinkedIn /posts/ paths trigger the negative `/posts/` signal but we make
    an explicit exception — the local pipeline relies on LinkedIn post URLs
    being valid signals on their own."""
    assert is_job_url_like("https://www.linkedin.com/posts/asaltech_hiring-software-engineer-activity-1234") is True


def test_accepts_employment_and_hire_paths():
    assert is_job_url_like("https://acme.com/employment/open-roles") is True
    assert is_job_url_like("https://acme.com/work-with-us/data-scientist") is True
    assert is_job_url_like("https://acme.com/join-us/backend-engineer") is True


# ---------------------------------------------------------------------------
# probe_url_alive / probe_urls_alive_batch — guard rails (no real network)
# ---------------------------------------------------------------------------

def test_probe_returns_false_on_empty_url():
    assert probe_url_alive("") is False
    assert probe_url_alive(None) is False


def test_batch_returns_empty_dict_on_empty_input():
    assert probe_urls_alive_batch([]) == {}
    assert probe_urls_alive_batch(None) == {}


def test_probe_treats_network_failure_as_alive():
    """We default to True on exceptions so a transient DNS / timeout doesn't
    silently delete a real job from today's email. Pointing at an unroutable
    address should raise immediately and the function should still return True."""
    # 192.0.2.0/24 is reserved TEST-NET-1 — guaranteed to never route.
    result = probe_url_alive("http://192.0.2.1/job/x", timeout=1)
    assert result is True
