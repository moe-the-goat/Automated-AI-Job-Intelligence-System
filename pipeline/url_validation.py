"""URL validation for DDG-discovered job results.

Two-layer defense against the two failure modes we see in the local pipeline:

  1. Path-pattern check — kills blog posts / marketing pages whose URL contains
     /blog/, /news/, /market-updates/, or a /YYYY/ year segment, while requiring
     a positive job-page signal (/job/, /jobs/, /careers/, /position/, /apply/,
     /vacancy/, /job_posting/, /opening/, etc.).

     Real example killed by this layer: freightos.com/freight-industry-updates/
     market-updates/the-data-behind-amazons-logistics-and-fulfillment-play/
     was getting accepted because DDG matched "logistics" in the slug.

  2. HEAD probe — kills "ghost" listings where the URL was once a real job but
     the company has since taken it down. Bing/DDG keep stale URLs in their
     index for weeks. A 404, 410, or redirect-to-not-found URL is a clear
     signal the listing is dead.

     Real example killed by this layer: innotech.factorialhr.com/job_posting/
     devops-engineer-20937 — URL pattern looked fine, but the job was no longer
     listed on Innotech's official careers page.

Both layers are pure functions where possible; the HEAD probe is the only
network-touching piece and is wrapped in a tight timeout + exception catch
so a flaky check never blocks the rest of the pipeline.
"""
from __future__ import annotations
import re
from concurrent.futures import ThreadPoolExecutor, as_completed


# ---------------------------------------------------------------------------
# Path-pattern check (no network)
# ---------------------------------------------------------------------------

# Positive signals: at least ONE of these segments must appear in the URL path.
# Anchored with `/` on both sides where possible so we don't get a false positive
# on words like "applepay" matching "/apply".
_JOB_PATH_SIGNALS = (
    r"/jobs?/",
    r"/career/",
    r"/careers/",
    r"/position/",
    r"/positions/",
    r"/vacanc[yi]e?s?/",            # vacancy, vacancies, vacancie
    r"/apply/",
    r"/openings?/",
    r"/role/",
    r"/roles/",
    r"/job[-_]posting/",
    r"/job[-_]post/",
    r"/posting/",
    r"/postings/",
    r"/employment/",
    r"/hire/",
    r"/work-with-us/",
    r"/join-us/",
    r"/we[-_]are[-_]hiring/",
)
_JOB_PATH_SIGNAL_RE = re.compile("|".join(_JOB_PATH_SIGNALS), re.IGNORECASE)

# Negative signals: if the URL path contains ANY of these, it's almost
# certainly NOT a job posting (blog / news / events / press / case studies).
_NON_JOB_PATH_SIGNALS = (
    r"/blog/",
    r"/blogs/",
    r"/news/",
    r"/press/",
    r"/press-release/",
    r"/press[-_]releases/",
    r"/article/",
    r"/articles/",
    r"/posts?/",                    # but not /job-posting/ — checked AFTER positive signals
    r"/whitepapers?/",
    r"/case-stud[yi]e?s?/",
    r"/insight/",
    r"/insights/",
    r"/event/",
    r"/events/",
    r"/research/",
    r"/about/",
    r"/contact/",
    r"/team/",
    r"/leadership/",
    r"/investor/",
    r"/investors/",
    r"/legal/",
    r"/privacy/",
    r"/terms/",
    r"/cookie",
    r"/podcast/",
    r"/webinar/",
    r"/webinars/",
    r"/market[-_]update/",
    r"/market[-_]updates/",
    r"/customer[-_]stor[yi]e?s?/",
    r"/help/",
    r"/support/",
    r"/faq/",
    r"/login",
    r"/sign[-_]in",
    r"/sign[-_]up",
    # Year segments — stale archive paths like /2017/03/some-post
    r"/(?:19|20)\d{2}/",
)
_NON_JOB_PATH_SIGNAL_RE = re.compile("|".join(_NON_JOB_PATH_SIGNALS), re.IGNORECASE)


def is_job_url_like(url):
    """Return True if `url` looks like an actual job-posting page.

    Logic: must contain at least one positive job-path signal AND zero negative
    signals. Returns False on empty / non-string input.

    Pure function — no network. Used to filter DDG website search results
    before they reach the AI evaluator.
    """
    if not url or not isinstance(url, str):
        return False
    # `/post/` would match both /blog-style/posts and /job_posting/ — we look
    # for negative signals BEFORE positives so /job_posting/ wins via the
    # positive-signal recheck.
    if _NON_JOB_PATH_SIGNAL_RE.search(url):
        # One escape hatch: /posts/ in a LinkedIn URL (e.g. linkedin.com/posts/
        # company-name_text...) is allowed because the local pipeline expects
        # LinkedIn posts as a separate signal source.
        if "linkedin.com/posts/" in url.lower():
            return True
        return False
    return bool(_JOB_PATH_SIGNAL_RE.search(url))


# ---------------------------------------------------------------------------
# HEAD probe (network, optional)
# ---------------------------------------------------------------------------

# Patterns observed in "this listing has been filled / removed / closed" URLs
# that some ATS platforms redirect to instead of returning a clean 404.
_DEAD_LISTING_URL_PATTERNS = re.compile(
    r"(?:not[-_]found|no[-_]longer[-_]available|position[-_]closed|"
    r"job[-_]closed|listing[-_]closed|expired|removed|"
    r"page[-_]not[-_]found|/404|/410|error)",
    re.IGNORECASE,
)

# Status codes that unambiguously mean "this URL is dead".
_DEAD_LISTING_STATUS_CODES = frozenset({404, 410, 451})

HEAD_PROBE_TIMEOUT = 5         # seconds per request; tight to keep batches fast


def probe_url_alive(url, timeout=HEAD_PROBE_TIMEOUT):
    """Return False if the URL is provably dead (404/410/451 or redirects to a
    well-known "not found" sink). Returns True otherwise — including on any
    network exception, because we DON'T want a transient timeout to silently
    drop a real job.

    Pure-policy choice: false positives (dead listings slipping through) are
    cheaper than false negatives (real jobs dropped due to flaky DDG-side
    DNS or our own connection limit being hit).
    """
    if not url or not isinstance(url, str):
        return False
    try:
        import requests
        r = requests.head(
            url,
            timeout=timeout,
            allow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; JobAlertsBot/1.0)"},
        )
        if r.status_code in _DEAD_LISTING_STATUS_CODES:
            return False
        # Some servers reject HEAD and return 405 / 501 — fall back to a tiny
        # GET so we don't false-positive-drop real jobs on those hosts.
        if r.status_code in (405, 501):
            r = requests.get(
                url, timeout=timeout, allow_redirects=True, stream=True,
                headers={"User-Agent": "Mozilla/5.0 (compatible; JobAlertsBot/1.0)"},
            )
            r.close()
            if r.status_code in _DEAD_LISTING_STATUS_CODES:
                return False
        final_url = r.url or url
        if _DEAD_LISTING_URL_PATTERNS.search(final_url):
            return False
    except Exception:
        # Network errors, timeouts, DNS failures — treat as "alive" to avoid
        # dropping real jobs on transient infrastructure issues.
        return True
    return True


def probe_urls_alive_batch(urls, max_workers=10, timeout=HEAD_PROBE_TIMEOUT):
    """Concurrent HEAD probes over a list of URLs.

    Returns a dict {url: bool} (True = alive). Order-preserving callers can
    map their input list against the result. Uses a thread pool so 30 URLs
    take ~1.5s instead of 30s sequential.
    """
    if not urls:
        return {}
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(probe_url_alive, u, timeout): u for u in urls}
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()
    return results
