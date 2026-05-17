"""End-to-end ATS flow: detection from HTML + fetcher dispatch + cache update.

Stubs `requests` at the module level so no actual HTTP fires. Exercises the
full `get_jobs_for_company` path including the cache write.
"""
import json
import os
import sys
import types
import pipeline.core_ats as core_ats


# ---------------------------------------------------------------------------
# Fake `requests` module — drop in via sys.modules['requests'] before the test
# calls into core_ats. Restored in a finally to avoid leaking state.
# ---------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, status_code=200, text="", json_payload=None):
        self.status_code = status_code
        self.text = text
        self._json = json_payload

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def _patch_requests(get_handler):
    """Replace sys.modules['requests'].get with a callable that takes (url) -> _FakeResponse."""
    fake_module = types.ModuleType("requests")
    fake_module.get = get_handler
    sys.modules["requests"] = fake_module


def _with_temp_cache(fn):
    """Swap the ATS_CACHE_FILE so tests don't trample on each other or real state."""
    path = "_qa_integration_ats_cache.json"
    original = core_ats.ATS_CACHE_FILE
    core_ats.ATS_CACHE_FILE = path
    try:
        if os.path.exists(path):
            os.remove(path)
        fn(path)
    finally:
        core_ats.ATS_CACHE_FILE = original
        if os.path.exists(path):
            os.remove(path)
        sys.modules.pop("requests", None)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_full_flow_greenhouse_detection_and_fetch():
    """Empty cache -> /careers fetched -> Greenhouse detected -> API hit -> jobs returned."""
    def body(cache_path):
        careers_html = '<iframe src="https://boards.greenhouse.io/asaltech-demo"></iframe>'
        ats_jobs_payload = {
            "jobs": [
                {
                    "id": 1, "title": "AI Engineer",
                    "location": {"name": "Remote"},
                    "absolute_url": "https://boards.greenhouse.io/asaltech-demo/jobs/1",
                    "updated_at": "2026-05-15T10:00:00Z",
                },
                {
                    "id": 2, "title": "Backend Intern",
                    "location": {"name": "Ramallah"},
                    "absolute_url": "https://boards.greenhouse.io/asaltech-demo/jobs/2",
                    "updated_at": "2026-05-16T10:00:00Z",
                },
            ]
        }

        def fake_get(url, **kwargs):
            if "boards-api.greenhouse.io" in url:
                return _FakeResponse(json_payload=ats_jobs_payload)
            # Otherwise it's the careers-page fetch
            return _FakeResponse(text=careers_html)

        _patch_requests(fake_get)
        cache = core_ats.AtsCache(filepath=cache_path)
        jobs = core_ats.get_jobs_for_company(
            "ASAL Demo", "https://asaltech.example/careers", cache=cache,
        )
        assert len(jobs) == 2
        assert jobs[0]["title"] == "AI Engineer"
        assert jobs[0]["company"] == "ASAL Demo"
        assert "greenhouse" in jobs[0]["job_url"]
        # Cache should now have an entry for this company.
        assert cache.get("ASAL Demo")["ats"] == "greenhouse"
        assert cache.get("ASAL Demo")["token"] == "asaltech-demo"
    _with_temp_cache(body)


def test_full_flow_uses_cached_token_no_redetection():
    """Pre-warmed cache should skip the /careers fetch entirely."""
    def body(cache_path):
        # Pre-warm
        cache = core_ats.AtsCache(filepath=cache_path)
        cache.set("CachedCo", "lever", "cachedco-token")

        ats_jobs_payload = [{
            "id": "u1", "text": "Software Engineer",
            "categories": {"location": "Remote"},
            "hostedUrl": "https://jobs.lever.co/cachedco-token/u1",
            "createdAt": 1747400000000,
        }]

        careers_fetched = {"called": False}

        def fake_get(url, **kwargs):
            if "api.lever.co" in url:
                return _FakeResponse(json_payload=ats_jobs_payload)
            careers_fetched["called"] = True
            return _FakeResponse(text="should not be requested")

        _patch_requests(fake_get)
        jobs = core_ats.get_jobs_for_company(
            "CachedCo", "https://cachedco.example/careers", cache=cache,
        )
        assert careers_fetched["called"] is False, "Cache hit must skip /careers fetch"
        assert len(jobs) == 1
        assert jobs[0]["title"] == "Software Engineer"
    _with_temp_cache(body)


def test_full_flow_no_ats_detected_returns_empty():
    """Careers page has no ATS signature -> empty list, and cache remembers (ats=None)."""
    def body(cache_path):
        def fake_get(url, **kwargs):
            return _FakeResponse(text="<html>Email careers@plain.com</html>")
        _patch_requests(fake_get)
        cache = core_ats.AtsCache(filepath=cache_path)
        jobs = core_ats.get_jobs_for_company(
            "PlainCo", "https://plain.example/careers", cache=cache,
        )
        assert jobs == []
        # We remember the negative result so we don't refetch every run.
        entry = cache.data.get("PlainCo")
        assert entry is not None and entry["ats"] is None
    _with_temp_cache(body)


def test_full_flow_cached_negative_skips_fetch():
    """If we previously detected no ATS for a company, we should NOT hit the network again."""
    def body(cache_path):
        cache = core_ats.AtsCache(filepath=cache_path)
        cache.set("NegCo", None, None)
        fetched = {"called": False}

        def fake_get(url, **kwargs):
            fetched["called"] = True
            return _FakeResponse(text="")

        _patch_requests(fake_get)
        jobs = core_ats.get_jobs_for_company(
            "NegCo", "https://anything.example/careers", cache=cache,
        )
        assert jobs == []
        assert fetched["called"] is False, "Negative cache must short-circuit network"
    _with_temp_cache(body)


def test_full_flow_workable_detection_and_url_construction():
    def body(cache_path):
        careers_html = '<a href="https://apply.workable.com/demoworkco/">Apply</a>'
        ats_jobs_payload = {"jobs": [
            {"shortcode": "ABC123", "title": "Engineer",
             "location": {"city": "Ramallah", "country": "PS"}},
        ]}

        def fake_get(url, **kwargs):
            if "apply.workable.com/api/" in url:
                return _FakeResponse(json_payload=ats_jobs_payload)
            return _FakeResponse(text=careers_html)

        _patch_requests(fake_get)
        cache = core_ats.AtsCache(filepath=cache_path)
        jobs = core_ats.get_jobs_for_company(
            "DemoWork", "https://demowork.example/careers", cache=cache,
        )
        assert len(jobs) == 1
        # URL constructed from the cached token + the job's shortcode
        assert jobs[0]["job_url"] == "https://apply.workable.com/demoworkco/j/ABC123/"
        assert jobs[0]["location"] == "Ramallah, PS"
    _with_temp_cache(body)
