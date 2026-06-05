"""core_websearch tests (Option A — Google Programmable Search + DDG fallback).

Locks: provider selection (Google when configured else DDG), the normalized
result shape, graceful failure to [], and the Google->DDG second-chance path.

We avoid pytest's `monkeypatch` fixture because the project's QA runner calls
test functions with no args — so we manually set env / swap attributes and
restore them in a finally block.
"""
import os

import pipeline.core_websearch as ws


class _Resp:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text or "{}"

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


def _enable_google():
    os.environ["GOOGLE_SEARCH_API_KEY"] = "k"
    os.environ["GOOGLE_SEARCH_CX"] = "cx"


def _disable_google():
    os.environ.pop("GOOGLE_SEARCH_API_KEY", None)
    os.environ.pop("GOOGLE_SEARCH_CX", None)


def _swap(get=None, ddg=None):
    """Swap requests.get and/or _ddg_search; return a restore() callable."""
    orig_get, orig_ddg = ws.requests.get, ws._ddg_search
    if get is not None:
        ws.requests.get = get
    if ddg is not None:
        ws._ddg_search = ddg

    def restore():
        ws.requests.get = orig_get
        ws._ddg_search = orig_ddg
        _disable_google()

    return restore


# ---------------------------------------------------------------------------
# google_search_configured
# ---------------------------------------------------------------------------

def test_configured_true_only_when_both_present():
    restore = _swap()
    try:
        _enable_google()
        assert ws.google_search_configured() is True
        os.environ.pop("GOOGLE_SEARCH_CX", None)
        assert ws.google_search_configured() is False
        _disable_google()
        assert ws.google_search_configured() is False
    finally:
        restore()


# ---------------------------------------------------------------------------
# Google path → normalized shape
# ---------------------------------------------------------------------------

def test_google_results_normalized():
    payload = {"items": [
        {"title": "We're hiring", "snippet": "Backend dev wanted", "link": "https://li/1"},
        {"title": "Role 2", "snippet": "snip2", "link": "https://li/2"},
    ]}
    restore = _swap(get=lambda *a, **k: _Resp(200, payload))
    try:
        _enable_google()
        out = ws.web_search("q", max_results=3)
        assert out == [
            {"title": "We're hiring", "body": "Backend dev wanted", "href": "https://li/1"},
            {"title": "Role 2", "body": "snip2", "href": "https://li/2"},
        ]
    finally:
        restore()


def test_google_quota_429_falls_back_to_ddg():
    restore = _swap(
        get=lambda *a, **k: _Resp(429, text="quota"),
        ddg=lambda q, n, t="w": [{"title": "ddg", "body": "", "href": "x"}],
    )
    try:
        _enable_google()
        assert ws.web_search("q") == [{"title": "ddg", "body": "", "href": "x"}]
    finally:
        restore()


def test_google_empty_results_second_chance_ddg():
    restore = _swap(
        get=lambda *a, **k: _Resp(200, {"items": []}),
        ddg=lambda q, n, t="w": [{"title": "via-ddg", "body": "", "href": "y"}],
    )
    try:
        _enable_google()
        assert ws.web_search("q") == [{"title": "via-ddg", "body": "", "href": "y"}]
    finally:
        restore()


# ---------------------------------------------------------------------------
# No Google config → DDG path
# ---------------------------------------------------------------------------

def test_no_google_uses_ddg():
    seen = {}

    def fake_ddg(q, n, t="w"):
        seen["q"] = q
        return [{"title": "d", "body": "b", "href": "h"}]

    restore = _swap(ddg=fake_ddg)
    try:
        _disable_google()
        out = ws.web_search("hello", max_results=2)
        assert out == [{"title": "d", "body": "b", "href": "h"}]
        assert seen["q"] == "hello"
    finally:
        restore()


# ---------------------------------------------------------------------------
# Graceful failure — never raises
# ---------------------------------------------------------------------------

def test_google_transport_error_returns_ddg_empty():
    def boom(*a, **k):
        raise ws.requests.RequestException("conn refused")

    restore = _swap(get=boom, ddg=lambda q, n, t="w": [])
    try:
        _enable_google()
        assert ws.web_search("q") == []   # both empty -> [], no raise
    finally:
        restore()


def test_google_non_json_returns_ddg():
    restore = _swap(
        get=lambda *a, **k: _Resp(200, payload=None),
        ddg=lambda q, n, t="w": [{"title": "fallback", "body": "", "href": "z"}],
    )
    try:
        _enable_google()
        assert ws.web_search("q") == [{"title": "fallback", "body": "", "href": "z"}]
    finally:
        restore()
