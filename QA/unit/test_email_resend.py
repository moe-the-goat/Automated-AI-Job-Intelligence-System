"""Resend email transport tests (B7a).

Locks the contract that send_email():
  * never raises — returns (False, "...") on any failure so the multi-user
    cron loop can keep moving;
  * refuses to send when RESEND_API_KEY is missing;
  * refuses to send to obviously-invalid recipients;
  * surfaces both 200 and 202 as success;
  * redacts the recipient local-part in log output.
"""

import pipeline.core_email_resend as cer
from pipeline.core_email_resend import send_email, _redact


class _FakeResponse:
    def __init__(self, status_code, json_body=None, text=""):
        self.status_code = status_code
        self._json_body = json_body or {}
        self.text = text or "{}"

    def json(self):
        return self._json_body


def _patch_post(post_fn):
    """Swap requests.post on the module under test. Returns a restore callable."""
    orig = cer.requests.post

    def restore():
        cer.requests.post = orig

    cer.requests.post = post_fn
    return restore


# ---------------------------------------------------------------------------
# Sad paths
# ---------------------------------------------------------------------------

def test_missing_api_key_returns_false_and_does_not_post():
    calls = []

    def fake_post(*a, **k):
        calls.append((a, k))
        return _FakeResponse(200)

    restore = _patch_post(fake_post)
    try:
        ok, info = send_email("user@example.com", "subject", "<p>body</p>", api_key="")
    finally:
        restore()

    assert ok is False
    assert "RESEND_API_KEY" in info
    assert calls == [], "should not have hit the network when API key is missing"


def test_invalid_recipient_returns_false():
    restore = _patch_post(lambda *a, **k: _FakeResponse(200))
    try:
        ok, info = send_email("not-an-email", "s", "<p>h</p>", api_key="rs_dummy")
        ok2, info2 = send_email("", "s", "<p>h</p>", api_key="rs_dummy")
    finally:
        restore()
    assert ok is False and "invalid recipient" in info
    assert ok2 is False


def test_http_error_response_returns_false_with_status_in_message():
    def fake_post(*a, **k):
        return _FakeResponse(401, text="invalid api key")

    restore = _patch_post(fake_post)
    try:
        ok, info = send_email("a@b.com", "s", "<p>h</p>", api_key="rs_bad")
    finally:
        restore()
    assert ok is False
    assert "HTTP 401" in info
    assert "invalid api key" in info


def test_transport_exception_is_caught():
    class Boom(Exception):
        pass

    def fake_post(*a, **k):
        import requests
        raise requests.RequestException("connection refused")

    restore = _patch_post(fake_post)
    try:
        ok, info = send_email("a@b.com", "s", "<p>h</p>", api_key="rs_ok")
    finally:
        restore()
    assert ok is False
    assert "transport error" in info


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------

def test_200_returns_ok_with_message_id():
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        return _FakeResponse(200, json_body={"id": "msg_abc"})

    restore = _patch_post(fake_post)
    try:
        ok, mid = send_email(
            "user@example.com", "subj", "<p>html</p>",
            text="plain", api_key="rs_ok",
        )
    finally:
        restore()

    assert ok is True
    assert mid == "msg_abc"
    assert captured["url"].endswith("/emails")
    assert captured["headers"]["Authorization"] == "Bearer rs_ok"
    assert captured["json"]["to"] == ["user@example.com"]
    assert captured["json"]["subject"] == "subj"
    assert captured["json"]["html"] == "<p>html</p>"
    assert captured["json"]["text"] == "plain"
    assert "from" in captured["json"]


def test_202_also_success_even_without_id():
    def fake_post(*a, **k):
        return _FakeResponse(202)

    restore = _patch_post(fake_post)
    try:
        ok, mid = send_email("a@b.com", "s", "<p>h</p>", api_key="rs_ok")
    finally:
        restore()
    assert ok is True
    assert mid == ""


def test_from_address_override_takes_precedence():
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["from"] = json["from"]
        return _FakeResponse(200, {"id": "x"})

    restore = _patch_post(fake_post)
    try:
        send_email(
            "a@b.com", "s", "<p>h</p>",
            api_key="rs_ok",
            from_address="Alerts <hello@example.com>",
        )
    finally:
        restore()
    assert captured["from"] == "Alerts <hello@example.com>"


# ---------------------------------------------------------------------------
# Redaction helper
# ---------------------------------------------------------------------------

def test_redact_preserves_domain_and_hides_local():
    assert _redact("mohaabuhijleh@gmail.com") == "m***h@gmail.com"
    assert _redact("ab@x.com") == "a***@x.com"
    assert _redact("") == "<invalid>"
    assert _redact("noatsign") == "<invalid>"
