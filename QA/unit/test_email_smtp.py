"""Gmail-SMTP email transport tests.

Locks the contract that core_email_smtp.send_email():
  * never raises — returns (False, "...") on any failure so the multi-user
    cron loop keeps moving;
  * refuses to send when SENDER_EMAIL / EMAIL_APP_PASSWORD are missing;
  * refuses to send to obviously-invalid recipients;
  * logs in and sends to the exact recipient on the happy path;
  * surfaces auth vs transport failures distinctly;
  * keeps the Resend module's exact (to, subject, html, ...) signature so it's
    a drop-in transport swap;
  * redacts the recipient local-part in log output.

No pytest: functions take no args and are called directly by QA/run_all.py.
SMTP_SSL is swapped for a fake context-manager so nothing hits the network.
"""

import os
import smtplib

import pipeline.core_email_smtp as smtp
from pipeline.core_email_smtp import send_email, _redact


class _FakeSMTP:
    """Stand-in for smtplib.SMTP_SSL usable as a context manager.

    Records login + send_message so tests can assert on them. Optionally
    raises on login/send to exercise the error branches.
    """

    last_instance = None

    def __init__(self, host, port, timeout=None):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.logged_in = None
        self.sent = []
        self.raise_on_login = None
        self.raise_on_send = None
        _FakeSMTP.last_instance = self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def login(self, user, password):
        if self.raise_on_login:
            raise self.raise_on_login
        self.logged_in = (user, password)

    def send_message(self, msg, from_addr=None, to_addrs=None):
        if self.raise_on_send:
            raise self.raise_on_send
        self.sent.append((msg, from_addr, to_addrs))


def _patch_smtp(factory):
    """Swap smtplib.SMTP_SSL on the module under test. Returns a restore fn."""
    orig = smtp.smtplib.SMTP_SSL

    def restore():
        smtp.smtplib.SMTP_SSL = orig

    smtp.smtplib.SMTP_SSL = factory
    return restore


def _set_creds():
    """Set sender creds, returning a restore callable to pop them back."""
    prev = (os.environ.get("SENDER_EMAIL"), os.environ.get("EMAIL_APP_PASSWORD"))
    os.environ["SENDER_EMAIL"] = "sender@gmail.com"
    os.environ["EMAIL_APP_PASSWORD"] = "app-pass-xyz"

    def restore():
        for k, v in zip(("SENDER_EMAIL", "EMAIL_APP_PASSWORD"), prev):
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    return restore


# ---------------------------------------------------------------------------
# Sad paths
# ---------------------------------------------------------------------------

def test_missing_credentials_returns_false_and_does_not_connect():
    connected = []

    def factory(*a, **k):
        connected.append((a, k))
        return _FakeSMTP(*a, **k)

    # Ensure creds are absent.
    prev = (os.environ.pop("SENDER_EMAIL", None),
            os.environ.pop("EMAIL_APP_PASSWORD", None))
    restore_smtp = _patch_smtp(factory)
    try:
        ok, info = send_email("user@example.com", "subj", "<p>body</p>")
    finally:
        restore_smtp()
        if prev[0] is not None:
            os.environ["SENDER_EMAIL"] = prev[0]
        if prev[1] is not None:
            os.environ["EMAIL_APP_PASSWORD"] = prev[1]

    assert ok is False
    assert "SENDER_EMAIL" in info or "EMAIL_APP_PASSWORD" in info
    assert connected == [], "must not open an SMTP connection without creds"


def test_invalid_recipient_returns_false():
    restore_creds = _set_creds()
    restore_smtp = _patch_smtp(lambda *a, **k: _FakeSMTP(*a, **k))
    try:
        ok, info = send_email("not-an-email", "s", "<p>h</p>")
        ok2, _ = send_email("", "s", "<p>h</p>")
    finally:
        restore_smtp()
        restore_creds()
    assert ok is False and "invalid recipient" in info
    assert ok2 is False


def test_auth_error_is_caught_and_labeled():
    restore_creds = _set_creds()

    def factory(*a, **k):
        inst = _FakeSMTP(*a, **k)
        inst.raise_on_login = smtplib.SMTPAuthenticationError(535, b"bad creds")
        return inst

    restore_smtp = _patch_smtp(factory)
    try:
        ok, info = send_email("a@b.com", "s", "<p>h</p>")
    finally:
        restore_smtp()
        restore_creds()
    assert ok is False
    assert "auth error" in info


def test_transport_error_is_caught_and_labeled():
    restore_creds = _set_creds()

    def factory(*a, **k):
        inst = _FakeSMTP(*a, **k)
        inst.raise_on_send = smtplib.SMTPException("connection dropped")
        return inst

    restore_smtp = _patch_smtp(factory)
    try:
        ok, info = send_email("a@b.com", "s", "<p>h</p>")
    finally:
        restore_smtp()
        restore_creds()
    assert ok is False
    assert "transport error" in info


def test_os_error_on_connect_is_caught():
    restore_creds = _set_creds()

    def factory(*a, **k):
        raise OSError("name resolution failed")

    restore_smtp = _patch_smtp(factory)
    try:
        ok, info = send_email("a@b.com", "s", "<p>h</p>")
    finally:
        restore_smtp()
        restore_creds()
    assert ok is False
    assert "transport error" in info


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------

def test_successful_send_logs_in_and_targets_the_recipient():
    restore_creds = _set_creds()
    restore_smtp = _patch_smtp(lambda *a, **k: _FakeSMTP(*a, **k))
    try:
        ok, info = send_email(
            "user@example.com", "subj", "<p>html</p>", text="plain",
        )
    finally:
        restore_smtp()
        restore_creds()

    inst = _FakeSMTP.last_instance
    assert ok is True
    assert info == ""
    # Authenticated with the configured sender creds.
    assert inst.logged_in == ("sender@gmail.com", "app-pass-xyz")
    # Exactly one message, addressed to exactly the requested recipient.
    assert len(inst.sent) == 1
    msg, from_addr, to_addrs = inst.sent[0]
    assert to_addrs == ["user@example.com"]
    assert from_addr == "sender@gmail.com"
    assert msg["To"] == "user@example.com"
    assert msg["Subject"] == "subj"
    # Multipart carries both the plain and HTML alternatives.
    payloads = [p.get_payload() for p in msg.get_payload()]
    assert "plain" in payloads
    assert "<p>html</p>" in payloads


def test_sends_to_any_recipient_not_just_the_owner():
    # The whole point of the swap: deliver to an arbitrary address.
    restore_creds = _set_creds()
    restore_smtp = _patch_smtp(lambda *a, **k: _FakeSMTP(*a, **k))
    try:
        ok, _ = send_email("someone.else@outlook.com", "s", "<p>h</p>")
    finally:
        restore_smtp()
        restore_creds()
    inst = _FakeSMTP.last_instance
    assert ok is True
    assert inst.sent[0][2] == ["someone.else@outlook.com"]


def test_from_address_override_sets_header_but_envelope_stays_sender():
    restore_creds = _set_creds()
    restore_smtp = _patch_smtp(lambda *a, **k: _FakeSMTP(*a, **k))
    try:
        send_email(
            "a@b.com", "s", "<p>h</p>",
            from_address="Alerts <hello@example.com>",
        )
    finally:
        restore_smtp()
        restore_creds()
    inst = _FakeSMTP.last_instance
    msg, from_addr, _ = inst.sent[0]
    assert msg["From"] == "Alerts <hello@example.com>"
    # Envelope sender stays the authenticated account (Gmail requires this).
    assert from_addr == "sender@gmail.com"


def test_api_key_kwarg_accepted_for_signature_parity():
    # multi_user_runner may pass api_key when swapping transports; it must be
    # silently accepted (and ignored) rather than raising a TypeError.
    restore_creds = _set_creds()
    restore_smtp = _patch_smtp(lambda *a, **k: _FakeSMTP(*a, **k))
    try:
        ok, _ = send_email("a@b.com", "s", "<p>h</p>", api_key="ignored")
    finally:
        restore_smtp()
        restore_creds()
    assert ok is True


def test_custom_smtp_server_and_port_are_honored():
    restore_creds = _set_creds()
    prev = (os.environ.get("SMTP_SERVER"), os.environ.get("SMTP_PORT"))
    os.environ["SMTP_SERVER"] = "smtp.example.org"
    os.environ["SMTP_PORT"] = "2465"
    restore_smtp = _patch_smtp(lambda *a, **k: _FakeSMTP(*a, **k))
    try:
        send_email("a@b.com", "s", "<p>h</p>")
    finally:
        restore_smtp()
        for k, v in zip(("SMTP_SERVER", "SMTP_PORT"), prev):
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        restore_creds()
    inst = _FakeSMTP.last_instance
    assert inst.host == "smtp.example.org"
    assert inst.port == 2465


# ---------------------------------------------------------------------------
# Redaction helper
# ---------------------------------------------------------------------------

def test_redact_preserves_domain_and_hides_local():
    assert _redact("mohaabuhijleh@gmail.com") == "m***h@gmail.com"
    assert _redact("ab@x.com") == "a***@x.com"
    assert _redact("") == "<invalid>"
    assert _redact("noatsign") == "<invalid>"


# ---------------------------------------------------------------------------
# Deliverability headers + MIME shape (anti-spam)
# ---------------------------------------------------------------------------
# These emails were landing in Gmail's spam folder. The fixes below are the
# free, code-side half of that: a well-formed multipart message with the
# headers bulk filters look for. Locked here so they can't silently regress.

def _send_and_capture(**kwargs):
    """Run one happy-path send and hand back the MIME message that went out."""
    restore_creds = _set_creds()
    restore_smtp = _patch_smtp(lambda *a, **k: _FakeSMTP(*a, **k))
    try:
        send_email("user@example.com", "subj", "<p>html</p>", **kwargs)
    finally:
        restore_smtp()
        restore_creds()
    return _FakeSMTP.last_instance.sent[0][0]


def test_never_sends_html_only_even_without_text_arg():
    """HTML-only mail trips MIME_HTML_ONLY in mainstream filters. A caller that
    forgets text= must still produce a text/plain alternative."""
    msg = _send_and_capture()
    subtypes = [p.get_content_subtype() for p in msg.get_payload()]
    assert "plain" in subtypes and "html" in subtypes


def test_from_carries_a_display_name_and_reply_to():
    msg = _send_and_capture(text="plain")
    assert "Job Alerts" in msg["From"]
    assert "sender@gmail.com" in msg["From"]
    assert msg["Reply-To"] == "sender@gmail.com"


def test_sets_required_rfc5322_headers():
    msg = _send_and_capture(text="plain")
    assert msg["Date"]
    assert msg["Message-ID"] and msg["Message-ID"].startswith("<")
    assert msg["Auto-Submitted"] == "auto-generated"


def test_advertises_a_working_unsubscribe_path():
    """A usable unsubscribe is one of the strongest legitimacy signals for
    recurring mail. mailto: always works and needs no endpoint."""
    msg = _send_and_capture(text="plain")
    assert "mailto:sender@gmail.com" in msg["List-Unsubscribe"]
    # We must NOT claim one-click POST support without an endpoint for it.
    assert msg["List-Unsubscribe-Post"] is None


def test_html_to_text_fallback_keeps_words_and_links():
    from pipeline.core_email_smtp import _html_to_text
    out = _html_to_text('<h2>Hi</h2><p>See <a href="https://x.co/j">this job</a></p>')
    assert "Hi" in out
    assert "this job" in out
    assert "https://x.co/j" in out
    assert "<" not in out and ">" not in out
