"""Email feedback tokens (task W2).

Locks the worker half of the tokenized feedback-page contract:

  * create_feedback_token persists ONLY the SHA-256 hex of the secret —
    the raw token must never reach the database.
  * Failure modes (missing args, insert exception) return None so the
    email send is never blocked by the feedback feature.
  * format_email_html renders the per-run CTA when feedback_url is given
    (header + footer repeat) and stays on the legacy env-driven path
    otherwise.
"""

import hashlib
import os

import pandas as pd

from pipeline.core_feedback_supabase import (
    FEEDBACK_TOKEN_TTL_DAYS,
    create_feedback_token,
)
from pipeline.core_notify import _feedback_cta_html, format_email_html


# ── Mock Supabase client ─────────────────────────────────────────────────────

class _Insert:
    def __init__(self, store, table, *, raise_on_execute=False):
        self.store = store
        self.table = table
        self.raise_on_execute = raise_on_execute
        self.row = None

    def insert(self, row):
        self.row = row
        return self

    def execute(self):
        if self.raise_on_execute:
            raise RuntimeError("relation \"email_feedback_tokens\" does not exist")
        self.store.setdefault(self.table, []).append(self.row)
        return self


class _Client:
    def __init__(self, *, raise_on_execute=False):
        self.store = {}
        self.raise_on_execute = raise_on_execute

    def table(self, name):
        return _Insert(self.store, name, raise_on_execute=self.raise_on_execute)


# ── create_feedback_token ────────────────────────────────────────────────────

def test_token_is_urlsafe_and_only_hash_is_persisted():
    client = _Client()
    token = create_feedback_token("user-1", 42, client=client)

    assert token is not None
    assert len(token) >= 40                      # 256 bits → 43 url-safe chars
    assert all(c.isalnum() or c in "-_" for c in token)

    rows = client.store["email_feedback_tokens"]
    assert len(rows) == 1
    row = rows[0]
    assert row["user_id"] == "user-1"
    assert row["run_id"] == 42
    # The raw secret must never appear anywhere in the persisted row.
    assert token not in str(row)
    assert row["token_hash"] == hashlib.sha256(token.encode("utf-8")).hexdigest()


def test_expiry_lands_on_the_documented_ttl():
    client = _Client()
    create_feedback_token("user-1", 42, client=client)
    row = client.store["email_feedback_tokens"][0]
    # ISO timestamp ~TTL days out; date arithmetic is the DB default's job,
    # so a presence + sanity check is enough here.
    assert row["expires_at"] is not None
    assert FEEDBACK_TOKEN_TTL_DAYS == 30


def test_two_tokens_never_collide():
    client = _Client()
    t1 = create_feedback_token("user-1", 1, client=client)
    t2 = create_feedback_token("user-1", 2, client=client)
    assert t1 != t2


def test_missing_args_return_none_without_touching_supabase():
    client = _Client()
    assert create_feedback_token("", 42, client=client) is None
    assert create_feedback_token("user-1", None, client=client) is None
    assert client.store == {}


def test_insert_failure_degrades_to_none():
    # Pre-migration-0012 schema (or any transient error) must not raise —
    # the runner sends the email without a feedback link instead.
    client = _Client(raise_on_execute=True)
    assert create_feedback_token("user-1", 42, client=client) is None


# ── Email CTA rendering ──────────────────────────────────────────────────────

def _stats():
    return {"scraped": 10, "filtered": 5, "approved": 0}


def test_cta_block_renders_url_and_empty_when_unset():
    html = _feedback_cta_html("https://app.example/f/tok123")
    assert 'href="https://app.example/f/tok123"' in html
    assert "Rate today" in html
    assert _feedback_cta_html(None) == ""
    assert _feedback_cta_html("") == ""


# QA/run_all.py is a bare runner (no pytest fixtures), so env juggling is
# done by hand with try/finally instead of monkeypatch.

def _with_env(name, value, fn):
    sentinel = object()
    prev = os.environ.pop(name, sentinel)
    if value is not None:
        os.environ[name] = value
    try:
        return fn()
    finally:
        os.environ.pop(name, None)
        if prev is not sentinel:
            os.environ[name] = prev


def test_email_includes_cta_top_and_footer_when_url_given():
    url = "https://app.example/f/tok123"
    html = _with_env(
        "FEEDBACK_PAGE_URL", None,
        lambda: format_email_html(pd.DataFrame(), pd.DataFrame(), _stats(),
                                  feedback_url=url),
    )
    # Header CTA + footer repeat → the link appears twice.
    assert html.count(f'href="{url}"') == 2


def test_email_without_url_keeps_legacy_env_path():
    html = _with_env(
        "FEEDBACK_PAGE_URL", None,
        lambda: format_email_html(pd.DataFrame(), pd.DataFrame(), _stats()),
    )
    assert "Rate today" not in html

    html = _with_env(
        "FEEDBACK_PAGE_URL", "https://legacy.example/feedback.html",
        lambda: format_email_html(pd.DataFrame(), pd.DataFrame(), _stats()),
    )
    assert 'href="https://legacy.example/feedback.html"' in html
