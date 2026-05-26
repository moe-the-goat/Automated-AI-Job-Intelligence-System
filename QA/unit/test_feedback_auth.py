"""LogsRepoAuthError tests.

Locks in the new behavior added 2026-05-26 after a silent 403 from the
GitHub Contents API made every feedback step a no-op for days. The
contract now:

  * 404 from the Contents API -> (None, None), normal "file missing" path.
  * 401 / 403 -> raise LogsRepoAuthError so the caller can log CRITICAL
    instead of mistaking the auth failure for an empty inbox.
  * verify_logs_repo_access -> returns False (and logs CRITICAL) on auth
    rejection so the entry-point pipeline gets one loud, actionable signal.
"""

import logging

import pipeline.core_feedback as cf
from pipeline.core_feedback import (
    LogsRepoAuthError,
    _read_file,
    _write_file,
    verify_logs_repo_access,
    ingest_pending_feedback,
    load_candidate_preferences,
    load_feedback_embeddings,
    count_feedback_entries,
)


class _FakeResponse:
    """Stand-in for requests.Response with only the bits the code touches."""

    def __init__(self, status_code, json_body=None):
        self.status_code = status_code
        self._json_body = json_body or {}

    def json(self):
        return self._json_body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


def _patch_requests(monkey_get=None, monkey_put=None):
    """Swap requests.get / requests.put on the module under test."""
    orig_get = cf.requests.get
    orig_put = cf.requests.put
    if monkey_get is not None:
        cf.requests.get = monkey_get
    if monkey_put is not None:
        cf.requests.put = monkey_put

    def restore():
        cf.requests.get = orig_get
        cf.requests.put = orig_put

    return restore


class _Raises:
    """Stdlib stand-in for pytest.raises — the QA runner is pure stdlib, no pytest."""

    def __init__(self, expected):
        self.expected = expected
        self.value = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            raise AssertionError(f"expected {self.expected.__name__} but no exception was raised")
        if not issubclass(exc_type, self.expected):
            return False
        self.value = exc
        return True


class _LogCapture:
    """Stdlib-only stand-in for pytest's caplog fixture.

    Records every LogRecord emitted by pipeline.core_feedback at >=DEBUG so the
    tests can assert on level and message. Use as a context manager.
    """

    def __init__(self, logger_name="pipeline.core_feedback"):
        self.logger_name = logger_name
        self.records = []
        self._handler = None
        self._orig_level = None

    def __enter__(self):
        lg = logging.getLogger(self.logger_name)
        self._orig_level = lg.level
        lg.setLevel(logging.DEBUG)

        class _H(logging.Handler):
            def __init__(self, sink):
                super().__init__(level=logging.DEBUG)
                self.sink = sink

            def emit(self, record):
                self.sink.append(record)

        self._handler = _H(self.records)
        lg.addHandler(self._handler)
        return self

    def __exit__(self, *exc):
        lg = logging.getLogger(self.logger_name)
        lg.removeHandler(self._handler)
        lg.setLevel(self._orig_level)
        return False

    def messages(self):
        return [r.getMessage() for r in self.records]

    def has_level(self, level_name):
        return any(r.levelname == level_name for r in self.records)


# ---------------------------------------------------------------------------
# _read_file — 404 vs 403 vs 401
# ---------------------------------------------------------------------------

def test_read_file_returns_none_on_404():
    restore = _patch_requests(monkey_get=lambda *a, **k: _FakeResponse(404))
    try:
        text, sha = _read_file("owner/repo", "data/missing.json", "tok")
        assert text is None and sha is None
    finally:
        restore()


def test_read_file_raises_on_403():
    restore = _patch_requests(monkey_get=lambda *a, **k: _FakeResponse(403))
    try:
        with _Raises(LogsRepoAuthError) as exc_info:
            _read_file("owner/repo", "data/x.json", "tok")
        # Message must mention the path + remediation so the log line is actionable.
        msg = str(exc_info.value)
        assert "data/x.json" in msg
        assert "owner/repo" in msg
        assert "403" in msg
    finally:
        restore()


def test_read_file_raises_on_401():
    restore = _patch_requests(monkey_get=lambda *a, **k: _FakeResponse(401))
    try:
        with _Raises(LogsRepoAuthError):
            _read_file("owner/repo", "data/x.json", "tok")
    finally:
        restore()


def test_read_file_swallows_500_to_preserve_pipeline_continuity():
    """Transient 5xx from GitHub should NOT crash the pipeline — only auth issues
    are loud. We log a warning and return (None, None) so the run continues."""
    restore = _patch_requests(monkey_get=lambda *a, **k: _FakeResponse(503))
    try:
        text, sha = _read_file("owner/repo", "data/x.json", "tok")
        assert text is None and sha is None
    finally:
        restore()


def test_read_file_short_circuits_with_no_repo_or_token():
    text, sha = _read_file(None, "data/x.json", "tok")
    assert (text, sha) == (None, None)
    text, sha = _read_file("owner/repo", "data/x.json", None)
    assert (text, sha) == (None, None)


# ---------------------------------------------------------------------------
# _write_file — 401 / 403 raise; transient errors return False
# ---------------------------------------------------------------------------

def test_write_file_raises_on_403():
    restore = _patch_requests(monkey_put=lambda *a, **k: _FakeResponse(403))
    try:
        with _Raises(LogsRepoAuthError):
            _write_file("owner/repo", "data/x.json", "content", "sha", "tok", "msg")
    finally:
        restore()


def test_write_file_returns_false_on_5xx():
    restore = _patch_requests(monkey_put=lambda *a, **k: _FakeResponse(500))
    try:
        ok = _write_file("owner/repo", "data/x.json", "content", "sha", "tok", "msg")
        assert ok is False
    finally:
        restore()


# ---------------------------------------------------------------------------
# verify_logs_repo_access — one-shot health check
# ---------------------------------------------------------------------------

def test_verify_returns_true_on_200():
    restore = _patch_requests(monkey_get=lambda *a, **k: _FakeResponse(200))
    try:
        assert verify_logs_repo_access("owner/repo", "tok") is True
    finally:
        restore()


def test_verify_returns_false_on_403_and_logs_critical():
    restore = _patch_requests(monkey_get=lambda *a, **k: _FakeResponse(403))
    try:
        with _LogCapture() as cap:
            ok = verify_logs_repo_access("owner/repo", "tok")
        assert ok is False
        # The remediation must be in the CRITICAL log line.
        joined = " ".join(cap.messages())
        assert "LOGS_REPO_TOKEN" in joined
        assert "owner/repo" in joined
        assert cap.has_level("CRITICAL")
    finally:
        restore()


def test_verify_returns_false_when_missing_credentials():
    assert verify_logs_repo_access(None, "tok") is False
    assert verify_logs_repo_access("owner/repo", None) is False


def test_verify_returns_false_on_unexpected_status():
    restore = _patch_requests(monkey_get=lambda *a, **k: _FakeResponse(500))
    try:
        assert verify_logs_repo_access("owner/repo", "tok") is False
    finally:
        restore()


# ---------------------------------------------------------------------------
# Top-level functions degrade gracefully on auth failure (don't crash pipeline)
# ---------------------------------------------------------------------------

def test_ingest_pending_feedback_returns_zero_on_403():
    restore = _patch_requests(monkey_get=lambda *a, **k: _FakeResponse(403))
    try:
        with _LogCapture() as cap:
            n = ingest_pending_feedback("owner/repo", "tok")
        assert n == 0
        # Must log CRITICAL with the word ABORTED — not just INFO/WARNING.
        # That silent INFO downgrade was the original bug.
        assert any(
            "ABORTED" in r.getMessage() and r.levelname == "CRITICAL"
            for r in cap.records
        )
    finally:
        restore()


def test_load_candidate_preferences_returns_empty_on_403():
    restore = _patch_requests(monkey_get=lambda *a, **k: _FakeResponse(403))
    try:
        with _LogCapture() as cap:
            out = load_candidate_preferences("owner/repo", "tok")
        assert out == ""
        assert cap.has_level("CRITICAL")
    finally:
        restore()


def test_count_feedback_entries_returns_zero_on_403():
    restore = _patch_requests(monkey_get=lambda *a, **k: _FakeResponse(403))
    try:
        with _LogCapture():
            n = count_feedback_entries("owner/repo", "tok")
        assert n == 0
    finally:
        restore()


def test_load_feedback_embeddings_returns_empty_on_403():
    restore = _patch_requests(monkey_get=lambda *a, **k: _FakeResponse(403))
    try:
        with _LogCapture():
            out = load_feedback_embeddings("owner/repo", "tok")
        assert out == {"entries": []}
    finally:
        restore()
