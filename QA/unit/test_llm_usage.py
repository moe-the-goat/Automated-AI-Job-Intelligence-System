"""core_llm_usage tracker + core_llm key rotation.

Locks:
  - the usage tracker tallies requests/failures/tokens per (provider, model)
  - snapshot_and_reset returns rows and clears state for the next user
  - extract_tokens reads both OpenAI-style and Gemini-style responses
  - key rotation (_parse_keys / _take_start) spreads calls across accounts so
    the happy path alternates accounts (not always the first)
  - adaptive pacing (_reserve / _throttle) spaces calls per account and absorbs
    call duration
"""

from pipeline.core_llm_usage import _UsageTracker, extract_tokens
from pipeline.core_llm import (
    _parse_keys,
    _take_start,
    _KEY_CURSOR,
    call_llm_with_fallback,
    call_gemini_verdict,
    _reserve,
    _throttle,
    _min_interval,
    _RATE_STATE,
)


# --- usage tracker ----------------------------------------------------------

def test_tracker_counts_requests_and_failures():
    t = _UsageTracker()
    t.record("Cerebras", "gpt-oss-120b", ok=True, tokens=100)
    t.record("Cerebras", "gpt-oss-120b", ok=True, tokens=50)
    t.record("Cerebras", "gpt-oss-120b", ok=False)
    rows = t.snapshot_and_reset()
    assert len(rows) == 1
    r = rows[0]
    assert r["provider"] == "Cerebras" and r["model"] == "gpt-oss-120b"
    assert r["requests"] == 3
    assert r["requests_failed"] == 1
    assert r["tokens"] == 150


def test_tracker_separates_models():
    t = _UsageTracker()
    t.record("Groq", "llama-3.3-70b-versatile", ok=True)
    t.record("Gemini", "gemini-embedding-001", ok=True)
    rows = {r["model"]: r for r in t.snapshot_and_reset()}
    assert set(rows) == {"llama-3.3-70b-versatile", "gemini-embedding-001"}


def test_snapshot_resets_state():
    t = _UsageTracker()
    t.record("Groq", "m", ok=True)
    assert len(t.snapshot_and_reset()) == 1
    assert t.snapshot_and_reset() == []  # cleared after the first snapshot


def test_peak_rpm_tracks_window_max():
    t = _UsageTracker()
    for _ in range(5):
        t.record("Cerebras", "m", ok=True)
    rows = t.snapshot_and_reset()
    assert rows[0]["peak_rpm"] == 5  # all within the same 60s window


def test_extract_tokens_openai_shape():
    class U:
        total_tokens = 321
    class R:
        usage = U()
    assert extract_tokens(R()) == 321


def test_extract_tokens_gemini_shape():
    class M:
        total_token_count = 99
    class R:
        usage_metadata = M()
    assert extract_tokens(R()) == 99


def test_extract_tokens_absent_returns_zero():
    assert extract_tokens(object()) == 0


# --- key rotation -----------------------------------------------------------

def test_parse_keys_variants():
    assert _parse_keys("k1,k2") == ["k1", "k2"]
    assert _parse_keys("k1") == ["k1"]
    assert _parse_keys("") == []
    assert _parse_keys(None) == []
    assert _parse_keys(["a", " b "]) == ["a", "b"]


def test_take_start_advances_one_per_call():
    # The start index moves by exactly ONE per call (not per attempt) so
    # consecutive calls begin on the next account.
    _KEY_CURSOR["Cerebras"] = 0
    seq = [_take_start("Cerebras", 2) for _ in range(4)]
    assert seq == [0, 1, 0, 1]


def test_take_start_single_key_always_zero():
    _KEY_CURSOR["Groq"] = 0
    seq = [_take_start("Groq", 1) for _ in range(3)]
    assert seq == [0, 0, 0]


def test_take_start_handles_no_keys():
    assert _take_start("Cerebras", 0) == 0


class _Resp:
    """Minimal stand-in for an OpenAI-style chat response."""
    class _Choice:
        class _Msg:
            content = "ok"
        message = _Msg()
    choices = [_Choice()]
    usage = None


def test_happy_path_alternates_cerebras_accounts():
    # The regression that motivated _take_start: with two Cerebras accounts the
    # NON-retry (happy-path) call must alternate accounts, not always use the
    # first. We capture which key each successful call used. (Manual patch/restore
    # — the QA runner has no pytest fixtures.)
    import pipeline.core_llm as m
    m._KEY_CURSOR["Cerebras"] = 0
    m._KEY_CURSOR["Groq"] = 0
    used = []
    original = m._call_cerebras

    def fake_cerebras(prompt, api_key):
        used.append(api_key)
        return "verdict"

    m._call_cerebras = fake_cerebras
    try:
        for _ in range(4):
            call_llm_with_fallback("p", cerebras_key="A,B", groq_key="g1,g2", label="t")
    finally:
        m._call_cerebras = original
    assert used == ["A", "B", "A", "B"]


# --- adaptive pacing (per-account rate limiter) -----------------------------

def test_reserve_first_call_no_wait():
    # An account that's never been called waits 0 and reserves now + interval.
    wait, nxt = _reserve(now=100.0, next_allowed=0.0, interval=12.0)
    assert wait == 0.0
    assert nxt == 112.0


def test_reserve_absorbs_call_duration():
    # If 5s of wall-clock already elapsed since the slot was reserved, the next
    # call waits only the REMAINDER of the interval, not a full interval.
    wait, nxt = _reserve(now=105.0, next_allowed=112.0, interval=12.0)
    assert wait == 7.0          # 112 - 105, not a fresh 12
    assert nxt == 124.0         # 112 + 12


def test_reserve_idle_gap_resets_baseline():
    # If the account has been idle past its next-allowed time, no wait and the
    # new baseline is anchored at now (not the stale past timestamp).
    wait, nxt = _reserve(now=200.0, next_allowed=112.0, interval=12.0)
    assert wait == 0.0
    assert nxt == 212.0


def test_min_interval_scales_with_rpm():
    # Higher RPM → shorter minimum gap. Groq (30) is far quicker than Cerebras (5).
    assert _min_interval("Cerebras") > _min_interval("Gemini") > _min_interval("Groq")


def test_throttle_spaces_two_calls_on_same_account():
    # Two back-to-back calls on the same account: the first doesn't wait (so
    # sleeper isn't called), the second sleeps ~one interval. _throttle only
    # invokes the sleeper when there's a positive wait.
    _RATE_STATE.clear()
    slept = []
    fake_clock = lambda: 1000.0
    fake_sleep = lambda s: slept.append(s)
    _throttle("Cerebras", "acctA", sleeper=fake_sleep, clock=fake_clock)
    _throttle("Cerebras", "acctA", sleeper=fake_sleep, clock=fake_clock)
    assert len(slept) == 1                           # only the second waited
    assert abs(slept[0] - _min_interval("Cerebras")) < 0.01


def test_throttle_independent_accounts_dont_block_each_other():
    # Round-robin's whole point: account B isn't throttled by account A's call.
    _RATE_STATE.clear()
    slept = []
    fake_clock = lambda: 5000.0
    fake_sleep = lambda s: slept.append(s)
    _throttle("Cerebras", "A", sleeper=fake_sleep, clock=fake_clock)
    _throttle("Cerebras", "B", sleeper=fake_sleep, clock=fake_clock)
    assert slept == []                               # neither waits (different accounts)


def test_throttle_noop_without_key():
    _RATE_STATE.clear()
    slept = []
    _throttle("Cerebras", "", sleeper=lambda s: slept.append(s), clock=lambda: 0.0)
    assert slept == []


# --- Gemini multi-account rotation (lower-ranked verdicts) ------------------

def test_gemini_verdict_rotates_across_accounts():
    # Two Gemini accounts → consecutive verdict calls alternate accounts on the
    # happy path (same machinery as Cerebras). Manual patch (no pytest fixtures).
    import pipeline.core_llm as m
    m._KEY_CURSOR["GeminiVerdict"] = 0
    used = []
    original = m._call_gemini

    def fake_gemini(prompt, api_key):
        used.append(api_key)
        return "verdict"

    m._call_gemini = fake_gemini
    try:
        for _ in range(4):
            call_gemini_verdict("p", "gA,gB", max_attempts=3, label="t")
    finally:
        m._call_gemini = original
    assert used == ["gA", "gB", "gA", "gB"]


def test_gemini_verdict_single_account_unchanged():
    import pipeline.core_llm as m
    m._KEY_CURSOR["GeminiVerdict"] = 0
    used = []
    original = m._call_gemini
    m._call_gemini = lambda prompt, api_key: (used.append(api_key) or "v")
    try:
        for _ in range(3):
            call_gemini_verdict("p", "only", max_attempts=3, label="t")
    finally:
        m._call_gemini = original
    assert used == ["only", "only", "only"]


def test_gemini_verdict_empty_key_raises():
    try:
        call_gemini_verdict("p", "", max_attempts=3)
        assert False, "Expected ValueError on empty Gemini key"
    except ValueError:
        pass
