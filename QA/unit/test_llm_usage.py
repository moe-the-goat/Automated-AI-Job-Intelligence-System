"""core_llm_usage tracker + core_llm key rotation.

Locks:
  - the usage tracker tallies requests/failures/tokens per (provider, model)
  - snapshot_and_reset returns rows and clears state for the next user
  - extract_tokens reads both OpenAI-style and Gemini-style responses
  - key rotation (_parse_keys / _take_start) spreads calls across accounts so
    the happy path alternates accounts (not always the first)
  - pacing (_eval_pace_seconds) shrinks as accounts are added
"""

from pipeline.core_llm_usage import _UsageTracker, extract_tokens
from pipeline.core_llm import (
    _parse_keys,
    _take_start,
    _KEY_CURSOR,
    call_llm_with_fallback,
)
from pipeline.core_ai import _count_keys, _eval_pace_seconds


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


# --- pacing -----------------------------------------------------------------

def test_count_keys():
    assert _count_keys("A,B") == 2
    assert _count_keys("A") == 1
    assert _count_keys("") == 0
    assert _count_keys(None) == 0
    assert _count_keys("A, , B") == 2


def test_pace_halves_with_two_cerebras_accounts():
    one = _eval_pace_seconds("A", "g1")
    two = _eval_pace_seconds("A,B", "g1")
    # One account ≈ the old ~13s; two accounts ≈ half that.
    assert 12.5 <= one <= 14.0
    assert abs(two - one / 2) < 0.01


def test_pace_falls_back_to_groq_when_no_cerebras():
    # Groq's 30 RPM is far higher → much shorter gap than Cerebras.
    pace = _eval_pace_seconds("", "g1")
    assert 1.0 <= pace <= 3.0
