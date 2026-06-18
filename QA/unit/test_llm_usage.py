"""core_llm_usage tracker + core_llm key rotation.

Locks:
  - the usage tracker tallies requests/failures/tokens per (provider, model)
  - snapshot_and_reset returns rows and clears state for the next user
  - extract_tokens reads both OpenAI-style and Gemini-style responses
  - key rotation (_parse_keys / _next_key) spreads calls across multiple keys
"""

from pipeline.core_llm_usage import _UsageTracker, extract_tokens
from pipeline.core_llm import _parse_keys, _next_key, _KEY_CURSOR


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


def test_next_key_round_robins_two_accounts():
    _KEY_CURSOR["Cerebras"] = 0
    seq = [_next_key("Cerebras", ["k1", "k2"]) for _ in range(4)]
    assert seq == ["k1", "k2", "k1", "k2"]


def test_next_key_single_key_always_same():
    _KEY_CURSOR["Groq"] = 0
    seq = [_next_key("Groq", ["only"]) for _ in range(3)]
    assert seq == ["only", "only", "only"]
