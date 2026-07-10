"""core_llm: Cerebras + Groq ping-pong fallback client.

We replace both provider call helpers via unittest.mock.patch and verify the
alternation, retry budget, and error classification. No real network calls.

(We avoid pytest's `monkeypatch` fixture because the project's QA runner calls
test functions directly rather than via pytest, so fixtures aren't injected.)
"""
from unittest.mock import patch

from pipeline import core_llm
from pipeline.core_llm import (
    call_llm_with_fallback,
    _is_retryable_error,
    _is_request_too_large,
    _GROQ_MAX_OUTPUT_TOKENS,
)


def test_raises_when_no_keys_provided():
    try:
        call_llm_with_fallback("any prompt", "", "")
        assert False, "Expected ValueError when no keys provided"
    except ValueError as e:
        assert "CEREBRAS_API_KEY" in str(e) or "GROQ_API_KEY" in str(e)


def test_uses_cerebras_on_first_attempt_when_both_keys_present():
    """Happy path: Cerebras succeeds first try, no Groq calls happen."""
    calls = []

    def fake_cerebras(prompt, key):
        calls.append(("cerebras", key))
        return '{"verdict": "ok"}'

    def fake_groq(prompt, key):
        calls.append(("groq", key))
        return '{"verdict": "groq-ok"}'

    with patch.object(core_llm, "_call_cerebras", fake_cerebras), \
         patch.object(core_llm, "_call_groq", fake_groq):
        result = call_llm_with_fallback("p", cerebras_key="csk-xxx", groq_key="gsk-yyy")

    assert result == '{"verdict": "ok"}'
    assert calls == [("cerebras", "csk-xxx")]


def test_groq_output_budget_fits_under_free_tier_tpm():
    """Groq pre-counts (prompt + max_tokens) against its 8K free-tier TPM and 413s
    if the sum exceeds it. The output budget must leave room for a realistic
    ~4-5K-token verdict prompt, so it stays well under 8K (this pins the value so
    a future bump back to 4096 — the size that caused the 413s — fails CI)."""
    assert _GROQ_MAX_OUTPUT_TOKENS <= 2048


def test_is_request_too_large_recognizes_413_but_not_plain_429():
    assert _is_request_too_large(Exception("Error code: 413 - request too large"))
    assert _is_request_too_large(Exception("rate_limit_exceeded on tokens per minute (TPM): Limit 8000"))
    # A plain 429 rate limit is transient, NOT a structural too-large — it should
    # stay retryable and not trigger the skip-Groq path.
    assert not _is_request_too_large(Exception("429 Too Many Requests"))


def test_groq_413_skips_further_groq_and_lets_cerebras_take_it():
    """A Groq 413 (request too large) must not be retried on Groq — the same
    prompt would 413 again. Cerebras (larger TPM) should score the job, and Groq
    should be hit at most once."""
    calls = []

    def fake_cerebras(prompt, key):
        calls.append(("cerebras", key))
        # Fail the first Cerebras attempt so the sequence reaches Groq, then
        # succeed on the later Cerebras attempt.
        if len([c for c in calls if c[0] == "cerebras"]) == 1:
            raise Exception("503 Service Unavailable")
        return '{"verdict": "cerebras-rescued"}'

    def fake_groq(prompt, key):
        calls.append(("groq", key))
        raise Exception("Error code: 413 - request too large for model on tokens per minute (TPM)")

    with patch.object(core_llm, "_call_cerebras", fake_cerebras), \
         patch.object(core_llm, "_call_groq", fake_groq), \
         patch.object(core_llm, "_INTER_ATTEMPT_BACKOFF_SECONDS", 0):
        result = call_llm_with_fallback("p", cerebras_key="csk", groq_key="gsk", max_attempts=4)

    assert result == '{"verdict": "cerebras-rescued"}'
    # Sequence is Cerebras, Groq, Cerebras, Groq — Groq 413s once on attempt 2,
    # then the 4th (Groq) attempt is skipped, so Groq is called exactly once.
    assert [c[0] for c in calls].count("groq") == 1


def test_falls_back_to_groq_when_cerebras_5xx():
    calls = []

    def fake_cerebras(prompt, key):
        calls.append("cerebras")
        raise Exception("503 Service Unavailable")

    def fake_groq(prompt, key):
        calls.append("groq")
        return '{"verdict": "groq-rescued"}'

    with patch.object(core_llm, "_call_cerebras", fake_cerebras), \
         patch.object(core_llm, "_call_groq", fake_groq), \
         patch.object(core_llm, "_INTER_ATTEMPT_BACKOFF_SECONDS", 0):
        result = call_llm_with_fallback("p", cerebras_key="csk-xxx", groq_key="gsk-yyy")

    assert result == '{"verdict": "groq-rescued"}'
    assert calls == ["cerebras", "groq"]


def test_ping_pong_alternation_4_attempts():
    """All four attempts fail with 503 -> should attempt cerebras, groq, cerebras, groq in order."""
    calls = []

    def fake_cerebras(prompt, key):
        calls.append("cerebras")
        raise Exception("503 Service Unavailable")

    def fake_groq(prompt, key):
        calls.append("groq")
        raise Exception("503 Service Unavailable")

    with patch.object(core_llm, "_call_cerebras", fake_cerebras), \
         patch.object(core_llm, "_call_groq", fake_groq), \
         patch.object(core_llm, "_INTER_ATTEMPT_BACKOFF_SECONDS", 0):
        try:
            call_llm_with_fallback("p", cerebras_key="csk-xxx", groq_key="gsk-yyy", max_attempts=4)
            assert False, "Expected exception after all attempts exhausted"
        except Exception as e:
            assert "503" in str(e)

    assert calls == ["cerebras", "groq", "cerebras", "groq"], (
        f"Expected ping-pong cerebras/groq alternation, got {calls}"
    )


def test_recovers_on_third_attempt():
    """Cerebras fails -> Groq fails -> Cerebras succeeds. Verifies > 2 attempts work."""
    attempts = {"cerebras": 0, "groq": 0}

    def fake_cerebras(prompt, key):
        attempts["cerebras"] += 1
        if attempts["cerebras"] == 1:
            raise Exception("503 backend overloaded")
        return '{"verdict": "third-time-lucky"}'

    def fake_groq(prompt, key):
        attempts["groq"] += 1
        raise Exception("503 server error")

    with patch.object(core_llm, "_call_cerebras", fake_cerebras), \
         patch.object(core_llm, "_call_groq", fake_groq), \
         patch.object(core_llm, "_INTER_ATTEMPT_BACKOFF_SECONDS", 0):
        result = call_llm_with_fallback("p", cerebras_key="csk-x", groq_key="gsk-y", max_attempts=4)

    assert result == '{"verdict": "third-time-lucky"}'
    assert attempts["cerebras"] == 2
    assert attempts["groq"] == 1


def test_uses_only_cerebras_when_groq_key_missing():
    calls = []

    def fake_cerebras(prompt, key):
        calls.append("cerebras")
        if len(calls) < 3:
            raise Exception("503 unavailable")
        return "success"

    def fake_groq(prompt, key):
        calls.append("groq")  # should not happen
        return "groq"

    with patch.object(core_llm, "_call_cerebras", fake_cerebras), \
         patch.object(core_llm, "_call_groq", fake_groq), \
         patch.object(core_llm, "_INTER_ATTEMPT_BACKOFF_SECONDS", 0):
        result = call_llm_with_fallback("p", cerebras_key="csk-x", groq_key="", max_attempts=4)

    assert result == "success"
    assert calls == ["cerebras", "cerebras", "cerebras"]


def test_uses_only_groq_when_cerebras_key_missing():
    calls = []

    def fake_groq(prompt, key):
        calls.append("groq")
        return "groq-result"

    with patch.object(core_llm, "_call_groq", fake_groq):
        result = call_llm_with_fallback("p", cerebras_key="", groq_key="gsk-y", max_attempts=4)

    assert result == "groq-result"
    assert calls == ["groq"]


def test_non_retryable_error_still_tries_other_provider():
    """A 400-class error from Cerebras should still let Groq attempt — provider-specific
    issues (model unavailable on one platform) can be rescued by the other."""
    calls = []

    def fake_cerebras(prompt, key):
        calls.append("cerebras")
        raise Exception("400 model not found")

    def fake_groq(prompt, key):
        calls.append("groq")
        return "groq-rescued"

    with patch.object(core_llm, "_call_cerebras", fake_cerebras), \
         patch.object(core_llm, "_call_groq", fake_groq), \
         patch.object(core_llm, "_INTER_ATTEMPT_BACKOFF_SECONDS", 0):
        result = call_llm_with_fallback("p", cerebras_key="csk-x", groq_key="gsk-y", max_attempts=4)

    assert result == "groq-rescued"


# ---------------------------------------------------------------------------
# _is_retryable_error classification
# ---------------------------------------------------------------------------

def test_classifies_503_as_retryable():
    assert _is_retryable_error(Exception("503 Service Unavailable")) is True


def test_classifies_429_rate_limit_as_retryable():
    assert _is_retryable_error(Exception("429 Too Many Requests rate limit hit")) is True


def test_classifies_timeout_as_retryable():
    assert _is_retryable_error(Exception("ConnectionError: timeout exceeded")) is True


def test_classifies_400_as_non_retryable():
    assert _is_retryable_error(Exception("400 Bad Request invalid prompt")) is False


def test_classifies_401_auth_as_non_retryable():
    assert _is_retryable_error(Exception("401 Unauthorized")) is False


# --- Groq model migration (llama-3.3-70b deprecated 2026-07, gone 2026-08-16) ---

def test_groq_default_model_is_gpt_oss_120b():
    """Locks the migration off llama-3.3-70b-versatile: after Groq's decommission
    date that id stops being served, so a regression here means dead fallbacks.
    The default must be Groq's recommended replacement (also our Cerebras primary,
    keeping verdict scales consistent); GROQ_MODEL env still overrides at import."""
    assert core_llm._GROQ_MODEL == "openai/gpt-oss-120b"


def test_groq_output_budget_fits_8k_tpm():
    """Groq pre-counts prompt + max output against its 8K TPM. With a ~3K-token
    verdict prompt, the output budget must stay <= 4096 or single calls start
    bouncing off the per-minute token ceiling."""
    assert core_llm._GROQ_MAX_OUTPUT_TOKENS <= 4096
