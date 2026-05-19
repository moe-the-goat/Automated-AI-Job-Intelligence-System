"""
CORE LLM MODULE
---------------
Cerebras (primary) + Groq (fallback) ping-pong client for the main job-verdict
generation. Both providers run llama-3.3-70b for free.

Fallback order on transient/retryable errors:
    Cerebras -> Groq -> Cerebras -> Groq (min 4 attempts before giving up)

Why split out from core_ai.py: the verdict prompt, JSON schema, and post-AI
caps in core_ai.py are stable; only the underlying API client changed. Keeping
the swap isolated here means a future provider switch touches one file.

Both Cerebras and Groq use OpenAI-compatible APIs. We use their official
Python SDKs for clean error handling — the SDKs are lazy-imported inside
the call helpers so unit tests don't need them installed.
"""
import time

from pipeline.logging_setup import get_logger

logger = get_logger(__name__)


# Model picks based on the actual free-tier menus on each provider (2026-05-19):
#
# Cerebras free tier: gpt-oss-120b is the most capable PRODUCTION model on offer.
#   * 65K context, 5 RPM, 150 RPH, 2400 RPD, 30K TPM, 1M TPH, 1M TPD.
#   * llama-3.3-70b (used previously) does NOT exist on Cerebras free tier — every
#     call silently fell through to Groq.
#
# Groq free tier: meta-llama/llama-4-scout-17b-16e-instruct.
#   * 30 RPM, 1K RPD, 30K TPM, 500K TPD.
#   * llama-3.3-70b-versatile (used previously) caps at only 100K TPD — too small
#     for ~60 evals/day at 3.5K tokens each (we exhausted it after ~28 jobs).
#   * llama-4-scout is Meta's newer 17B instruction-tuned model with 5x the daily
#     token quota of llama-3.3-70b-versatile.
_CEREBRAS_MODEL = "gpt-oss-120b"
_GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# Short backoff between fallback attempts. The user wants "immediate" switching
# between providers — long exponential backoff would defeat the point of having
# a fallback. 1.5s gives the failing provider a moment to settle without
# blocking the pipeline for minutes.
_INTER_ATTEMPT_BACKOFF_SECONDS = 1.5

# Retry markers that justify trying the OTHER provider. Includes 5xx server
# errors, rate limits, timeouts, and connection drops — all of which are
# typically per-provider transient conditions.
_RETRYABLE_MARKERS = (
    "500", "502", "503", "504",
    "UNAVAILABLE", "INTERNAL", "BAD_GATEWAY", "GATEWAY_TIMEOUT",
    "timeout", "Timeout", "TIMEOUT",
    "ConnectionError", "ConnectError", "connection error",
    "rate limit", "Rate limit", "rate_limit", "RATE_LIMIT", "429",
    "overloaded", "Overloaded", "OVERLOADED",
    "service unavailable", "Service Unavailable",
)


def _is_retryable_error(exc):
    """Decide whether the exception is a transient condition worth retrying.

    True for 5xx-class server errors, rate limits, timeouts, and connection
    failures. False for 4xx auth errors and other non-recoverable cases.
    """
    msg = str(exc)
    return any(marker in msg for marker in _RETRYABLE_MARKERS)


def _call_cerebras(prompt, api_key):
    """Single Cerebras call. Raises on any error (caller decides retry policy)."""
    from cerebras.cloud.sdk import Cerebras
    client = Cerebras(api_key=api_key)
    response = client.chat.completions.create(
        model=_CEREBRAS_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_completion_tokens=800,
    )
    return response.choices[0].message.content


def _call_groq(prompt, api_key):
    """Single Groq call. Raises on any error (caller decides retry policy)."""
    from groq import Groq
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=_GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=800,
    )
    return response.choices[0].message.content


def call_llm_with_fallback(prompt, cerebras_key, groq_key, max_attempts=4, label=""):
    """Run an LLM completion with Cerebras<->Groq ping-pong fallback.

    Sequence (when both keys are present): Cerebras, Groq, Cerebras, Groq, ...
    On a non-retryable error from one provider, still tries the other once in
    case the failure is provider-specific (model unavailable, region block, etc.).

    Args:
        prompt: full prompt string (the existing verdict prompt from core_ai.py).
        cerebras_key: CEREBRAS_API_KEY (may be empty/None to skip).
        groq_key: GROQ_API_KEY (may be empty/None to skip).
        max_attempts: minimum total attempts across both providers (default 4).
        label: short tag for log lines, usually the job title.

    Returns:
        Response text content on success.

    Raises:
        ValueError: if both keys are empty.
        Exception: re-raises the LAST exception after all attempts fail.
    """
    if not cerebras_key and not groq_key:
        raise ValueError("Neither CEREBRAS_API_KEY nor GROQ_API_KEY is set")

    # Build the alternating sequence. If only one key is present, repeat that
    # provider for all attempts.
    providers = []
    if cerebras_key and groq_key:
        for i in range(max_attempts):
            if i % 2 == 0:
                providers.append(("Cerebras", _call_cerebras, cerebras_key))
            else:
                providers.append(("Groq", _call_groq, groq_key))
    elif cerebras_key:
        providers = [("Cerebras", _call_cerebras, cerebras_key)] * max_attempts
    else:
        providers = [("Groq", _call_groq, groq_key)] * max_attempts

    last_exc = None
    for idx, (name, fn, key) in enumerate(providers):
        try:
            if idx > 0:
                logger.warning(
                    "[LLM RETRY %d/%d] switching to %s after %.1fs for %s",
                    idx + 1, max_attempts, name, _INTER_ATTEMPT_BACKOFF_SECONDS,
                    label[:55],
                )
                time.sleep(_INTER_ATTEMPT_BACKOFF_SECONDS)
            return fn(prompt, key)
        except Exception as e:
            last_exc = e
            err_str = str(e)[:200]
            if _is_retryable_error(e):
                logger.warning("[LLM %s] retryable error attempt %d: %s", name, idx + 1, err_str)
            else:
                logger.warning(
                    "[LLM %s] non-retryable error: %s (will still try the other provider)",
                    name, err_str,
                )

    # All attempts exhausted.
    raise last_exc if last_exc else RuntimeError("LLM fallback exhausted with no exception")
