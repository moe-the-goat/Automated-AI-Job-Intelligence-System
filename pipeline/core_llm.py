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
from pipeline.core_llm_usage import get_tracker, extract_tokens

logger = get_logger(__name__)


def _record_usage(provider, model, response):
    """Tally a successful call into the in-memory usage tracker. Best-effort —
    usage tracking must never break a verdict, so any failure is swallowed."""
    try:
        get_tracker().record(
            provider, model, ok=True, tokens=extract_tokens(response)
        )
    except Exception:
        pass


# Model picks based on the actual free-tier menus on each provider.
#
# 2026-05-29 update: Cerebras retired qwen-3-235b-a22b-instruct-2507 from the
# free tier (calls now 404 "model does not exist or you do not have access").
# The free menu is gpt-oss-120b (Production) + zai-glm-4.7 (Preview). We move to
# gpt-oss-120b — Production tier, 1M TPD (10x Groq), 65K context.
#
# gpt-oss-120b is a REASONING model, which previously truncated/emptied the JSON
# (hidden reasoning ate the 2048-token budget). We tame that with two levers:
#   1. reasoning_effort="low"  — minimise hidden chain-of-thought.
#   2. a larger Cerebras output budget (below) so reasoning + the ~500-token JSON
#      verdict both fit.
# core_ai._parse_ai_response also now extracts the JSON object even if the model
# wraps it in prose — defence in depth against a stray reasoning preamble.
#
# Override per-deployment with the CEREBRAS_MODEL env var (e.g. to try
# zai-glm-4.7) without a code change.
#
# Groq free tier: llama-3.3-70b-versatile.
#   * 30 RPM, 1K RPD, 12K TPM, 100K TPD. Standard non-reasoning instruct model,
#     reliable clean JSON. Only fires on Cerebras failures, so its small TPD is
#     fine as long as Cerebras carries the bulk.
import os

_CEREBRAS_MODEL = os.environ.get("CEREBRAS_MODEL", "gpt-oss-120b")
_GROQ_MODEL = "llama-3.3-70b-versatile"

# gpt-oss reasoning needs headroom beyond the ~500-token JSON answer, so the
# Cerebras call gets a larger budget than the non-reasoning Groq/Gemini paths.
_CEREBRAS_MAX_OUTPUT_TOKENS = 8192

# Gemini Flash Lite handles the cheaper second-pass verdict for "Also Found"
# lower-ranked jobs. 15 RPM / 500 RPD free-tier budget — plenty for ~25 calls
# per run. Same JSON-output behavior as Cerebras/Groq, just smaller model.
_GEMINI_VERDICT_MODEL = "gemini-3.1-flash-lite"

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


# Output budget: 2048 tokens. The actual JSON verdict is ~500 tokens (200-token
# "verdict" string + ~8 numeric/short fields), but we keep headroom so the model
# can preface with a brief preamble before the JSON without truncating it.
_MAX_OUTPUT_TOKENS = 2048


def _call_cerebras(prompt, api_key):
    """Single Cerebras call. Raises on any error (caller decides retry policy)."""
    from cerebras.cloud.sdk import Cerebras
    client = Cerebras(api_key=api_key)
    kwargs = dict(
        model=_CEREBRAS_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_completion_tokens=_CEREBRAS_MAX_OUTPUT_TOKENS,
    )
    # gpt-oss is a reasoning model — keep hidden reasoning minimal so the budget
    # goes to the JSON answer. Guarded: if the installed SDK doesn't accept the
    # kwarg, fall back to a plain call (a larger token budget still covers it).
    try:
        response = client.chat.completions.create(reasoning_effort="low", **kwargs)
    except TypeError:
        response = client.chat.completions.create(**kwargs)
    _record_usage("Cerebras", _CEREBRAS_MODEL, response)
    return response.choices[0].message.content


def _call_groq(prompt, api_key):
    """Single Groq call. Raises on any error (caller decides retry policy)."""
    from groq import Groq
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=_GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=_MAX_OUTPUT_TOKENS,
    )
    _record_usage("Groq", _GROQ_MODEL, response)
    return response.choices[0].message.content


def _call_gemini(prompt, api_key):
    """Single Gemini Flash Lite call. Raises on any error (caller decides retry policy)."""
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=_GEMINI_VERDICT_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=_MAX_OUTPUT_TOKENS,
        ),
    )
    _record_usage("Gemini", _GEMINI_VERDICT_MODEL, response)
    return response.text


def call_gemini_verdict(prompt, api_key, max_attempts=3, label=""):
    """Run a Gemini Flash Lite verdict call with simple retry on transient errors.

    Used for the lower-ranked "Also Found" section verdicts. Only one provider
    (no ping-pong), so we just sleep-and-retry on 5xx/429.
    """
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set")

    last_exc = None
    for idx in range(max_attempts):
        try:
            if idx > 0:
                backoff = _INTER_ATTEMPT_BACKOFF_SECONDS * (2 ** (idx - 1))
                logger.warning("[LLM Gemini RETRY %d/%d] backing off %.1fs for %s",
                               idx + 1, max_attempts, backoff, label[:55])
                time.sleep(backoff)
            return _call_gemini(prompt, api_key)
        except Exception as e:
            last_exc = e
            err_str = str(e)[:200]
            if _is_retryable_error(e):
                logger.warning("[LLM Gemini] retryable error attempt %d: %s", idx + 1, err_str)
            else:
                logger.warning("[LLM Gemini] non-retryable error: %s", err_str)
                break

    raise last_exc if last_exc else RuntimeError("Gemini call exhausted with no exception")


# Round-robin cursors so multiple keys per provider (multiple accounts) are
# spread evenly across calls — this is what makes a 2nd Cerebras/Groq account
# actually carry load instead of sitting idle. Module-level so rotation persists
# across the many calls in a run.
_KEY_CURSOR = {"Cerebras": 0, "Groq": 0}


def _parse_keys(raw):
    """Normalize a key input into a list of non-empty keys.

    Accepts a single string, a comma-separated string ("k1,k2" for multiple
    accounts), or a list. Empty / None → []. So a caller can pass one key (works
    as before) or several (rotation kicks in)."""
    if not raw:
        return []
    if isinstance(raw, (list, tuple)):
        items = raw
    else:
        items = str(raw).split(",")
    return [k.strip() for k in items if k and k.strip()]


def _next_key(provider, keys):
    """Pick the next key for a provider in round-robin order, advancing the
    cursor. With one key this always returns it; with two it alternates."""
    if not keys:
        return None
    i = _KEY_CURSOR[provider] % len(keys)
    _KEY_CURSOR[provider] = (i + 1) % len(keys)
    return keys[i]


def call_llm_with_fallback(prompt, cerebras_key, groq_key, max_attempts=4, label=""):
    """Run an LLM completion with Cerebras<->Groq ping-pong fallback.

    Sequence (when both providers have keys): Cerebras, Groq, Cerebras, Groq...
    Each provider's keys are ROUND-ROBINED across calls, so passing two keys for
    a provider (comma-separated, e.g. "k1,k2" — two accounts) spreads the load
    evenly and effectively doubles that provider's rate-limit headroom. On a
    non-retryable error from one provider, still tries the other once in case the
    failure is provider-specific.

    Args:
        prompt: full prompt string (the existing verdict prompt from core_ai.py).
        cerebras_key: CEREBRAS_API_KEY — one key, or comma-separated for multiple
            accounts (may be empty/None to skip Cerebras).
        groq_key: GROQ_API_KEY — same (one or comma-separated).
        max_attempts: minimum total attempts across both providers (default 4).
        label: short tag for log lines, usually the job title.

    Returns:
        Response text content on success.

    Raises:
        ValueError: if both keys are empty.
        Exception: re-raises the LAST exception after all attempts fail.
    """
    cerebras_keys = _parse_keys(cerebras_key)
    groq_keys = _parse_keys(groq_key)
    if not cerebras_keys and not groq_keys:
        raise ValueError("Neither CEREBRAS_API_KEY nor GROQ_API_KEY is set")

    # Build the alternating provider sequence; pick each attempt's key in
    # round-robin within that provider so multiple accounts share the load.
    providers = []
    if cerebras_keys and groq_keys:
        for i in range(max_attempts):
            if i % 2 == 0:
                providers.append(("Cerebras", _call_cerebras, _next_key("Cerebras", cerebras_keys)))
            else:
                providers.append(("Groq", _call_groq, _next_key("Groq", groq_keys)))
    elif cerebras_keys:
        providers = [
            ("Cerebras", _call_cerebras, _next_key("Cerebras", cerebras_keys))
            for _ in range(max_attempts)
        ]
    else:
        providers = [
            ("Groq", _call_groq, _next_key("Groq", groq_keys))
            for _ in range(max_attempts)
        ]

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
            # Record the failed attempt against this provider (best-effort).
            try:
                model = _CEREBRAS_MODEL if name == "Cerebras" else _GROQ_MODEL
                get_tracker().record(name, model, ok=False)
            except Exception:
                pass
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
