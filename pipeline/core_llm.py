"""
CORE LLM MODULE
---------------
Cerebras (primary) + Groq (fallback) ping-pong client for the main job-verdict
generation. Both providers now run gpt-oss-120b for free, so the fallback
scores on the same scale as the primary.

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


_usage_record_warned = False


def _record_usage(provider, model, response):
    """Tally a successful call into the in-memory usage tracker. Best-effort —
    usage tracking must never break a verdict, so failures are swallowed. But we
    log the FIRST failure (once) instead of silently doing nothing forever, so a
    broken tracker is diagnosable rather than invisible."""
    global _usage_record_warned
    try:
        get_tracker().record(
            provider, model, ok=True, tokens=extract_tokens(response)
        )
    except Exception as e:
        if not _usage_record_warned:
            _usage_record_warned = True
            logger.warning("LLM usage record failed (first occurrence): %s", str(e)[:200])


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
# Groq free tier: openai/gpt-oss-120b.
#
# 2026-07-03 update: Groq deprecated llama-3.3-70b-versatile (decommission
# 2026-08-16). Their recommended replacement is gpt-oss-120b — which is also the
# model our Cerebras primary runs, so primary and fallback now score on the SAME
# scale (no mixed verdict behavior in one digest). Published org limits:
#   * 30 RPM, 1K RPD, 8K TPM, 200K TPD — same RPM/RPD as the old llama, double
#     the TPD; the smaller TPM is handled by the output budget below. Fallback-
#     only traffic, so the ceilings are comfortable.
# Override per-deployment with the GROQ_MODEL env var (e.g. qwen/qwen3.6-27b)
# without a code change — same escape hatch as CEREBRAS_MODEL.
import os

_CEREBRAS_MODEL = os.environ.get("CEREBRAS_MODEL", "gpt-oss-120b")
_GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

# gpt-oss reasoning needs headroom beyond the ~500-token JSON answer, so the
# Cerebras call gets a larger budget than the non-reasoning Gemini path.
_CEREBRAS_MAX_OUTPUT_TOKENS = 8192

# Groq's gpt-oss-120b shares the 8K TPM free-tier ceiling — the tightest budget in
# the fleet — and Groq PRE-COUNTS (prompt + max_tokens) against it, rejecting the
# request with an HTTP 413 "rate_limit_exceeded" BEFORE running (0 tokens billed)
# if the sum exceeds 8K. The verdict prompt is ~2.9K tokens of fixed structure
# (CV capped at 3K chars + description at 5K chars + the instructions) PLUS a
# VARIABLE learned-preferences block: for a heavy-feedback user the RAG/digest
# profile can add another 1-3K tokens. So a 4096 output budget pushed real
# requests past 8K and 413'd. 2000 keeps (prompt + output) under the ceiling for
# realistic prompts while still fitting low-effort hidden reasoning + the
# ~500-token JSON verdict; a prompt still too big falls through to Cerebras (which
# has a far larger TPM), handled in the fallback loop below.
_GROQ_MAX_OUTPUT_TOKENS = 2000

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


def _is_request_too_large(exc):
    """A Groq HTTP 413 (labeled `rate_limit_exceeded`) means the single request's
    (prompt + max_tokens) exceeds the per-minute TPM ceiling — a STRUCTURAL limit,
    not a transient one. Retrying the same request always 413s again, so the
    fallback stops trying that provider for this job and lets the other provider
    (with a far larger TPM) take it. Distinct from a plain 429, which does clear
    with time and stays retryable."""
    msg = str(exc).lower()
    return (
        "413" in msg
        or "request too large" in msg
        or "tokens per minute" in msg
        or "reduce your message" in msg
    )


# Output budget: 2048 tokens. The actual JSON verdict is ~500 tokens (200-token
# "verdict" string + ~8 numeric/short fields), but we keep headroom so the model
# can preface with a brief preamble before the JSON without truncating it.
_MAX_OUTPUT_TOKENS = 2048


# ---------------------------------------------------------------------------
# Adaptive per-account rate limiting.
# ---------------------------------------------------------------------------
# Pacing lives HERE, at the call site, keyed by (provider, account) — not as a
# fixed sleep in the caller. Why this is better than a flat pre-call sleep:
#   * Per ACCOUNT: round-robined accounts each get their own RPM budget, so two
#     accounts genuinely run at 2× the rate (the budget isn't shared away).
#   * Absorbs call duration: we sleep only until the account's next slot, so the
#     time the previous call spent on the network counts toward the interval
#     instead of being added on top — shorter wall-clock when calls aren't free.
#   * Provider-aware: the Groq fallback (30 RPM) waits far less than Cerebras
#     (5 RPM), and Gemini (15 RPM) gets its own budget — all from one mechanism.
# State is module-level so the budget is enforced across every call in the run
# (and across users in a tick — the RPM ceiling is global per account).
import threading

_RPM_PER_ACCOUNT = {"Cerebras": 5, "Groq": 30, "Gemini": 15}
_RATE_SAFETY = 1.10  # 10% headroom under the published per-account RPM
_RATE_STATE = {}     # (provider, api_key) -> next-allowed monotonic timestamp
_RATE_LOCK = threading.Lock()


def _min_interval(provider):
    """Minimum seconds between calls on ONE account for this provider."""
    rpm = _RPM_PER_ACCOUNT.get(provider, 5)
    return (60.0 / rpm) * _RATE_SAFETY


def _reserve(now, next_allowed, interval):
    """Pure core of the limiter: given the clock and an account's next-allowed
    time, return (wait_seconds, new_next_allowed). Reserves this call's slot so
    concurrent callers serialize. Kept pure so it's unit-testable without sleeping."""
    wait = max(0.0, next_allowed - now)
    new_next = max(now, next_allowed) + interval
    return wait, new_next


def _throttle(provider, api_key, *, sleeper=time.sleep, clock=time.monotonic):
    """Block until this (provider, account) may make another call under its
    free-tier RPM. No-op when api_key is falsy (nothing to bill)."""
    if not api_key:
        return
    interval = _min_interval(provider)
    key = (provider, api_key)
    with _RATE_LOCK:
        wait, new_next = _reserve(clock(), _RATE_STATE.get(key, 0.0), interval)
        _RATE_STATE[key] = new_next
    if wait > 0:
        sleeper(wait)


def _call_cerebras(prompt, api_key):
    """Single Cerebras call. Raises on any error (caller decides retry policy)."""
    _throttle("Cerebras", api_key)
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
    _throttle("Groq", api_key)
    from groq import Groq
    client = Groq(api_key=api_key)
    kwargs = dict(
        model=_GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=_GROQ_MAX_OUTPUT_TOKENS,
    )
    # gpt-oss is a reasoning model. By default Groq returns its chain-of-thought
    # INSIDE message.content (wrapped in <think> tags), which pollutes the JSON
    # answer with stray prose/braces and eats the token budget — the cause of
    # the "truncated/emptied JSON" this model is known for. So:
    #   * reasoning_effort="low"    — minimise the hidden reasoning.
    #   * reasoning_format="hidden" — keep it OUT of content; content is only
    #                                 the final answer (clean JSON).
    # Guarded: an older groq SDK (or a non-reasoning GROQ_MODEL override) that
    # rejects the kwargs falls back to a plain call; core_ai._parse_ai_response
    # still brace-extracts the JSON if a reasoning preamble sneaks through.
    try:
        response = client.chat.completions.create(
            reasoning_effort="low", reasoning_format="hidden", **kwargs
        )
    except TypeError:
        response = client.chat.completions.create(**kwargs)
    _record_usage("Groq", _GROQ_MODEL, response)
    return response.choices[0].message.content


def _call_gemini(prompt, api_key):
    """Single Gemini Flash Lite call. Raises on any error (caller decides retry policy)."""
    _throttle("Gemini", api_key)
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

    Used for the lower-ranked "Also Found" section verdicts. `api_key` may be a
    single key or a comma-separated list of accounts. Multiple accounts are
    round-robined the same way Cerebras/Groq are: consecutive calls start on the
    next account (spreading the steady load), and a retry within a call advances
    to a different account than the one that just failed. Per-account RPM is then
    enforced in _call_gemini via _throttle.
    """
    keys = _parse_keys(api_key)
    if not keys:
        raise ValueError("GEMINI_API_KEY is not set")

    start = _take_start("GeminiVerdict", len(keys))
    last_exc = None
    for idx in range(max_attempts):
        key = keys[(start + idx) % len(keys)]
        try:
            if idx > 0:
                backoff = _INTER_ATTEMPT_BACKOFF_SECONDS * (2 ** (idx - 1))
                logger.warning("[LLM Gemini RETRY %d/%d] backing off %.1fs for %s",
                               idx + 1, max_attempts, backoff, label[:55])
                time.sleep(backoff)
            return _call_gemini(prompt, key)
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


def _take_start(provider, n):
    """Return the round-robin START index for THIS call and advance the cursor by
    exactly ONE, so consecutive calls begin on the next account.

    Why per-call (not per-attempt): a single call_llm_with_fallback builds a
    multi-attempt plan that may include the same provider twice (Cerebras on
    attempts 0 and 2). The old _next_key advanced once PER attempt, so with two
    accounts the cursor moved by 2 each call and always landed back on account A
    for attempt 0 — meaning the happy path (no retry) used account A every time
    and account B only ever saw retries. Advancing by one per CALL instead makes
    attempt-0 alternate A, B, A, B… so both accounts truly share the steady load
    (which is what lets us halve the pacing). Within a call, extra attempts on the
    same provider rotate onward from this start, so a retry hits a DIFFERENT
    account than the one that just failed."""
    if n <= 0:
        return 0
    start = _KEY_CURSOR.get(provider, 0) % n
    _KEY_CURSOR[provider] = (start + 1) % n
    return start


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

    # Each provider gets ONE round-robin start index for this whole call, so
    # consecutive calls begin on the next account (balancing the happy-path load
    # across accounts). Within this call, repeated attempts on a provider rotate
    # onward from its start, so a retry lands on a different account than the one
    # that just failed.
    cer_start = _take_start("Cerebras", len(cerebras_keys))
    groq_start = _take_start("Groq", len(groq_keys))

    # Build the alternating provider sequence (Cerebras, Groq, Cerebras, Groq…),
    # falling back to whichever provider has keys when only one is configured.
    providers = []
    cer_used = 0
    groq_used = 0
    for i in range(max_attempts):
        if cerebras_keys and (i % 2 == 0 or not groq_keys):
            key = cerebras_keys[(cer_start + cer_used) % len(cerebras_keys)]
            cer_used += 1
            providers.append(("Cerebras", _call_cerebras, key))
        elif groq_keys:
            key = groq_keys[(groq_start + groq_used) % len(groq_keys)]
            groq_used += 1
            providers.append(("Groq", _call_groq, key))

    last_exc = None
    groq_too_large = False  # a Groq 413 means this prompt structurally can't fit
    for idx, (name, fn, key) in enumerate(providers):
        # A prior Groq attempt already 413'd on request size — retrying Groq with
        # the SAME prompt will 413 again, so skip it and let Cerebras take the job.
        if name == "Groq" and groq_too_large:
            continue
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
            if name == "Groq" and _is_request_too_large(e):
                # Structural, not transient — stop trying Groq for this job.
                groq_too_large = True
                logger.warning(
                    "[LLM Groq] request too large for the free-tier TPM (413) — "
                    "skipping Groq for this job; Cerebras will handle it: %s",
                    err_str,
                )
            elif _is_retryable_error(e):
                logger.warning("[LLM %s] retryable error attempt %d: %s", name, idx + 1, err_str)
            else:
                logger.warning(
                    "[LLM %s] non-retryable error: %s (will still try the other provider)",
                    name, err_str,
                )

    # All attempts exhausted.
    raise last_exc if last_exc else RuntimeError("LLM fallback exhausted with no exception")
