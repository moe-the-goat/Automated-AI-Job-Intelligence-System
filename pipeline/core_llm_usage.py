"""
LLM USAGE TRACKER
-----------------
Lightweight, in-memory tally of LLM/embedding API calls, flushed to Supabase
once per user-run. Keeps the hot path cheap: the verdict loop makes dozens of
calls per run, so we count in memory and write a single upsert at the end
instead of a DB round-trip per call.

What it records, per (provider, model):
  - requests        total calls
  - requests_failed calls that raised
  - tokens          summed token usage where the provider reports it (best effort)
  - peak_rpm        max calls observed in any rolling 60s window (a rate proxy —
                    a dashboard can't show a live RPM, so we surface the peak)

Usage:
    from pipeline.core_llm_usage import get_tracker
    t = get_tracker()                       # process-wide singleton
    t.record("Cerebras", "gpt-oss-120b", ok=True, tokens=812)
    ...
    rows = t.snapshot_and_reset()           # at end of a user-run
    # caller writes rows to Supabase keyed by (user_id, provider, model, day)

The tracker is provider-agnostic and has NO Supabase dependency — the runner
owns persistence, so core_llm / core_embedding can import this without dragging
the DB client into them.
"""

import time
from collections import defaultdict
from threading import Lock


class _UsageTracker:
    def __init__(self):
        self._lock = Lock()
        # key: (provider, model) -> dict of counters
        self._counts = defaultdict(
            lambda: {"requests": 0, "requests_failed": 0, "tokens": 0, "peak_rpm": 0}
        )
        # key: (provider, model) -> list of recent call timestamps (for peak RPM)
        self._timestamps = defaultdict(list)

    def record(self, provider, model, *, ok=True, tokens=0):
        """Record one call. `tokens` is best-effort (0 when unknown)."""
        if not provider or not model:
            return
        key = (provider, model)
        now = time.time()
        with self._lock:
            c = self._counts[key]
            c["requests"] += 1
            if not ok:
                c["requests_failed"] += 1
            if tokens:
                c["tokens"] += int(tokens)
            # Rolling 60s window for peak-RPM: keep only the last minute of
            # timestamps, then the window length is the current RPM; track its max.
            ts = self._timestamps[key]
            ts.append(now)
            cutoff = now - 60
            while ts and ts[0] < cutoff:
                ts.pop(0)
            if len(ts) > c["peak_rpm"]:
                c["peak_rpm"] = len(ts)

    def snapshot_and_reset(self):
        """Return a list of per-(provider, model) usage dicts and clear state.

        Each row: {provider, model, requests, requests_failed, tokens, peak_rpm}.
        Called once at the end of a user-run; the caller persists + resets for the
        next user so counts are attributed to the right user_id.
        """
        with self._lock:
            rows = [
                {"provider": p, "model": m, **vals}
                for (p, m), vals in self._counts.items()
            ]
            self._counts.clear()
            self._timestamps.clear()
            return rows


_TRACKER = None


def get_tracker():
    """Process-wide singleton tracker."""
    global _TRACKER
    if _TRACKER is None:
        _TRACKER = _UsageTracker()
    return _TRACKER


def extract_tokens(response):
    """Best-effort total-token count from a provider response. 0 when absent.

    Handles the shapes we use:
      - OpenAI-style (Cerebras, Groq): response.usage.total_tokens
      - Gemini: response.usage_metadata.total_token_count
    """
    try:
        usage = getattr(response, "usage", None)
        if usage is not None:
            tot = getattr(usage, "total_tokens", None)
            if isinstance(tot, int):
                return tot
        meta = getattr(response, "usage_metadata", None)
        if meta is not None:
            tot = getattr(meta, "total_token_count", None)
            if isinstance(tot, int):
                return tot
    except Exception:
        pass
    return 0
