"""
CORE EMAIL (Resend) MODULE
--------------------------
Thin wrapper around the Resend REST API for the multi-user pipeline. Replaces
Gmail SMTP from core_notify.send_email() — SMTP is per-recipient and tied to
Mohammad's personal account, neither of which scales to N users.

Stays a separate module (not bolted onto core_notify) so the single-user
pipeline keeps using SMTP unchanged during the B10 dual-run window.

Env:
  RESEND_API_KEY        required — server-side API key from resend.com
  RESEND_FROM_ADDRESS   optional — overrides the default `noreply@resend.dev`
                        sender. Use this after B12 (custom domain verified).

Reuses format_email_html() from core_notify so the email body is byte-identical
to the single-user version. Only the transport differs.
"""

import os
from typing import Optional, Tuple

import requests

from pipeline.logging_setup import get_logger

logger = get_logger(__name__)


RESEND_API_URL = "https://api.resend.com/emails"
DEFAULT_FROM = "Job Alerts <noreply@resend.dev>"
REQUEST_TIMEOUT_SECONDS = 30


class ResendError(RuntimeError):
    """Raised when the Resend API rejects a send.

    The caller (multi_user_runner) catches this, marks the run as failed,
    and continues with the next user instead of crashing the whole cron tick.
    """


def send_email(
    to: str,
    subject: str,
    html: str,
    *,
    text: Optional[str] = None,
    from_address: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Tuple[bool, Optional[str]]:
    """POST one email through Resend. Returns (success, message_id_or_error).

    Returns (True, message_id) on a 200/202 response.
    Returns (False, error_text) on any failure — never raises, so a single
    user's email failure can't take down the multi-user cron loop.
    """
    key = api_key or os.environ.get("RESEND_API_KEY", "")
    if not key:
        logger.error("RESEND_API_KEY not set — skipping email to %s.", _redact(to))
        return False, "RESEND_API_KEY not set"

    if not to or "@" not in to:
        logger.error("Refusing to send to invalid address: %r", to)
        return False, f"invalid recipient: {to!r}"

    sender = from_address or os.environ.get("RESEND_FROM_ADDRESS") or DEFAULT_FROM

    payload = {
        "from": sender,
        "to": [to],
        "subject": subject,
        "html": html,
    }
    if text:
        payload["text"] = text

    try:
        resp = requests.post(
            RESEND_API_URL,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as e:
        logger.error("Resend transport error sending to %s: %s", _redact(to), e)
        return False, f"transport error: {e}"

    if resp.status_code in (200, 202):
        message_id = ""
        try:
            message_id = resp.json().get("id", "") or ""
        except ValueError:
            pass
        logger.info("Resend OK -> %s (id=%s)", _redact(to), message_id or "<no id>")
        return True, message_id

    # Surface the body so 401/403 (bad key) and 422 (bad sender) are distinguishable
    # in logs. Truncated to keep CI logs readable.
    body = (resp.text or "")[:500]
    logger.error(
        "Resend rejected send to %s: %d %s",
        _redact(to), resp.status_code, body,
    )
    return False, f"HTTP {resp.status_code}: {body}"


def _redact(email: str) -> str:
    """Log a recipient address without leaking the full local-part.

    `mohaabuhijleh@gmail.com` -> `m***h@gmail.com`. We log enough to
    identify rows in audit grep but not enough to dump a usable mailing
    list into CI output if the worker logs ever leak.
    """
    if not email or "@" not in email:
        return "<invalid>"
    local, _, domain = email.partition("@")
    if len(local) <= 2:
        return f"{local[:1]}***@{domain}"
    return f"{local[0]}***{local[-1]}@{domain}"
