"""
CORE EMAIL (SMTP) MODULE — multi-user transport
-----------------------------------------------
Gmail-SMTP transport for the multi-user pipeline. Replaced the former Resend
transport, which required a verified custom domain to send to arbitrary
recipients (test mode only delivers to the account owner) — a paid/domain
dependency we don't have.

Gmail SMTP with an app password sends to ANY recipient the user specifies,
needs no domain, and reuses the SENDER_EMAIL / EMAIL_APP_PASSWORD secrets the
single-user pipeline already has. Trade-off: Gmail's ~500-recipients/day cap,
which is irrelevant for a handful of known users.

This module mirrors the Resend module's signature exactly:
    send_email(to, subject, html, *, text=None, from_address=None,
               api_key=None) -> (success: bool, message_id_or_error: str|None)
so multi_user_runner only has to swap the import. `api_key` is accepted and
ignored for signature parity with the old transport.

Env:
  SENDER_EMAIL          required — the Gmail address that authenticates + sends.
  EMAIL_APP_PASSWORD    required — a Google app password for that account.
  SMTP_SERVER           optional — defaults to smtp.gmail.com.
  SMTP_PORT             optional — defaults to 465 (implicit TLS / SMTP_SSL).

Reuses format_email_html() from core_notify upstream so the body is identical;
only the transport differs.
"""

import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional, Tuple

from pipeline.logging_setup import get_logger

logger = get_logger(__name__)


DEFAULT_SMTP_SERVER = "smtp.gmail.com"
DEFAULT_SMTP_PORT = 465
SMTP_TIMEOUT_SECONDS = 30


class SmtpEmailError(RuntimeError):
    """Raised when SMTP setup is fundamentally misconfigured.

    Kept for parity with ResendError, but send_email never raises it — it
    returns (False, msg) instead so one user's failure can't crash the
    multi-user cron loop.
    """


def send_email(
    to: str,
    subject: str,
    html: str,
    *,
    text: Optional[str] = None,
    from_address: Optional[str] = None,
    api_key: Optional[str] = None,  # accepted for signature parity; unused
) -> Tuple[bool, Optional[str]]:
    """Send one email through Gmail SMTP. Returns (success, info).

    Returns (True, "") on a successful send (SMTP gives no message id).
    Returns (False, error_text) on any failure — never raises, so a single
    user's email failure can't take down the multi-user cron loop.

    `from_address` overrides the From header; otherwise the authenticated
    SENDER_EMAIL is used. (Gmail rewrites From to the authenticated account
    unless it's a verified alias, so this is mostly cosmetic.)
    """
    del api_key  # unused — present only so callers can swap transports freely

    sender_email = os.environ.get("SENDER_EMAIL", "")
    app_password = os.environ.get("EMAIL_APP_PASSWORD", "")
    if not sender_email or not app_password:
        logger.error(
            "SENDER_EMAIL / EMAIL_APP_PASSWORD not set — skipping email to %s.",
            _redact(to),
        )
        return False, "SENDER_EMAIL or EMAIL_APP_PASSWORD not set"

    if not to or "@" not in to:
        logger.error("Refusing to send to invalid address: %r", to)
        return False, f"invalid recipient: {to!r}"

    server_host = os.environ.get("SMTP_SERVER") or DEFAULT_SMTP_SERVER
    try:
        server_port = int(os.environ.get("SMTP_PORT") or DEFAULT_SMTP_PORT)
    except ValueError:
        server_port = DEFAULT_SMTP_PORT

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = from_address or sender_email
    msg["To"] = to
    # A plain-text part first, HTML second: clients render the last (richest)
    # part, but a text alternative improves deliverability and accessibility.
    if text:
        msg.attach(MIMEText(text, "plain"))
    msg.attach(MIMEText(html, "html"))

    try:
        with smtplib.SMTP_SSL(
            server_host, server_port, timeout=SMTP_TIMEOUT_SECONDS
        ) as server:
            server.login(sender_email, app_password)
            server.send_message(msg, from_addr=sender_email, to_addrs=[to])
    except smtplib.SMTPAuthenticationError as e:
        logger.error("SMTP auth failed sending to %s: %s", _redact(to), e)
        return False, f"auth error: {e}"
    except (smtplib.SMTPException, OSError) as e:
        logger.error("SMTP transport error sending to %s: %s", _redact(to), e)
        return False, f"transport error: {e}"

    logger.info("SMTP OK -> %s", _redact(to))
    return True, ""


def _redact(email: str) -> str:
    """Log a recipient address without leaking the full local-part.

    `mohaabuhijleh@gmail.com` -> `m***h@gmail.com`. Identical behavior to the
    Resend module's redactor so audit greps work across both transports.
    """
    if not email or "@" not in email:
        return "<invalid>"
    local, _, domain = email.partition("@")
    if len(local) <= 2:
        return f"{local[:1]}***@{domain}"
    return f"{local[0]}***{local[-1]}@{domain}"
