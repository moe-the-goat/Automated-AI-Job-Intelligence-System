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
from email.utils import formataddr, formatdate, make_msgid
from typing import Optional, Tuple

from pipeline.logging_setup import get_logger

logger = get_logger(__name__)


DEFAULT_SMTP_SERVER = "smtp.gmail.com"
DEFAULT_SMTP_PORT = 465
SMTP_TIMEOUT_SECONDS = 30

# The friendly name shown instead of a bare address. A recognizable sender is
# one of the cheapest deliverability wins there is — filters and humans both
# treat "Job Alerts <...>" better than a raw gmail address.
SENDER_DISPLAY_NAME = "Job Alerts"


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
    # Display name + Reply-To: a bare address reads as machine-generated bulk.
    msg["From"] = from_address or formataddr((SENDER_DISPLAY_NAME, sender_email))
    msg["To"] = to
    msg["Reply-To"] = sender_email
    # Date and Message-ID are REQUIRED by RFC 5322. Gmail backfills them, but a
    # message that arrives without them is scored as malformed by some filters,
    # so we set them ourselves rather than relying on the relay.
    msg["Date"] = formatdate(localtime=True)
    msg["Message-ID"] = make_msgid(domain=sender_email.rpartition("@")[2] or None)
    # Marks the mail as machine-generated so recipients' auto-responders don't
    # reply to it (a reply loop looks like spam traffic from our address).
    msg["Auto-Submitted"] = "auto-generated"
    # A working unsubscribe path is one of the strongest "this is legitimate
    # bulk mail" signals to Gmail. mailto: needs no endpoint and always works;
    # the app link lets people pause delivery themselves. Deliberately NOT
    # advertising List-Unsubscribe-Post — that promises a one-click POST
    # endpoint, and claiming it without one is worse than omitting it.
    unsub = [f"<mailto:{sender_email}?subject=unsubscribe>"]
    app_base = os.environ.get("APP_BASE_URL", "").strip().rstrip("/")
    if app_base.startswith("http"):
        unsub.append(f"<{app_base}/preferences>")
    msg["List-Unsubscribe"] = ", ".join(unsub)

    # A plain-text part first, HTML second: clients render the last (richest)
    # part, but a text alternative improves deliverability and accessibility.
    # HTML-only mail trips MIME_HTML_ONLY in every mainstream filter, so fall
    # back to a stripped rendering of the HTML rather than sending none.
    # Charset is left to MIMEText: it picks us-ascii (readable, 7bit) when the
    # body allows and utf-8 only when it must — Arabic local listings included.
    msg.attach(MIMEText(text or _html_to_text(html), "plain"))
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


def _html_to_text(html: str) -> str:
    """Last-resort text/plain rendering of an HTML body.

    Callers should pass a purpose-written `text=`; this only exists so a caller
    that forgets one still produces a multipart/alternative message instead of
    an HTML-only one. Keeps link targets, since a text part whose links are
    invisible is useless to whoever is reading it.
    """
    import re
    from html import unescape

    if not html:
        return ""
    out = re.sub(r"(?is)<(script|style|head)\b.*?</\1>", " ", html)
    # Surface hrefs as "label (url)" before the tags are stripped away.
    out = re.sub(
        r'(?is)<a\b[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>',
        lambda m: f"{re.sub(r'(?s)<[^>]+>', '', m.group(2)).strip()} ({m.group(1)})",
        out,
    )
    out = re.sub(r"(?i)<br\s*/?>", "\n", out)
    out = re.sub(r"(?i)</(p|div|tr|h[1-6]|li)>", "\n", out)
    out = re.sub(r"(?s)<[^>]+>", " ", out)
    out = unescape(out)
    out = re.sub(r"[ \t ]+", " ", out)
    out = re.sub(r"\n\s*\n\s*\n+", "\n\n", out)
    return "\n".join(line.strip() for line in out.splitlines()).strip()


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
