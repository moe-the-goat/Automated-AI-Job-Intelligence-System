"""
CORE TELEGRAM MODULE
--------------------
Reads job posts from PUBLIC Telegram channels via their web-preview page
(https://t.me/s/<channel>) — plain public HTML, no login, no bot token, no API
key, no ToS gray area. This is the cleanest local-jobs source we have: Telegram
job channels are *built* to be read, unlike LinkedIn feed posts.

Each channel message becomes one job dict shaped like the rest of the pipeline:
    {title, company, location, job_url, description, job_type, source, date_posted}

Design notes:
  * `job_url` is the stable per-post permalink (t.me/<channel>/<id>) parsed from
    the `data-post` attribute — perfect for seen_jobs dedup and the email's
    "open job" link.
  * Posts are free-form text (often Arabic/English mixed, sometimes image-only).
    We pull the first non-trivial line as a provisional title and keep the full
    text as the description; the AI verdict does the real relevance judgement.
  * Image-only posts with no caption are skipped (nothing to evaluate).
  * Recency is enforced from the post's <time datetime=...> against a lookback
    window, mirroring the rest of the local pipeline.

This source is consumed by local_companies.py and runs through the LOCAL filter
path (local=True), which is Arabic-safe — the global English-only language filter
would otherwise drop every Arabic post before the AI sees it.
"""

import re
from datetime import datetime, timezone, timedelta
from html import unescape
from typing import List, Dict, Optional

import requests

from pipeline.logging_setup import get_logger

logger = get_logger(__name__)


TELEGRAM_PREVIEW_BASE = "https://t.me/s/"
HTTP_TIMEOUT = 20
USER_AGENT = "Mozilla/5.0 (compatible; JobIntelligenceBot/1.0)"

# Hiring-signal keywords (English + Arabic) — a cheap pre-screen so we don't feed
# the AI every announcement/event post in a mixed channel. The AI still judges
# the survivors; this just trims obvious non-jobs. Arabic: وظيفة (job), توظيف
# (hiring), مطلوب (wanted), فرصة (opportunity), تدريب (training), شاغر (vacancy).
_HIRING_SIGNALS = [
    "hiring", "vacancy", "vacancies", "job", "jobs", "position", "opening",
    "opportunit", "internship", "intern", "developer", "engineer", "we are looking",
    "we're looking", "apply", "recruit", "train to hire", "training program",
    "وظيفة", "وظائف", "توظيف", "مطلوب", "فرصة", "فرص", "تدريب", "شاغر", "شواغر",
]


def _strip_html(fragment: str) -> str:
    """Turn a message-text HTML fragment into clean plain text.

    <br> and block tags become spaces/newlines; entities are unescaped; runs of
    whitespace collapse. Keeps Arabic + emoji intact.
    """
    if not fragment:
        return ""
    text = re.sub(r"<br\s*/?>", "\n", fragment, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = unescape(text)
    # collapse spaces but preserve line breaks for title extraction
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


def _first_meaningful_line(text: str, fallback: str) -> str:
    """Pick a provisional job title: the first line with real words.

    Skips lines that are just emoji/punctuation/hashtags. Caps length so a wall
    of text doesn't become a giant title. Falls back to a channel-based label
    for caption-less posts.
    """
    for raw in text.split("\n"):
        line = raw.strip(" \t•—-–*=>#").strip()
        # need at least a few letter characters (Latin or Arabic)
        if len(re.sub(r"[^A-Za-z؀-ۿ]", "", line)) >= 4:
            return line[:140]
    return fallback


def _extract_apply_link(fragment: str) -> Optional[str]:
    """Return the first outbound application link in a message, if any.

    Prefers a real apply target (forms, lnkd.in, company sites) over the post's
    own permalink. Returns None when the post has no link (then the caller uses
    the t.me permalink as job_url).
    """
    hrefs = re.findall(r'href="(https?://[^"]+)"', fragment or "")
    for h in hrefs:
        # skip telegram-internal + share links
        if "t.me/" in h or "telegram." in h:
            continue
        return h
    return None


def parse_telegram_preview(html: str, channel: str, lookback_days: int,
                           now: Optional[datetime] = None) -> List[Dict]:
    """Parse a t.me/s/<channel> preview page into job dicts. Pure — no I/O.

    Filters to posts that (a) are within the lookback window, (b) carry a hiring
    signal, and (c) have evaluable text (not image-only). Order preserved.
    """
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=lookback_days)
    out: List[Dict] = []

    # Each message is a <div class="tgme_widget_message ..." data-post="ch/ID">...
    # We split on the wrapper so text, time, and links stay grouped per message.
    blocks = re.split(r'(?=<div class="tgme_widget_message[ "])', html)
    for block in blocks:
        post_m = re.search(r'data-post="([^"]+)"', block)
        if not post_m:
            continue
        post_id = post_m.group(1)  # e.g. "fromcodetocareer/253"

        # Recency from <time datetime="...">
        time_m = re.search(r'<time[^>]*datetime="([^"]+)"', block)
        post_dt = None
        if time_m:
            try:
                post_dt = datetime.fromisoformat(time_m.group(1))
                if post_dt.tzinfo is None:
                    post_dt = post_dt.replace(tzinfo=timezone.utc)
            except ValueError:
                post_dt = None
        if post_dt and post_dt < cutoff:
            continue

        # Message text fragment
        text_m = re.search(
            r'class="tgme_widget_message_text[^"]*"[^>]*>(.*?)</div>', block, re.S
        )
        fragment = text_m.group(1) if text_m else ""
        text = _strip_html(fragment)
        if not text:
            continue  # image-only / caption-less — nothing for the AI to read

        low = text.lower()
        if not any(sig in low for sig in _HIRING_SIGNALS):
            continue  # not a hiring post

        permalink = f"https://t.me/{post_id}"
        apply_link = _extract_apply_link(fragment)
        title = _first_meaningful_line(text, fallback=f"Telegram job ({channel})")

        out.append({
            "title": title,
            "company": "",                      # free-form posts rarely name a clean company
            "location": "Palestine/Remote",
            "job_url": apply_link or permalink,  # prefer the apply target, else the post
            "description": text[:4000],
            "job_type": "fulltime",
            "source": "telegram",
            "date_posted": post_dt.isoformat() if post_dt else "",
        })
    return out


def fetch_telegram_jobs(channels, lookback_days: int = 7) -> List[Dict]:
    """Fetch + parse job posts from one or more public Telegram channels.

    `channels` is a channel handle ("fromcodetocareer") or a list of them.
    Never raises — a failed channel is logged and skipped so one bad handle
    can't take down the run. Returns a combined list of job dicts.
    """
    if isinstance(channels, str):
        channels = [channels]

    all_jobs: List[Dict] = []
    for ch in channels:
        ch = str(ch).strip().lstrip("@").strip("/")
        if not ch:
            continue
        # Accept full URLs too: pull the handle out of t.me/<handle> or t.me/s/<handle>
        m = re.search(r"t\.me/(?:s/)?([A-Za-z0-9_]+)", ch)
        if m:
            ch = m.group(1)

        url = f"{TELEGRAM_PREVIEW_BASE}{ch}"
        try:
            resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=HTTP_TIMEOUT)
        except requests.RequestException as e:
            logger.warning("Telegram fetch failed for %s: %s", ch, e)
            continue
        if resp.status_code != 200:
            logger.warning("Telegram %s: HTTP %d (channel private or renamed?)", ch, resp.status_code)
            continue

        jobs = parse_telegram_preview(resp.text, ch, lookback_days)
        logger.info("Telegram %s: %d hiring post(s) within %dd.", ch, len(jobs), lookback_days)
        all_jobs.extend(jobs)

    return all_jobs
