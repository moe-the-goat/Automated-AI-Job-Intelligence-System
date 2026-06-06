"""core_telegram parser tests (public Telegram channel web-preview).

Locks the pure parser: recency window, hiring-signal pre-screen (EN + Arabic),
image-only skip, apply-link preference over permalink, and field shape. No
network — feeds synthetic t.me/s preview HTML straight to parse_telegram_preview.
"""
from datetime import datetime, timezone, timedelta

from pipeline.core_telegram import (
    parse_telegram_preview,
    _strip_html,
    _first_meaningful_line,
    _extract_apply_link,
)

_NOW = datetime(2026, 5, 20, 12, 0, 0, tzinfo=timezone.utc)


def _msg(post_id, dt, text_html, *, photo=False):
    """Build one tgme_widget_message block like the real preview page."""
    photo_html = '<a class="tgme_widget_message_photo_wrap"></a>' if photo else ""
    text_block = (
        f'<div class="tgme_widget_message_text js-message_text">{text_html}</div>'
        if text_html is not None else ""
    )
    return (
        f'<div class="tgme_widget_message" data-post="{post_id}">'
        f'{photo_html}{text_block}'
        f'<time datetime="{dt.isoformat()}"></time>'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def test_strip_html_keeps_arabic_and_collapses():
    out = _strip_html("Backend &amp; Frontend<br>مطلوب مبرمج <b>الآن</b>")
    assert "Backend & Frontend" in out
    assert "مطلوب مبرمج" in out


def test_first_meaningful_line_skips_emoji_and_punctuation_lines():
    # Leading emoji/separator lines are skipped; the first line with real words wins.
    text = "🚀🔥\n———\nFront-End Developer at IzTechValley\nmore details"
    assert _first_meaningful_line(text, "fallback") == "Front-End Developer at IzTechValley"


def test_first_meaningful_line_fallback_for_emoji_only():
    assert _first_meaningful_line("🚀 🔥 ✨", "fb") == "fb"


def test_extract_apply_link_prefers_external_over_telegram():
    frag = '<a href="https://t.me/share">x</a> <a href="https://forms.gle/abc">apply</a>'
    assert _extract_apply_link(frag) == "https://forms.gle/abc"


def test_extract_apply_link_none_when_only_telegram():
    assert _extract_apply_link('<a href="https://t.me/foo/1">x</a>') is None


# ---------------------------------------------------------------------------
# parse_telegram_preview
# ---------------------------------------------------------------------------

def test_parses_english_hiring_post():
    html = _msg("ch/253", _NOW - timedelta(days=1),
                'Backend Developer Internship Opportunity at Gaza Sky Geeks. '
                '<a href="https://forms.gle/apply">Apply here</a>')
    jobs = parse_telegram_preview(html, "ch", lookback_days=7, now=_NOW)
    assert len(jobs) == 1
    j = jobs[0]
    assert j["source"] == "telegram"
    assert j["job_url"] == "https://forms.gle/apply"     # apply link preferred
    assert "Backend Developer Internship" in j["title"]
    assert j["date_posted"].startswith("2026-05-19")


def test_parses_arabic_hiring_post():
    html = _msg("ch/254", _NOW - timedelta(days=2),
                "مطلوب مطور واجهات أمامية للعمل في رام الله. للتقديم راسلونا.")
    jobs = parse_telegram_preview(html, "ch", lookback_days=7, now=_NOW)
    assert len(jobs) == 1
    assert jobs[0]["job_url"] == "https://t.me/ch/254"   # no external link -> permalink
    assert "مطور واجهات" in jobs[0]["description"]


def test_drops_post_outside_lookback():
    html = _msg("ch/100", _NOW - timedelta(days=30),
                "We are hiring a Python developer. Apply now.")
    assert parse_telegram_preview(html, "ch", lookback_days=7, now=_NOW) == []


def test_drops_non_hiring_post():
    html = _msg("ch/200", _NOW - timedelta(days=1),
                "Congratulations to our graduates on completing the bootcamp! 🎉")
    assert parse_telegram_preview(html, "ch", lookback_days=7, now=_NOW) == []


def test_skips_image_only_post():
    html = _msg("ch/300", _NOW - timedelta(days=1), None, photo=True)
    assert parse_telegram_preview(html, "ch", lookback_days=7, now=_NOW) == []


def test_multiple_posts_mixed():
    html = (
        _msg("ch/1", _NOW - timedelta(days=1), "We are hiring a Backend Engineer. Apply.")
        + _msg("ch/2", _NOW - timedelta(days=1), "Just a community update, nothing to see.")
        + _msg("ch/3", _NOW - timedelta(days=2), "مطلوب مبرمج Python. فرصة عمل.")
        + _msg("ch/4", _NOW - timedelta(days=40), "Old hiring post for a developer role.")
    )
    jobs = parse_telegram_preview(html, "ch", lookback_days=7, now=_NOW)
    # keeps #1 (EN hiring) and #3 (AR hiring); drops #2 (no signal) and #4 (too old)
    assert len(jobs) == 2
    urls = {j["job_url"] for j in jobs}
    assert "https://t.me/ch/3" in urls


def test_empty_html():
    assert parse_telegram_preview("", "ch", lookback_days=7, now=_NOW) == []
