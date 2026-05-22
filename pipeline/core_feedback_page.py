import html
import json
import os
from datetime import datetime, timezone

from pipeline.logging_setup import get_logger

logger = get_logger(__name__)

"""
CORE FEEDBACK PAGE MODULE
-------------------------
Renders the static feedback HTML page committed to `docs/` after every
pipeline run. The page lists every job that reached the email — top section
plus lower-ranked — with a feedback dropdown and an optional note field per
job. When the user hits "Submit All", embedded JavaScript calls the GitHub
Contents API to write `data/feedback_pending.json` into the private logs
repo. Tomorrow's pipeline reads that file, applies the signals, and archives
the entries.

The embedded token is a fine-grained PAT scoped to a single file path in a
single private repo — the worst case for token leakage is a stranger
overwriting your pending feedback with garbage, which the ingestion
sanitizer then drops. No information is exposed by the page being public.
"""

FEEDBACK_OPTIONS = [
    ("", "— Select feedback —"),
    ("applied", "Applied"),
    ("bookmarked", "Bookmarked"),
    ("not_relevant", "Not relevant"),
    ("block_company", "Block company"),
    ("wrong_location", "Wrong location"),
    ("other", "Other"),
]


def _match_class(pct):
    """Color class for the match badge (mirrors the email's three-tier scheme)."""
    try:
        n = int(pct)
    except (TypeError, ValueError):
        return "match-grey"
    if n >= 85:
        return "match-green"
    if n >= 70:
        return "match-amber"
    if n >= 0:
        return "match-red"
    return "match-grey"


def _render_job_block(row, date_str):
    title = html.escape(str(row.get("title", "Untitled")))
    company = html.escape(str(row.get("company", "Unknown")))
    location = html.escape(str(row.get("location", "Remote / Unspecified")))
    verdict = html.escape(str(row.get("ai_verdict", "")))
    job_url = html.escape(str(row.get("job_url", "#")))
    pct = row.get("match_percentage", "N/A")
    badge_cls = _match_class(pct)
    pct_label = f"{pct}%" if str(pct).upper() != "N/A" else "N/A"

    options_html = "\n".join(
        f'<option value="{html.escape(val)}">{html.escape(label)}</option>'
        for val, label in FEEDBACK_OPTIONS
    )

    return (
        f'<div class="job" '
        f'data-url="{job_url}" '
        f'data-company="{company}" '
        f'data-title="{title}" '
        f'data-location="{location}" '
        f'data-date="{date_str}">'
        '<div class="job-header">'
        '<div class="job-info">'
        f'<div class="job-title"><a href="{job_url}" target="_blank" rel="noopener">{title}</a></div>'
        f'<div class="job-company">{company}</div>'
        f'<div class="job-location">{location}</div>'
        '</div>'
        f'<span class="match {badge_cls}">{pct_label}</span>'
        '</div>'
        f'<div class="verdict">{verdict}</div>'
        '<div class="feedback-row">'
        f'<select class="feedback-select">{options_html}</select>'
        '</div>'
        '<textarea class="feedback-note" placeholder="Optional note explaining your choice"></textarea>'
        '</div>'
    )


def render_feedback_page(*, dfs, logs_repo, write_token, page_title="Job Feedback", date_str=None):
    """Render the full feedback HTML page from one or more job DataFrames.

    `dfs` is an iterable of `pandas.DataFrame` (typically [internships, jobs,
    lower_ranked]). Empty or None entries are skipped silently. Returns the
    full HTML string ready to be written to `docs/`.
    """
    date_str = date_str or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    blocks = []
    for df in dfs:
        if df is None or getattr(df, "empty", True):
            continue
        for _, row in df.iterrows():
            blocks.append(_render_job_block(row, date_str))

    job_blocks_html = (
        "\n".join(blocks)
        if blocks
        else '<p class="empty">No jobs in today\'s pipeline run.</p>'
    )

    token_safe = (write_token or "").strip()
    logs_repo_safe = (logs_repo or "").strip()

    return (
        _TEMPLATE
        .replace("__PAGE_TITLE__", html.escape(page_title))
        .replace("__DATE__", html.escape(date_str))
        .replace("__JOB_BLOCKS__", job_blocks_html)
        .replace("__TOKEN__", json.dumps(token_safe))
        .replace("__LOGS_REPO__", json.dumps(logs_repo_safe))
        .replace("__PENDING_PATH__", json.dumps("data/feedback_pending.json"))
    )


def write_feedback_page(path, page_html):
    """Persist the rendered HTML to disk, creating parent directories as needed."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(page_html)
    logger.info("Feedback page written to %s (%d job blocks).", path, page_html.count('class="job"'))


_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>__PAGE_TITLE__ — __DATE__</title>
<style>
  :root { color-scheme: light dark; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
    max-width: 900px; margin: 0 auto; padding: 32px 20px 120px;
    color: #1f2937; background: #f9fafb; line-height: 1.5;
  }
  h1 { margin: 0 0 4px; font-size: 26px; color: #111827; }
  .meta { color: #6b7280; font-size: 14px; margin-bottom: 24px; }
  .job {
    background: #ffffff; border: 1px solid #e5e7eb; border-radius: 10px;
    padding: 16px 18px; margin-bottom: 14px; box-shadow: 0 1px 2px rgba(0,0,0,0.04);
  }
  .job-header { display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; }
  .job-info { flex: 1; min-width: 0; }
  .job-title { font-size: 17px; font-weight: 600; }
  .job-title a { color: #1d4ed8; text-decoration: none; }
  .job-title a:hover { text-decoration: underline; }
  .job-company { color: #374151; font-size: 14px; margin-top: 2px; }
  .job-location { color: #6b7280; font-size: 13px; margin-top: 2px; }
  .match {
    padding: 4px 10px; border-radius: 6px; color: #ffffff;
    font-size: 13px; font-weight: 600; white-space: nowrap; flex-shrink: 0;
  }
  .match-green { background: #22c55e; }
  .match-amber { background: #eab308; }
  .match-red   { background: #ef4444; }
  .match-grey  { background: #9ca3af; }
  .verdict { color: #374151; font-size: 14px; margin: 12px 0; white-space: pre-wrap; }
  .feedback-row { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 8px; }
  select, textarea {
    font-family: inherit; font-size: 14px; padding: 8px 10px;
    border-radius: 6px; border: 1px solid #d1d5db; background: #ffffff; color: #1f2937;
  }
  select:focus, textarea:focus { outline: 2px solid #2563eb; outline-offset: -1px; border-color: #2563eb; }
  textarea { width: 100%; box-sizing: border-box; min-height: 52px; margin-top: 8px; resize: vertical; }
  .empty { color: #6b7280; font-style: italic; }
  .submit-bar {
    position: fixed; bottom: 0; left: 0; right: 0;
    background: #ffffff; border-top: 1px solid #e5e7eb;
    padding: 14px 20px; text-align: center; box-shadow: 0 -2px 8px rgba(0,0,0,0.06);
  }
  button {
    background: #2563eb; color: #ffffff; padding: 10px 28px;
    border: none; border-radius: 6px; font-size: 15px; font-weight: 600; cursor: pointer;
  }
  button:hover:not(:disabled) { background: #1d4ed8; }
  button:disabled { background: #9ca3af; cursor: not-allowed; }
  .status { margin-left: 16px; font-weight: 600; font-size: 14px; }
  .status.success { color: #16a34a; }
  .status.error { color: #dc2626; }
  .status.working { color: #6b7280; }
  @media (prefers-color-scheme: dark) {
    body { color: #e5e7eb; background: #0f172a; }
    h1 { color: #f1f5f9; }
    .meta { color: #94a3b8; }
    .job { background: #1e293b; border-color: #334155; box-shadow: none; }
    .job-title a { color: #60a5fa; }
    .job-company { color: #cbd5f5; }
    .job-location { color: #94a3b8; }
    .verdict { color: #cbd5f5; }
    select, textarea { background: #0f172a; color: #e5e7eb; border-color: #475569; }
    .submit-bar { background: #1e293b; border-color: #334155; }
  }
</style>
</head>
<body>
  <h1>__PAGE_TITLE__</h1>
  <div class="meta">
    Generated __DATE__ · Pick a feedback type for any job you want to influence future scoring on. Skip the rest.
  </div>

  <div id="jobs">
    __JOB_BLOCKS__
  </div>

  <div class="submit-bar">
    <button id="submit-btn">Submit All Feedback</button>
    <span id="status" class="status"></span>
  </div>

<script>
const TOKEN = __TOKEN__;
const REPO = __LOGS_REPO__;
const PENDING_PATH = __PENDING_PATH__;

function utf8ToBase64(str) {
  const bytes = new TextEncoder().encode(str);
  let binary = "";
  for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
  return btoa(binary);
}

async function getCurrentSha() {
  const url = `https://api.github.com/repos/${REPO}/contents/${PENDING_PATH}`;
  const r = await fetch(url, {
    headers: {
      "Authorization": `Bearer ${TOKEN}`,
      "Accept": "application/vnd.github.v3+json",
    },
  });
  if (r.status === 404) return null;
  if (!r.ok) throw new Error(`Read failed (${r.status})`);
  const data = await r.json();
  return data.sha || null;
}

function collectEntries() {
  const out = [];
  document.querySelectorAll(".job").forEach((el) => {
    const fb = el.querySelector(".feedback-select").value;
    if (!fb) return;
    const note = el.querySelector(".feedback-note").value.trim();
    out.push({
      job_url: el.dataset.url,
      company: el.dataset.company,
      title: el.dataset.title,
      location: el.dataset.location,
      feedback: fb,
      note: note,
      date: el.dataset.date,
    });
  });
  return out;
}

async function submitFeedback() {
  const btn = document.getElementById("submit-btn");
  const status = document.getElementById("status");
  btn.disabled = true;
  status.textContent = "Submitting…";
  status.className = "status working";

  if (!TOKEN || !REPO) {
    status.textContent = "Feedback page not configured (missing token or repo).";
    status.className = "status error";
    btn.disabled = false;
    return;
  }

  const entries = collectEntries();
  if (entries.length === 0) {
    status.textContent = "No feedback selected.";
    status.className = "status error";
    btn.disabled = false;
    return;
  }

  try {
    const sha = await getCurrentSha();
    const body = { entries: entries };
    const content = utf8ToBase64(JSON.stringify(body, null, 2));
    const url = `https://api.github.com/repos/${REPO}/contents/${PENDING_PATH}`;
    const payload = {
      message: `Submitted ${entries.length} feedback entries from feedback page`,
      content: content,
    };
    if (sha) payload.sha = sha;
    const r = await fetch(url, {
      method: "PUT",
      headers: {
        "Authorization": `Bearer ${TOKEN}`,
        "Accept": "application/vnd.github.v3+json",
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });
    if (!r.ok) {
      const text = await r.text();
      throw new Error(`Write failed (${r.status}): ${text.slice(0, 140)}`);
    }
    status.textContent = `Submitted ${entries.length} entr${entries.length === 1 ? "y" : "ies"}. They'll apply on the next pipeline run.`;
    status.className = "status success";
  } catch (err) {
    status.textContent = `Error: ${err.message}`;
    status.className = "status error";
    btn.disabled = false;
  }
}

document.getElementById("submit-btn").addEventListener("click", submitFeedback);
</script>
</body>
</html>
"""
