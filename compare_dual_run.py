"""
DUAL-RUN COMPARISON HELPER (B10)
--------------------------------
During the 7-day dual-run, the legacy single-user pipeline (scraper.py →
email + GitHub issue) and the multi-user runner (multi_user_runner.py →
Supabase) both run for Mohammad. This script answers: "is the multi-user
path producing equivalent results?"

It compares:
  * LEGACY  — the latest "Automated AI Job Alerts" GitHub issue in LOGS_REPO,
              whose markdown tables list each surfaced job (title, company,
              match %, URL).
  * MULTI   — the latest (or a chosen) run's job_results rows in Supabase.

IMPORTANT — what a "good" result looks like:
  The two pipelines scrape LIVE job boards at DIFFERENT times (legacy at
  06:55 UTC, multi-user hourly), so the raw available jobs genuinely differ
  between runs. An exact match is NOT expected and NOT the goal. What we
  validate is:
    1. Sanity — multi-user surfaced a plausible count (not 0, not 10x legacy).
    2. Dedup integrity — no duplicate URLs within the multi-user run.
    3. Score agreement — for jobs that DO appear in both (same URL), the
       match % should be close (same CV, prompt, models; small drift from
       LLM nondeterminism is fine).
  Large score gaps on overlapping jobs, or zero overlap across several days,
  are the real red flags.

CLI:
    python compare_dual_run.py --user-id <uuid>
    python compare_dual_run.py --email you@example.com --run-id 42
    python compare_dual_run.py --email you@example.com --out report.csv

Env:
    SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
    LOGS_REPO, LOGS_REPO_TOKEN
"""

import argparse
import csv
import os
import re
import sys
from typing import Optional
from urllib.parse import urlsplit, urlunsplit

import requests

from pipeline.logging_setup import configure_logging, get_logger
from pipeline.core_supabase import SupabaseConfigError, get_service_client

logger = get_logger(__name__)


LEGACY_ISSUE_TITLE_PREFIX = "Automated AI Job Alerts"
SCORE_GAP_WARN = 15  # |legacy − multi| above this on an overlapping job is flagged


# ─────────────────────────────────────────────────────────────────────────────
# Pure helpers (unit-tested)
# ─────────────────────────────────────────────────────────────────────────────

def normalize_url(url: Optional[str]) -> str:
    """Canonicalize a job URL for cross-pipeline matching.

    Job boards append per-session tracking params (utm_*, refId, trk, ...) that
    differ between two scrapes of the same posting, so query + fragment are
    dropped. Host is lowercased and a trailing slash removed. The path is the
    stable identity of a posting.
    """
    if not url:
        return ""
    s = str(url).strip()
    if not s:
        return ""
    try:
        parts = urlsplit(s if "//" in s else f"//{s}", scheme="https")
    except ValueError:
        return s.lower()
    host = (parts.netloc or "").lower()
    path = (parts.path or "").rstrip("/")
    return urlunsplit((parts.scheme or "https", host, path, "", ""))


_APPLY_RE = re.compile(r"\[[^\]]*\]\((?P<url>[^)]+)\)")
_MATCH_RE = re.compile(r"\*\*\s*(?P<pct>\d{1,3})\s*%\s*\*\*")


def extract_apply_url(cell: str) -> Optional[str]:
    """Pull the URL out of a `[Apply](https://…)` markdown link cell."""
    if not cell:
        return None
    m = _APPLY_RE.search(cell)
    return m.group("url").strip() if m else None


def parse_match_cell(cell: str) -> Optional[int]:
    """Pull the integer percent out of a `**NN%** (T:.. E:.. L:..)` cell.

    Returns None for `**N/A**` or anything without a percent.
    """
    if not cell:
        return None
    m = _MATCH_RE.search(cell)
    if not m:
        return None
    try:
        n = int(m.group("pct"))
    except ValueError:
        return None
    return n if 0 <= n <= 100 else None


def _strip_severity_prefix(title: str) -> str:
    """Drop the 🚨 / 🚫 / ⚠️ severity glyphs the renderer prepends to titles."""
    return re.sub(r"^[\U0001F6A8\U0001F6AB⚠️\s]+", "", title or "").strip()


def parse_legacy_issue_markdown(md: str) -> list:
    """Parse the job rows out of a legacy GitHub-issue body.

    Returns a list of {url, norm_url, title, company, match} dicts. Only data
    rows of the rich-schema table are kept — the header, separator, and prose
    lines are skipped. A row is recognised by having the 8 pipe-delimited
    columns AND an `[Apply](url)` link in the last cell.
    """
    out = []
    for line in (md or "").splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        # Split into cells, dropping the leading/trailing empty strings.
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) != 8:
            continue
        if cells[0].lower() == "title" or set(cells[0]) <= {"-", ":"}:
            continue  # header or separator row
        url = extract_apply_url(cells[7])
        if not url or url == "#":
            continue
        out.append({
            "url": url,
            "norm_url": normalize_url(url),
            "title": _strip_severity_prefix(cells[0]),
            "company": cells[1],
            "match": parse_match_cell(cells[3]),
        })
    return out


def compare(legacy: list, multi: list) -> dict:
    """Diff two job lists keyed on normalized URL. Pure — no I/O.

    `legacy` items have {norm_url, title, match}; `multi` items have
    {norm_url, title, match_percentage}. Returns a report dict with counts,
    the overlapping jobs (with score deltas), the symmetric differences, and
    any duplicate URLs found within each side.
    """
    legacy_by_url = {}
    legacy_dupes = []
    for j in legacy:
        u = j["norm_url"]
        if not u:
            continue
        if u in legacy_by_url:
            legacy_dupes.append(u)
        else:
            legacy_by_url[u] = j

    multi_by_url = {}
    multi_dupes = []
    for j in multi:
        u = j["norm_url"]
        if not u:
            continue
        if u in multi_by_url:
            multi_dupes.append(u)
        else:
            multi_by_url[u] = j

    overlap_urls = set(legacy_by_url) & set(multi_by_url)
    overlaps = []
    for u in overlap_urls:
        lm = legacy_by_url[u].get("match")
        mm = multi_by_url[u].get("match_percentage")
        delta = abs(lm - mm) if (lm is not None and mm is not None) else None
        overlaps.append({
            "norm_url": u,
            "title": multi_by_url[u].get("title") or legacy_by_url[u].get("title"),
            "legacy_match": lm,
            "multi_match": mm,
            "delta": delta,
            "flagged": delta is not None and delta > SCORE_GAP_WARN,
        })
    overlaps.sort(key=lambda r: (r["delta"] is None, -(r["delta"] or 0)))

    return {
        "legacy_count": len(legacy_by_url),
        "multi_count": len(multi_by_url),
        "overlap_count": len(overlap_urls),
        "only_legacy": sorted(set(legacy_by_url) - set(multi_by_url)),
        "only_multi": sorted(set(multi_by_url) - set(legacy_by_url)),
        "overlaps": overlaps,
        "legacy_dupes": legacy_dupes,
        "multi_dupes": multi_dupes,
    }


# ─────────────────────────────────────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────────────────────────────────────

def resolve_user_id(client, *, user_id, email) -> Optional[str]:
    if user_id:
        return user_id
    if not email:
        return None
    email_lc = email.strip().lower()
    try:
        users = client.auth.admin.list_users()
    except Exception as e:
        logger.critical("Could not list auth users: %s", e)
        return None
    if hasattr(users, "users"):
        users = users.users
    for u in users or []:
        u_email = getattr(u, "email", None) or (u.get("email") if isinstance(u, dict) else None)
        if u_email and u_email.strip().lower() == email_lc:
            return getattr(u, "id", None) or (u.get("id") if isinstance(u, dict) else None)
    logger.critical("No auth user with email %s.", email)
    return None


def fetch_multiuser_jobs(client, user_id: str, run_id: Optional[int]) -> tuple:
    """Return (run_meta, [job dicts]) for the chosen run (latest if run_id is None)."""
    if run_id is None:
        runs = (
            client.table("runs")
            .select("id, status, started_at, scraped, filtered, approved")
            .eq("user_id", user_id)
            .order("started_at", desc=True)
            .limit(1)
            .execute()
        ).data or []
        if not runs:
            return None, []
        run = runs[0]
        run_id = run["id"]
    else:
        rows = (
            client.table("runs")
            .select("id, status, started_at, scraped, filtered, approved")
            .eq("user_id", user_id).eq("id", run_id).limit(1).execute()
        ).data or []
        run = rows[0] if rows else {"id": run_id}

    jobs = (
        client.table("job_results")
        .select("job_url, title, company, match_percentage, ai_evaluated, is_valid")
        .eq("user_id", user_id)
        .eq("run_id", run_id)
        .eq("is_valid", True)
        .execute()
    ).data or []
    for j in jobs:
        j["norm_url"] = normalize_url(j.get("job_url"))
    return run, jobs


def fetch_legacy_issue_markdown(repo: str, token: str, date: Optional[str]) -> Optional[str]:
    """Fetch the body of the latest (or date-matching) legacy job-alert issue."""
    from pipeline.core_notify import _normalize_repo

    repo = _normalize_repo(repo)
    if not repo or not token:
        logger.critical("LOGS_REPO and LOGS_REPO_TOKEN are required to read the legacy issue.")
        return None
    url = f"https://api.github.com/repos/{repo}/issues"
    try:
        resp = requests.get(
            url,
            headers={"Authorization": f"token {token}", "Accept": "application/vnd.github+json"},
            params={"state": "all", "per_page": 30, "sort": "created", "direction": "desc"},
            timeout=30,
        )
    except requests.RequestException as e:
        logger.critical("GitHub issues fetch failed: %s", e)
        return None
    if resp.status_code != 200:
        logger.critical("GitHub issues fetch: HTTP %d %s", resp.status_code, resp.text[:200])
        return None

    candidates = [
        it for it in resp.json()
        if isinstance(it, dict)
        and "pull_request" not in it
        and str(it.get("title", "")).startswith(LEGACY_ISSUE_TITLE_PREFIX)
    ]
    if date:
        candidates = [it for it in candidates if date in str(it.get("title", ""))]
    if not candidates:
        logger.warning("No legacy '%s' issue found%s.", LEGACY_ISSUE_TITLE_PREFIX,
                       f" for {date}" if date else "")
        return None
    chosen = candidates[0]  # already sorted newest-first by the API
    logger.info("Legacy issue: #%s %r", chosen.get("number"), chosen.get("title"))
    return chosen.get("body") or ""


def write_csv(path: str, report: dict):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["bucket", "norm_url", "title", "legacy_match", "multi_match", "delta", "flagged"])
        for o in report["overlaps"]:
            w.writerow(["both", o["norm_url"], o["title"], o["legacy_match"],
                        o["multi_match"], o["delta"], o["flagged"]])
        for u in report["only_legacy"]:
            w.writerow(["legacy_only", u, "", "", "", "", ""])
        for u in report["only_multi"]:
            w.writerow(["multi_only", u, "", "", "", "", ""])
    logger.info("Wrote CSV: %s", path)


def print_report(report: dict, run_meta: Optional[dict]):
    print("\n================ DUAL-RUN COMPARISON ================")
    if run_meta:
        print(f"Multi-user run #{run_meta.get('id')} · status={run_meta.get('status')} · "
              f"started={run_meta.get('started_at')}")
        print(f"  run stats: scraped={run_meta.get('scraped')} "
              f"filtered={run_meta.get('filtered')} approved={run_meta.get('approved')}")
    print(f"Legacy jobs:      {report['legacy_count']}")
    print(f"Multi-user jobs:  {report['multi_count']}")
    print(f"Overlap (same URL): {report['overlap_count']}")
    print(f"  only in legacy:  {len(report['only_legacy'])}")
    print(f"  only in multi:   {len(report['only_multi'])}")

    if report["multi_dupes"]:
        print(f"\n  ⚠️  DUPLICATE URLs in multi-user output ({len(report['multi_dupes'])}) "
              f"— dedup may be broken:")
        for u in report["multi_dupes"][:10]:
            print(f"      {u}")
    else:
        print("\n  ✓ No duplicate URLs in the multi-user run (dedup intact).")

    if report["overlaps"]:
        print(f"\n  Score agreement on {len(report['overlaps'])} overlapping job(s) "
              f"(flagged if |Δ| > {SCORE_GAP_WARN}):")
        for o in report["overlaps"][:25]:
            flag = "  ⚠️" if o["flagged"] else ""
            print(f"      Δ={str(o['delta']):>4}  legacy={str(o['legacy_match']):>4} "
                  f"multi={str(o['multi_match']):>4}  {(o['title'] or '')[:50]}{flag}")
        flagged = [o for o in report["overlaps"] if o["flagged"]]
        if flagged:
            print(f"\n  ⚠️  {len(flagged)} overlapping job(s) disagree by more than {SCORE_GAP_WARN} points.")
    else:
        print("\n  (No overlapping URLs — expected when the two runs scraped at different times.)")
    print("====================================================\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Compare legacy vs multi-user run output (B10).")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--user-id")
    target.add_argument("--email")
    parser.add_argument("--run-id", type=int, default=None, help="Specific multi-user run; default latest.")
    parser.add_argument("--date", default=None, help="Match the legacy issue for this YYYY-MM-DD.")
    parser.add_argument("--out", default=None, help="Optional CSV output path.")
    args = parser.parse_args(argv)

    configure_logging()

    repo = os.environ.get("LOGS_REPO")
    token = os.environ.get("LOGS_REPO_TOKEN")

    try:
        client = get_service_client()
    except SupabaseConfigError as e:
        logger.critical(str(e))
        return 2

    user_id = resolve_user_id(client, user_id=args.user_id, email=args.email)
    if not user_id:
        return 2

    run_meta, mu_jobs = fetch_multiuser_jobs(client, user_id, args.run_id)
    if run_meta is None:
        logger.critical("No multi-user runs found for this user yet.")
        return 1

    legacy_md = fetch_legacy_issue_markdown(repo, token, args.date)
    legacy_jobs = parse_legacy_issue_markdown(legacy_md) if legacy_md else []
    if not legacy_jobs:
        logger.warning("No legacy jobs parsed — comparison will show multi-user side only.")

    report = compare(legacy_jobs, mu_jobs)
    print_report(report, run_meta)
    if args.out:
        write_csv(args.out, report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
