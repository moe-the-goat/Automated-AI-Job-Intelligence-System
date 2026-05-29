"""
PROVISION USER — set up a multi-user account from the service role (B9)
-----------------------------------------------------------------------
Idempotently fills the per-user rows the web onboarding UI would normally
write, for cases where the account is created directly (Supabase dashboard
"Add user") instead of going through the signup → CV → preferences flow.

What it writes (all scoped to one user_id):
  * profiles : cv_text, first_name, last_name, display_name,
               is_whitelisted, is_admin
  * preferences (upsert) : notification_email, frequency_hours,
               is_active, next_run_at
  * search_queries : one row per entry in a config.json `searches` list,
               mapping site_name -> sites. Deduped on (search_term, location)
               so re-running adds only what's missing.

It deliberately does NOT touch the historical feedback corpus — that's
migrate_to_multi_user.py (B9a), which needs the logs-repo credentials.

The dashboard's "ready" gate only needs cv_text + a preferences row with
notification_email + >=1 active search + is_active, so after this runs the
account is fully onboarded in the UI without ever visiting onboarding.

CLI:
    python provision_user.py --user-id <uuid> \
        --cv-file cv_text.txt \
        --first-name Mohammad --last-name "Abu Hijleh" \
        --notification-email results@example.com \
        --searches-config config.json --make-admin
    python provision_user.py --email you@example.com --cv-file cv.txt --dry-run

Env:
    SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from typing import Optional

from pipeline.logging_setup import configure_logging, get_logger
from pipeline.core_supabase import SupabaseConfigError, get_service_client

logger = get_logger(__name__)


VALID_JOB_TYPES = {"fulltime", "internship", "contract", "parttime"}
VALID_FREQUENCIES = {1, 24, 48, 168}


# ─────────────────────────────────────────────────────────────────────────────
# Pure helpers (unit-tested directly)
# ─────────────────────────────────────────────────────────────────────────────

def config_to_search_rows(config: dict, user_id: str) -> list:
    """Map a config.json `searches` list into search_queries insert rows.

    The single-user config key is `site_name`; the multi-user column is
    `sites` (jsonb). job_type is dropped when it isn't one of the schema's
    allowed values (the constraint permits NULL). Everything else maps 1:1.
    """
    rows = []
    for s in (config or {}).get("searches", []):
        if not isinstance(s, dict):
            continue
        term = str(s.get("search_term", "")).strip()
        if not term:
            continue
        sites = s.get("site_name") or ["linkedin", "indeed"]
        if isinstance(sites, str):
            sites = [sites]
        job_type = s.get("job_type")
        if job_type not in VALID_JOB_TYPES:
            job_type = None
        rows.append({
            "user_id": user_id,
            "search_term": term,
            "location": str(s.get("location", "Worldwide")).strip() or "Worldwide",
            "sites": list(sites),
            "job_type": job_type,
            "is_remote": bool(s.get("is_remote", True)),
            "results_wanted": _clamp_int(s.get("results_wanted", 30), 1, 100, 30),
            "hours_old": _clamp_int(s.get("hours_old", 24), 1, 720, 24),
            "country_indeed": str(s.get("country_indeed", "USA")).strip() or "USA",
            "is_active": True,
        })
    return rows


def search_key(row: dict) -> tuple:
    """Dedup key for a search row: (search_term, location), case-normalized."""
    return (
        str(row.get("search_term", "")).strip().lower(),
        str(row.get("location", "")).strip().lower(),
    )


def plan_search_inserts(rows: list, existing: list) -> list:
    """Return the subset of `rows` whose (term, location) isn't already present."""
    have = {search_key(r) for r in existing}
    out = []
    for r in rows:
        k = search_key(r)
        if k in have:
            continue
        have.add(k)  # also dedup within the incoming list
        out.append(r)
    return out


def derive_display_name(first: Optional[str], last: Optional[str]) -> Optional[str]:
    parts = [p for p in [(first or "").strip(), (last or "").strip()] if p]
    return " ".join(parts) or None


def _clamp_int(v, lo, hi, default) -> int:
    try:
        n = int(v)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, n))


# ─────────────────────────────────────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────────────────────────────────────

def resolve_user_id(client, *, user_id: Optional[str], email: Optional[str]) -> Optional[str]:
    if user_id:
        return user_id
    if not email:
        return None
    email_lc = email.strip().lower()
    try:
        users = client.auth.admin.list_users()
    except Exception as e:
        logger.critical("Could not list auth users to resolve %s: %s", email, e)
        return None
    if hasattr(users, "users"):
        users = users.users
    for u in users or []:
        u_email = getattr(u, "email", None) or (u.get("email") if isinstance(u, dict) else None)
        u_id = getattr(u, "id", None) or (u.get("id") if isinstance(u, dict) else None)
        if u_email and u_email.strip().lower() == email_lc:
            return u_id
    logger.critical("No auth user found with email %s.", email)
    return None


def update_profile(client, user_id, *, cv_text, first_name, last_name, make_admin, dry_run) -> bool:
    patch = {}
    if cv_text is not None:
        patch["cv_text"] = cv_text
        patch["cv_uploaded_at"] = datetime.now(timezone.utc).isoformat()
    if first_name is not None:
        patch["first_name"] = first_name
    if last_name is not None:
        patch["last_name"] = last_name
    display = derive_display_name(first_name, last_name)
    if display:
        patch["display_name"] = display
    patch["is_whitelisted"] = True
    if make_admin:
        patch["is_admin"] = True

    preview = {k: (f"<{len(v)} chars>" if k == "cv_text" else v) for k, v in patch.items()}
    if dry_run:
        logger.info("[dry-run] profiles update: %s", preview)
        return True
    try:
        client.table("profiles").update(patch).eq("user_id", user_id).execute()
        logger.info("profiles updated: %s", preview)
        return True
    except Exception as e:
        logger.error("profiles update failed: %s", e)
        return False


def upsert_preferences(client, user_id, *, notification_email, frequency_hours, dry_run) -> bool:
    if frequency_hours not in VALID_FREQUENCIES:
        logger.warning("frequency_hours %s not in %s — defaulting to 24.", frequency_hours, VALID_FREQUENCIES)
        frequency_hours = 24
    row = {
        "user_id": user_id,
        "notification_email": notification_email,
        "frequency_hours": frequency_hours,
        "is_active": True,
        "next_run_at": datetime.now(timezone.utc).isoformat(),
    }
    if dry_run:
        logger.info("[dry-run] preferences upsert: %s", row)
        return True
    try:
        client.table("preferences").upsert(row, on_conflict="user_id").execute()
        logger.info("preferences upserted (notify=%s, freq=%dh).", notification_email, frequency_hours)
        return True
    except Exception as e:
        logger.error("preferences upsert failed: %s", e)
        return False


def seed_searches(client, user_id, config, *, dry_run) -> dict:
    rows = config_to_search_rows(config, user_id)
    stats = {"in_config": len(rows), "inserted": 0, "skipped": 0}
    if not rows:
        return stats
    try:
        existing = (
            client.table("search_queries")
            .select("search_term, location")
            .eq("user_id", user_id)
            .execute()
        ).data or []
    except Exception as e:
        logger.error("Could not read existing searches: %s", e)
        existing = []
    to_insert = plan_search_inserts(rows, existing)
    stats["skipped"] = len(rows) - len(to_insert)
    if dry_run:
        logger.info("[dry-run] would insert %d search(es), %d already present.",
                    len(to_insert), stats["skipped"])
        return stats
    if to_insert:
        try:
            client.table("search_queries").insert(to_insert).execute()
            stats["inserted"] = len(to_insert)
        except Exception as e:
            logger.error("search_queries insert failed: %s", e)
    logger.info("searches: %d inserted, %d already present.", stats["inserted"], stats["skipped"])
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Provision a multi-user account (B9).")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--user-id", help="Target user UUID.")
    target.add_argument("--email", help="Target user email (looked up via auth admin).")
    parser.add_argument("--cv-file", help="Path to a UTF-8 text file with the user's CV.")
    parser.add_argument("--first-name", default=None)
    parser.add_argument("--last-name", default=None)
    parser.add_argument("--notification-email", required=True,
                        help="Where job-alert emails are delivered (can differ from the login email).")
    parser.add_argument("--frequency-hours", type=int, default=24, choices=sorted(VALID_FREQUENCIES))
    parser.add_argument("--searches-config", help="Path to a config.json whose `searches` seed search_queries.")
    parser.add_argument("--make-admin", action="store_true", help="Set profiles.is_admin = true.")
    parser.add_argument("--dry-run", action="store_true", help="Report what would change; write nothing.")
    args = parser.parse_args(argv)

    configure_logging()

    cv_text = None
    if args.cv_file:
        try:
            with open(args.cv_file, "r", encoding="utf-8") as f:
                cv_text = f.read().strip()
        except OSError as e:
            logger.critical("Could not read CV file %s: %s", args.cv_file, e)
            return 2
        if not cv_text:
            logger.critical("CV file %s is empty.", args.cv_file)
            return 2

    config = {}
    if args.searches_config:
        try:
            with open(args.searches_config, "r", encoding="utf-8") as f:
                config = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.critical("Could not read searches config %s: %s", args.searches_config, e)
            return 2

    try:
        client = get_service_client()
    except SupabaseConfigError as e:
        logger.critical(str(e))
        return 2

    user_id = resolve_user_id(client, user_id=args.user_id, email=args.email)
    if not user_id:
        return 2
    logger.info("Provisioning user_id=%s%s.", user_id, " [dry-run]" if args.dry_run else "")

    ok = update_profile(
        client, user_id,
        cv_text=cv_text, first_name=args.first_name, last_name=args.last_name,
        make_admin=args.make_admin, dry_run=args.dry_run,
    )
    ok = upsert_preferences(
        client, user_id,
        notification_email=args.notification_email,
        frequency_hours=args.frequency_hours,
        dry_run=args.dry_run,
    ) and ok
    if config:
        seed_searches(client, user_id, config, dry_run=args.dry_run)

    logger.info("Provisioning %s.", "preview complete" if args.dry_run else "complete")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
