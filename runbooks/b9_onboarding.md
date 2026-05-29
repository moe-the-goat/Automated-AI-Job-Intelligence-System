# Runbook — B9: Onboard a user and migrate their corpus

One-time procedure to bring a user onto the multi-user platform: sign-up →
CV → preferences → whitelist → data migration → validation. Written for
onboarding **user #1 (Mohammad)**, but the steps generalise to any user.

> **Prerequisites (must already be true):**
> - Migrations `0001`–`0006` applied in the Supabase SQL Editor.
> - GitHub repo secrets set: `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, `RESEND_API_KEY`.
> - The Multi-User workflows are **disabled** in the Actions UI (they stay off until B10).
> - The web app is deployed (Vercel) and reachable.

---

## Step 1 — Sign up

1. Open the deployed app URL.
2. Sign up with the target email (`mohaabuhijleh@gmail.com`) and confirm via the
   email link if confirmation is on.
3. The `on_auth_user_created` trigger (migration 0001) auto-creates the
   `profiles` row. Nothing to do here but verify it exists in Step 4.

## Step 2 — Upload CV

1. Go through onboarding → upload your CV (PDF).
2. Confirm `profiles.cv_text` is populated (it's parsed server-side on upload).
   The worker **skips any user with an empty `cv_text`**, so this gates everything.

## Step 3 — Set preferences + searches

Port the searches from the worker repo's `config.json` into the Preferences UI.
The field mapping is 1:1 except `site_name → sites`:

| `config.json` field | `search_queries` column |
| ------------------- | ----------------------- |
| `site_name`         | `sites`                 |
| `search_term`       | `search_term`           |
| `location`          | `location`              |
| `job_type`          | `job_type`              |
| `is_remote`         | `is_remote`             |
| `results_wanted`    | `results_wanted`        |
| `hours_old`         | `hours_old`             |
| `country_indeed`    | `country_indeed`        |

`config.json` currently holds **16 searches**. Entering them by hand is
tedious — if you'd rather seed them programmatically, see
*[Optional: seed searches from config.json](#optional-seed-searches-from-configjson)*
at the end.

Also set, in the Preferences UI:
- **Frequency** (24h is the default cadence).
- **Notification email** (defaults to your account email).

## Step 4 — Whitelist (and optionally admin) the account

Run in the Supabase **SQL Editor**. This flips the closed-beta gate on and,
optionally, grants `/admin` access (used later in B14):

```sql
update public.profiles p
   set is_whitelisted = true,
       is_admin       = true          -- optional; drop this line for a non-admin user
  from auth.users u
 where u.id = p.user_id
   and u.email = 'mohaabuhijleh@gmail.com';

-- Verify the row looks right (cv_text present, whitelisted, searches counted):
select p.user_id, p.is_whitelisted, p.is_admin,
       (p.cv_text is not null) as has_cv,
       (select count(*) from public.search_queries s
         where s.user_id = p.user_id and s.is_active) as active_searches
  from public.profiles p
  join auth.users u on u.id = p.user_id
 where u.email = 'mohaabuhijleh@gmail.com';
```

You want: `has_cv = true`, `is_whitelisted = true`, `active_searches > 0`.

## Step 5 — Local env for the migration

The migration reads the private logs repo and writes Supabase, so it needs
both sets of credentials. In your local shell (PowerShell):

```powershell
$env:SUPABASE_URL              = "https://axyuiaxchlcshcvqqnws.supabase.co"
$env:SUPABASE_SERVICE_ROLE_KEY = "<service-role key from Supabase dashboard>"
$env:LOGS_REPO                 = "<owner>/job-scrapper-logs"
$env:LOGS_REPO_TOKEN           = "<PAT with repo access to the logs repo>"
```

> The service-role key bypasses RLS — keep it out of any committed file. These
> are session-only env vars; close the shell when you're done.

## Step 6 — Dry-run the migration

Always preview first. This reads everything and reports the plan but **writes
nothing**:

```powershell
python migrate_to_multi_user.py --email mohaabuhijleh@gmail.com --dry-run
```

Read the log summary:
- `Feedback summary: {'in_log': N, 'inserted': 0, 'skipped': 0, ...}` — on a
  first run `inserted` in the dry-run shows how many *would* be inserted (it
  reports the plan size; nothing is written).
- `Reputation summary: {'rows': M, ...}`.
- `Would set profiles.feedback_count = N`.

If the email can't be resolved, the account hasn't signed up yet (Step 1).

## Step 7 — Run it for real

```powershell
python migrate_to_multi_user.py --email mohaabuhijleh@gmail.com
```

The script is **idempotent** — safe to re-run if it's interrupted. It uses
multiset dedup, so a second run inserts nothing and a half-finished run
resumes exactly where it stopped.

## Step 8 — Verify in Supabase

```sql
-- Feedback rows + how many already have embeddings (the rest backfill on the
-- next worker run via ensure_feedback_embeddings).
select
  (select count(*) from public.feedback           where user_id = :uid) as feedback_rows,
  (select count(*) from public.feedback_embeddings where user_id = :uid) as embedded_rows,
  (select feedback_count from public.profiles      where user_id = :uid) as counter,
  (select length(candidate_preferences) from public.preferences where user_id = :uid) as pref_chars,
  (select count(*) from public.reputation)                                            as reputation_rows;
```

Expect: `feedback_rows == counter`, `embedded_rows ≤ feedback_rows` (any gap
self-heals on the next run), `pref_chars > 0` if a digest existed, and
`reputation_rows` matching the three lists in `reputation.json`.

> Replace `:uid` with the user's UUID (from the Step 4 verify query).

## Step 9 — Validation dispatch (no email)

Before B10, prove the end-to-end worker path against just this user without
sending mail:

1. GitHub → Actions → **Multi-User Job Alerts** → **Run workflow**.
2. Set `dry_run = true`, `user_id = <the UUID>`, `skip_due_check = true`.
3. Run it. Confirm in Supabase that a `runs` row was created and
   `job_results` were persisted. No email should arrive (dry-run).

If that's clean, B9 is done and you're ready for **B10** (enable the hourly
cron, drop `--dry-run`, run alongside the legacy pipeline for 7 days).

---

## Rollback / re-run notes

- **Re-running the migration** is always safe (idempotent). It never deletes.
- **Undo a migration** (rare — only if you imported into the wrong account):
  ```sql
  delete from public.feedback where user_id = :uid;   -- cascades to feedback_embeddings
  update public.preferences set candidate_preferences = '' where user_id = :uid;
  update public.profiles set feedback_count = 0 where user_id = :uid;
  -- reputation is global; only clear it if this user seeded it alone:
  -- delete from public.reputation;
  ```
- **`feedback_count` looks wrong?** Re-run the migration — its final reconcile
  step does an absolute `SET` to the true row count regardless of the 0006
  trigger.

---

## Optional: seed searches from config.json

If you'd rather not hand-enter the 16 searches in the UI, ask for a
`--seed-searches <config.json>` flag on `migrate_to_multi_user.py`. It's a
small, isolated addition (read the `searches` list, map `site_name → sites`,
upsert into `search_queries` for the user). It's intentionally **not** in the
script today because the plan scopes search setup to the Preferences UI (B9),
keeping the migration focused on the historical corpus.
