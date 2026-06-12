# Web App Redesign — Consolidated Task List

> ## ✅ STATUS (2026-06-12): 37 of 38 tasks implemented and green.
>
> All of Phases 0–6 plus **W1** and **W2** are built and pass both QA gates:
> - **Web** (`job-alerts-app`): tsc ✓ · ESLint ✓ · Vitest **126/126** ✓ · `next build` ✓
> - **Worker**: `python QA/run_all.py` → **742 passed, 0 failed**
>
> **W2 shipped as mechanism (a)** — tokenized standalone page, no app login:
> migration `0012_email_feedback_tokens.sql` (hash-only token table + two
> SECURITY DEFINER RPCs), public `/f/<token>` page + `/api/email-feedback`
> route in the web app, and the worker mints a per-(user,run) token and
> embeds the "Rate today's matches" CTA in the email.
>
> **⚠ Manual activation steps (deploy order matters):**
> 1. Apply `job-alerts-app/migrations/0011_job_results_origin.sql` then
>    `0012_email_feedback_tokens.sql` in the Supabase SQL Editor.
> 2. Add `, origin` to `JOB_FIELDS` in
>    `src/app/dashboard/(workspace)/feedback/_lib/feedback-data.ts` (a marked one-line TODO).
> 3. Deploy the web app, then set the repo Variable **APP_BASE_URL** (e.g.
>    `https://<app>.vercel.app`) on the worker repo and push the worker change.
> Until then: grid shows one untagged section, emails go out without the
> feedback link — by design, nothing breaks.
>
> **Still open:** 0.2 (before/after screenshots — needs the app running).

> **Source:** the genuinely-useful ideas distilled from BOTH design research docs
> ("Redesigning Multi-User Workflow Web Design" — the JavaFX-hallucination one, ~40% useful;
> and "Multi-User Web Design Enhancement Plan" — the architecture-grounded one, ~70% useful),
> deduplicated and filtered against the REAL app (see [ARCHITECTURE.md](ARCHITECTURE.md)).
>
> **Scope:** `job-alerts-app` (Next.js 16 / React 19 / Tailwind v4 / Supabase). This is the
> WEB APP only — it does not touch the worker pipeline or the cutover.
>
> **Hard exclusions (do NOT build — both docs got these wrong):**
> - ❌ Real-time collaboration: live cursors, presence avatars, inline comments, co-editing.
>   *(Your "multi-user" = many isolated individuals. No shared data exists to collaborate on.)*
> - ❌ Recruiter/candidate/coach role modes, "Dev Mode," persona matrix-mapping.
>   *(One persona: an individual job seeker. Admin is a thin gate, not a second dashboard.)*
> - ❌ A user-facing "Run Scrape Now" button. *(Scraping is GitHub Actions cron — there is no
>   on-demand trigger the web app can fire today. Don't imply a capability you don't have.)*
> - ❌ Scraper-mechanic controls (XPath editor, proxy rotation, throttle slider, headless toggle,
>   live RAM/CPU/process monitor, "Kill Process"). *(Not user-facing in this product.)*
> - ❌ Voice/gesture/AI-command interfaces, adversarial/AI-hostile components.
>
> **Guiding tension (important):** both docs preach "Linear-style data density" AND
> "claymorphism / squishy maximalism" — these fight each other. For a data-dense job
> dashboard, **side with Linear's restraint.** Use tactile depth as a light accent, never as
> the dominant language. Clarity and scan-speed beat squish.
>
> **Sequencing note:** this is a SEPARATE, LATER track. Finish the cutover first
> (real email delivery via a verified Resend domain → disable legacy workflows). Don't polish
> a dashboard for a system that can't yet email its users.

---

## Phase 0 — Decide & guard (do before any code)

- [x] **0.1 — Agree the scope.** Confirm we're building the filtered list below, not the
  excluded items. One persona, isolated users, no collaboration.
- [ ] **0.2 — Snapshot the current UI.** Screenshot the current dashboard / results / preferences
  pages so "before vs after" is provable (honest, not invented "10x" claims).
- [x] **0.3 — Confirm no scope creep into the worker.** All tasks here are frontend; the DB
  schema already exposes everything we need (`match_percentage`, `tech_fit`,
  `experience_fit`, `logistics_fit`, `ai_verdict`, `description_excerpt`, `similarity`,
  `suspicious`, `pre_flagged_*`).

---

## Phase 1 — Design foundation (Tailwind v4 tokens, type, color)

*Both docs agree on this and it's the correct, native Tailwind-v4 way. Highest ROI, lowest risk.*

- [x] **1.1 — Typographic scale.** Add a display **serif** (e.g. Instrument Serif / Playfair) for
  page titles, empty-state messages, and the AI-verdict summary; a clean **sans** for body/UI
  labels; a **monospace** for numeric/tabular data (match %, dates, salary). Define all three
  via the `@theme` directive in `src/app/globals.css`. *(Benefit: instant "crafted" feel; mono
  aligns numeric columns so they scan cleanly.)*
- [x] **1.2 — Warm dark palette.** Replace any default/purple-gradient SaaS look with a deep
  charcoal/espresso dark base + warm cream text. Define **semantic elevation tokens**
  (`--color-surface-base`, `--color-surface-raised`, `--color-surface-recessed`) and
  **functional state colors**: sage = success/good match, amber = warning/rate-limit,
  terracotta = destructive/block. Use `color-mix()` for hover states instead of hardcoding.
  *(Benefit: differentiates from generic AI dashboards; warmth reduces eye strain.)*
- [x] **1.3 — Elevation shadow tokens (restrained).** Define soft, multi-layer shadow tokens for
  depth (`--shadow-raised`, `--shadow-recessed`) to replace harsh 1px borders. Keep subtle —
  this is "Linear depth," not heavy claymorphism. *(Benefit: hierarchy without visual noise.)*
- [x] **1.4 — Motion tokens + `motion-safe:` guard.** Define a small set of transition
  durations (150–250ms ease) and ONE optional "press" transform for buttons. Wrap ALL
  animation in Tailwind's `motion-safe:` so reduced-motion users get instant states.
  *(Benefit: feels alive but stays accessible — non-negotiable.)*
- [x] **1.5 — Monoline icon pass.** Standardize on `lucide-react` (already a dependency) with
  consistent 1.6–1.8px stroke; use `currentColor` so icons inherit text color. Reserve filled
  icons for the active sidebar item only. *(Benefit: icons read as part of the type, not
  competing clip-art.)*
- [x] **1.6 — Honest empty states.** Every empty view (no runs yet, no results this run, no
  bookmarks) gets a real, domain-specific message — never Lorem Ipsum or fake metrics.
  *(Benefit: structural integrity; both docs insist on data veracity.)*

---

## Phase 2 — Atomic components

- [x] **2.1 — Rebuild `Button`** (`src/components/ui/button.tsx`) on the new tokens via `cn()`.
  Default = raised shadow; `:active` = recessed + a *subtle* scale press (`motion-safe` only).
  *(Benefit: tactile feedback on the actions users repeat most.)*
- [x] **2.2 — Inputs / textarea / switch** (`src/components/ui/`): recessed surface, soft inner
  focus glow instead of a harsh outline ring. *(Benefit: cohesive, premium form feel.)*
- [x] **2.3 — New `ContextMenu` + `Tooltip` primitives** (build on Radix or headless
  primitives). Glassmorphic float, 1px translucent edge, clear drop shadow for z-separation.
  *(Benefit: powers the per-row actions in Phase 3 without cluttering the grid.)*
- [x] **2.4 — `MatchScore` component.** Render `match_percentage` as a visual (segmented bar or
  dot-matrix gauge) with heatmap color (sage→amber→terracotta), not a raw integer. Tooltip
  reveals the tech/experience/logistics breakdown. *(Benefit: match quality readable at a
  glance across a long list — top idea from both docs.)*
- [x] **2.5 — `Kbd` keycap badge.** Small component that renders a shortcut like a physical
  keycap, shown next to actions in menus/tooltips. *(Benefit: teaches the Phase 4 shortcuts
  ambiently, no tutorial overlay.)*

---

## Phase 3 — The Job Results screen (the core value screen)

*This is where most of the benefit lives. Today it's the feedback/results tab.*

- [x] **3.1 — Convert results to a high-density data grid.** Move from big airy cards to a
  Linear-style table: tight 4–8px cell padding, no zebra striping, no vertical borders —
  rows separated by a barely-there low-opacity line, generous horizontal breathing room.
  Columns: Title · Company · Source (icon) · Location · **MatchScore** · status. *(Benefit:
  scan 14–300 jobs/run without fatigue; the single biggest readability gain.)*
- [x] **3.1b — Split GLOBAL vs LOCAL into separate sections.** Render the results in two clearly
  separated groups — **Global / Remote** jobs and **Local (Palestinian)** jobs — each with its
  own heading and count, so it's obvious at a glance what's local vs global-remote. Drive the
  split from the new `origin` field on `job_results` (see task **W1** — the worker must persist
  whether a row came from the local sources or the global ones; today it doesn't). UI options:
  two stacked sections (recommended), a tab toggle, or a sticky group header. *(Benefit: matches
  how you actually triage — local market and global remote are different decisions; mixing them
  buries the local ones.)* **Depends on W1.**
- [x] **3.2 — Inline-expandable rows.** Click a row → it expands **in place** (no page nav),
  revealing the **`ai_verdict` justification + `description_excerpt`** in a small bento
  sub-layout (reasoning separated from raw description). Animate open with Tailwind v4
  `@starting-style` (no JS animation lib needed). *(Benefit: deep detail without losing your
  place in the list — directly maps to data you already store.)*
- [x] **3.3 — Severity / status signaling.** Use the existing `suspicious`,
  `pre_flagged_low_quality`, `pre_flagged_trusted` flags to tint a row's status subtly
  (terracotta pulse for suspicious, sage for trusted). Human-readable tooltip on hover.
  *(Benefit: surfaces the AI's risk judgment you already compute but barely show.)*
- [x] **3.4 — Collapsible secondary columns.** Use Tailwind v4 container queries to auto-collapse
  lower-priority columns (e.g. source/location) into icons on narrow/laptop widths, expanding
  on hover. *(Benefit: kills horizontal scroll on 13" screens.)*
- [x] **3.5 — Per-row actions in a context menu** (not a row of always-visible buttons): Mark
  Applied, Not Relevant, Block Company, Bookmark → write to Supabase `feedback`. Show the
  `Kbd` badge next to each. *(Benefit: keeps the grid clean; powerful actions one click away.)*
- [x] **3.6 — Optional full-screen focus toggle** for the grid (hide sidebar/nav).
  *(Benefit: deep-review mode; low effort, nice-to-have.)*

---

## Phase 4 — Keyboard-first flow (power-user feel)

- [x] **4.1 — Cmd/Ctrl+K command palette.** Highest-z glassmorphic modal, fuzzy search,
  grouped results: **Navigation** (go to Results / Tracker / Preferences), **Job actions**
  (Mark Applied / Block Company on the focused row), **Jump to job** (filter the user's
  fetched results). *(Benefit: premium, fast; the standout "this isn't a template" signal.)*
  - ⚠️ Do **not** add "Run Scrape Now" — no on-demand trigger exists.
- [x] **4.2 — Row navigation + single-key feedback.** `J`/`K` move focus down/up the grid;
  `Enter` toggles inline expand; `A` = mark applied; `B` = block company (with a confirm,
  since it's irreversible). *(Benefit: review a whole run's worth of jobs without the mouse.)*
- [x] **4.3 — Compound nav shortcuts** (Linear-style): `G` then `R` → Results, `G` then `T` →
  Tracker, `G` then `P` → Preferences. *(Benefit: instant traversal; cheap once the listener
  exists.)*
- [x] **4.4 — Ambient shortcut hints.** Show the `Kbd` badge in every context menu/tooltip so
  users discover shortcuts naturally. *(Benefit: novice→power-user without a tutorial.)*

---

## Phase 5 — Other screens (lighter touch)

- [x] **5.1 — Onboarding / CV upload.** Turn the dropzone into a recessed tactile "well" that
  responds when a file is dragged over; while `cv-parser.ts` parses, show real progress /
  extracted-field glimpses instead of a generic spinner. *(Benefit: the system feels like it's
  "reading" your CV — first impression matters.)*
- [x] **5.2 — Preferences & searches.** Apply the new tokens/components; group search-query
  cards cleanly; on save, give kinetic feedback (button "Save"→"Saving…"→success ring) instead
  of only a toast. *(Benefit: cohesive feel; confirms the write landed.)*
- [x] **5.3 — Run-status summary panel.** A small 5–9 metric strip answering "what happened
  in my last run?" from `runs` (scraped → filtered → ai_evaluated → approved, status,
  started_at, RAG vs digest mode if exposed, next_run_at). Progressive disclosure for detail.
  *(Benefit: turns the opaque cron into something the user can see and trust.)*
- [x] **5.4 — Application Tracker (Tab B) polish.** Apply tokens to the bookmark board; tasteful
  status transitions. No new features — just consistency. *(Benefit: the two tabs feel like one
  product.)*

---

## Phase 6 — Optimization, polish & QA

- [x] **6.1 — Optimistic UI for mutations.** Feedback clicks and bookmark status changes update
  the DOM immediately (row shrinks/updates) while the Supabase write resolves in the
  background; roll back on error. *(Benefit: instant feel, no network-latency stalls — the one
  legitimately useful point from the "87% fewer reads" section, minus the fabricated number.)*
- [x] **6.2 — Memoize data, avoid needless refetch.** Switching Results↔Tracker↔Preferences
  shouldn't tear down and re-subscribe to the results stream. Centralize fetched data so nav
  re-renders don't re-hit Supabase. *(Benefit: snappier app, fewer reads — real, just don't
  quote invented percentages.)*
- [x] **6.3 — Micro-interaction audit.** Sweep every hover/click: 150–250ms ease transitions,
  consistent press feedback, no jarring pop-in (use `@starting-style`). Keep it restrained.
- [x] **6.4 — Motion-safe verification.** Confirm every animation/transform degrades to an
  instant state under `prefers-reduced-motion`. *(Accessibility gate.)*
- [x] **6.5 — Update the test suite.** Extend the Vitest + RTL suites (`QA/run_all.mjs`,
  `vitest.config.ts`) for the new DOM: expandable rows render, command palette portal mounts,
  keyboard handlers fire, MatchScore renders for 0 and 100. *(Matches the project's QA gate
  discipline — verify before commit.)*
- [x] **6.6 — RLS sanity re-check.** Confirm none of the new client-side state or fetching
  weakens row isolation — the grid/palette must only ever read the caller's own rows.
  *(Benefit: the new UI must not become a data-leak vector.)*
- [x] **6.7 — "No-slop" final audit.** No fake browser chrome around any panel; no Lorem
  Ipsum; no fabricated metrics; no left-bordered "AI tile" cards; honest empty states; one
  consistent icon set. *(Benefit: the whole point — looks crafted, not generated.)*

---

## Phase W — Worker-side prerequisites (CROSS-REPO — touches `multi_user_runner.py`, not just the web app)

> ⚠️ These are NOT frontend tasks. They change the Python worker and the Supabase schema.
> They're listed here because two of your requested features depend on them. Each is a real
> pipeline change with its own QA gate (`python QA/run_all.py`) — sequence them like cutover
> work, not like CSS.

- [x] **W1 — Persist a `local` vs `global` origin on each result.** *(Prereq for 3.1b.)*
  Today `multi_user_runner.py` merges global jobs and shared local jobs into one DataFrame
  before scoring, and `_jobs_to_rows` does **not** record which source a row came from. To
  split them in the UI we need that provenance saved.
  - Add an `origin` column to `public.job_results` (e.g. `text check (origin in
    ('global','local'))`) via a new idempotent migration in `job-alerts-app/migrations/`,
    ending with `notify pgrst, 'reload schema';`.
  - In `_run_for_user`, tag rows before the merge: global jobs `origin='global'`, local-cache
    jobs `origin='local'`. (The local raw jobs already carry a `source` field like
    `ddg_linkedin` / `telegram` / `jobs_ps` from `core_local_sources.py` — collapse any of
    those to `local`.) Make sure the tag survives the URL-dedup merge.
  - Persist it in `_jobs_to_rows` / `_persist_job_results`.
  - Surface `origin` in the web app's results query (`feedback-data.ts`) so 3.1b can group on it.
  - Add a QA test that a local-sourced row lands with `origin='local'` and a global one with
    `'global'`.
  *(Benefit: the UI can finally separate local vs global-remote — the whole point of 3.1b.)*

- [x] **W2 — Email-embedded feedback page (phone-friendly, port the legacy feature to multi-user).**
  *(Shipped 2026-06-12 as mechanism (a): tokenized standalone page. See the STATUS block at the top
  for the activation steps — migration 0012 + APP_BASE_URL repo variable.)*
  *(Your explicit request: keep the "give feedback from a page linked in the email" flow, which
  is easier on a phone than opening the app.)*
  - **Context / honest caveat:** this is a **legacy-only** feature today. `scraper.py` /
    `local_companies.py` render `docs/feedback_*.html` via `core_feedback_page.py` and submit
    through the Cloudflare Worker (`feedback_worker/worker.js`) into the old GitHub-issue flow.
    `multi_user_runner.py` does **none** of this — it has no feedback page at all. So "keep it"
    really means **build it into the multi-user pipeline, per-user, writing back to Supabase
    `feedback`** (not the legacy GitHub flow).
  - Decide the mechanism (pick one):
    - **(a) Reuse `core_feedback_page.py`** to render a per-run, per-user static page, host it
      somewhere public (e.g. the existing GitHub Pages `docs/` or a Supabase Storage URL), and
      point the submit at a small endpoint that writes to Supabase scoped to that user. Needs a
      **signed/tokenized link** so feedback can't be forged for another user (no app login on
      the page). **Recommended** for true "no app needed" phone use.
    - **(b) Make the email's feedback buttons deep-link into the web app** (e.g.
      `/dashboard/...feedback?job=<id>&action=applied`) so a tap opens the app pre-filled.
      Simpler + reuses RLS/auth, but it *does* require being logged into the app on the phone.
  - If (a): add a tokenized feedback endpoint (web app route or the Cloudflare Worker)
    that validates the token → writes one row to `feedback` for the right `user_id` →
    append-only, RLS-safe. Reuse the embedding/RAG path so this feedback still trains the AI.
  - Wire `multi_user_runner.py`'s email step (`format_email_html`) to include the per-user
    feedback links (global + local sections).
  - QA: token validation (can't write to another user's feedback), page renders this run's
    jobs, a submit creates exactly one `feedback` row.
  *(Benefit: feedback from the phone without opening/logging into the app — your stated, real
  workflow; preserves the convenience the legacy pipeline had.)*

---

## Quick reference — value vs. effort

| Task cluster | User-visible benefit | Effort | Priority |
|---|---|---|---|
| Phase 1 (tokens/type/color) | Whole app looks crafted, warmer, on-brand | Low | **Do first** |
| 3.1–3.2 (dense grid + inline rows) | Core screen far easier to scan + read | Med | **Highest payoff** |
| 2.4 / 3.3 (MatchScore + severity) | AI judgment readable at a glance | Low-Med | **High** |
| 4.1–4.2 (Cmd+K + J/K/A) | Premium power-user review flow | Med | High |
| 6.1–6.2 (optimistic + memoized) | Feels instant | Med | High |
| 5.1–5.3 (onboarding/prefs/run-status) | Trust + first impression | Med | Medium |
| 3.4 / 3.6 / 5.4 (column collapse, full-screen, tracker polish) | Comfort + consistency | Low-Med | Medium |
| 6.3–6.7 (audit/QA/RLS) | Quality + safety | Med | **Gate before ship** |
| **W1 + 3.1b (local/global split)** | See local vs global-remote at a glance | Med (worker+migration+UI) | **High — your request** |
| **W2 (email feedback page)** | Give feedback from the phone, no app login | Med-High (worker+endpoint+token) | **High — your request** |

> Note: **W1** and **W2** are cross-repo (worker + schema), so they carry more risk/effort than
> the pure-CSS frontend tasks and must pass `python QA/run_all.py` before commit. Their *visible*
> benefit is high because you asked for both specifically.

---

## Explicitly dropped (and why) — so we don't re-litigate

| Dropped idea | From which doc | Why |
|---|---|---|
| Live cursors / presence avatars / inline comments / co-editing | Doc 2, Phase 5 | No shared data; users are isolated. |
| Recruiter/candidate/coach role modes, persona matrix-mapping | Doc 2 | One persona only. |
| "Run Scrape Now" command | Doc 2, Phase 4 | Scraping is cron-only; no on-demand trigger. |
| XPath editor, proxy/throttle/headless controls, RAM/CPU monitor, Kill Process | Doc 1 | Not user-facing; describes an imaginary scraper console. |
| Heavy claymorphism/neumorphism "squishy everything," film grain on all containers | Both | Fights data-density; becomes its own slop. Keep tactile as a light accent only. |
| Voice/gesture/AI-command + adversarial/AI-hostile UI | Doc 2 | Filler / counterproductive. |
| Fabricated metrics ("$50M rebellion," "10x," "87%", "82% dark-mode adoption") | Both | Use the techniques, ignore the invented numbers. |
