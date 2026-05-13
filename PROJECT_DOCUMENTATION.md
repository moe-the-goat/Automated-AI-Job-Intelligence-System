# 🧠 Automated AI Job Intelligence System: Deep Documentation

## 🎯 1. Project Goal & Philosophy
The core objective of this project is to build an **autonomous, highly intelligent recruitment pipeline** that completely replaces the grueling process of manual job searching.

Instead of a standard "dumb scraper" that bombards the user with thousands of irrelevant or non-remote jobs, this system operates like an elite human recruiter. It scrapes jobs globally and locally, aggressively filters out garbage, performs deep web searches to verify ambiguous geographic restrictions (specifically ensuring remote eligibility for a candidate in Palestine/Middle East), and uses Google's Gemini AI to mathematically score each job against the candidate's CV.

The ultimate output is a clean, daily digest of **highly curated, highly probable job matches** delivered directly to the user's Email and GitHub Issues.

---

## 🏗️ 2. High-Level Architecture
The system uses a **Modular Hybrid Architecture** — two independent workflows that share the same underlying "brain" and "filters".

1. **Global Remote Pipeline (`scraper.py`)**: Runs daily to hunt for global remote opportunities across major platforms and APIs.
2. **Local Companies Pipeline (`local_companies.py`)**: Runs daily to deeply scan specific Palestinian IT companies via their LinkedIn posts and websites.

Both pipelines share a single **seen-jobs tracker** persisted via GitHub Actions cache, so a job evaluated by one is automatically skipped by the other.

### The Core Modules
*   **`core_search.py`** — The Hunter. Pulls raw job data from JobSpy (LinkedIn, Indeed) and 4 free public APIs (Remotive, Arbeitnow, Jobicy, RemoteOK).
*   **`core_filter.py`** — The Bouncer. Deterministic rules + the `JobTracker` class that drops URLs we've already evaluated.
*   **`core_ai.py`** — The Brain. DuckDuckGo deep-search + Gemini 3.1 Flash Lite evaluation, with 3-attempt retry on transient 5xx errors.
*   **`core_notify.py`** — The Presenter. HTML/Markdown formatting, email dispatch, GitHub Issue creation, automated stale-issue cleanup.

---

## 🌍 3. The Global Pipeline (`scraper.py`)

### A. The Search Strategy
The `config.json` uses **8 laser-focused search queries** (e.g., `"junior software engineer"`, `"AI intern"`).
Sources:
1. **JobSpy Engine:** LinkedIn, Indeed. (Glassdoor was removed — JobSpy's Glassdoor connector errors on every call.)
2. **Free Public APIs:** Remotive, Arbeitnow, Jobicy, RemoteOK.

### B. Per-Source Recency Windows
- **JobSpy (LinkedIn/Indeed):** `hours_old=24` (server-side filter). LinkedIn already pre-filters to the last 24 h.
- **Public APIs:** `hours_old=72` (client-side filter inside `filter_api_jobs`). The APIs don't support server-side date filtering — they dump full inventory — so we look back further to surface more non-LinkedIn jobs.

### C. The Deterministic Filter Gauntlet (`core_filter.py`)
Raw jobs pass through `apply_pipeline_filters(jobs, tracker=…)`:
0. **Seen-jobs filter** (only if a tracker is passed in) — drops URLs we've already evaluated. Runs FIRST so seen jobs never burn any later filter or AI quota.
1. **URL deduplication**.
2. **Smart normalized dedup** — strips tags like `(Brazil)`/`(LATAM)` from titles and normalizes company names to drop duplicates of the same role posted slightly differently.
3. **Language pre-filter** — rejects titles containing Chinese, Japanese, or Korean characters.
4. **Location pre-filter** — rejects `Shanghai`, `Moscow`, `Bangalore`, `India`, `Texas`, etc. UNLESS the title or location says `Remote`.
5. **Seniority filter** — drops `senior`, `lead`, `manager`, `principal`, `vp`, `staff`, `head`, `director`.
6. **Tightened keyword filter** — title must contain a relevant tech keyword (`software engineer`, `ml engineer`, `backend`, etc.).

### D. The AI Engine (`core_ai.py`)
Survivors are evaluated one-by-one by **Gemini 3.1 Flash Lite** (`gemini-3.1-flash-lite`) via the modern `google-genai` SDK.
1. **Authwall detection** — if a LinkedIn login wall truncates the description, a `[DESCRIPTION TRUNCATED…]` flag is passed to the AI, triggering the "Limited Info Protocol" to avoid hallucinations.
2. **DuckDuckGo deep search** — if the description contains a tightened set of trigger phrases (e.g. `"eligible countries"`, `"must be based"`, `"candidates based"`, `"remote in the"`, `"us only"`, etc.), a live DDG search queries the company's remote policy and feeds the results into the AI prompt. The trigger list was tightened to avoid false-positive web searches on phrases like `"based in NYC"` (which used to fire on nearly every job).
3. **CV matching & scoring** — the AI reads `cv_text.txt` and mathematically calculates a Match Percentage (0–100).
4. **Rate-limit protection** — 4-second sleep between AI calls keeps us under the 15 RPM free-tier limit (500 RPD).
5. **3-attempt retry on transient errors** — `503 UNAVAILABLE` and `500 INTERNAL` (Gemini demand spikes are common) are retried up to 2 extra times with 10 s + 20 s backoff so transient failures don't lose jobs to tomorrow's 24-hour cutoff. Non-retryable errors (quota 429, auth, parse) break out immediately.
6. **Fail-closed error handling** — exhausted retries return `is_valid=False, evaluated=False`. Jobs are NOT marked seen on errors, so they get another chance next run.

---

## 🧠 4. The Seen-Jobs Tracker (`core_filter.JobTracker`)

### What it does
A lightweight on-disk set of every URL we've already evaluated. When `apply_pipeline_filters` runs, it drops jobs whose URL is already in the set — saving AI quota and avoiding duplicate notifications.

### Persistence across CI runs
GitHub Actions runners are ephemeral — anything written to disk vanishes when the runner shuts down. To persist the tracker between runs without polluting git history, both workflows use the **`actions/cache@v4`** step:

```yaml
- uses: actions/cache@v4
  with:
    path: seen_jobs.json
    key: seen-jobs-${{ github.run_id }}
    restore-keys: seen-jobs-
```

The unique `run_id` makes every save immutable; the `restore-keys: seen-jobs-` prefix matches the most recent prior save so the next run boots with the freshest tracker.

### Cross-pipeline state sharing
Both workflows use the **same** cache key prefix. After the global pipeline runs in the morning and saves the cache, the local pipeline restores it later that day — so a job seen by one is automatically skipped by the other.

### Safe mark-seen timing
URLs are marked seen **only when Gemini returns a real verdict**. If a call raises an exception (quota / 500 / 503 / parse fail), `evaluated=False` is returned and the URL stays unmarked so the job gets another chance the next run.

---

## 📍 5. The Local Companies Pipeline (`local_companies.py`)

### The Challenge
Local Palestinian IT companies often post jobs as regular LinkedIn **Posts** (not in the official "Jobs" section) and on custom-built websites. Standard scrapers fail because LinkedIn blocks bots with login walls and every custom site has a different HTML layout.

### The Solution: Triple Radar
Instead of building 50 custom scrapers, this pipeline reads `IT Companies - Nablus.xlsx` and `IT Companies - Ramallah.xlsx` (parsed via `pandas` + `openpyxl`) and executes stealth searches:
1. **LinkedIn Post Radar** — `site:linkedin.com/posts [Company Name] (hiring OR vacancy OR job)`
2. **Website Radar** — `site:[company_website.com] (hiring OR careers OR jobs OR vacancy)`
3. **JobSpy Fallback** — standard JobSpy targeting the company name on LinkedIn Jobs.

**Search relaxation:** the company name is algorithmically shortened to its first highly identifiable word (e.g., `ASAL Technologies` → `ASAL`) so DDG catches posts by handles like `@asaltech`.

**6-day catch-up window:** DuckDuckGo relies on Bing's index, which can take 1–3 days to ingest a new LinkedIn post. The pipeline looks back 6 days (`hours_old=144`) so indexing lag doesn't cost us jobs.

### Routing Logic
*   **0 Raw Jobs Found:** Script quietly shuts down. Cleanup still runs (in `finally`).
*   **Raw Jobs Found, 0 Passed AI:** A small GitHub Issue titled `Local Companies Scan - YYYY-MM-DD (0 Passed)` is created so you know the script is healthy without cluttering email.
*   **Jobs Passed AI:** HTML email + GitHub Issue dispatched.

---

## ✉️ 6. Notification & Infrastructure (`core_notify.py`)

### The Final Presentation
1. **Match % sorting** — final dataframe sorted so highest Match % appears at the top.
2. **Daily Stats Dashboard** — every report begins with `Scraped: X → Filtered: Y → AI Approved: Z`.
3. **Split tables** — `🎓 Internships` and `💼 Full-Time Jobs` shown separately.

### Automated Housekeeping
*   **The "Inbox Method":** GitHub Issues are a rolling inbox.
*   **Auto-cleanup:** `cleanup_old_github_issues(days_old=5)` runs at the end of EVERY workflow run — wrapped in `try/finally`, so it survives even if the pipeline crashes or AI approves zero jobs.
*   **Title matching:** the cleanup recognises every issue prefix the bot has ever used: `Automated AI Job Alerts`, `Automated Job Alerts` (older), `Local Companies Job Alerts`, `Local Companies Scan`.
*   **Calendar-day age math:** ages are computed by date difference, NOT 24-hour periods. An issue from May 8 counts as 5 days old on May 13 regardless of clock time.

### GitHub Actions (Cron Jobs)
Both pipelines are automated via GitHub Actions on Ubuntu runners:
*   `.github/workflows/job_alert.yml` — daily at 06:55 UTC (09:55 AM Jerusalem time).
*   `.github/workflows/local_companies.yml` — daily at 08:50 UTC (11:50 AM Jerusalem time).

Each workflow has 4 steps: checkout, set up Python, restore seen-jobs cache, run the script. The cache action automatically saves the updated `seen_jobs.json` in a post-job step.

---

## 🧱 7. Dependencies (`requirements.txt`)
- `python-jobspy` — LinkedIn / Indeed scraping
- `pandas` — dataframe handling
- `requests` — HTTP for APIs + GitHub Issues
- `google-genai` — modern Gemini SDK (replaces the deprecated `google-generativeai`)
- `beautifulsoup4` — fallback HTML parsing for authwall detection
- `ddgs` — DuckDuckGo Search (replaces deprecated `duckduckgo-search`)
- `openpyxl` — reads the Palestinian company Excel sheets

---

## 📌 8. Summary of Evolution

*   **Tier 1:** Moved from a generic scraper to targeted multi-API architecture.
*   **Tier 2:** Hardened filtering (language, location, dedup), added authwall detection, deep web search, mathematical Match %.
*   **Tier 3:** Professional modular architecture. Stateful tracking (`seen_jobs.json`), stats dashboards, local companies deep-scraper, automated issue cleanup.
*   **Tier 4 (current):** Re-enabled tracker with Actions cache persistence + safer mark-seen timing. Added 3-attempt retry on transient Gemini 5xx errors. Fixed calendar-day age math in cleanup. Per-source logging. Removed broken Glassdoor source. Tightened DDG search triggers to reduce false-positive web searches. Bumped API recency to 72h to compensate for the lack of server-side date filtering on public APIs.
