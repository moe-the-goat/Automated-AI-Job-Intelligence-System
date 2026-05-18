# Automated AI Job Intelligence System

A daily, fully-autonomous pipeline that scrapes the global remote job market, filters out the noise, evaluates each surviving role against a candidate CV with a Gemini-powered recruiter heuristic, and delivers a short, honest shortlist by email every morning.

The system is the answer to a specific problem: existing job aggregators assume geographic centrality and a default candidate profile. Their idea of "best fit for you" is calibrated for that profile. I built a sifter that injects explicit constraints into an unoptimized search market — replacing manual triage with a deterministic, observable, tested pipeline that runs on free infrastructure and costs nothing to operate.

This repository is what I built.

---

## Overview

Every day at 06:55 UTC (09:55 Jerusalem), a GitHub Actions workflow runs [scraper.py](scraper.py). Roughly five to seven minutes later, a single HTML email lands in my inbox: a stats header, two ranked tables (internships and full-time roles), and an optional "lower-ranked" section. A second workflow at 08:50 UTC (11:50 Jerusalem) runs [local_companies.py](local_companies.py) against a curated list of Palestinian IT companies pulled from two Excel sheets in this repo. Both share the same brain. Both share a seen-jobs cache, so a job evaluated by one is automatically skipped by the other. Neither runs unless the test suite passes first.

The pipeline talks to nine external job sources, six ATS platforms, a free LLM (Gemini 3.1 Flash Lite), a free embedding API (also Gemini), DuckDuckGo for occasional fact-checks, and Jina Reader as a fallback for careers pages no scraper can parse alone. It runs entirely on free tiers. The total monthly cost of operating it is zero.

Inside, there are nine modules under [pipeline/](pipeline/), two thin orchestrators at the repo root, a curated reputation list, **408 tests across 26 files**, and three GitHub Actions workflows that wire the whole thing into a self-managing loop. The rest of this document is a tour of why each piece is shaped the way it is.

---

## The Problem It Solves

I started this project because I had reached a frustrating equilibrium with manual job hunting. I would spend an hour on LinkedIn, open thirty tabs, read each posting carefully, and discover that twenty-five of them required US work authorization, three were senior roles disguised under a "junior" filter, one was a clear scam, and the remaining one had been deluged with two thousand applicants. The signal-to-noise ratio was punishing, and the cost of being wrong about any single posting — wasting an hour writing a tailored application for a role I had no legal way to take — was real.

The deeper observation: ranking algorithms on existing aggregators assume a default candidate. When the candidate sits outside that profile (in my case, a Computer Engineering student based in Palestine), even careful filters — "Remote," "Junior," "Internship" — leak in both directions. They keep US-only postings that happen to mention "remote" in the description, and they hide global-remote postings whose titles miss a narrow keyword.

The system in this repository inverts that. It assumes nothing about the candidate but what the CV says, and forces every other piece of information — location, company reputation, scam probability, work-auth restrictions, level seniority — to be inferred explicitly, with traceable rules, before a job is allowed near my inbox.

---

## How It Works

At the highest level, the pipeline is a five-stage funnel:

```
   raw scrape (~150-300 jobs/day)
          |
          v
   deterministic filter gauntlet
          |
          v
   embedding pre-rank with region & trust weighting
          |
          v
   AI evaluation (top 45 + 5 wildcards)
          |
          v
   final dispatch (email + private GitHub Issue)
```

Each stage exists for a reason and was added in response to a specific failure mode I observed in production. The funnel is sequential and largely deterministic until the AI step, which is the only place a black-box model gets a voice — and even there, the verdicts are post-processed by deterministic caps before they reach the user.

**Stage 1 — Raw scrape.** Nine external sources, each behind its own fetcher. JobSpy hits LinkedIn, Indeed, and ZipRecruiter across fifteen region-specific queries. Six public APIs supplement it (Remotive, Arbeitnow, Jobicy, RemoteOK, Himalayas, The Muse, WeWorkRemotely via RSS). YC's Work at a Startup is pulled via Jina Reader plus Gemini extraction. The local pipeline runs a four-mode search per Palestinian company: handle-precise DDG against LinkedIn posts, domain-restricted DDG against the company website, a JobSpy fallback, and an ATS lookup.

**Stage 2 — Filter gauntlet.** Eight deterministic checks: seen-jobs cache, reputation prefilter, dedup, language detection, location prefilter, seniority filter (including FAANG-style level codes), tech-keyword whitelist, and a non-tech intern blocker. A typical run filters 300 raw jobs down to 40-50 survivors.

**Stage 3 — Embedding pre-rank.** Survivors are embedded against the CV, then sorted by `weighted_score = cosine_similarity * region_weight * trust_weight`. The weights inject the inductive bias the embedding model lacks: EU and Americas roles get 1.15x, fully-remote and Middle East roles get 1.30x, India gets 0.70x, sanctioned regions get 0.50x. Trusted companies stack a 1.25x multiplier.

**Stage 4 — AI evaluation.** Top 45 plus 5 wildcards reach a heuristic pre-screen, then Gemini. The model returns a structured verdict with sub-scores, an opinionated text verdict in a recruiter's voice, and a `suspicious` flag. Deterministic caps post-process the result before it reaches the user.

**Stage 5 — Dispatch.** Color-coded match badges, bolded `MATCH:` and `GAP:` markers in each verdict, severity badges on titles, and a compact lower-ranked table at the bottom. Same data renders as Markdown into a GitHub Issue in a private logs repo. Cleanup runs in a `finally` so even crashes don't leak orphaned issues.

That is the system, end to end. The next section is the deep tour.

---

## The Pipeline in Detail

<details>
<summary><strong>Search — the nine sources</strong></summary>

The first version of this project used only JobSpy. It was unreliable: LinkedIn's anti-bot detection produced anywhere between two hundred jobs and zero jobs on consecutive runs, and the source was opaque about why. The fix was redundancy. I added Remotive, Arbeitnow, Jobicy, and RemoteOK first because each exposes a clean public REST API with no authentication required and has been stable across the entire history of this project. Then came Himalayas, The Muse, and WeWorkRemotely (RSS, parsed via `feedparser`). Each source has its own fetcher in [pipeline/core_search.py](pipeline/core_search.py), each returning a `pandas.DataFrame` with a normalized schema of `title, company, location, job_url, description, date_posted`. Every fetcher is wrapped in a try/except so a single dead vendor cannot stop the pipeline.

Recency is handled per-source because the APIs differ. JobSpy filters server-side with `hours_old=24`. The public APIs do not support server-side recency filtering, so [filter_api_jobs](pipeline/core_filter.py) does a client-side date filter at `hours_old=72` to give more headroom. Arbeitnow returns Unix timestamps; everyone else returns ISO strings; the filter handles both.

For the local pipeline, the search story is different. Most Palestinian companies don't post jobs on standard aggregators at all. They hire through LinkedIn posts (not the Jobs section), private referrals, or custom careers pages with no SEO. To reach them, [local_companies.py](local_companies.py) reads the IT-company list from two Excel sheets, extracts each company's LinkedIn handle via [pipeline/core_ats.py](pipeline/core_ats.py)'s `extract_linkedin_handle`, and runs four search modes per company: a handle-precise DDG query against `linkedin.com/company/{handle}/posts`, a domain-restricted DDG query against the website, a JobSpy fallback, and a direct ATS lookup. Each LinkedIn post URL has its publication date decoded from the embedded snowflake activity ID — a tiny piece of LinkedIn's internal data model that proves whether the post is fresh or whether Bing has cached it from 2017. This decoder had a bug in an earlier version (it treated the timestamp as seconds instead of milliseconds), which caused multi-year-old posts to slip through for weeks. The fix is the single line `datetime.fromtimestamp(ts_milliseconds / 1000, tz=timezone.utc)` and is one of the things I am proudest of catching, because the failure mode was completely silent.

</details>

<details>
<summary><strong>Filter — the gauntlet</strong></summary>

The deterministic filter chain is the cheapest part of the pipeline and does the most work. Each step exists because something specific went wrong without it.

1. **Seen-jobs**. URLs we evaluated yesterday don't need to be evaluated again. [JobTracker](pipeline/core_filter.py) is a thin set wrapped around a JSON file on disk; the file persists across CI runs via `actions/cache@v4` so two ephemeral runners share state. Both pipelines write to and read from the same cache key prefix.

2. **Reputation prefilter**. Some companies are reliably terrible — the "Skillfied Mentor" / "Webs IT Solution" / "Inficore Soft" cluster that has produced exactly zero legitimate offers. [data/reputation.json](data/reputation.json) holds a curated list of name and handle patterns. Matches are tagged `pre_flagged_low_quality=True` and get a 55% match cap downstream. Trusted companies (Anthropic, GitLab, Stripe, Cohere, Spotify, Klarna, and 75 others) carry the inverse flag.

3. **Smart deduplication**. URL dedup catches the easy case. Normalized title-plus-company dedup catches the case where the same role is posted as "Software Engineer (Remote)" and "Software Engineer" by the same company — they look distinct via URL but are the same job.

4. **Language**. CJK characters in the title kill the row immediately. For everything else, `langdetect` runs on the title (with a 30-character minimum to avoid false positives on short tech titles loaded with proper nouns) and on the description (with a 300-character minimum, conservative by design — false positives here delete real jobs from the daily email).

5. **Location**. Explicit location-locked postings that don't say "remote" in the title or location are dropped.

6. **Seniority**. The obvious words — senior, principal, staff, manager, director, VP, lead, head, president, architect. Plus FAANG-style internal level codes: Netflix's L4-L12, Meta's IC4-IC12, Stripe's E4-E9, Amazon's SDE II-9, Google's G7-G12. Level codes were added after a Netflix posting titled "Software Engineer (L5)" made it into a daily email and reminded me that L5 is not entry-level work no matter how clean the title looks.

7. **Tech keyword whitelist**. The title must contain at least one of about thirty tech signals.

8. **Non-tech intern blocker**. The catch-all `intern` keyword above lets through "Graduate Research Intern, Biology" and "Business Analyst Intern" — both of which I observed in a real email. This final step requires that any title containing `intern` AND any of 24 non-tech signals (biology, biotech, business analyst, social media, HR, finance, legal, etc.) be dropped.

After this chain, a typical run has filtered three hundred raw jobs down to forty or fifty survivors. The AI never sees the other two hundred and fifty.

</details>

<details>
<summary><strong>Embedding — the ranker</strong></summary>

Pre-ranking via embedding similarity has two purposes. The first is quota management: Gemini's free tier gives 500 requests per day, and one full AI verdict costs one request plus roughly four seconds of throttle. With seventy filtered survivors, we cannot afford to evaluate everyone. The second purpose is more interesting: the embedding ranker gives us a place to inject inductive bias that the AI cannot. The AI sees one job at a time and scores it on the merits. The ranker sees all jobs at once and can answer the cross-cutting question: "of these, which are most worth burning my limited AI budget on?"

The implementation is straightforward. The CV is embedded once per run (cached by SHA-256 hash so we only regenerate when `cv_text.txt` changes) and each job is embedded as `title + description[:7000]`. The character limit was originally 1000; I raised it to 7000 after observing that most job descriptions open with two or three paragraphs of HR boilerplate before the actual requirements section. At 1000 characters, the embedding was learning the company's marketing voice instead of the role's technical content.

Then the inductive bias. [pipeline/region_weighting.py](pipeline/region_weighting.py) classifies every job into a five-tier region category based on location signals across title, company, and description:

| Tier | Multiplier | Examples |
| --- | --- | --- |
| Highly preferred | 1.30x | Worldwide / Anywhere / Fully Remote, Middle East, Africa |
| Preferred | 1.15x | EU (15+ countries), LATAM, Canada, non-India South Asia |
| Neutral | 1.00x | US, unspecified |
| Deweighted | 0.70x | India / Pvt Ltd / Indian metros |
| Heavily deweighted | 0.50x | Sanctioned regions (Russia, China, Iran, North Korea, Belarus) |

A second multiplier (1.25x) stacks on top for the 81 trusted companies in [data/reputation.json](data/reputation.json). The final `weighted_score` drives the ranking; the raw `similarity` is preserved for visibility. The result: a 0.57 raw-similarity "Junior Software Engineer — Remote, Brazil" now sorts well above a 0.57 raw-similarity Indian sus-internship, because Brazil's 1.15x beats India's 0.70x.

Top 45 jobs by weighted score plus 5 deterministically-sampled wildcards reach the AI. The rest appear in the email under a "Lower-Ranked Matches" section with their similarity scores but no AI verdict — a hedge against the ranker being wrong about something the AI would catch.

</details>

<details>
<summary><strong>AI — the recruiter</strong></summary>

Gemini 3.1 Flash Lite was chosen because it is free, fast, and accurate enough for this task. The prompt is engineered to make the model think like a senior recruiter, not a friendly career-coach AI. It is forbidden from using the word "strong" or any other generic praise. Every match claim must cite a specific CV asset by name, and every gap claim must name the missing requirement specifically. The verdict structure is anchored: `MATCH:`, optionally `SECOND MATCH/GAP:`, then `GAP:`, optionally `CLOSING REASON:` when the model decides to disqualify outright.

The prompt also demands explicit math. The model is told that `match_percentage = 0.5 * tech_fit + 0.3 * experience_fit + 0.2 * logistics_fit`, capped at 60 if `suspicious=True`. This makes the verdict auditable: a 92% match should be inspectable as roughly 100% tech, 80% experience, 90% logistics. When the math doesn't add up, I know the model lied about its sub-scores.

Two failure modes shaped the post-processing. First, when a job description is missing or truncated by LinkedIn's authwall, the model used to invent confidence it didn't have. The "Limited Info Protocol" — a paragraph in the prompt that instructs the model to deduct ten from each sub-score and explicitly note the missing description — fixed it. Second, the model sometimes flagged a posting as `suspicious=True` and then proceeded to give it 80%. [apply_post_ai_caps](pipeline/core_ai.py) is the deterministic guard: any AI-self-flagged suspicious result above 55% gets clamped, and the verdict gets an `[AI-SUSPICIOUS]` prefix.

For India-located suspicious companies, a third layer fires: [detect_company_scam](pipeline/core_ai.py) runs three short DuckDuckGo queries against the company name (scam, fake job complaints, reddit review). Two or more scam keywords across the combined snippets, and the match drops to 30% with a `[SCAM]` tag. The check is conservative because false positives here would damage real Indian companies; the threshold was tuned by examining historical reviews of known job mills.

</details>

<details>
<summary><strong>ATS — direct integration</strong></summary>

A late but important discovery: most companies don't write their own careers pages. They use a SaaS hiring platform — Greenhouse, Lever, Workable, Ashby, Workday, FactorialHR — and each of these has a free public JSON API that returns current openings without authentication. Scraping the HTML careers page is brittle and triggers anti-bot; hitting the structured API is clean and gives us live data that, by construction, contains only currently-open positions.

[pipeline/core_ats.py](pipeline/core_ats.py) implements six platforms. For each, the URL pattern is detected once via a regex against the careers-page HTML, the result is cached in `data/ats_cache.json` for thirty days, and subsequent runs hit the platform's API directly. Greenhouse uses `?content=true` which inlines descriptions in the same call (no N+1 pagination). Workday is the most complex: its endpoint requires a composite `tenant|cluster|site` token packed from a regex against URLs like `nvidia.wd5.myworkdayjobs.com/en-US/NVIDIAExternalCareerSite`. FactorialHR (which Innotech, a Palestinian company, uses) was added specifically after a ghost listing from their indexed page kept appearing in my daily emails for weeks; integrating their public API meant we only ever see roles that are actually open today.

For companies that don't use any of these platforms (a real condition for many small EU and Palestinian businesses), a **tiered fallback** runs: first, BeautifulSoup extracts visible text from the already-fetched careers HTML; if that yields fewer than 500 characters (meaning the page is a JavaScript-rendered SPA whose body is empty), Jina Reader (a free third-party service at `r.jina.ai`) renders the page to markdown; whatever text we have is then sent to Gemini with an extraction prompt that produces a structured job list. The tiered design means the fast, free Tier 1 handles the common case, and the slower, rate-limited Tier 2 only fires for genuinely SPA pages.

</details>

<details>
<summary><strong>URL validation — defending against blog posts and ghost listings</strong></summary>

[pipeline/url_validation.py](pipeline/url_validation.py) does two things, both in response to specific failure modes observed in real emails.

**Path-pattern check.** A pure-Python `is_job_url_like(url)` rejects DDG search results whose paths don't look like job pages. It requires positive signals (`/job/`, `/careers/`, `/positions/`, `/job_posting/`, `/apply/`, etc.) AND zero negative signals (`/blog/`, `/news/`, `/market-updates/`, `/article/`, year segments like `/2017/`). This single rule killed a class of failure where blog posts were treated as jobs — including a real case where a Freightos market-update article matched a "logistics" keyword search.

**HEAD-probe ghost-listing filter.** A concurrent batch probe (`probe_urls_alive_batch`, ten worker threads, five-second timeout each) checks every DDG-discovered URL for 404, 410, or 451 status codes — and for redirects to known "no-longer-available" sink URLs. Catches ghost listings that companies removed but Bing still indexes. Runs once per batch after collection, not per row, adding about 1.5 seconds to the pipeline runtime.

</details>

<details>
<summary><strong>Notify — the polish</strong></summary>

The email is the final product, and how it looks matters more than I expected when I started. The first version was a plain text dump. It was unreadable: I would scan it for thirty seconds, see nothing obviously good, and close it. The current version is HTML, with:

- Color-coded match badges (green ≥85, amber 70-84, red <70, grey N/A)
- Bolded `MATCH:` / `GAP:` / `CLOSING-REASON:` keywords inside each verdict
- Sub-score breakdowns under each headline percentage (T:tech E:experience L:logistics)
- Severity badges on titles in decreasing order: web-confirmed scam, reputation blacklist, AI-flagged suspicious
- Inline color legend at the top so first-time readers don't need to guess
- A compact "lower-ranked" table at the bottom for the long tail

The GitHub Issue version uses Markdown with the same structure. It exists for two reasons: persistence (emails get archived, issues stay searchable), and audit trail (I can look back at any day's run and see exactly what the AI thought). Issues land in a private repo via a fine-grained PAT so the public repo doesn't accumulate hundreds of stale issues. Automated cleanup closes anything older than two calendar days, with the calendar-day math computed via date subtraction (not 24-hour periods) to avoid edge cases at midnight.

</details>

---

## Design Decisions That Mattered

<details>
<summary><strong>Two-tier evaluation: heuristic pre-screen before the LLM</strong></summary>

The heuristic pre-screen in `quick_viability_check` saves roughly 30-40% of AI calls. It runs five gates in sequence: reputation blacklist, non-tech title signal, description-length sanity check (with bypasses for legitimate cases like LinkedIn-radar snippets and authwall placeholders), explicit senior-experience requirement, and a hard work-authorization disqualifier. Cost of the heuristic: microseconds per job. Cost of the AI call it replaces: 4-6 seconds plus one of 500 daily quota slots. The asymmetry justified the work.

</details>

<details>
<summary><strong>Region and trust weighting: where the inductive bias lives</strong></summary>

The single biggest quality lever in the pipeline. Without it, a 0.85 raw-similarity "Backend Intern — Bangalore" outranks a 0.55 raw-similarity "Junior Engineer — Egypt" even though the Egypt job is actually actionable and the Bangalore one is not. With multiplicative weights (region tier from 0.50 to 1.30, trust boost 1.25x), the order inverts. The weights are tuned conservatively so India is *down*-weighted, not *banned* — a genuinely strong Indian job with no suspicion flags can still surface, it just has to clear a higher bar.

</details>

<details>
<summary><strong>Tiered Jina fallback for custom careers pages</strong></summary>

For ATS-less careers pages, the obvious solution was Jina Reader plus Gemini extraction. The smarter solution was to try cheap BeautifulSoup extraction first on the HTML we already fetched during ATS detection. BS4 is free, instant, and works perfectly for the common case (server-rendered pages). Only the actual JavaScript-rendered SPAs need Jina. The tier structure cut the average fallback time by an order of magnitude and reduced our exposure to Jina's rate limits.

</details>

<details>
<summary><strong>URL validation as a separate defensive layer</strong></summary>

[pipeline/url_validation.py](pipeline/url_validation.py) lives in its own module rather than bolted onto `local_companies.py` because the same two-layer defense (path pattern plus HEAD probe) will be useful for any future caller — `scraper.py`, the eventual multi-user runner, an ad-hoc CLI. Separating policy from invocation is a habit worth keeping; it makes the pipeline easier to extend without growing fragile.

</details>

<details>
<summary><strong>Mark-on-success seen-jobs semantics</strong></summary>

The tracker only marks a URL as "seen" when the AI returns a real verdict. If a Gemini call fails with a quota error or a 5xx, the URL stays unmarked so we can retry it tomorrow. This sounds obvious in retrospect; the original implementation marked seen unconditionally, which meant a single bad afternoon at Gemini's data center would silently lose a day's worth of jobs forever. The current behavior is encoded in the second return value of `evaluate_job_with_ai`: a tuple of `(result_dict, evaluated_bool)`, where the caller only marks seen when `evaluated_bool` is True.

</details>

<details>
<summary><strong>Self-cap on AI-flagged suspicious verdicts</strong></summary>

The AI sometimes flags a job as suspicious AND gives it 80% match because the tech keywords overlap with the CV. [apply_post_ai_caps](pipeline/core_ai.py) is the deterministic clamp: any AI-self-flagged suspicious result above 55% gets clamped to 55, and the verdict gets an `[AI-SUSPICIOUS]` prefix that makes the flag visible in the rendered email. This pattern — keeping the AI's sub-score breakdown faithful while letting deterministic policy adjust the headline number — is one I'd apply to any LLM-in-the-loop pipeline.

</details>

<details>
<summary><strong>CV in GitHub Secrets, not in the repo</strong></summary>

The CV used to be committed as `cv_text.txt`. Once the repo went public, that became a problem. The current arrangement keeps the CV as a `CV_TEXT` GitHub Actions secret; the workflow writes it to disk at runtime before invoking the scraper. The file is gitignored. Anyone with read access to the workflow logs sees nothing — secrets are masked automatically. Standard CI/CD pattern, applied correctly here.

</details>

---

## Engineering Discipline

Everything above describes what the system does. This section describes how I made sure it keeps working — and is the part of the project I would point a hiring manager toward first.

**408 tests across three tiers.** The suite in [QA/](QA/) follows the testing pyramid. Most tests are unit tests in `QA/unit/` (pure functions, deterministic, sub-millisecond). Some are integration tests in `QA/integration/` (multi-module flows with external services mocked). The rest are regression tests in `QA/regression/`. The test runner at [QA/run_all.py](QA/run_all.py) is pure stdlib — no pytest dependency required to run it — though every test is pytest-compatible too, so you can run `pytest QA/` if you prefer. The whole suite executes in roughly twenty-three seconds locally.

**Regression tests named for production bugs.** Each file in `QA/regression/` freezes a specific failure that once made it into a real email — `test_date_decoder_ms_bug.py`, `test_logs_repo_url_normalization.py`, `test_short_description_bypass.py`, `test_zero_similarity_render.py`. When something breaks in production, the fix isn't complete until there's a regression test pinning it. This is the discipline that turns code into something you can maintain rather than something you have to babysit, and it's one of the things I'm most deliberate about in this repository.

**Two CI gates.** [.github/workflows/qa.yml](.github/workflows/qa.yml) runs on every push and pull request. It's the first gate; if the test suite fails on a push, the diff doesn't reach `main`. Separately, the daily cron workflows ([job_alert.yml](.github/workflows/job_alert.yml), [local_companies.yml](.github/workflows/local_companies.yml)) each include a "QA gate" step that runs the suite *again* before the scraper itself starts. This is defense in depth: a regression that somehow slips through code review, or a non-code change (like a `data/reputation.json` edit, or a `pip` resolver picking up a breaking dependency update overnight) still gets caught before the system dispatches a garbage email. The extra cost is twenty-five seconds per run.

**Structured logging across every module.** [pipeline/logging_setup.py](pipeline/logging_setup.py) replaces every `print()` in the codebase with a properly-leveled logger call. The format (`HH:MM:SS LEVEL module: message`) is scannable in the GitHub Actions console. The level mapping is honest: progress messages are INFO, recoverable failures (retries, missing keys, fetch errors) are WARNING, dispatched failures (AI exhausted retries, email/issue failures) are ERROR. Chatty third-party libraries (urllib3, requests, httpx, httpcore) are automatically silenced to WARNING regardless of the root level. `configure_logging()` is idempotent and respects a `LOG_LEVEL` env-var override for debugging.

**Secrets isolation and operational hygiene.** The candidate's CV lives as a `CV_TEXT` GitHub Actions secret, written to disk at runtime and gitignored. Daily-run summaries route to a separate private repository (`moe-the-goat/job-scrapper-logs`) via a fine-grained PAT scoped to a single repository's issue-write permission — minimum necessary access. Automated cleanup closes issues older than two calendar days on both the private repo and a legacy public one, running in a `finally` block so even pipeline crashes don't leak orphaned issues. The reputation list at [data/reputation.json](data/reputation.json) is curated data, not code — editing it doesn't require a code review, just a passing test suite. The QA gate ensures even that lightweight workflow can't ship a broken JSON file by accident.

---

## Project Layout

<details>
<summary>Click to expand the directory tree</summary>

```
.
|-- pipeline/                         core modules, each a single responsibility
|   |-- core_search.py                fetchers for nine external sources
|   |-- core_ats.py                   six ATS integrations + tiered Jina fallback
|   |-- core_filter.py                deterministic filter gauntlet + JobTracker
|   |-- core_embedding.py             CV-similarity pre-ranker
|   |-- core_ai.py                    Gemini evaluation, pre-screen, scam check
|   |-- core_notify.py                email + GitHub Issue rendering and dispatch
|   |-- region_weighting.py           geographic + trust weighting for the ranker
|   |-- url_validation.py             URL-pattern check + HEAD probe for ghost listings
|   `-- logging_setup.py              configure_logging + get_logger helpers
|
|-- scraper.py                        global pipeline entry point (cron 06:55 UTC)
|-- local_companies.py                Palestinian-companies pipeline (cron 08:50 UTC)
|
|-- data/
|   `-- reputation.json               curated blacklist + trust-boost list
|
|-- IT Companies - Nablus.xlsx        target list (Palestinian local pipeline)
|-- IT Companies - Ramallah.xlsx      target list (Palestinian local pipeline)
|
|-- config.json                       search queries + email destination
|
|-- QA/
|   |-- unit/                         pure-function tests (most of the suite)
|   |-- integration/                  multi-module flow tests with mocks
|   |-- regression/                   each file pins a specific shipped bug
|   |-- fixtures/sample_jobs.py       shared test data
|   |-- conftest.py                   pytest path setup
|   `-- run_all.py                    stdlib-only test runner
|
|-- requirements.txt                  runtime deps
|-- requirements-qa.txt               test-only deps (pytest)
|
`-- .github/workflows/
    |-- qa.yml                        runs QA on every push/PR
    |-- job_alert.yml                 daily global scraper (cron 06:55 UTC)
    `-- local_companies.yml           daily local scraper (cron 08:50 UTC)
```

</details>

---

## Running It Locally

```bash
# Install runtime + test deps
pip install -r requirements.txt
pip install -r requirements-qa.txt

# Set the required secrets
export GEMINI_API_KEY=...
export SENDER_EMAIL=...
export EMAIL_APP_PASSWORD=...                    # Gmail app password, not your account password
export LOGS_REPO=owner/repo-name                 # optional — private logs repo
export LOGS_REPO_TOKEN=ghp_...                   # optional — fine-grained PAT

# Provide CV text (gitignored)
echo "..." > cv_text.txt

# Run the test suite (must pass before either pipeline runs)
python QA/run_all.py

# Run the global pipeline
python scraper.py

# Or the local one
python local_companies.py
```

Both pipelines are designed to be safe to run repeatedly. The seen-jobs tracker prevents duplicate AI evaluations, and the GitHub Issue cleanup is idempotent.

---

## Numbers

A snapshot of the current state:

| Metric                                      | Value                          |
| ------------------------------------------- | ------------------------------ |
| External job sources integrated             | 9                              |
| ATS platforms with direct API integration   | 6                              |
| Region tiers in the weighting system        | 5                              |
| Companies in the trust-boost list           | 81                             |
| Companies in the reputation blacklist       | 12                             |
| Total tests in the QA suite                 | 408                            |
| Test files                                  | 26                             |
| QA suite runtime (local, sequential)        | ~23 seconds                    |
| Daily pipeline runtime (GitHub Actions)     | ~5-7 minutes                   |
| LLM provider                                | Gemini 3.1 Flash Lite          |
| LLM quota used per day (typical)            | ~50 / 500 RPD                  |
| Embedding model                             | gemini-embedding-001           |
| Embedding throttle                          | 50ms (~20 RPS)                 |
| AI evaluation top-N                         | 45 + 5 wildcards               |
| Email frequency                             | once per day per pipeline      |
| Monthly cost                                | $0                             |

---

## What's Next

The single-user version of this system is mature. The path forward is making it serve other people, which is fundamentally a different architecture: a Next.js front end on Vercel, a Supabase Postgres back end with row-level security so each user's data is isolated, the existing `pipeline/` package as the worker (run via the same GitHub Actions cron, just looping over users from the database), and Resend for transactional email. The current pipeline is being shaped so that this transition is straightforward — the package is import-clean, the entry points are thin orchestrators, the configuration is data not code. When the multi-user variant ships, the single-user version will still work; it will just be one user in the database.

A few smaller items also remain on the list. A user-feedback loop ("I applied to this; I got an interview"; "I rejected this") that biases future rankings is a natural follow-on but requires the database. A logging upgrade to ship structured JSON logs to an aggregator like Better Stack or Grafana Loki is appropriate when multiple users are running concurrently. A Cerebras-as-fallback LLM for when Gemini quota is exhausted is a safety net I have not yet needed.

For now, the system runs every morning and tells me the truth about that day's market. That was the goal.

---

## A Personal Note

This project is, in its own small way, evidence of an engineering temperament: when an existing tool wastes time in proportion to a structural gap, the productive response is to build a better tool. The choice was to spend hours every week sifting through low-signal job feeds, or to spend a few months building a sifter that does it deterministically, testably, and for free. I chose the latter, and the result is here.

If you're an engineer reading this and want to understand how the pipeline works, every module is documented inline, every design decision has a comment explaining the reasoning, and every shipped bug has a regression test pinning it down. If you're a recruiter or hiring manager looking for evidence of how I think — about architecture, about trade-offs, about quality, about maintaining production software — this repository is the most honest portfolio I can offer.

Thank you for reading.
