# 🧠 Automated AI Job Intelligence System: Deep Documentation

## 🎯 1. Project Goal & Philosophy
The core objective of this project is to build an **autonomous, highly intelligent recruitment pipeline** that completely replaces the grueling process of manual job searching. 

Instead of a standard "dumb scraper" that bombards the user with thousands of irrelevant or non-remote jobs, this system operates like an elite human recruiter. It scrapes jobs globally and locally, aggressively filters out garbage, performs deep web searches to verify ambiguous geographic restrictions (specifically ensuring remote eligibility for a candidate in Palestine/Middle East), and uses Google's Gemini AI to mathematically score each job against the candidate's CV.

The ultimate output is a clean, daily digest of **highly curated, highly probable job matches** delivered directly to the user's Email and GitHub Issues.

---

## 🏗️ 2. High-Level Architecture
To maintain a clean codebase, the system is designed using a **Modular Hybrid Architecture**. It is split into two independent workflows that share the same underlying "brain" and "filters".

1. **Global Remote Pipeline (`scraper.py`)**: Runs daily to hunt for global remote opportunities across major platforms and APIs.
2. **Local Companies Pipeline (`local_companies.py`)**: Runs daily to deeply scan specific Palestinian IT companies via their LinkedIn posts and websites.

### The Core Modules
Instead of one massive, unreadable script, the logic is separated into highly focused modules:
*   **`core_search.py`**: The Hunter. Responsible solely for gathering raw job data from external sources.
*   **`core_filter.py`**: The Bouncer. A gauntlet of strict, deterministic rules that instantly delete bad jobs before they cost us API quota.
*   **`core_ai.py`**: The Brain. Handles deep web searching and AI evaluation via Google Gemini.
*   **`core_notify.py`**: The Presenter. Formats the surviving jobs into beautiful HTML/Markdown and dispatches them.

---

## 🌍 3. The Global Pipeline (`scraper.py`)

### A. The Search Strategy
Instead of relying on broad, fragile boolean queries, the `config.json` uses **8 laser-focused, specific search queries** (e.g., `"junior software engineer"`, `"AI intern"`).
It gathers data from 7 different sources:
1.  **JobSpy Engine:** LinkedIn, Indeed, Glassdoor.
2.  **Free Public APIs:** Remotive, Arbeitnow, Jobicy, RemoteOK.

### B. The Deterministic Filter Gauntlet (`core_filter.py`)
To protect the AI from evaluating garbage data, all raw jobs pass through these filters:
1.  **Seen Jobs Tracker:** Checks `seen_jobs.json`. If a job URL was evaluated on a previous day, it is instantly dropped.
2.  **URL Deduplication:** Removes duplicate links.
3.  **Smart Normalized Deduplication:** Strips tags like `(Brazil)` or `(LATAM)` from titles, normalizes the company name, and drops duplicates of the same role posted slightly differently.
4.  **Language Pre-Filter:** Uses Regex to instantly reject jobs containing Chinese, Japanese, or Korean characters in the title.
5.  **Location Pre-Filter:** Rejects jobs explicitly located in places like `Shanghai`, `Moscow`, or `Texas` unless the title explicitly says "Remote".
6.  **Seniority Filter:** Drops jobs containing words like `senior`, `lead`, `vp`, `manager`.
7.  **Tightened Keyword Filter:** Ensures the title strictly contains relevant tech keywords (`software engineer`, `ml engineer`, `data scientist`, `backend`, etc.).

### C. The AI Engine (`core_ai.py`)
Jobs that survive the gauntlet are passed one-by-one to **Gemini 3.1 Flash Lite**.
1.  **Authwall Detection:** If a job description is truncated by a LinkedIn Login Wall, the scraper passes a specific `[DESCRIPTION TRUNCATED...]` flag to the AI, triggering the "Limited Info Protocol" to prevent hallucinations.
2.  **DuckDuckGo Deep Search:** If the job description contains phrases like `"eligible countries"`, `"US only"`, or `"based in"`, the script automatically triggers a live DuckDuckGo web search querying the company's remote policy. This live web data is fed into the AI prompt so it knows exactly if Palestine/EMEA is excluded.
3.  **CV Matching & Scoring:** The AI reads `cv_text.txt` and mathematically calculates a Match Percentage (0-100) based strictly on how the candidate's skills align with the job requirements.
4.  **Rate Limit Protection:** The script enforces a strict 4-second sleep between AI calls to stay safely under the free tier limit of 15 Requests Per Minute.
5.  **Fail-Closed Error Handling:** If the AI crashes or times out, it defaults to `is_valid=False` to prevent broken jobs from reaching the inbox.

---

## 📍 4. The Local Companies Pipeline (`local_companies.py`)

### The Challenge
Local companies often post jobs as regular LinkedIn **Posts** (not in the official "Jobs" section) and on custom-built websites. Standard scrapers fail here because LinkedIn blocks bots with login walls, and every custom website has a different HTML layout.

### The Solution: Dual DDG Radar
Instead of building 50 custom scrapers, this pipeline reads `IT Companies - Nablus.xlsx` and `IT Companies - Ramallah.xlsx` (parsed via `pandas` and `openpyxl`) and executes stealth searches:
1.  **LinkedIn Post Radar:** `site:linkedin.com/posts [Company Name] (hiring OR vacancy OR job)`
2.  **Website Radar:** `site:[company_website.com] (hiring OR careers OR jobs OR vacancy)`
3.  **JobSpy Fallback:** Runs standard JobSpy targeted strictly at the company's name.

**Search Strictness:** The company name is algorithmically shortened to its most identifiable first word (e.g., `ASAL Technologies` -> `ASAL`) so DuckDuckGo can catch posts by handles like `@asaltech`.

**6-Day Catch-Up Window:** Because DuckDuckGo relies on Bing's search index (which can take 1-3 days to index a new LinkedIn post), the pipeline looks back exactly 6 days (`hours_old=144`) to ensure no job slips through the indexing lag.

### Routing Logic
*   **0 Raw Jobs:** Script quietly shuts down.
*   **Raw Jobs Found, 0 Passed AI:** No email sent. Opens a GitHub Issue saying "X jobs found, 0 passed" to confirm the script is working without cluttering the inbox.
*   **Jobs Passed AI:** Sends the standard HTML email and GitHub Issue.

---

## ✉️ 5. Notification & Infrastructure (`core_notify.py`)

### The Final Presentation
1.  **Match % Sorting:** Before rendering, the final dataframe is sorted so jobs with the highest AI Match % appear at the absolute top of the email/issue.
2.  **Daily Stats Dashboard:** The top of every report includes a pipeline breakdown: `Scraped: X -> Filtered: Y -> AI Approved: Z`.
3.  **Split Tables:** Jobs are cleanly separated into a "🎓 Internships" table and a "💼 Full-Time Jobs" table.

### Automated Housekeeping
*   **The "Inbox Method":** GitHub Issues are treated as a rolling inbox. 
*   **Auto-Cleanup:** At the end of every run, the `cleanup_old_github_issues(5)` function triggers. It scans the repository for any issue titled "Automated AI Job Alerts" or "Local Companies Job Alerts". If the issue is older than 5 days, it automatically closes it. This ensures the repo stays clean permanently.

### GitHub Actions (Cron Jobs)
Both pipelines are fully automated via GitHub Actions running on Ubuntu runners:
*   `.github/workflows/job_alert.yml`: Runs daily at 06:55 UTC (09:55 AM Jerusalem Time).
*   `.github/workflows/local_companies.yml`: Runs daily at 08:50 UTC (11:50 AM Jerusalem Time).

---

## 📌 Summary of Evolution (How We Got Here)
*   **Tier 1:** Transitioned from a generic scraper to a highly targeted, multi-API architecture.
*   **Tier 2:** Hardened the filtering logic (Language, Location, Dedup) and added extreme resilience to the AI (Authwall detection, failure defaults, deep web search triggers).
*   **Tier 3:** Achieved a professional, modular software architecture. Implemented stateful tracking (`seen_jobs.json`) to save API calls, built statistical dashboards, and deployed the advanced Local Companies deep-scraper.
