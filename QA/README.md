# QA — test suite for the Job Scraping Automation Tool

Every meaningful behaviour of the pipeline is locked down here. New features
should ship with new tests in the appropriate subdirectory; new bug fixes
should ship with a regression test that fails before the fix.

## Running the suite

**Locally (no dependencies needed — pure stdlib):**
```
python QA/run_all.py
```

Run only a slice:
```
python QA/run_all.py unit
python QA/run_all.py integration regression
```

**Locally with richer output (optional):**
```
pip install -r requirements-qa.txt
python -m pytest QA/
```

**On CI:** `.github/workflows/qa.yml` runs the suite automatically on every
push and pull request. A failing run blocks the badge on GitHub.

## Layout

```
QA/
├── run_all.py                 # stdlib runner, used by CI
├── conftest.py                # pytest path setup (if you install pytest)
├── unit/                      # pure-function tests, no network, no I/O
│   ├── test_ai_schema.py            DEFAULT_AI_RESULT shape, _normalize_result, _parse_ai_response
│   ├── test_normalizers.py          _safe_int / _safe_bool / _safe_str / _normalize_effort
│   ├── test_viability.py            quick_viability_check pre-screen rules
│   ├── test_reputation.py           _pre_flag_reputation / _load_reputation
│   ├── test_embedding_math.py       cosine_similarity / rank_by_similarity / CV hash + cache
│   ├── test_linkedin_helpers.py     linkedin_post_date + linkedin_handle_matches
│   ├── test_scam_helpers.py         looks_like_india_employer + scan_for_scam_signals
│   └── test_rendering.py            match-cell, badge precedence, sort, _normalize_repo
├── integration/               # tests that wire multiple modules + use fixtures
│   ├── test_email_render.py         format_email_html end-to-end
│   ├── test_filter_chain.py         apply_pipeline_filters on a 20-row mixed dataframe
│   └── test_scam_flow.py            detect_company_scam with monkey-patched DDGS
├── regression/                # bug-specific tests with full context comments
│   ├── test_date_decoder_ms_bug.py        ms-vs-seconds bug in LinkedIn activity decoder
│   ├── test_zero_similarity_render.py     0.0 must render as '0.00', not '—'
│   ├── test_logs_repo_url_normalization.py   full-URL inputs must be normalized
│   └── test_short_description_bypass.py   pre-screen must not skip API-missing descriptions
└── fixtures/
    └── sample_jobs.py         realistic dataframes shared across integration tests
```

## Adding new tests

1. Pick the right subdirectory:
   - **unit/** — a pure function with no I/O, no network.
   - **integration/** — multiple modules wired together (renderers, filter chain, scam flow).
   - **regression/** — a specific bug from history. Always add a multi-line
     comment at the top of the file explaining the symptom, root cause, and fix
     so a future reader knows why the test exists.
2. Name the file `test_<area>.py` and put each test in a function `test_<scenario>(...)`
3. Use bare `assert` — works with both `run_all.py` and `pytest`.
4. Avoid network calls. Mock DDGS / Gemini / requests via `sys.modules` or
   `unittest.mock` if the code path needs them (see `integration/test_scam_flow.py`
   for the pattern).
5. Run `python QA/run_all.py` before committing.

## When a test fails on CI

The CI run output shows the failing test name, file, and traceback. Re-run
locally with `python QA/run_all.py` to reproduce. If you can't reproduce
locally, check that your virtualenv has the same versions as `requirements.txt`.
