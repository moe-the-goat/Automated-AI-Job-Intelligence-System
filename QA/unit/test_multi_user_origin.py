"""multi_user_runner origin provenance (W1).

The web app renders Local (Palestinian) jobs and Global / Remote jobs as
separate sections, driven by job_results.origin (migration 0011). The runner
tags each scraped frame BEFORE the merge — global jobs 'global', shared
local-cache jobs 'local' — and _jobs_to_rows persists the tag. Locking:
  * _safe_origin only ever emits 'global', 'local', or None (schema CHECK)
  * _jobs_to_rows carries the tag through to the insert payload
  * the tag survives the concat + URL-dedup merge step
"""

import numpy as np
import pandas as pd

import multi_user_runner as mur


def test_safe_origin_enforces_schema_check():
    assert mur._safe_origin("global") == "global"
    assert mur._safe_origin("local") == "local"
    assert mur._safe_origin("Local") == "local"      # case-insensitive
    assert mur._safe_origin("  GLOBAL  ") == "global"
    assert mur._safe_origin(None) is None
    assert mur._safe_origin("") is None
    assert mur._safe_origin("telegram") is None      # raw source ≠ origin
    assert mur._safe_origin(np.nan) is None          # pandas NaN → 'nan' → None


def _job(url: str, origin=None) -> dict:
    row = {
        "title": "Eng", "company": "Acme", "location": "Remote",
        "job_url": url, "description": "x",
    }
    if origin is not None:
        row["origin"] = origin
    return row


def test_jobs_to_rows_persists_origin():
    df = pd.DataFrame([
        _job("https://x/1", origin="global"),
        _job("https://x/2", origin="local"),
    ])
    rows = mur._jobs_to_rows(df, run_id=1, user_id="u1", ai_evaluated=True)
    assert rows[0]["origin"] == "global"
    assert rows[1]["origin"] == "local"


def test_jobs_to_rows_origin_none_for_untagged_frames():
    # A frame with no origin column at all (pre-W1 shape) must not crash
    # and must write NULL, not garbage.
    df = pd.DataFrame([_job("https://x/3")])
    rows = mur._jobs_to_rows(df, run_id=1, user_id="u1", ai_evaluated=False)
    assert rows[0]["origin"] is None


def test_origin_survives_merge_and_url_dedup():
    # Mirror of _run_for_user step 3: tag → concat → drop_duplicates(job_url).
    global_jobs = pd.DataFrame([_job("https://x/1"), _job("https://x/both")])
    global_jobs["origin"] = "global"
    local_jobs = pd.DataFrame([_job("https://x/2"), _job("https://x/both")])
    local_jobs["origin"] = "local"

    combined = pd.concat([global_jobs, local_jobs], ignore_index=True)
    combined = combined.drop_duplicates(subset=["job_url"]).reset_index(drop=True)

    by_url = {r["job_url"]: r["origin"]
              for r in mur._jobs_to_rows(combined, run_id=1, user_id="u1",
                                         ai_evaluated=True)}
    assert by_url["https://x/1"] == "global"
    assert by_url["https://x/2"] == "local"
    # Duplicate URL: keep-first semantics → the global copy wins.
    assert by_url["https://x/both"] == "global"
    assert len(by_url) == 3
