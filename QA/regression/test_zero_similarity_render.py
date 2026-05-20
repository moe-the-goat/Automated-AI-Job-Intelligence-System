"""REGRESSION: a similarity of 0.0 used to render as the em-dash placeholder.

Symptom: when the embedding API call failed (404 on the wrong model name),
every job got similarity=0.0. The renderer used `if sim` which is falsy for
0.0, so all rows showed '—'. Should show as a numeric score.

Fix: explicit None-check so legitimate zero values render numerically.

Note (2026-05-20): The Also-Found rewrite changed the displayed score from
raw cosine ("0.00") to a weighted-score percentage ("0%"). The regression
behavior is preserved — zero must still render as a number, not em-dash.
"""
import pandas as pd
from pipeline.core_notify import _render_lower_ranked_html, _render_lower_ranked_md


def test_zero_score_renders_as_percent_in_html():
    df = pd.DataFrame([{
        "title": "X", "company": "Y", "location": "R",
        "similarity": 0.0, "weighted_score": 0.0, "job_url": "#",
    }])
    html = _render_lower_ranked_html(df)
    assert "0%" in html
    assert "—" not in html


def test_zero_score_renders_as_percent_in_markdown():
    df = pd.DataFrame([{
        "title": "X", "company": "Y", "location": "R",
        "similarity": 0.0, "weighted_score": 0.0, "job_url": "#",
    }])
    md = _render_lower_ranked_md(df)
    assert "0%" in md
    assert "—" not in md


def test_missing_score_falls_back_to_em_dash():
    """When BOTH weighted_score and similarity are None, em-dash is the right fallback."""
    df = pd.DataFrame([{
        "title": "X", "company": "Y", "location": "R",
        "similarity": None, "weighted_score": None, "job_url": "#",
    }])
    html = _render_lower_ranked_html(df)
    assert "—" in html
