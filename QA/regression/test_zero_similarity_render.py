"""REGRESSION: a similarity of 0.0 used to render as the em-dash placeholder.

Symptom: when the embedding API call failed (404 on the wrong model name),
every job got similarity=0.0. The renderer used `if sim` which is falsy for
0.0, so all rows showed '—'. Should show '0.00'.

Fix: `if sim is not None` so legitimate zero values render numerically.
"""
import pandas as pd
from core_notify import _render_lower_ranked_html, _render_lower_ranked_md


def test_zero_similarity_renders_as_number_in_html():
    df = pd.DataFrame([{
        "title": "X", "company": "Y", "location": "R",
        "similarity": 0.0, "job_url": "#",
    }])
    html = _render_lower_ranked_html(df)
    assert "0.00" in html
    assert "—" not in html


def test_zero_similarity_renders_as_number_in_markdown():
    df = pd.DataFrame([{
        "title": "X", "company": "Y", "location": "R",
        "similarity": 0.0, "job_url": "#",
    }])
    md = _render_lower_ranked_md(df)
    assert "0.00" in md
    assert "—" not in md


def test_missing_similarity_falls_back_to_em_dash():
    """When similarity is actually None (not 0.0), em-dash is the right fallback."""
    df = pd.DataFrame([{
        "title": "X", "company": "Y", "location": "R",
        "similarity": None, "job_url": "#",
    }])
    html = _render_lower_ranked_html(df)
    assert "—" in html
