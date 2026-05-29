"""Dual-run comparison pure-helper tests (B10).

Locks the parsing + diff logic that the report rests on: URL normalization
(so the same posting matches across two scrapes), legacy markdown-table
parsing, match-cell extraction, and the overlap/delta/dedup computation.
"""

import compare_dual_run as cmp


# ---------------------------------------------------------------------------
# normalize_url
# ---------------------------------------------------------------------------

def test_normalize_strips_query_and_trailing_slash():
    a = cmp.normalize_url("https://www.linkedin.com/jobs/view/123/?refId=abc&trk=xyz")
    b = cmp.normalize_url("https://www.linkedin.com/jobs/view/123")
    assert a == b == "https://www.linkedin.com/jobs/view/123"


def test_normalize_lowercases_host_only_not_path():
    out = cmp.normalize_url("https://Indeed.COM/viewjob?jk=AbC")
    assert out == "https://indeed.com/viewjob"


def test_normalize_handles_empty_and_fragment():
    assert cmp.normalize_url("") == ""
    assert cmp.normalize_url(None) == ""
    assert cmp.normalize_url("https://x.com/a#section") == "https://x.com/a"


# ---------------------------------------------------------------------------
# match cell + apply url extraction
# ---------------------------------------------------------------------------

def test_parse_match_cell():
    assert cmp.parse_match_cell("**88%** (T:90 E:70 L:85)") == 88
    assert cmp.parse_match_cell("**5%**") == 5
    assert cmp.parse_match_cell("**N/A**") is None
    assert cmp.parse_match_cell("") is None
    assert cmp.parse_match_cell("no percent here") is None


def test_extract_apply_url():
    assert cmp.extract_apply_url("[Apply](https://x.com/job/1)") == "https://x.com/job/1"
    assert cmp.extract_apply_url("[Apply](#)") == "#"
    assert cmp.extract_apply_url("no link") is None


# ---------------------------------------------------------------------------
# parse_legacy_issue_markdown
# ---------------------------------------------------------------------------

LEGACY_MD = """## Automated AI Job Alerts

**Pipeline Stats:** Scraped: 100 -> Filtered to: 40 -> AI Approved: 3

### Junior & Entry-Level Jobs (CV-Matched)

| Title | Company | Location | Match | Pay | Effort | AI Verdict | Link |
|---|---|---|---|---|---|---|---|
| Backend Engineer | Stripe | Remote | **88%** (T:90 E:70 L:85) | $120k | medium | MATCH: strong | [Apply](https://stripe.com/jobs/1?utm=x) |
| ⚠️ Sketchy Role | FooCorp | India | **30%** | N/A | high | GAP: junior | [Apply](https://foo.com/2) |
| No Score Job | BarCo | Remote | **N/A** | N/A | unknown | skipped | [Apply](https://bar.com/3) |
"""


def test_parse_legacy_issue_extracts_rows():
    rows = cmp.parse_legacy_issue_markdown(LEGACY_MD)
    assert len(rows) == 3
    first = rows[0]
    assert first["company"] == "Stripe"
    assert first["title"] == "Backend Engineer"
    assert first["match"] == 88
    # query string stripped in norm_url
    assert first["norm_url"] == "https://stripe.com/jobs/1"


def test_parse_legacy_strips_severity_glyph_and_handles_na():
    rows = cmp.parse_legacy_issue_markdown(LEGACY_MD)
    assert rows[1]["title"] == "Sketchy Role"   # ⚠️ stripped
    assert rows[2]["match"] is None              # **N/A**


def test_parse_legacy_skips_header_separator_and_prose():
    rows = cmp.parse_legacy_issue_markdown(LEGACY_MD)
    titles = {r["title"] for r in rows}
    assert "Title" not in titles
    assert all(r["url"] for r in rows)


def test_parse_legacy_empty():
    assert cmp.parse_legacy_issue_markdown("") == []
    assert cmp.parse_legacy_issue_markdown("no tables here") == []


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------

def _legacy(url, match, title="t"):
    return {"norm_url": cmp.normalize_url(url), "title": title, "match": match}


def _multi(url, match, title="t"):
    return {"norm_url": cmp.normalize_url(url), "title": title, "match_percentage": match}


def test_compare_overlap_and_deltas():
    legacy = [_legacy("https://x.com/1", 88), _legacy("https://x.com/2", 70)]
    multi = [_multi("https://x.com/1?ref=a", 85), _multi("https://x.com/3", 60)]
    rep = cmp.compare(legacy, multi)
    assert rep["legacy_count"] == 2
    assert rep["multi_count"] == 2
    assert rep["overlap_count"] == 1
    assert rep["only_legacy"] == ["https://x.com/2"]
    assert rep["only_multi"] == ["https://x.com/3"]
    o = rep["overlaps"][0]
    assert o["legacy_match"] == 88 and o["multi_match"] == 85
    assert o["delta"] == 3
    assert o["flagged"] is False


def test_compare_flags_large_score_gap():
    legacy = [_legacy("https://x.com/1", 90)]
    multi = [_multi("https://x.com/1", 60)]  # delta 30 > 15
    rep = cmp.compare(legacy, multi)
    assert rep["overlaps"][0]["flagged"] is True


def test_compare_detects_multiuser_duplicates():
    legacy = []
    multi = [_multi("https://x.com/1", 80), _multi("https://x.com/1?x=2", 80)]
    rep = cmp.compare(legacy, multi)
    assert len(rep["multi_dupes"]) == 1
    assert rep["multi_count"] == 1  # deduped in the map


def test_compare_handles_missing_scores_without_crashing():
    legacy = [_legacy("https://x.com/1", None)]
    multi = [_multi("https://x.com/1", 80)]
    rep = cmp.compare(legacy, multi)
    assert rep["overlaps"][0]["delta"] is None
    assert rep["overlaps"][0]["flagged"] is False
