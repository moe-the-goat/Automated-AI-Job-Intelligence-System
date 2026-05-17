"""Wave-1 source parsers: Himalayas, The Muse, WeWorkRemotely.

Pure parsers — fed canonical / edge-case JSON or RSS dicts. If a vendor
changes their response shape we'll see it as a red test, not as silently
empty job lists in the daily email.
"""
import pandas as pd
from pipeline.core_search import (
    parse_himalayas_payload,
    parse_themuse_payload,
    parse_wwr_feed,
    _split_wwr_title,
)


# ---------------------------------------------------------------------------
# Himalayas
# ---------------------------------------------------------------------------

def test_himalayas_parses_canonical_response():
    payload = {
        "totalCount": 2,
        "offset": 0,
        "limit": 100,
        "jobs": [
            {
                "title": "Senior Python Developer",
                "companyName": "Acme Remote",
                "locationRestrictions": ["EU", "Worldwide"],
                "applicationLink": "https://himalayas.app/jobs/acme/12345",
                "pubDate": "2026-05-15T10:00:00Z",
                "description": "Full-time Python role at a remote-first company.",
                "excerpt": "Short excerpt",
            },
            {
                "title": "ML Engineering Intern",
                "companyName": "OtherCo",
                "locationRestrictions": ["Worldwide"],
                "applicationLink": "https://himalayas.app/jobs/otherco/67890",
                "pubDate": "2026-05-16T08:00:00Z",
                "description": "Internship role.",
            },
        ],
    }
    df = parse_himalayas_payload(payload)
    assert len(df) == 2
    assert df.iloc[0]["title"] == "Senior Python Developer"
    assert df.iloc[0]["company"] == "Acme Remote"
    assert df.iloc[0]["location"] == "EU, Worldwide"
    assert df.iloc[0]["job_url"].startswith("https://himalayas.app")
    assert df.iloc[1]["title"] == "ML Engineering Intern"


def test_himalayas_handles_empty_jobs():
    df = parse_himalayas_payload({"jobs": []})
    assert df.empty


def test_himalayas_handles_missing_jobs_key():
    df = parse_himalayas_payload({})
    assert df.empty


def test_himalayas_handles_none_payload():
    df = parse_himalayas_payload(None)
    assert df.empty


def test_himalayas_handles_string_location_restrictions():
    """Defensive: if API ever returns a string instead of list, we shouldn't crash."""
    payload = {"jobs": [{"title": "X", "companyName": "Y",
                         "locationRestrictions": "Remote",
                         "applicationLink": "u", "pubDate": ""}]}
    df = parse_himalayas_payload(payload)
    assert len(df) == 1
    assert df.iloc[0]["location"] == "Remote"


def test_himalayas_falls_back_to_excerpt_when_no_description():
    payload = {"jobs": [{"title": "T", "companyName": "C",
                         "excerpt": "from excerpt",
                         "applicationLink": "u", "pubDate": ""}]}
    df = parse_himalayas_payload(payload)
    assert df.iloc[0]["description"] == "from excerpt"


# ---------------------------------------------------------------------------
# The Muse
# ---------------------------------------------------------------------------

def test_themuse_parses_canonical_response():
    payload = {
        "page": 0, "page_count": 100, "items_per_page": 20, "total": 2000,
        "results": [
            {
                "name": "Software Engineer",
                "company": {"name": "Bechtel"},
                "locations": [{"name": "Remote"}, {"name": "Berlin, Germany"}],
                "refs": {"landing_page": "https://www.themuse.com/jobs/bechtel/se-abc"},
                "publication_date": "2026-03-30T16:51:26Z",
                "contents": "Full job description...",
            },
            {
                "name": "Data Scientist",
                "company": {"name": "AcmeCorp"},
                "locations": [{"name": "Worldwide"}],
                "refs": {"landing_page": "https://www.themuse.com/jobs/acme/ds-xyz"},
                "publication_date": "2026-05-01T10:00:00Z",
            },
        ],
    }
    df = parse_themuse_payload(payload)
    assert len(df) == 2
    assert df.iloc[0]["title"] == "Software Engineer"
    assert df.iloc[0]["company"] == "Bechtel"
    assert df.iloc[0]["location"] == "Remote, Berlin, Germany"
    assert df.iloc[0]["job_url"].startswith("https://www.themuse.com")
    assert df.iloc[1]["title"] == "Data Scientist"
    assert df.iloc[1]["location"] == "Worldwide"


def test_themuse_handles_empty_results():
    df = parse_themuse_payload({"results": []})
    assert df.empty


def test_themuse_handles_missing_results_key():
    df = parse_themuse_payload({})
    assert df.empty


def test_themuse_handles_none_payload():
    df = parse_themuse_payload(None)
    assert df.empty


def test_themuse_handles_missing_company_dict():
    payload = {"results": [{"name": "Job", "locations": [{"name": "Remote"}],
                            "refs": {"landing_page": "u"}}]}
    df = parse_themuse_payload(payload)
    assert df.iloc[0]["company"] == ""


def test_themuse_handles_missing_locations():
    payload = {"results": [{"name": "Job", "company": {"name": "C"},
                            "refs": {"landing_page": "u"}}]}
    df = parse_themuse_payload(payload)
    assert df.iloc[0]["location"] == "Remote/Unspecified"


def test_themuse_handles_missing_refs():
    payload = {"results": [{"name": "Job", "company": {"name": "C"},
                            "locations": [{"name": "Remote"}]}]}
    df = parse_themuse_payload(payload)
    assert df.iloc[0]["job_url"] == ""


# ---------------------------------------------------------------------------
# WeWorkRemotely RSS
# ---------------------------------------------------------------------------

def test_wwr_split_title_canonical_format():
    """WWR's <title> is usually 'CompanyName: Job Title'."""
    company, title = _split_wwr_title("Acme Corp: Senior Backend Engineer")
    assert company == "Acme Corp"
    assert title == "Senior Backend Engineer"


def test_wwr_split_title_no_colon():
    """If no colon present, company is empty and the whole string is the title."""
    company, title = _split_wwr_title("Just A Title Without Colon")
    assert company == ""
    assert title == "Just A Title Without Colon"


def test_wwr_split_title_empty_input():
    company, title = _split_wwr_title("")
    assert company == ""
    assert title == ""


def test_wwr_split_title_extra_colons():
    """Only split on the FIRST colon so titles like 'Co: Title: Subtitle' work."""
    company, title = _split_wwr_title("Acme: Senior Engineer: Backend")
    assert company == "Acme"
    assert title == "Senior Engineer: Backend"


def test_wwr_parses_feed_with_canonical_entries():
    fake_feed = {
        "entries": [
            {
                "title": "Acme Corp: Senior Backend Engineer",
                "link": "https://weworkremotely.com/listings/abc",
                "summary": "<p>Remote role at Acme.</p>",
                "published": "Mon, 15 May 2026 12:00:00 +0000",
            },
            {
                "title": "OtherCo: Frontend Developer",
                "link": "https://weworkremotely.com/listings/xyz",
                "summary": "<p>Remote role at OtherCo.</p>",
                "published": "Tue, 16 May 2026 09:00:00 +0000",
            },
        ]
    }
    df = parse_wwr_feed(fake_feed)
    assert len(df) == 2
    assert df.iloc[0]["company"] == "Acme Corp"
    assert df.iloc[0]["title"] == "Senior Backend Engineer"
    assert df.iloc[0]["location"] == "Remote"
    assert df.iloc[0]["job_url"].startswith("https://weworkremotely.com")
    assert df.iloc[1]["company"] == "OtherCo"


def test_wwr_handles_empty_feed():
    df = parse_wwr_feed({"entries": []})
    assert df.empty


def test_wwr_handles_missing_entries_key():
    df = parse_wwr_feed({})
    assert df.empty


def test_wwr_handles_missing_optional_fields():
    """Entry with no summary or published date shouldn't crash the parser."""
    fake_feed = {"entries": [{
        "title": "Co: Title",
        "link": "https://example.com/job",
    }]}
    df = parse_wwr_feed(fake_feed)
    assert len(df) == 1
    assert df.iloc[0]["description"] == ""
    assert df.iloc[0]["date_posted"] == ""


def test_wwr_accepts_feedparser_dict_like_object():
    """feedparser returns FeedParserDict with attribute access; ensure we handle both."""
    class _Entry:
        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)
        def get(self, k, default=""):
            return getattr(self, k, default)

    class _Feed:
        entries = [_Entry(title="X: Y", link="u", summary="s", published="p")]

    df = parse_wwr_feed(_Feed())
    assert len(df) == 1
    assert df.iloc[0]["company"] == "X"
    assert df.iloc[0]["title"] == "Y"
