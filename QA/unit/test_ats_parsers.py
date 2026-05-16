"""Per-platform payload parsers: Greenhouse, Lever, Workable.

The parsers are pure functions — given a JSON dict (the shape the public API
returns), they produce normalized job records matching our pipeline schema.

If a vendor changes their response shape, these tests fail loudly instead of
silently producing empty job lists in production.
"""
from core_ats import (
    parse_greenhouse_payload,
    parse_lever_payload,
    parse_workable_payload,
)


# --- Greenhouse ---

def test_greenhouse_parses_canonical_response():
    payload = {
        "jobs": [
            {
                "id": 12345,
                "title": "Senior Software Engineer",
                "location": {"name": "Remote, EMEA"},
                "absolute_url": "https://boards.greenhouse.io/asaltech/jobs/12345",
                "updated_at": "2026-05-15T10:00:00Z",
            },
            {
                "id": 12346,
                "title": "ML Engineering Intern",
                "location": {"name": "Ramallah, Palestine"},
                "absolute_url": "https://boards.greenhouse.io/asaltech/jobs/12346",
                "updated_at": "2026-05-16T08:00:00Z",
            },
        ]
    }
    out = parse_greenhouse_payload(payload, "ASAL Technologies")
    assert len(out) == 2
    assert out[0]["title"] == "Senior Software Engineer"
    assert out[0]["company"] == "ASAL Technologies"
    assert out[0]["location"] == "Remote, EMEA"
    assert out[0]["job_url"].startswith("https://boards.greenhouse.io")
    assert out[1]["title"] == "ML Engineering Intern"


def test_greenhouse_handles_empty_jobs_list():
    assert parse_greenhouse_payload({"jobs": []}, "AnyCo") == []


def test_greenhouse_handles_missing_jobs_key():
    assert parse_greenhouse_payload({}, "AnyCo") == []


def test_greenhouse_handles_none_payload():
    assert parse_greenhouse_payload(None, "AnyCo") == []


def test_greenhouse_handles_string_location_gracefully():
    """If the API returns a string location instead of dict, we fall back to empty."""
    payload = {"jobs": [{"id": 1, "title": "X", "location": "weird-string-form"}]}
    out = parse_greenhouse_payload(payload, "Co")
    assert len(out) == 1
    assert out[0]["title"] == "X"
    assert out[0]["location"] == "Remote/Unspecified"  # _normalize_job default


# --- Lever ---

def test_lever_parses_canonical_response():
    payload = [
        {
            "id": "abc-uuid",
            "text": "Software Engineer",
            "categories": {"location": "Remote", "team": "Engineering"},
            "hostedUrl": "https://jobs.lever.co/anthropic/abc-uuid",
            "createdAt": 1747400000000,  # ms timestamp
            "descriptionPlain": "Build great products.",
        }
    ]
    out = parse_lever_payload(payload, "Anthropic")
    assert len(out) == 1
    assert out[0]["title"] == "Software Engineer"
    assert out[0]["company"] == "Anthropic"
    assert out[0]["location"] == "Remote"
    assert "anthropic" in out[0]["job_url"]
    assert out[0]["description"] == "Build great products."
    # createdAt was a ms timestamp; should now be an ISO string.
    assert out[0]["date_posted"].startswith("2025-")


def test_lever_handles_empty_list():
    assert parse_lever_payload([], "Co") == []


def test_lever_handles_none_payload():
    assert parse_lever_payload(None, "Co") == []


def test_lever_skips_non_dict_entries():
    payload = [None, "garbage", {"text": "Real Job", "hostedUrl": "x"}]
    out = parse_lever_payload(payload, "Co")
    assert len(out) == 1
    assert out[0]["title"] == "Real Job"


def test_lever_handles_missing_createdAt():
    payload = [{"text": "Job", "hostedUrl": "x"}]
    out = parse_lever_payload(payload, "Co")
    assert out[0]["date_posted"] == ""


# --- Workable ---

def test_workable_parses_canonical_response():
    payload = {
        "jobs": [
            {
                "id": "wxyz",
                "shortcode": "WXYZ123",
                "title": "Backend Developer",
                "location": {"city": "Ramallah", "country": "Palestine"},
                "published_on": "2026-05-15",
                "description": "Join our team.",
            }
        ]
    }
    out = parse_workable_payload(payload, "Workable Co", token="example-co")
    assert len(out) == 1
    assert out[0]["title"] == "Backend Developer"
    assert out[0]["location"] == "Ramallah, Palestine"
    # URL constructed from the account token + job shortcode
    assert out[0]["job_url"] == "https://apply.workable.com/example-co/j/WXYZ123/"


def test_workable_constructs_url_from_shortcode_when_no_direct_url():
    payload = {"jobs": [{"shortcode": "AB12", "title": "X"}]}
    out = parse_workable_payload(payload, "Co", token="example")
    assert out[0]["job_url"] == "https://apply.workable.com/example/j/AB12/"


def test_workable_falls_back_to_explicit_url_when_no_shortcode():
    payload = {"jobs": [{
        "title": "X", "url": "https://example.com/jobs/x",
    }]}
    out = parse_workable_payload(payload, "Co", token="")
    assert out[0]["job_url"] == "https://example.com/jobs/x"


def test_workable_handles_missing_location_dict():
    payload = {"jobs": [{"title": "X", "shortcode": "Y"}]}
    out = parse_workable_payload(payload, "Co", token="example")
    # No location info -> normalized default
    assert out[0]["location"] == "Remote/Unspecified"


def test_workable_handles_empty_jobs():
    assert parse_workable_payload({"jobs": []}, "Co") == []
    assert parse_workable_payload({}, "Co") == []
    assert parse_workable_payload(None, "Co") == []
