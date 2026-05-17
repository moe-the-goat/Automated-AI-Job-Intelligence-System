"""Per-platform payload parsers: Greenhouse, Lever, Workable.

The parsers are pure functions — given a JSON dict (the shape the public API
returns), they produce normalized job records matching our pipeline schema.

If a vendor changes their response shape, these tests fail loudly instead of
silently producing empty job lists in production.
"""
from pipeline.core_ats import (
    parse_greenhouse_payload,
    parse_lever_payload,
    parse_workable_payload,
    parse_ashby_payload,
    parse_workday_payload,
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


def test_greenhouse_extracts_content_html_to_plain_text():
    """With ?content=true, Greenhouse inlines HTML-escaped HTML in `content`. We
    must unescape entities AND strip tags so downstream consumers see plain text."""
    payload = {
        "jobs": [
            {
                "id": 99,
                "title": "Backend Developer",
                "location": {"name": "Remote"},
                "absolute_url": "https://boards.greenhouse.io/co/jobs/99",
                "content": "&lt;p&gt;Build &lt;strong&gt;great&lt;/strong&gt; APIs in &lt;em&gt;Python&lt;/em&gt;.&lt;/p&gt;&lt;ul&gt;&lt;li&gt;Requirements: 3+ yrs&lt;/li&gt;&lt;/ul&gt;",
            }
        ]
    }
    out = parse_greenhouse_payload(payload, "Co")
    assert len(out) == 1
    desc = out[0]["description"]
    # HTML tags removed, entities unescaped, whitespace collapsed.
    assert "<p>" not in desc
    assert "&lt;" not in desc
    assert "Build great APIs in Python" in desc
    assert "3+ yrs" in desc


def test_greenhouse_handles_empty_content_field():
    """If `content` is missing or empty, description should be empty (backward compat)."""
    payload = {"jobs": [{"id": 1, "title": "X", "location": {"name": "Remote"}, "absolute_url": "u"}]}
    out = parse_greenhouse_payload(payload, "Co")
    assert out[0]["description"] == ""

    payload2 = {"jobs": [{"id": 1, "title": "X", "content": ""}]}
    out2 = parse_greenhouse_payload(payload2, "Co")
    assert out2[0]["description"] == ""


def test_greenhouse_content_with_nested_html_lists():
    """Verify whitespace collapsing for multi-line HTML content."""
    payload = {
        "jobs": [
            {
                "title": "Engineer",
                "content": "&lt;h2&gt;Role&lt;/h2&gt;\n&lt;p&gt;Description here.&lt;/p&gt;\n&lt;ul&gt;\n  &lt;li&gt;Python&lt;/li&gt;\n  &lt;li&gt;Docker&lt;/li&gt;\n&lt;/ul&gt;",
            }
        ]
    }
    out = parse_greenhouse_payload(payload, "Co")
    desc = out[0]["description"]
    # No double-spaces or newlines after collapsing.
    assert "  " not in desc
    assert "\n" not in desc
    assert "Python" in desc and "Docker" in desc


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


# --- Ashby ---

def test_ashby_parses_canonical_response():
    payload = {
        "apiVersion": "1",
        "jobs": [
            {
                "id": "uuid-1",
                "title": "Senior Backend Engineer",
                "location": "Remote (Worldwide)",
                "department": "Engineering",
                "employmentType": "FullTime",
                "jobUrl": "https://jobs.ashbyhq.com/anthropic/uuid-1",
                "publishedAt": "2026-05-15T10:00:00.000Z",
                "descriptionPlain": "Build agents.",
            },
            {
                "id": "uuid-2",
                "title": "ML Intern",
                "location": "Remote (Americas)",
                "employmentType": "Intern",
                "jobUrl": "https://jobs.ashbyhq.com/anthropic/uuid-2",
                "publishedAt": "2026-05-16T09:00:00.000Z",
            },
        ],
    }
    out = parse_ashby_payload(payload, "Anthropic")
    assert len(out) == 2
    assert out[0]["title"] == "Senior Backend Engineer"
    assert out[0]["company"] == "Anthropic"
    assert out[0]["location"] == "Remote (Worldwide)"
    assert out[0]["job_url"].startswith("https://jobs.ashbyhq.com")
    assert out[0]["job_type"] == "fulltime"
    # Intern employmentType should map to internship job_type
    assert out[1]["job_type"] == "internship"


def test_ashby_handles_nested_location_dict():
    """Some Ashby payloads nest location instead of returning a string."""
    payload = {"jobs": [{"title": "Engineer", "location": {"name": "Berlin, Germany"}, "jobUrl": "u"}]}
    out = parse_ashby_payload(payload, "Co")
    assert out[0]["location"] == "Berlin, Germany"


def test_ashby_handles_empty_jobs():
    assert parse_ashby_payload({"jobs": []}, "Co") == []
    assert parse_ashby_payload({}, "Co") == []
    assert parse_ashby_payload(None, "Co") == []


def test_ashby_skips_non_dict_entries():
    payload = {"jobs": [None, "garbage", {"title": "Real", "jobUrl": "u"}]}
    out = parse_ashby_payload(payload, "Co")
    assert len(out) == 1
    assert out[0]["title"] == "Real"


def test_ashby_falls_back_to_html_description():
    payload = {"jobs": [{"title": "T", "descriptionHtml": "<p>html only</p>", "jobUrl": "u"}]}
    out = parse_ashby_payload(payload, "Co")
    assert "html" in out[0]["description"]


# --- Workday ---

def test_workday_parses_canonical_response():
    payload = {
        "total": 2,
        "jobPostings": [
            {
                "title": "Senior Software Engineer",
                "externalPath": "/job/Ramallah/Senior-Software-Engineer_R12345",
                "locationsText": "Ramallah, Palestine",
                "postedOn": "Posted Yesterday",
                "bulletFields": [],
            },
            {
                "title": "ML Engineering Intern",
                "externalPath": "/job/Remote/ML-Intern_R67890",
                "locationsText": "Remote, Worldwide",
                "postedOn": "Posted 5 Days Ago",
            },
        ],
    }
    out = parse_workday_payload(payload, "Acme Corp", tenant="acme", cluster="wd5", site="AcmeCareers")
    assert len(out) == 2
    assert out[0]["title"] == "Senior Software Engineer"
    assert out[0]["company"] == "Acme Corp"
    assert out[0]["location"] == "Ramallah, Palestine"
    # job_url must be a full URL combining origin + site + externalPath
    assert out[0]["job_url"] == "https://acme.wd5.myworkdayjobs.com/AcmeCareers/job/Ramallah/Senior-Software-Engineer_R12345"


def test_workday_handles_empty_postings():
    assert parse_workday_payload({"jobPostings": []}, "Co") == []
    assert parse_workday_payload({}, "Co") == []
    assert parse_workday_payload(None, "Co") == []


def test_workday_skips_non_dict_entries():
    payload = {"jobPostings": [None, "garbage", {"title": "Real", "externalPath": "/job/x"}]}
    out = parse_workday_payload(payload, "Co", tenant="t", cluster="wd1", site="s")
    assert len(out) == 1
    assert out[0]["title"] == "Real"


def test_workday_keeps_externalPath_as_is_when_no_tenant_info():
    """If we don't know the tenant/cluster, we can't rebuild the absolute URL — just keep the relative path."""
    payload = {"jobPostings": [{"title": "X", "externalPath": "/job/foo"}]}
    out = parse_workday_payload(payload, "Co")
    assert out[0]["job_url"] == "/job/foo"
