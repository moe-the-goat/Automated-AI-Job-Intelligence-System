"""provision_user pure-helper tests (B9).

Locks the config.json → search_queries mapping (the site_name→sites rename,
job_type constraint handling, clamps) and the (term, location) dedup that
makes seeding idempotent.
"""

import provision_user as pu


# ---------------------------------------------------------------------------
# config_to_search_rows
# ---------------------------------------------------------------------------

def test_maps_site_name_to_sites_and_passes_fields():
    config = {"searches": [{
        "site_name": ["linkedin", "indeed"],
        "search_term": "python developer",
        "location": "Germany",
        "job_type": "fulltime",
        "is_remote": True,
        "results_wanted": 25,
        "hours_old": 24,
        "country_indeed": "Germany",
    }]}
    rows = pu.config_to_search_rows(config, "u1")
    assert len(rows) == 1
    r = rows[0]
    assert r["user_id"] == "u1"
    assert r["sites"] == ["linkedin", "indeed"]
    assert "site_name" not in r
    assert r["search_term"] == "python developer"
    assert r["location"] == "Germany"
    assert r["job_type"] == "fulltime"
    assert r["country_indeed"] == "Germany"
    assert r["is_active"] is True


def test_drops_invalid_job_type_to_none():
    # Config entries that omit job_type, or use a non-enum value, must become NULL
    # (the schema CHECK only allows fulltime/internship/contract/parttime or null).
    config = {"searches": [
        {"search_term": "swe", "location": "Palestine"},                      # no job_type
        {"search_term": "swe2", "location": "X", "job_type": "permanent"},    # invalid
        {"search_term": "swe3", "location": "Y", "job_type": "internship"},   # valid
    ]}
    rows = pu.config_to_search_rows(config, "u1")
    assert rows[0]["job_type"] is None
    assert rows[1]["job_type"] is None
    assert rows[2]["job_type"] == "internship"


def test_defaults_and_clamps():
    config = {"searches": [{
        "search_term": "x",
        "results_wanted": 9999,   # clamp to 100
        "hours_old": -5,          # clamp to 1
    }]}
    r = pu.config_to_search_rows(config, "u1")[0]
    assert r["location"] == "Worldwide"
    assert r["sites"] == ["linkedin", "indeed"]
    assert r["is_remote"] is True
    assert r["results_wanted"] == 100
    assert r["hours_old"] == 1
    assert r["country_indeed"] == "USA"


def test_skips_entries_without_search_term():
    config = {"searches": [{"location": "X"}, {"search_term": "  "}, {"search_term": "real"}]}
    rows = pu.config_to_search_rows(config, "u1")
    assert len(rows) == 1
    assert rows[0]["search_term"] == "real"


def test_string_site_name_is_wrapped():
    config = {"searches": [{"search_term": "x", "site_name": "linkedin"}]}
    assert pu.config_to_search_rows(config, "u1")[0]["sites"] == ["linkedin"]


def test_empty_or_missing_searches():
    assert pu.config_to_search_rows({}, "u1") == []
    assert pu.config_to_search_rows({"searches": []}, "u1") == []
    assert pu.config_to_search_rows(None, "u1") == []


# ---------------------------------------------------------------------------
# plan_search_inserts — idempotent seeding
# ---------------------------------------------------------------------------

def _row(term, loc):
    return {"search_term": term, "location": loc}


def test_plan_inserts_everything_when_none_exist():
    rows = [_row("a", "X"), _row("b", "Y")]
    assert pu.plan_search_inserts(rows, []) == rows


def test_plan_skips_existing_by_term_and_location():
    rows = [_row("a", "X"), _row("b", "Y")]
    existing = [{"search_term": "A", "location": "x"}]  # case-insensitive match
    planned = pu.plan_search_inserts(rows, existing)
    assert len(planned) == 1
    assert planned[0]["search_term"] == "b"


def test_plan_dedups_within_incoming_list():
    rows = [_row("a", "X"), _row("a", "X"), _row("a", "Y")]
    planned = pu.plan_search_inserts(rows, [])
    # same term+location collapses; different location kept
    assert len(planned) == 2


def test_plan_is_idempotent_on_rerun():
    rows = [_row("a", "X"), _row("b", "Y")]
    existing = [{"search_term": "a", "location": "X"}, {"search_term": "b", "location": "Y"}]
    assert pu.plan_search_inserts(rows, existing) == []


# ---------------------------------------------------------------------------
# derive_display_name
# ---------------------------------------------------------------------------

def test_derive_display_name():
    assert pu.derive_display_name("Mohammad", "Abu Hijleh") == "Mohammad Abu Hijleh"
    assert pu.derive_display_name("Ada", None) == "Ada"
    assert pu.derive_display_name(None, "Lovelace") == "Lovelace"
    assert pu.derive_display_name(None, None) is None
    assert pu.derive_display_name("  ", "  ") is None
