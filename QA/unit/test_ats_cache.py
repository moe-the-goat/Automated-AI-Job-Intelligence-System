"""AtsCache: persists ATS detection results so we don't re-fetch /careers every run.

TTL logic — cached entries older than CACHE_TTL_DAYS are treated as misses.
"""
import json
import os
from datetime import datetime, timezone, timedelta
import core_ats
from core_ats import AtsCache, CACHE_TTL_DAYS


def _tmp_cache_path():
    """Use a project-relative tmp path that round-trips even on Windows path quirks."""
    return "_qa_test_ats_cache.json"


def _with_temp_cache(fn):
    """Tiny context-helper: swap CACHE_FILE module attr around a callable."""
    path = _tmp_cache_path()
    original = core_ats.ATS_CACHE_FILE
    core_ats.ATS_CACHE_FILE = path
    try:
        if os.path.exists(path):
            os.remove(path)
        fn(path)
    finally:
        core_ats.ATS_CACHE_FILE = original
        if os.path.exists(path):
            os.remove(path)


def test_cache_roundtrip_persists_entries():
    def body(path):
        c = AtsCache(filepath=path)
        c.set("CompanyA", "greenhouse", "company-a-token")
        c.save()
        # Re-load from disk and assert the entry is there.
        c2 = AtsCache(filepath=path)
        entry = c2.get("CompanyA")
        assert entry is not None
        assert entry["ats"] == "greenhouse"
        assert entry["token"] == "company-a-token"
    _with_temp_cache(body)


def test_cache_miss_for_unknown_company():
    def body(path):
        c = AtsCache(filepath=path)
        c.set("CompanyA", "greenhouse", "tok")
        assert c.get("CompanyB") is None
    _with_temp_cache(body)


def test_cache_expires_stale_entry():
    """An entry older than CACHE_TTL_DAYS should be treated as a miss."""
    def body(path):
        c = AtsCache(filepath=path)
        stale_iso = (datetime.now(timezone.utc) - timedelta(days=CACHE_TTL_DAYS + 5)).isoformat()
        c.data["StaleCo"] = {"ats": "greenhouse", "token": "x", "detected_at": stale_iso}
        assert c.get("StaleCo") is None  # expired -> treated as miss
    _with_temp_cache(body)


def test_cache_keeps_fresh_entry():
    def body(path):
        c = AtsCache(filepath=path)
        fresh_iso = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        c.data["FreshCo"] = {"ats": "lever", "token": "x", "detected_at": fresh_iso}
        entry = c.get("FreshCo")
        assert entry is not None
        assert entry["ats"] == "lever"
    _with_temp_cache(body)


def test_cache_load_survives_corrupted_file():
    """Malformed JSON shouldn't crash the loader."""
    def body(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write("{ this is not valid json ")
        c = AtsCache(filepath=path)
        # Should fall back to empty dict, not raise.
        assert c.data == {}
    _with_temp_cache(body)


def test_cache_save_creates_parent_directory():
    """If the cache filepath includes a directory that doesn't exist yet, save() makes it."""
    nested_path = "_qa_test_subdir/ats.json"
    original = core_ats.ATS_CACHE_FILE
    core_ats.ATS_CACHE_FILE = nested_path
    try:
        c = AtsCache(filepath=nested_path)
        c.set("X", "lever", "tok")
        c.save()
        assert os.path.exists(nested_path)
        with open(nested_path) as f:
            payload = json.load(f)
        assert payload["X"]["ats"] == "lever"
    finally:
        core_ats.ATS_CACHE_FILE = original
        if os.path.exists(nested_path):
            os.remove(nested_path)
        if os.path.isdir("_qa_test_subdir"):
            os.rmdir("_qa_test_subdir")
