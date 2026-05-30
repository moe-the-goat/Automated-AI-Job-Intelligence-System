"""Retention cleanup pure-helper tests.

The destructive logic lives in partition_deletable (never delete a bookmarked
job_result), chunks (batch deletes), and cutoff_iso (the age boundary). These
are what a bug would turn into data loss, so they're locked here.
"""

from datetime import datetime, timedelta, timezone

import cleanup_retention as cr


# ---------------------------------------------------------------------------
# partition_deletable — bookmarked rows must NEVER be returned for deletion
# ---------------------------------------------------------------------------

def test_partition_excludes_bookmarked():
    candidates = [1, 2, 3, 4, 5]
    bookmarked = {2, 4}
    assert cr.partition_deletable(candidates, bookmarked) == [1, 3, 5]


def test_partition_all_bookmarked_deletes_nothing():
    assert cr.partition_deletable([1, 2, 3], {1, 2, 3}) == []


def test_partition_none_bookmarked_deletes_all():
    assert cr.partition_deletable([1, 2, 3], set()) == [1, 2, 3]


def test_partition_preserves_order():
    assert cr.partition_deletable([5, 3, 9, 1], {3}) == [5, 9, 1]


# ---------------------------------------------------------------------------
# chunks
# ---------------------------------------------------------------------------

def test_chunks_splits_evenly_and_remainder():
    assert list(cr.chunks([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]


def test_chunks_empty():
    assert list(cr.chunks([], 100)) == []


def test_chunks_smaller_than_size():
    assert list(cr.chunks([1, 2], 100)) == [[1, 2]]


# ---------------------------------------------------------------------------
# cutoff_iso
# ---------------------------------------------------------------------------

def test_cutoff_iso_subtracts_days():
    now = datetime(2026, 5, 30, 12, 0, 0, tzinfo=timezone.utc)
    out = cr.cutoff_iso(90, now=now)
    assert out == (now - timedelta(days=90)).isoformat()
    # sanity: it's in the past relative to now
    assert datetime.fromisoformat(out) < now
