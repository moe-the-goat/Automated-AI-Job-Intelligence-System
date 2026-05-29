"""Supabase-backed feedback helpers tests (B7b).

Covers the pure helpers (format_entry_text, _parse_pgvector) and the
data-shape contract of load_feedback_embeddings — specifically that it
returns the same {"entries": [{"text", "embedding"}]} envelope that
core_embedding.retrieve_relevant_feedback expects, so the RAG path stays
backward-compatible with the GitHub-Contents version.
"""

import pipeline.core_feedback_supabase as cfs
from pipeline.core_feedback_supabase import (
    format_entry_text,
    load_feedback_embeddings,
    count_feedback_entries,
    load_candidate_preferences,
    to_pgvector_literal,
    _parse_pgvector,
)


class _Resp:
    def __init__(self, data):
        self.data = data


class _Query:
    def __init__(self, table, store):
        self.table_name = table
        self.store = store
        self._filters = {}
        self._select = None
        self._order = None

    def select(self, cols):
        self._select = cols
        return self

    def eq(self, col, val):
        self._filters[("eq", col)] = val
        return self

    def order(self, *a, **k):
        return self

    def limit(self, *a, **k):
        return self

    def execute(self):
        rows = list(self.store.get(self.table_name, []))
        for (op, col), val in self._filters.items():
            if op == "eq":
                rows = [r for r in rows if r.get(col) == val]
        return _Resp(rows)


class _Client:
    def __init__(self, store):
        self.store = store

    def table(self, name):
        return _Query(name, self.store)


# ---------------------------------------------------------------------------
# format_entry_text
# ---------------------------------------------------------------------------

def test_format_entry_text_supabase_shape():
    row = {"feedback_type": "applied", "title": "AI Engineer", "company": "Acme", "note": "loved the role"}
    assert format_entry_text(row) == "[applied] AI Engineer @ Acme — note: loved the role"


def test_format_entry_text_falls_back_to_legacy_feedback_key():
    # GitHub-Contents log entries use `feedback`, not `feedback_type`.
    # The helper accepts either so RAG corpora migrated from the old format
    # render identically.
    row = {"feedback": "not_relevant", "title": "Sales Rep", "company": "BigCo"}
    assert format_entry_text(row) == "[not_relevant] Sales Rep @ BigCo"


def test_format_entry_text_handles_missing_fields():
    assert format_entry_text({"feedback_type": "other"}) == "[other] ? @ ?"
    assert format_entry_text({}) == "[unknown] ? @ ?"
    assert format_entry_text(None) == ""


# ---------------------------------------------------------------------------
# _parse_pgvector — the only fiddly cross-protocol detail
# ---------------------------------------------------------------------------

def test_parse_pgvector_handles_string_form():
    # PostgREST default serialization of a vector column.
    assert _parse_pgvector("[0.1,0.2,0.3]") == [0.1, 0.2, 0.3]
    assert _parse_pgvector("[1.0, -0.5, 0.0]") == [1.0, -0.5, 0.0]


def test_parse_pgvector_handles_list_form():
    assert _parse_pgvector([0.1, 0.2]) == [0.1, 0.2]


def test_parse_pgvector_returns_none_on_garbage():
    assert _parse_pgvector(None) is None
    assert _parse_pgvector("") is None
    assert _parse_pgvector("not a vector") is None
    assert _parse_pgvector("[a,b,c]") is None
    assert _parse_pgvector(42) is None


# ---------------------------------------------------------------------------
# to_pgvector_literal — the INSERT-side format (must be a "[...]" string, not
# a JSON array, or PostgREST sends a PG array literal that fails the cast)
# ---------------------------------------------------------------------------

def test_to_pgvector_literal_formats_string():
    assert to_pgvector_literal([0.1, 0.2, 0.3]) == "[0.1,0.2,0.3]"
    assert to_pgvector_literal([1, 2]) == "[1.0,2.0]"


def test_to_pgvector_literal_none_on_bad_input():
    assert to_pgvector_literal(None) is None
    assert to_pgvector_literal([]) is None
    assert to_pgvector_literal("[0.1,0.2]") is None  # already a string, not a list


def test_pgvector_round_trip():
    # What we write must parse back to the same vector on read.
    vec = [0.123, -0.456, 0.789]
    literal = to_pgvector_literal(vec)
    assert _parse_pgvector(literal) == vec


# ---------------------------------------------------------------------------
# load_candidate_preferences / count_feedback_entries
# ---------------------------------------------------------------------------

def test_load_candidate_preferences_returns_value():
    client = _Client({
        "preferences": [
            {"user_id": "u1", "candidate_preferences": "Prefers backend roles."},
            {"user_id": "u2", "candidate_preferences": "Wants ML."},
        ],
    })
    assert load_candidate_preferences("u1", client=client) == "Prefers backend roles."
    assert load_candidate_preferences("u2", client=client) == "Wants ML."


def test_load_candidate_preferences_returns_empty_on_miss():
    client = _Client({"preferences": []})
    assert load_candidate_preferences("nobody", client=client) == ""
    assert load_candidate_preferences("", client=client) == ""


def test_count_feedback_entries_reads_denormalized_counter():
    client = _Client({
        "profiles": [
            {"user_id": "u1", "feedback_count": 42},
            {"user_id": "u2", "feedback_count": 0},
        ],
    })
    assert count_feedback_entries("u1", client=client) == 42
    assert count_feedback_entries("u2", client=client) == 0
    assert count_feedback_entries("missing", client=client) == 0


def test_count_feedback_entries_coerces_string_counter():
    # Defensive: PostgREST occasionally serializes ints as strings under
    # certain RPC paths. Stay tolerant.
    client = _Client({"profiles": [{"user_id": "u1", "feedback_count": "7"}]})
    assert count_feedback_entries("u1", client=client) == 7


# ---------------------------------------------------------------------------
# load_feedback_embeddings — shape contract for RAG retrieval
# ---------------------------------------------------------------------------

def test_load_feedback_embeddings_shape_matches_rag_consumer():
    # core_embedding.retrieve_relevant_feedback expects
    # {"entries": [{"text": str, "embedding": list[float]}, ...]}.
    client = _Client({
        "feedback": [
            {
                "id": 1, "user_id": "u1",
                "feedback_type": "applied", "title": "Backend Eng", "company": "Stripe",
                "note": "great culture",
                "feedback_embeddings": {"embedding": "[0.1, 0.2, 0.3]"},
            },
            {
                "id": 2, "user_id": "u1",
                "feedback_type": "not_relevant", "title": "Sales", "company": "Foo",
                "note": None,
                "feedback_embeddings": {"embedding": "[0.4, 0.5]"},
            },
            # Row without embedding payload — must be dropped, not crash.
            {
                "id": 3, "user_id": "u1",
                "feedback_type": "bookmarked", "title": "?", "company": "?",
                "note": None,
                "feedback_embeddings": None,
            },
        ],
    })
    out = load_feedback_embeddings("u1", client=client)
    assert "entries" in out
    assert len(out["entries"]) == 2
    assert out["entries"][0]["text"].startswith("[applied] Backend Eng @ Stripe")
    assert out["entries"][0]["embedding"] == [0.1, 0.2, 0.3]
    assert out["entries"][1]["embedding"] == [0.4, 0.5]


def test_load_feedback_embeddings_tolerates_joined_list_form():
    # supabase-py sometimes returns the join as a single-element list when
    # an inner join hits exactly one row. Tolerate both shapes.
    client = _Client({
        "feedback": [
            {
                "id": 1, "user_id": "u1",
                "feedback_type": "applied", "title": "Eng", "company": "X",
                "note": None,
                "feedback_embeddings": [{"embedding": "[0.9]"}],
            },
        ],
    })
    out = load_feedback_embeddings("u1", client=client)
    assert len(out["entries"]) == 1
    assert out["entries"][0]["embedding"] == [0.9]


def test_load_feedback_embeddings_empty_user():
    client = _Client({"feedback": []})
    assert load_feedback_embeddings("u1", client=client) == {"entries": []}
    assert load_feedback_embeddings("", client=client) == {"entries": []}
