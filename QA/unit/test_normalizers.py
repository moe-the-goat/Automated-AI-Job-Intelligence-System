"""Coercion helpers in core_ai: _safe_int, _safe_bool, _safe_str, _normalize_effort.

These guard every dimension of the AI response against malformed values (the
model occasionally returns "82%" instead of 82, "true" instead of true, etc.).
A regression here would corrupt every downstream column.
"""
from pipeline.core_ai import _safe_int, _safe_bool, _safe_str, _normalize_effort


def test_safe_int_passthrough():
    assert _safe_int(82) == 82


def test_safe_int_clamps_high():
    assert _safe_int(150) == 100


def test_safe_int_clamps_low():
    assert _safe_int(-5) == 0


def test_safe_int_float_truncates():
    assert _safe_int(82.7) == 82


def test_safe_int_string_clean():
    assert _safe_int("82") == 82


def test_safe_int_string_with_percent():
    assert _safe_int("82%") == 82


def test_safe_int_string_decimal():
    assert _safe_int("82.5") == 82


def test_safe_int_string_decimal_with_percent():
    assert _safe_int("82.5%") == 82


def test_safe_int_garbage_returns_default():
    assert _safe_int("abc") == 0


def test_safe_int_empty_returns_default():
    assert _safe_int("") == 0


def test_safe_int_none_returns_default():
    assert _safe_int(None) == 0


def test_safe_int_custom_default():
    assert _safe_int(None, 50) == 50


def test_safe_int_bool_true():
    assert _safe_int(True) == 1


def test_safe_bool_true_passthrough():
    assert _safe_bool(True) is True


def test_safe_bool_false_passthrough():
    assert _safe_bool(False) is False


def test_safe_bool_string_variants():
    assert _safe_bool("true") is True
    assert _safe_bool("True") is True
    assert _safe_bool("yes") is True
    assert _safe_bool("1") is True
    assert _safe_bool("no") is False
    assert _safe_bool("") is False


def test_safe_bool_none_default():
    assert _safe_bool(None) is False


def test_safe_bool_numeric():
    assert _safe_bool(0) is False
    assert _safe_bool(7) is True


def test_safe_str_passthrough():
    assert _safe_str("hello") == "hello"


def test_safe_str_none_uses_default():
    assert _safe_str(None) == ""


def test_safe_str_strips_whitespace():
    assert _safe_str("  trimmed  ") == "trimmed"


def test_normalize_effort_valid_values():
    assert _normalize_effort("low") == "low"
    assert _normalize_effort("Medium") == "medium"
    assert _normalize_effort("HIGH") == "high"


def test_normalize_effort_invalid_returns_unknown():
    assert _normalize_effort("extreme") == "unknown"
    assert _normalize_effort(None) == "unknown"
    assert _normalize_effort("") == "unknown"
