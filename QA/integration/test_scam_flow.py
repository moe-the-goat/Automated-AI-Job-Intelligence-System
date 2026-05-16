"""Scam detection flow without hitting DDG.

Monkey-patches sys.modules['ddgs'] before importing detect_company_scam so the
real network call is replaced with a controllable fake.
"""
import sys
import types


class _FakeResultsForScam:
    """Returns body snippets that contain multiple scam keywords."""
    def text(self, query, **kwargs):
        return [
            {"body": "This company is a scam, they never paid me my final salary."},
            {"body": "Reddit thread warns this is a fake job and ghost job posting."},
        ]


class _FakeResultsClean:
    """Returns innocuous review snippets with no scam keywords."""
    def text(self, query, **kwargs):
        return [
            {"body": "Great place to work, friendly team, decent benefits."},
            {"body": "Solid mid-size tech company with growth potential."},
        ]


class _FakeResultsSingleMention:
    """One scam keyword in total — below the threshold."""
    def text(self, query, **kwargs):
        return [
            {"body": "An article titled 'How to spot a scam job in 2026'."},
            {"body": "Generic listicle, no other red flags."},
        ]


def _install_fake_ddgs(fake_class):
    fake_module = types.ModuleType("ddgs")
    fake_module.DDGS = fake_class
    sys.modules["ddgs"] = fake_module


def test_scam_detected_when_signals_present():
    _install_fake_ddgs(_FakeResultsForScam)
    from core_ai import detect_company_scam
    assert detect_company_scam("Zetheta Algorithms Pvt Ltd") is True


def test_scam_not_detected_when_results_clean():
    _install_fake_ddgs(_FakeResultsClean)
    from core_ai import detect_company_scam
    assert detect_company_scam("Real Indian Tech Pvt Ltd") is False


def test_scam_not_detected_below_threshold():
    """A single 'scam' mention across all queries shouldn't trip the flag."""
    _install_fake_ddgs(_FakeResultsSingleMention)
    from core_ai import detect_company_scam
    assert detect_company_scam("Generic Co") is False


def test_scam_handles_empty_company():
    _install_fake_ddgs(_FakeResultsForScam)
    from core_ai import detect_company_scam
    assert detect_company_scam("") is False
    assert detect_company_scam(None) is False


def test_scam_handles_ddg_exception():
    """If DDGS raises (network down, rate-limited), function returns False, not throws."""
    class _ExplodingDDGS:
        def text(self, query, **kwargs):
            raise RuntimeError("simulated network failure")
    _install_fake_ddgs(_ExplodingDDGS)
    from core_ai import detect_company_scam
    assert detect_company_scam("Whatever Co") is False
