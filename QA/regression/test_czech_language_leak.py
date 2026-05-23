"""Regression: Czech-language descriptions were slipping through _is_english_description
because langdetect misidentified mixed Czech/English tech descriptions as English.
Fixed by adding a pre-check for ř/ů/ő which are characters unique to Czech, Slovak,
and Hungarian that never appear in English text."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from pipeline.core_filter import _is_english_description


def test_pure_czech_description_rejected():
    czech = (
        "Hledáme zkušeného vývojáře softwaru pro náš tým. "
        "Pracovní náplň zahrnuje vývoj webových aplikací v Pythonu. "
        "Požadujeme znalost Reactu a TypeScriptu. Nabízíme skvělé podmínky."
    )
    assert _is_english_description(czech) is False


def test_mixed_czech_english_with_tech_keywords_rejected():
    # Simulates a real Czech job posting: Czech prose + English tech keywords.
    # This is the case that fooled langdetect before the ř/ů fix.
    mixed = (
        "Hledáme junior vývojáře. Požadujeme zkušenosti s Pythonem, React, FastAPI. "
        "Nabízíme práci na dálku a příjemné pracovní prostředí. Plat závisí na zkušenostech."
    )
    assert _is_english_description(mixed) is False


def test_english_description_with_czech_company_name_kept():
    # English description from a Czech company must NOT be rejected.
    english = (
        "We are looking for a junior software engineer to join our Prague-based team. "
        "Requirements: Python, FastAPI, Docker. The role is fully remote worldwide. "
        "1-2 years of experience preferred."
    )
    assert _is_english_description(english) is True


def test_short_description_under_threshold_kept():
    # Descriptions under 100 chars are kept regardless of language.
    short_czech = "Hledáme vývojáře."
    assert _is_english_description(short_czech) is True


def test_english_description_not_rejected():
    english = (
        "We are hiring a machine learning engineer with experience in PyTorch and "
        "LangChain. The role involves building RAG pipelines and deploying models "
        "to production via FastAPI. Remote worldwide, no restrictions."
    )
    assert _is_english_description(english) is True
