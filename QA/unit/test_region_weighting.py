"""Region + trust weighting that biases the embedding pre-ranker.

These weights are the mechanism that moves EU / Americas / Middle East /
fully-remote postings above India-only postings of equal raw similarity, so
the AI top-N cutoff lands on actionable jobs first.
"""
from pipeline.region_weighting import (
    infer_region,
    compute_region_weight,
    compute_trust_weight,
    compute_combined_weight,
    REGION_WEIGHTS,
    TRUST_BOOST_MULTIPLIER,
)


# ---------------------------------------------------------------------------
# infer_region — top-tier (highly preferred)
# ---------------------------------------------------------------------------

def test_worldwide_remote_is_highly_preferred():
    row = {"location": "Remote, Worldwide", "title": "Software Engineer", "description": ""}
    assert infer_region(row) == "highly_preferred"


def test_anywhere_is_highly_preferred():
    row = {"location": "Anywhere", "title": "Engineer", "description": ""}
    assert infer_region(row) == "highly_preferred"


def test_fully_remote_marker_is_highly_preferred():
    row = {"location": "", "title": "Fully Remote ML Engineer", "description": ""}
    assert infer_region(row) == "highly_preferred"


def test_uae_is_highly_preferred():
    row = {"location": "Dubai, UAE", "title": "Backend Engineer", "description": ""}
    assert infer_region(row) == "highly_preferred"


def test_egypt_is_highly_preferred():
    """MixRank's "Junior Software Engineer — Remote, Egypt" — Middle East timezone-friendly."""
    row = {"location": "", "title": "Junior Software Engineer — Remote, Egypt", "description": ""}
    assert infer_region(row) == "highly_preferred"


def test_turkey_is_highly_preferred():
    row = {"location": "", "title": "Junior Software Engineer — Remote, Türkiye", "description": ""}
    assert infer_region(row) == "highly_preferred"


def test_jordan_is_highly_preferred():
    row = {"location": "Amman, Jordan", "title": "AI Engineer", "description": ""}
    assert infer_region(row) == "highly_preferred"


def test_kenya_is_highly_preferred():
    row = {"location": "Nairobi, Kenya", "title": "Engineer", "description": ""}
    assert infer_region(row) == "highly_preferred"


# ---------------------------------------------------------------------------
# infer_region — preferred (EU / Americas / non-India Asia)
# ---------------------------------------------------------------------------

def test_germany_is_preferred():
    row = {"location": "Berlin, Germany", "title": "Data Scientist", "description": ""}
    assert infer_region(row) == "preferred"


def test_brazil_is_preferred():
    """MixRank's "Junior Software Engineer — Remote, Brazil" — LATAM."""
    row = {"location": "", "title": "Junior Software Engineer — Remote, Brazil", "description": ""}
    assert infer_region(row) == "preferred"


def test_mexico_is_preferred():
    row = {"location": "", "title": "Junior Software Engineer — Remote, Mexico", "description": ""}
    assert infer_region(row) == "preferred"


def test_canada_is_preferred():
    row = {"location": "Toronto, Canada", "title": "ML Engineer", "description": ""}
    assert infer_region(row) == "preferred"


def test_romania_is_preferred():
    """ING Hubs Romania case from the 2026-05-17 email."""
    row = {"location": "Bucharest, Romania", "title": "Data Scientist | KYC", "description": ""}
    assert infer_region(row) == "preferred"


def test_georgia_country_is_preferred():
    """The country Georgia (not the US state) is in EU's neighbourhood."""
    row = {"location": "", "title": "Junior Software Engineer — Remote, Georgia", "description": ""}
    assert infer_region(row) == "preferred"


def test_latam_signal_alone_is_preferred():
    row = {"location": "LATAM", "title": "Engineer", "description": ""}
    assert infer_region(row) == "preferred"


def test_emea_signal_is_preferred():
    row = {"location": "EMEA", "title": "Engineer", "description": ""}
    assert infer_region(row) == "preferred"


# ---------------------------------------------------------------------------
# infer_region — deweighted (India)
# ---------------------------------------------------------------------------

def test_india_explicit_is_deweighted():
    row = {"location": "Bangalore, India", "title": "Software Intern", "description": ""}
    assert infer_region(row) == "deweighted"


def test_pvt_ltd_in_company_is_deweighted():
    """The "Pvt Ltd" suffix is a strong India signal."""
    row = {"location": "", "title": "Data Analyst", "company": "Zetheta Algorithms Private Limited", "description": ""}
    assert infer_region(row) == "deweighted"


def test_bengaluru_is_deweighted():
    row = {"location": "Bengaluru, KA", "title": "Engineer", "company": "Co", "description": ""}
    assert infer_region(row) == "deweighted"


def test_hyderabad_is_deweighted():
    row = {"location": "Hyderabad", "title": "Engineer", "company": "Co", "description": ""}
    assert infer_region(row) == "deweighted"


# ---------------------------------------------------------------------------
# infer_region — heavily deweighted (sanctioned / blocked)
# ---------------------------------------------------------------------------

def test_russia_is_heavily_deweighted():
    row = {"location": "Moscow, Russia", "title": "Engineer", "description": ""}
    assert infer_region(row) == "heavily_deweighted"


def test_china_is_heavily_deweighted():
    row = {"location": "Beijing, China", "title": "Engineer", "description": ""}
    assert infer_region(row) == "heavily_deweighted"


def test_iran_is_heavily_deweighted():
    row = {"location": "Tehran, Iran", "title": "Engineer", "description": ""}
    assert infer_region(row) == "heavily_deweighted"


# ---------------------------------------------------------------------------
# infer_region — priority + neutral fallback
# ---------------------------------------------------------------------------

def test_worldwide_remote_overrides_india_mention():
    """An India-HQ company offering a worldwide-remote role is still actionable."""
    row = {"location": "Remote, Worldwide",
           "title": "Engineer", "company": "Some India Pvt Ltd", "description": ""}
    assert infer_region(row) == "highly_preferred"


def test_heavily_deweighted_overrides_preferred():
    """A 'Remote (Europe / Russia)' posting still gets blocked — Russia is sanctioned."""
    row = {"location": "", "title": "Engineer", "description": "We hire from Germany, France, and Russia."}
    assert infer_region(row) == "heavily_deweighted"


def test_neutral_when_no_signal():
    row = {"location": "", "title": "Engineer", "company": "", "description": ""}
    assert infer_region(row) == "neutral"


def test_blank_row_is_neutral():
    assert infer_region({}) == "neutral"


# ---------------------------------------------------------------------------
# compute_region_weight — multipliers
# ---------------------------------------------------------------------------

def test_region_weight_matches_constants():
    assert compute_region_weight({"location": "Worldwide"}) == REGION_WEIGHTS["highly_preferred"]
    assert compute_region_weight({"location": "Germany"}) == REGION_WEIGHTS["preferred"]
    assert compute_region_weight({"location": ""}) == REGION_WEIGHTS["neutral"]
    assert compute_region_weight({"location": "Bangalore"}) == REGION_WEIGHTS["deweighted"]
    assert compute_region_weight({"location": "Moscow"}) == REGION_WEIGHTS["heavily_deweighted"]


def test_region_weight_ordering():
    """Highest > preferred > neutral > deweighted > heavily_deweighted."""
    h  = REGION_WEIGHTS["highly_preferred"]
    p  = REGION_WEIGHTS["preferred"]
    n  = REGION_WEIGHTS["neutral"]
    d  = REGION_WEIGHTS["deweighted"]
    hd = REGION_WEIGHTS["heavily_deweighted"]
    assert h > p > n > d > hd


def test_region_weight_neutral_is_one():
    """Neutral must be exactly 1.0 so it preserves the raw similarity ordering."""
    assert REGION_WEIGHTS["neutral"] == 1.0


# ---------------------------------------------------------------------------
# compute_trust_weight
# ---------------------------------------------------------------------------

def test_trust_weight_applies_only_when_flagged():
    assert compute_trust_weight({"pre_flagged_trusted": True}) == TRUST_BOOST_MULTIPLIER
    assert compute_trust_weight({"pre_flagged_trusted": False}) == 1.0
    assert compute_trust_weight({}) == 1.0


def test_trust_weight_is_strictly_greater_than_one():
    """A trusted company should always edge out an untrusted one at equal raw similarity."""
    assert TRUST_BOOST_MULTIPLIER > 1.0


# ---------------------------------------------------------------------------
# compute_combined_weight — region * trust
# ---------------------------------------------------------------------------

def test_trusted_eu_company_beats_untrusted_neutral_at_equal_similarity():
    """The whole point of the weighting: bias the top-N toward known good geos + companies."""
    trusted_eu = {"location": "Berlin", "pre_flagged_trusted": True}
    untrusted_neutral = {"location": "", "pre_flagged_trusted": False}
    assert compute_combined_weight(trusted_eu) > compute_combined_weight(untrusted_neutral)


def test_combined_weight_for_highly_preferred_and_trusted_is_highest():
    row = {"location": "Worldwide", "pre_flagged_trusted": True}
    expected = REGION_WEIGHTS["highly_preferred"] * TRUST_BOOST_MULTIPLIER
    assert compute_combined_weight(row) == expected


def test_combined_weight_for_heavily_deweighted_no_trust_is_lowest():
    row = {"location": "Beijing", "pre_flagged_trusted": False}
    assert compute_combined_weight(row) == REGION_WEIGHTS["heavily_deweighted"]


def test_india_with_trust_boost_still_below_eu():
    """A trusted India company (1.25 * 0.70 = 0.875) should NOT beat a plain EU
    job (1.0 * 1.15 = 1.15). Sanity-check the magnitudes are right."""
    india_trusted = {"location": "Bangalore", "pre_flagged_trusted": True}
    eu_plain = {"location": "Berlin", "pre_flagged_trusted": False}
    assert compute_combined_weight(india_trusted) < compute_combined_weight(eu_plain)
