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
    infer_role_tier,
    compute_role_weight,
    REGION_WEIGHTS,
    ROLE_WEIGHTS,
    TRUST_BOOST_MULTIPLIER,
    PATH_LABELS,
    PATH_STARTER_PROFILES,
    starter_profile_text,
)


# ---------------------------------------------------------------------------
# starter_profile_text — the per-path cold-start scoring prior
# ---------------------------------------------------------------------------

def test_every_path_label_has_a_starter_profile():
    # Sync guard: the starter map and the label map must not drift apart.
    assert set(PATH_STARTER_PROFILES) == set(PATH_LABELS)


def test_starter_profile_empty_for_no_paths():
    assert starter_profile_text([]) == ""
    assert starter_profile_text(None) == ""
    assert starter_profile_text(["not_a_real_path"]) == ""


def test_starter_profile_includes_each_chosen_path():
    out = starter_profile_text(["ai_ml", "backend"])
    assert "AI/ML" in out
    assert "Backend" in out
    # one line per path
    assert out.count("\n") == 1


def test_starter_profile_dedupes_and_caps():
    dupes = starter_profile_text(["backend", "backend", "backend"])
    assert dupes.count("Backend:") == 1
    many = starter_profile_text(list(PATH_LABELS.keys()))
    assert len([ln for ln in many.splitlines() if ln.strip()]) <= 5


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


def test_palestine_home_market_is_highly_preferred():
    """The candidate's home market — local on-site roles must get the top tier,
    not the neutral 1.00 that let global-remote/EU jobs out-rank them."""
    for loc in ("Ramallah", "Nablus, Palestine", "Gaza", "Hebron", "West Bank", "Bethlehem"):
        row = {"location": loc, "title": "Backend Developer", "description": ""}
        assert infer_region(row) == "highly_preferred", loc


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


# ---------------------------------------------------------------------------
# Role-tier weighting — AI > SWE > Other (added 2026-05-20)
# ---------------------------------------------------------------------------

def test_role_weight_ordering_ai_above_swe_above_other():
    """AI roles outweigh SWE roles, which outweigh everything else."""
    assert ROLE_WEIGHTS["ai"] > ROLE_WEIGHTS["swe"] > ROLE_WEIGHTS["other"]


def test_role_tier_ai_engineer():
    row = {"title": "AI Engineer"}
    assert infer_role_tier(row) == "ai"


def test_role_tier_ml_engineer():
    row = {"title": "Machine Learning Engineer"}
    assert infer_role_tier(row) == "ai"


def test_role_tier_data_scientist():
    row = {"title": "Junior Data Scientist"}
    assert infer_role_tier(row) == "ai"


def test_role_tier_nlp_engineer():
    row = {"title": "NLP Engineer (Remote)"}
    assert infer_role_tier(row) == "ai"


def test_role_tier_genai_engineer():
    row = {"title": "Generative AI Engineer"}
    assert infer_role_tier(row) == "ai"


def test_role_tier_llm_engineer():
    row = {"title": "LLM Engineer"}
    assert infer_role_tier(row) == "ai"


def test_role_tier_software_engineer():
    row = {"title": "Software Engineer"}
    assert infer_role_tier(row) == "swe"


def test_role_tier_backend_developer():
    row = {"title": "Backend Developer"}
    assert infer_role_tier(row) == "swe"


def test_role_tier_full_stack_engineer():
    row = {"title": "Full-Stack Engineer"}
    assert infer_role_tier(row) == "swe"


def test_role_tier_web_developer():
    row = {"title": "Web Developer (Remote)"}
    assert infer_role_tier(row) == "swe"


def test_role_tier_devops_is_other():
    """DevOps / SRE / Platform roles deliberately land in 'other' — not negative,
    just unboosted relative to AI/SWE."""
    row = {"title": "DevOps Engineer"}
    assert infer_role_tier(row) == "other"


def test_role_tier_data_analyst_is_other():
    row = {"title": "Data Analyst"}
    assert infer_role_tier(row) == "other"


def test_role_tier_ai_wins_over_swe_in_compound_title():
    """A title that matches both AI and SWE keywords should land in AI tier —
    e.g. 'Machine Learning Software Engineer' is fundamentally an AI role."""
    row = {"title": "Machine Learning Software Engineer"}
    assert infer_role_tier(row) == "ai"


def test_role_tier_empty_title_is_other():
    assert infer_role_tier({"title": ""}) == "other"
    assert infer_role_tier({}) == "other"


def test_compute_role_weight_matches_table():
    assert compute_role_weight({"title": "AI Engineer"}) == ROLE_WEIGHTS["ai"]
    assert compute_role_weight({"title": "Software Engineer"}) == ROLE_WEIGHTS["swe"]
    assert compute_role_weight({"title": "DevOps"}) == ROLE_WEIGHTS["other"]


def test_combined_weight_now_includes_role():
    """compute_combined_weight = region * trust * role.

    An AI engineer in Berlin (trusted) should beat the equivalent untitled
    row by the AI role multiplier alone — proving role is in the product.
    """
    ai_eu_trusted = {"location": "Berlin", "title": "AI Engineer", "pre_flagged_trusted": True}
    untitled_eu_trusted = {"location": "Berlin", "title": "", "pre_flagged_trusted": True}
    ratio = compute_combined_weight(ai_eu_trusted) / compute_combined_weight(untitled_eu_trusted)
    assert abs(ratio - ROLE_WEIGHTS["ai"] / ROLE_WEIGHTS["other"]) < 1e-9


# --- Per-user path weighting (Tier 5b) ---

def test_path_weight_boosts_on_path_title_and_neutral_off_path():
    from pipeline.region_weighting import PATH_MATCH_WEIGHT
    assert compute_role_weight({"title": "Backend Engineer"}, ["backend"]) == PATH_MATCH_WEIGHT
    assert compute_role_weight({"title": "DevOps Engineer"}, ["devops"]) == PATH_MATCH_WEIGHT
    assert compute_role_weight({"title": "Backend Engineer"}, ["frontend"]) == 1.00


def test_path_weight_empty_falls_back_to_legacy_tiers():
    """No paths → the legacy ai>swe>other behavior, unchanged for existing users."""
    assert compute_role_weight({"title": "AI Engineer"}, []) == ROLE_WEIGHTS["ai"]
    assert compute_role_weight({"title": "AI Engineer"}, None) == ROLE_WEIGHTS["ai"]
    assert compute_role_weight({"title": "Software Engineer"}, []) == ROLE_WEIGHTS["swe"]


def test_path_weight_lifts_devops_off_the_other_floor():
    """The point of the feature: a DevOps user's DevOps role is boosted, where the
    hardcoded tiers left every DevOps role at 'other' (1.00)."""
    from pipeline.region_weighting import PATH_MATCH_WEIGHT
    legacy = compute_role_weight({"title": "DevOps Engineer"})            # no paths -> other
    per_user = compute_role_weight({"title": "DevOps Engineer"}, ["devops"])
    assert legacy == ROLE_WEIGHTS["other"]
    assert per_user == PATH_MATCH_WEIGHT > legacy


def test_combined_weight_threads_paths():
    from pipeline.region_weighting import PATH_MATCH_WEIGHT
    row = {"location": "Berlin", "title": "DevOps Engineer"}
    ratio = compute_combined_weight(row, ["devops"]) / compute_combined_weight(row)
    assert abs(ratio - PATH_MATCH_WEIGHT / ROLE_WEIGHTS["other"]) < 1e-9


def test_format_paths_labels():
    from pipeline.region_weighting import format_paths
    assert format_paths(["backend", "ai_ml"]) == "Backend, AI/ML"
    assert format_paths([]) == ""
    assert format_paths(None) == ""
    assert format_paths(["something_new"]) == "Something New"  # unknown slug title-cased


def test_ai_role_beats_swe_role_at_equal_geography_and_trust():
    """The whole point of role-tier weighting: AI roles outrank SWE roles
    at otherwise-identical regional and trust weighting."""
    ai_role = {"location": "Berlin", "title": "AI Engineer", "pre_flagged_trusted": False}
    swe_role = {"location": "Berlin", "title": "Software Engineer", "pre_flagged_trusted": False}
    assert compute_combined_weight(ai_role) > compute_combined_weight(swe_role)


def test_swe_role_beats_other_role_at_equal_geography():
    swe = {"location": "Berlin", "title": "Software Engineer"}
    other = {"location": "Berlin", "title": "Data Analyst"}
    assert compute_combined_weight(swe) > compute_combined_weight(other)
