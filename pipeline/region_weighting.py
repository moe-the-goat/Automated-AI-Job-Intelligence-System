"""Region + trust weighting for the embedding-similarity pre-ranker.

The raw cosine similarity between the CV embedding and a job-description
embedding tells us "how textually similar is this job to my CV." It does NOT
know that:
  - A "Junior Software Engineer — Remote, Egypt" is far more actionable for a
    Palestine-based candidate than a "Software Engineer Intern" at an Indian
    placement agency, even if both happen to mention Python.
  - A Stripe / GitLab / Spotify role deserves a head start over a random
    unknown company's posting at the same similarity score.

We apply two multiplicative weights to the raw similarity AFTER it's computed,
and rank by `weighted_score = similarity * region_weight * trust_weight`.

  region_weight    in [0.50, 1.30]   — based on inferred geography
  trust_weight     in [1.00, 1.30]   — boosts trust_boost matches

The raw `similarity` column is kept for visibility/debug; ranking uses
`weighted_score`. The AI evaluator sees neither — its job is to score the
match on the merits.

Why multiplicative not additive: similarity already ranges 0..1, and we want
percentage swings that preserve the ordering within a region tier. A 1.30x
boost on a 0.55 (good fit) Egypt role yields 0.715, jumping it ahead of a
1.0x 0.60 unrelated US role. Additive offsets would compress the dynamic
range; multiplication keeps strong fits strong.
"""
from __future__ import annotations
import re


# ---------------------------------------------------------------------------
# Region inference — keyword lists tuned for the Palestine-based candidate
# ---------------------------------------------------------------------------
#
# Categories are CHECK-IN-ORDER: the first category that matches wins. So if a
# job description mentions both "India" and "remote worldwide", we DO want
# India to lose — hence "deweighted" checked AFTER "highly_preferred" so a
# job that's explicitly worldwide-remote keeps its boost even if the company
# happens to have an India office.

# Tier 1: jobs that are essentially location-independent for a Palestine-based
# candidate (full remote, Middle East, Africa proper). Biggest boost.
_HIGHLY_PREFERRED_PATTERNS = (
    r"\bworldwide\b", r"\banywhere\b", r"\bglobal(?:ly)?\b",
    r"\bfully\s+remote\b", r"\b100%\s+remote\b",
    r"\bremote[\s\-]*first\b",
    # Middle East: time-zone friendly, sometimes Palestine-hiring directly
    r"\buae\b", r"\bu\.a\.e\b", r"\bdubai\b", r"\babu\s*dhabi\b",
    r"\bjordan\b", r"\bamman\b",
    r"\begypt\b", r"\bcairo\b", r"\balexandria\b",
    r"\bsaudi\s+arabia\b", r"\briyadh\b", r"\bjeddah\b",
    r"\bqatar\b", r"\bdoha\b",
    r"\bbahrain\b", r"\bkuwait\b", r"\boman\b", r"\bmuscat\b",
    r"\blebanon\b", r"\bbeirut\b",
    r"\bturkey\b", r"\btürkiye\b", r"\bistanbul\b", r"\bankara\b",
    # Africa (English-speaking + Maghreb)
    r"\bsouth\s+africa\b", r"\bjohannesburg\b", r"\bcape\s+town\b",
    r"\bkenya\b", r"\bnairobi\b",
    r"\bnigeria\b", r"\blagos\b", r"\babuja\b",
    r"\bghana\b", r"\baccra\b",
    r"\bmorocco\b", r"\bcasablanca\b", r"\brabat\b",
    r"\btunisia\b", r"\btunis\b",
)

# Tier 2: EU and other Americas. Strong fit, work-auth usually manageable.
_PREFERRED_PATTERNS = (
    # EU core
    r"\bgermany\b", r"\bberlin\b", r"\bmunich\b", r"\bmunchen\b", r"\bhamburg\b",
    r"\bfrance\b", r"\bparis\b", r"\blyon\b",
    r"\bnetherlands\b", r"\bamsterdam\b", r"\brotterdam\b", r"\butrecht\b",
    r"\bireland\b", r"\bdublin\b",
    r"\bspain\b", r"\bmadrid\b", r"\bbarcelona\b",
    r"\bitaly\b", r"\brome\b", r"\bmilan\b",
    r"\bportugal\b", r"\blisbon\b", r"\bporto\b",
    r"\bpoland\b", r"\bwarsaw\b", r"\bkrakow\b",
    r"\bsweden\b", r"\bstockholm\b",
    r"\bdenmark\b", r"\bcopenhagen\b",
    r"\bfinland\b", r"\bhelsinki\b",
    r"\bnorway\b", r"\boslo\b",
    r"\baustria\b", r"\bvienna\b",
    r"\bbelgium\b", r"\bbrussels\b",
    r"\bswitzerland\b", r"\bzurich\b", r"\bgeneva\b",
    r"\bunited\s+kingdom\b", r"\bengland\b", r"\bscotland\b", r"\blondon\b",
    r"\b(?<!new\s)manchester\b", r"\bedinburgh\b",
    r"\bczech\s+republic\b", r"\bczechia\b", r"\bprague\b",
    r"\bromania\b", r"\bbucharest\b",
    r"\bhungary\b", r"\bbudapest\b",
    r"\bgreece\b", r"\bathens\b",
    r"\bbulgaria\b", r"\bsofia\b",
    r"\bcroatia\b", r"\bzagreb\b",
    r"\bserbia\b", r"\bbelgrade\b",
    r"\bestonia\b", r"\btallinn\b",
    r"\blithuania\b", r"\bvilnius\b",
    r"\blatvia\b", r"\briga\b",
    r"\bgeorgia\b",                                            # the country
    # LATAM + Canada
    r"\bcanada\b", r"\btoronto\b", r"\bvancouver\b", r"\bmontreal\b",
    r"\bbrazil\b", r"\bbrasil\b", r"\bsao\s+paulo\b", r"\bsão\s+paulo\b",
    r"\brio\s+de\s+janeiro\b",
    r"\bargentina\b", r"\bbuenos\s+aires\b",
    r"\bmexico\b", r"\bméxico\b", r"\bmexico\s+city\b",
    r"\bchile\b", r"\bsantiago\b",
    r"\bcolombia\b", r"\bbogota\b", r"\bbogotá\b", r"\bmedellin\b", r"\bmedellín\b",
    r"\bperu\b", r"\blima\b",
    r"\buruguay\b", r"\bmontevideo\b",
    r"\bcosta\s+rica\b",
    r"\blatam\b", r"\blatin\s+america\b",
    r"\beurope\b", r"\bemea\b", r"\beu[\-\s]based\b",
    # South Asia EXCLUDING India
    r"\bsri\s+lanka\b", r"\bcolombo\b",
    r"\bbangladesh\b", r"\bdhaka\b",
    r"\bnepal\b", r"\bkathmandu\b",
    r"\bpakistan\b", r"\blahore\b", r"\bislamabad\b",
)

# Tier 3 (deweight): India + a few hubs where our experience shows mostly
# low-quality postings. NOT a hard reject — just a 0.7 multiplier so a
# genuinely strong Indian job (high similarity, no suspicious flags) can
# still surface, while the noise floor gets pushed down.
_DEWEIGHTED_PATTERNS = (
    r"\bindia\b", r"\bindian\b",
    r"\bpvt\.?\s*ltd\b", r"\bpvt\.?\s*limited\b", r"\bprivate\s+limited\b",
    r"\b\(p\)\s*ltd\b",
    r"\bbangalore\b", r"\bbengaluru\b",
    r"\bmumbai\b", r"\bdelhi\b", r"\bnew\s+delhi\b",
    r"\bnoida\b", r"\bgurgaon\b", r"\bgurugram\b",
    r"\bchennai\b", r"\bpune\b", r"\bkolkata\b",
    r"\bhyderabad\b", r"\bahmedabad\b", r"\bjaipur\b",
)

# Tier 4 (heavy deweight): sanctioned / politically obstructed regions where
# Palestine-based work is effectively impossible. Cap rather than zero so
# trivia-level matches still appear at the very bottom.
_HEAVILY_DEWEIGHTED_PATTERNS = (
    r"\brussia\b", r"\bmoscow\b", r"\bst[\.\s]+petersburg\b",
    r"\bchina\b", r"\bbeijing\b", r"\bshanghai\b", r"\bshenzhen\b",
    r"\bnorth\s+korea\b", r"\bdprk\b",
    r"\bbelarus\b", r"\bminsk\b",
    r"\biran\b", r"\btehran\b",
)

# Pre-compile each tier as one OR'd regex for speed (we run this on every job).
_HIGHLY_PREFERRED_RE = re.compile("|".join(_HIGHLY_PREFERRED_PATTERNS), re.IGNORECASE)
_PREFERRED_RE = re.compile("|".join(_PREFERRED_PATTERNS), re.IGNORECASE)
_DEWEIGHTED_RE = re.compile("|".join(_DEWEIGHTED_PATTERNS), re.IGNORECASE)
_HEAVILY_DEWEIGHTED_RE = re.compile("|".join(_HEAVILY_DEWEIGHTED_PATTERNS), re.IGNORECASE)


# Weight values. Tuned so:
#  - 1.30 boost on a 0.55 similarity = 0.715 (clearly jumps past 0.65 raw)
#  - 0.70 deweight on a 0.65 similarity = 0.455 (clearly sinks below 0.55 raw)
#  - 0.50 heavy deweight effectively floors a 0.80 to 0.40
REGION_WEIGHTS = {
    "highly_preferred":   1.30,
    "preferred":          1.15,
    "neutral":            1.00,
    "deweighted":         0.70,
    "heavily_deweighted": 0.50,
}

# Trust boost is independent and multiplicative — a trusted EU company stacks
# both bonuses (1.15 * 1.25 = 1.4375).
TRUST_BOOST_MULTIPLIER = 1.25
NEUTRAL_TRUST_MULTIPLIER = 1.00


# ---------------------------------------------------------------------------
# Role-tier weighting (added 2026-05-20)
# ---------------------------------------------------------------------------
#
# On top of region + trust, we now weight the role itself based on how
# central it is to the candidate's profile. The CV centers on AI / GenAI /
# RAG / LLM systems, with strong general software engineering as the
# foundation. So we want:
#   AI roles  > Software Engineering roles  > Everything else
#
# The role weight is applied AFTER region and trust in compute_combined_weight,
# so the final multiplier is (region * trust * role). Maxed out: a trusted
# EU AI Engineer role gets 1.15 * 1.25 * 1.20 = 1.725x — clearly ahead of
# a US generic role at 1.00x.
#
# Why these tiers (not finer-grained):
#  - AI (1.20x): The candidate's strongest, most differentiated experience.
#    Anything matching ML/AI/NLP/CV/LLM/GenAI/Data Science deserves the lead.
#  - SWE (1.10x): The candidate is a competent generalist. Strong fit for any
#    software / backend / frontend / fullstack / web role even when AI isn't
#    the explicit topic.
#  - Other (1.00x): Everything else. DevOps, SRE, platform, analyst, etc. —
#    not negative, just unboosted. Those roles still ride the region weight.
_AI_ROLE_PATTERNS = (
    r"\bai\s+(?:engineer|developer|scientist|intern|researcher|architect|lead)\b",
    r"\bartificial\s+intelligence\b",
    r"\bml\s+(?:engineer|ops|intern|developer|researcher|scientist)\b",
    r"\bmlops\b",
    r"\bmachine\s+learning\b",
    r"\bdeep\s+learning\b",
    r"\bneural\s+network\b",
    r"\bnlp\b",
    r"\bnatural\s+language\s+processing\b",
    r"\bcomputer\s+vision\b",
    r"\bllm\s+(?:engineer|intern|developer|researcher)\b",
    r"\blarge\s+language\s+model\b",
    r"\bgenerative\s+ai\b",
    r"\bgen\s*ai\b",
    r"\bdata\s+scien(?:tist|ce)\b",
    r"\bresearch\s+(?:engineer|scientist)\b",
    r"\bapplied\s+scientist\b",
    r"\brag\s+(?:engineer|developer)\b",
    r"\bai/ml\b", r"\bml/ai\b",
)

_SWE_ROLE_PATTERNS = (
    r"\bsoftware\s+(?:engineer|developer)\b",
    r"\b(?:backend|back[\-\s]end)\s+(?:engineer|developer)\b",
    r"\b(?:frontend|front[\-\s]end)\s+(?:engineer|developer)\b",
    r"\bfull[\-\s]?stack\s+(?:engineer|developer)\b",
    r"\bfullstack\b",
    r"\bweb\s+(?:developer|engineer)\b",
    r"\bmobile\s+(?:developer|engineer)\b",
    r"\bios\s+developer\b", r"\bandroid\s+developer\b",
    r"\bjava\s+(?:developer|engineer)\b",
    r"\bpython\s+developer\b",
    r"\bjavascript\s+(?:developer|engineer)\b",
    r"\btypescript\s+developer\b",
    r"\breact\s+developer\b",
    r"\bnode\.?js\s+developer\b",
    r"\bdjango\s+developer\b",
    r"\bmember\s+of\s+technical\s+staff\b",
    r"\bsystems?\s+engineer\b",
    r"\bembedded\s+(?:developer|engineer)\b",
    # bare "developer" / "engineer" as fallback (least specific, last on the list)
    r"\bjunior\s+(?:developer|engineer)\b",
    r"\bentry[\-\s]level\s+(?:developer|engineer)\b",
)

_AI_ROLE_RE = re.compile("|".join(_AI_ROLE_PATTERNS), re.IGNORECASE)
_SWE_ROLE_RE = re.compile("|".join(_SWE_ROLE_PATTERNS), re.IGNORECASE)

ROLE_WEIGHTS = {
    "ai":    1.20,
    "swe":   1.10,
    "other": 1.00,
}


def infer_role_tier(row):
    """Categorize a job row into "ai", "swe", or "other" based on the title.

    Order matters: AI is checked first so "Machine Learning Software Engineer"
    correctly lands in the AI tier (not SWE) even though both keywords match.
    """
    title = str(row.get("title", "") or "")
    if not title.strip():
        return "other"

    if _AI_ROLE_RE.search(title):
        return "ai"
    if _SWE_ROLE_RE.search(title):
        return "swe"
    return "other"


def compute_role_weight(row):
    """Multiplier for role tier, applied alongside region and trust weights."""
    return ROLE_WEIGHTS[infer_role_tier(row)]


def _row_text(row):
    """Pool the row fields most likely to carry geographic signal."""
    parts = [
        str(row.get("location", "") or ""),
        str(row.get("title", "") or ""),
        str(row.get("company", "") or ""),
        # Description gets the smallest weight because it often mentions many
        # locations (HQ, offices, applicant pools). We cap to 1200 chars to
        # avoid pollution by unrelated geo mentions deep in the listing.
        str(row.get("description", "") or "")[:1200],
    ]
    return " ".join(parts)


def infer_region(row):
    """Categorize a job row into one of the five region tiers.

    Returns one of: "highly_preferred", "preferred", "neutral",
    "deweighted", "heavily_deweighted".

    Order of checks matters: a job mentioning both India AND worldwide-remote
    should be "highly_preferred" because the remote eligibility overrides the
    employer's geography for our purposes. Same logic for EU companies that
    sometimes name India in their office list — the remote-friendly tier wins.
    """
    text = _row_text(row)
    if not text.strip():
        return "neutral"

    # 1. Heavy deweight wins over everything else (sanctions / blocked).
    if _HEAVILY_DEWEIGHTED_RE.search(text):
        return "heavily_deweighted"

    # 2. Highly preferred wins over India deweight. A "Remote Worldwide"
    # job from an India HQ company is still actionable.
    if _HIGHLY_PREFERRED_RE.search(text):
        return "highly_preferred"

    # 3. Preferred (EU/Americas/non-India Asia) also wins over India deweight.
    if _PREFERRED_RE.search(text):
        return "preferred"

    # 4. India / placement-mill markers.
    if _DEWEIGHTED_RE.search(text):
        return "deweighted"

    # 5. No geographic signal -> neutral (typically US-based or unspecified).
    return "neutral"


def compute_region_weight(row):
    """Multiplier for region tier, applied to similarity for ranking."""
    return REGION_WEIGHTS[infer_region(row)]


def compute_trust_weight(row):
    """Multiplier for trust_boost flag, applied alongside the region weight.

    The flag is set by `core_filter._pre_flag_reputation` against
    `data/reputation.json` trust_boost entries — typically big-name
    Palestine-friendly remote employers (GitLab, Anthropic, Stripe, Spotify,
    Mistral, etc.).
    """
    if bool(row.get("pre_flagged_trusted", False)):
        return TRUST_BOOST_MULTIPLIER
    return NEUTRAL_TRUST_MULTIPLIER


def compute_combined_weight(row):
    """Final multiplier = region * trust * role. Used by the embedding ranker."""
    return compute_region_weight(row) * compute_trust_weight(row) * compute_role_weight(row)
