"""
scoring.py
==========
Combines the per-candidate components into a single, explainable score.

Design (defensible at the Stage-5 interview):

  fit      = Σ_c  w_c · component_c            (JD-weighted additive blend)
  fit      = fit · disqualifier_penalty        (JD's explicit "do NOT want")
  final    = fit · behavioral_multiplier       (availability, from signals doc)
  final    = 0 (tier 0) if honeypot gate fires

The additive blend is intentionally transparent rather than a black-box model:
there is no labelled training data shipped with the challenge (ground truth is
hidden), so a calibrated, inspectable scoring function — whose every weight maps
to a sentence in the JD — is both more honest and more defensible than a model
fit to self-invented labels. ``calibrate.py`` documents how the weights were
sanity-checked against unambiguous archetypes.

A coarse tier (0..5) is derived from the final score purely to drive the *tone*
of the generated reasoning (so a rank-5 candidate reads confident and a rank-95
reads hedged — the Stage-4 "rank consistency" check).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from .features import CandidateFeatures
from . import taxonomy as tax


@dataclass
class ScoredCandidate:
    features: CandidateFeatures
    components: Dict[str, float]
    fit: float
    disq_penalty: float
    disq_reasons: List[str]
    final: float
    tier: int


def _skill_match_component(f: CandidateFeatures) -> float:
    """Corroborated, JD-relevant skill coverage, weighted by group importance.

    Crucially this is the *trust*-weighted match, so a keyword-stuffer who lists
    every core term but with no endorsements / months / assessment scores barely
    moves this component.
    """
    total = 0.0
    for group, weight in tax.SKILL_GROUP_WEIGHTS.items():
        trust = f.skill_trust.get(group, 0.0)
        total += weight * trust
    max_possible = sum(tax.SKILL_GROUP_WEIGHTS.values())
    return round(total / max_possible, 4)


def _role_career_component(f: CandidateFeatures) -> float:
    """Title alignment + trajectory + concrete shipped-evidence.

    This is the heaviest single lever and the decisive defence against the
    "Marketing Manager with AI keywords" trap: a non-technical title and
    trajectory cap this near zero no matter what the skills array says.
    """
    # current title matters most (are they doing this work *now*?), trajectory
    # next (have they been on this path?), evidence confirms they shipped it.
    base = (0.45 * f.current_title_align
            + 0.30 * f.trajectory_align
            + 0.25 * f.shipped_evidence)
    # product-company experience is a JD-stated positive
    if f.n_product_roles >= 1:
        base = min(1.0, base + 0.06)
    return round(base, 4)


def compute_components(f: CandidateFeatures, semantic_score: float) -> Dict[str, float]:
    return {
        "role_and_career": _role_career_component(f),
        "skill_trust": _skill_match_component(f),
        "semantic": round(float(semantic_score), 4),
        "experience": f.exp_fit,
        "location": round(f.loc_fit, 4),
        "education_extras": f.edu_extra,
    }


def _disqualifier_penalty(f: CandidateFeatures,
                          active: Optional[Set[str]] = None) -> Tuple[float, List[str]]:
    """Return (multiplicative penalty in (0,1], reasons) per JD 'do NOT want'.

    ``active`` is the set of disqualifier names the JD actually declares (from
    role_dna). A penalty applies only if its name is in that set, so the scorer
    generalises to JDs that don't share this role's exclusions. When ``active``
    is None, all penalties apply (backward-compatible default).

    Penalties stack multiplicatively but are floored so a single soft flag does
    not nuke an otherwise-strong candidate — the JD frames most of these as
    "probably not", not "never".
    """
    def on(name: str) -> bool:
        return active is None or name in active

    penalty = 1.0
    reasons: List[str] = []

    if f.research_only and on("research_only"):
        penalty *= 0.35
        reasons.append("appears research-only with no production deployment")
    if f.services_only and on("services_only"):
        penalty *= 0.55
        reasons.append("entire career at IT-services firms (no product-company experience)")
    if f.cv_speech_robotics_primary and on("cv_speech_robotics"):
        penalty *= 0.50
        reasons.append("expertise is primarily CV/speech/robotics, not NLP/IR")
    if f.job_hopper and on("job_hopper"):
        penalty *= 0.85
        reasons.append("short average tenure (job-hopping pattern)")
    if f.noise_ratio >= 0.5 and f.current_title_align < 0.4 and on("keyword_pad"):
        penalty *= 0.7
        reasons.append("skills list reads as keyword padding around a non-technical role")

    return round(penalty, 4), reasons


def _tier_of(final: float) -> int:
    if final >= 0.62:
        return 5
    if final >= 0.50:
        return 4
    if final >= 0.38:
        return 3
    if final >= 0.24:
        return 2
    if final >= 0.10:
        return 1
    return 0


def score(f: CandidateFeatures, semantic_score: float,
          jd: tax.JDRequirements,
          active_disqualifiers: Optional[Set[str]] = None) -> ScoredCandidate:
    comps = compute_components(f, semantic_score)
    fit = sum(jd.component_weights[k] * v for k, v in comps.items())

    disq_penalty, disq_reasons = _disqualifier_penalty(f, active_disqualifiers)
    fit *= disq_penalty

    final = fit * f.behavioral_mult

    # Honeypot gate: an impossible profile is forced to tier 0.
    if f.honeypot_score >= 0.6:
        final = min(final, 0.01)

    final = round(min(1.0, max(0.0, final)), 6)
    return ScoredCandidate(
        features=f, components=comps, fit=round(fit, 6),
        disq_penalty=disq_penalty, disq_reasons=disq_reasons,
        final=final, tier=_tier_of(final),
    )
