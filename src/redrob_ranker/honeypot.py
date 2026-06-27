"""
honeypot.py
===========
The dataset hides ~80 "honeypot" candidates with subtly *impossible* profiles
(e.g. 8 years at a company founded 3 years ago; 'expert' in 10 skills with 0
months of use). Ground truth forces them to relevance tier 0, and a top-100
honeypot rate > 10% disqualifies the whole submission at Stage 3.

The spec says: "You can identify honeypots through careful profile inspection.
We expect a good ranking system to naturally avoid them; you don't need to
special-case them."

We do both: the structured scorer naturally down-ranks weird profiles, AND this
module adds an explicit internal-consistency gate as defence-in-depth. The gate
is deliberately built from *impossibilities* (claims that cannot all be true at
once), not from "looks too good", so it does not punish strong genuine
candidates — which is the precision concern when only 0.08% of the pool is a
honeypot.

Each check returns evidence in [0,1]. We combine them and expose a single
``honeypot_score`` plus the list of reasons, so the reasoning layer can be
honest about *why* a profile was suppressed.
"""

from __future__ import annotations

from datetime import date
from typing import List, Tuple

from .features import parse_date


def _years_between(d1: date, d2: date) -> float:
    return (d2 - d1).days / 365.25


def detect(candidate: dict) -> Tuple[float, List[str]]:
    """Return (honeypot_score in [0,1], reasons).

    honeypot_score >= ~0.6 should be treated as "force to tier 0".
    """
    reasons: List[str] = []
    evidence = 0.0

    profile = candidate.get("profile", {})
    yoe = float(profile.get("years_of_experience", 0) or 0)
    history = candidate.get("career_history", []) or []
    skills = candidate.get("skills", []) or []
    signals = candidate.get("redrob_signals", {})

    # --- Check 1: a single role lasts longer than the whole career ----------
    # (the "8 years at a 3-year-old company" family, internal version)
    for h in history:
        dur_m = h.get("duration_months", 0) or 0
        if dur_m > (yoe * 12) + 9:  # +9mo slack for rounding
            reasons.append(
                f"a single role claims {dur_m//12}y{dur_m%12}m, exceeding the "
                f"stated {yoe:.1f}y total experience")
            evidence = max(evidence, 0.9)
            break

    # --- Check 2: summed tenure wildly exceeds total experience -------------
    total_m = sum((h.get("duration_months", 0) or 0) for h in history)
    if total_m > (yoe * 12) + 30:  # 2.5y slack for overlapping roles
        reasons.append(
            f"role durations sum to {total_m/12:.1f}y vs a stated {yoe:.1f}y "
            f"of experience")
        evidence = max(evidence, 0.75)

    # --- Check 3: duration_months contradicts its own start/end dates -------
    for h in history:
        sd = parse_date(h.get("start_date"))
        ed = parse_date(h.get("end_date"))
        dur_m = h.get("duration_months", 0) or 0
        if sd and ed:
            if ed < sd:
                reasons.append("a role ends before it starts")
                evidence = max(evidence, 0.95)
            span_m = max(0.0, _years_between(sd, ed) * 12)
            if dur_m > span_m + 12:  # claimed duration >> actual span
                reasons.append(
                    "a role's claimed duration far exceeds its start/end span")
                evidence = max(evidence, 0.7)
        # is_current rows should not carry a (non-null) past end_date
        if h.get("is_current") and ed is not None:
            reasons.append("a role is marked current yet has an end date")
            evidence = max(evidence, 0.55)

    # --- Check 4: high-proficiency skills with zero substantiation ----------
    # 'expert'/'advanced' but 0 months used AND 0 endorsements -> fabricated.
    fabricated = [
        s for s in skills
        if s.get("proficiency") in ("expert", "advanced")
        and (s.get("duration_months", 0) or 0) == 0
        and (s.get("endorsements", 0) or 0) == 0
    ]
    if len(fabricated) >= 3:
        reasons.append(
            f"{len(fabricated)} skills marked advanced/expert with 0 months of "
            f"use and 0 endorsements")
        evidence = max(evidence, 0.85)
    elif len(fabricated) >= 1:
        # weak signal on its own; contributes but does not condemn
        evidence = max(evidence, 0.30)

    # --- Check 5: experience inconsistent with earliest career start --------
    starts = [parse_date(h.get("start_date")) for h in history]
    starts = [s for s in starts if s]
    if starts:
        earliest = min(starts)
        career_span = _years_between(earliest, date(2026, 6, 1))
        # Claiming far more experience than the career timeline allows.
        if yoe > career_span + 4:
            reasons.append(
                f"claims {yoe:.1f}y experience but earliest role began only "
                f"{career_span:.1f}y ago")
            evidence = max(evidence, 0.7)

    # NOTE: signup-vs-last-active ordering and salary-band ordering were
    # evaluated as honeypot signals but rejected: in this synthetic pool those
    # two fields are generated independently of profile quality and fire on
    # ~7-8% of *all* candidates, so they are dataset noise, not the ~80
    # deliberately-impossible profiles. Using them would wrongly suppress
    # thousands of legitimate candidates. We keep only checks rooted in
    # *internal career/skill impossibility*, which is what the spec describes.

    return evidence, reasons
