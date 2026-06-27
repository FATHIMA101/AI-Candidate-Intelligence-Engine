"""
behavioral.py
=============
The signals doc's thesis: behavioural signals are often *more predictive of
whether a candidate can actually be hired* than the static profile. A
perfect-on-paper candidate who hasn't logged in for 6 months with a 5% response
rate is, for hiring purposes, not available.

So behaviour is applied as a **multiplier** on the skill/role fit, bounded
between ``behavioral_floor`` and ``behavioral_ceiling`` (config in taxonomy.py).
It can sink a strong-on-paper ghost, and give a small lift to a hot, responsive,
ready-to-move candidate — but it can never, by itself, turn a non-fit into a
fit. That asymmetry is deliberate.
"""

from __future__ import annotations

from datetime import date
from typing import List, Tuple

from .features import CandidateFeatures, _TODAY
from . import taxonomy as tax


def _recency_factor(last_active) -> Tuple[float, str]:
    if not last_active:
        return 0.5, "no recent activity recorded"
    days = (_TODAY - last_active).days
    if days <= 30:
        return 1.0, "active in the last month"
    if days <= 90:
        return 0.85, "active within 3 months"
    if days <= 180:
        return 0.6, "last active 3-6 months ago"
    return 0.3, f"dormant (~{days} days since last login)"


def compute(f: CandidateFeatures, candidate: dict, jd: tax.JDRequirements) -> None:
    """Mutates ``f``: sets ``behavioral_mult`` and ``behavioral_notes``."""
    signals = candidate.get("redrob_signals", {})
    notes: List[str] = []

    recency, rnote = _recency_factor(f.last_active)
    notes.append(rnote)

    # Responsiveness: a recruiter cannot hire someone who never replies.
    resp = f.response_rate
    resp_factor = 0.4 + 0.6 * min(1.0, resp / 0.6)  # 0%→0.4, 60%+→1.0
    if resp < 0.15:
        notes.append(f"low recruiter response rate ({resp:.0%})")
    elif resp >= 0.6:
        notes.append(f"highly responsive ({resp:.0%})")

    # Interview reliability + offer behaviour.
    icr = float(signals.get("interview_completion_rate", 0) or 0)
    oar = signals.get("offer_acceptance_rate", -1)
    icr_factor = 0.7 + 0.3 * icr
    oar_factor = 1.0
    if oar is not None and oar >= 0:
        # candidates who decline most offers are a weaker bet to actually close
        oar_factor = 0.85 + 0.15 * oar

    # Availability intent.
    intent = 1.0
    if f.open_to_work:
        intent *= 1.04
        notes.append("open to work")
    else:
        intent *= 0.94
    # Notice period: JD wants sub-30-day, can buy out up to 30.
    if f.notice_days <= 30:
        intent *= 1.03
    elif f.notice_days >= 90:
        intent *= 0.92
        notes.append(f"long notice period ({f.notice_days}d)")

    # Recruiter pull: profile saved / searched recently is mild positive demand.
    saved = int(signals.get("saved_by_recruiters_30d", 0) or 0)
    pull = 1.0 + min(0.04, saved / 250.0)

    raw = recency * resp_factor * icr_factor * oar_factor * intent * pull

    # Clamp into the configured envelope.
    lo, hi = jd.behavioral_floor, jd.behavioral_ceiling
    f.behavioral_mult = round(min(hi, max(lo, raw)), 4)
    f.behavioral_notes = notes
