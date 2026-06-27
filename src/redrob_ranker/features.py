"""
features.py
===========
Turns one raw candidate dict into a compact ``CandidateFeatures`` record.

This is where "reading the profile" happens. The extractor never trusts a skill
string on its own: it cross-references the skill against endorsements, months of
use, the platform's own skill-assessment scores, and whether the skill actually
shows up in the candidate's role descriptions. That corroboration is what
separates a real engineer from a keyword-stuffer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional

from . import taxonomy as tax

_TODAY = date(2026, 6, 1)  # fixed "now" for deterministic recency scoring


def parse_date(s: Optional[str]) -> Optional[date]:
    if not s or not isinstance(s, str):
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            return datetime.strptime(s.strip(), fmt).date()
        except ValueError:
            continue
    return None


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9+#. ]", " ", (s or "").lower())


@dataclass
class CandidateFeatures:
    candidate_id: str
    name: str
    title: str
    yoe: float
    location: str
    country: str

    # role / career
    title_align: float = 0.0          # best title alignment across current+history
    current_title_align: float = 0.0
    trajectory_align: float = 0.0     # weighted by recency across roles
    shipped_evidence: float = 0.0     # 0..1, strongest description evidence
    n_product_roles: int = 0
    n_services_roles: int = 0
    services_only: bool = False
    research_only: bool = False
    job_hopper: bool = False

    # skills
    skill_trust: Dict[str, float] = field(default_factory=dict)  # group -> 0..1
    matched_core_skills: List[str] = field(default_factory=list)
    noise_ratio: float = 0.0          # fraction of listed skills that are noise
    cv_speech_robotics_primary: bool = False

    # experience / location / extras
    exp_fit: float = 0.0
    loc_fit: float = 0.0
    loc_note: str = ""
    edu_extra: float = 0.0
    edu_tier_best: str = "unknown"
    github: float = -1.0

    # behavioral
    behavioral_mult: float = 1.0
    behavioral_notes: List[str] = field(default_factory=list)
    last_active: Optional[date] = None
    response_rate: float = 0.0
    open_to_work: bool = False
    notice_days: int = 0

    # semantic doc (built once, consumed by semantic.py)
    doc: str = ""

    # honeypot
    honeypot_score: float = 0.0
    honeypot_reasons: List[str] = field(default_factory=list)


def _skill_corroboration(skill: dict, descriptions_blob: str,
                         assessment_scores: Dict[str, float]) -> float:
    """How much do we *trust* that this listed skill is real? 0..1.

    A keyword-stuffer lists 'expert' skills with no endorsements, no months of
    use, no assessment, never mentioned in any role description. A real engineer
    has at least some of those.
    """
    prof = {"beginner": 0.25, "intermediate": 0.5,
            "advanced": 0.8, "expert": 1.0}.get(skill.get("proficiency"), 0.4)
    endorse = min(1.0, (skill.get("endorsements", 0) or 0) / 20.0)
    dur = min(1.0, (skill.get("duration_months", 0) or 0) / 36.0)
    name = (skill.get("name") or "")
    in_desc = 1.0 if name and _norm(name) in descriptions_blob else 0.0
    assessed = 0.0
    if name in assessment_scores:
        assessed = min(1.0, assessment_scores[name] / 100.0)
    # Corroboration is the evidence side; proficiency is the claim side.
    # Trust = claim tempered by how much independent evidence backs it.
    evidence = max(endorse, dur, in_desc, assessed)
    # If a skill is claimed 'expert' but has *zero* evidence, trust collapses.
    if evidence == 0.0 and prof >= 0.8:
        return 0.10
    return round(0.35 * prof + 0.65 * evidence, 4)


def _skill_group_of(name: str) -> Optional[str]:
    n = name.lower().strip()
    for group, members in tax.SKILL_GROUPS.items():
        if n in members:
            return group
    return None


def extract(candidate: dict, jd: tax.JDRequirements) -> CandidateFeatures:
    profile = candidate.get("profile", {})
    history = candidate.get("career_history", []) or []
    skills = candidate.get("skills", []) or []
    edu = candidate.get("education", []) or []
    certs = candidate.get("certifications", []) or []
    signals = candidate.get("redrob_signals", {})

    cid = candidate.get("candidate_id", "")
    title = profile.get("current_title", "") or ""
    yoe = float(profile.get("years_of_experience", 0) or 0)

    f = CandidateFeatures(
        candidate_id=cid,
        name=profile.get("anonymized_name", ""),
        title=title,
        yoe=yoe,
        location=profile.get("location", "") or "",
        country=profile.get("country", "") or "",
    )

    # ---- text blobs -------------------------------------------------------
    summary = profile.get("summary", "") or ""
    headline = profile.get("headline", "") or ""
    descriptions = " ".join(_norm(h.get("description", "")) for h in history)
    blob = _norm(headline + " " + summary + " " + descriptions)

    # ---- role / career ----------------------------------------------------
    f.current_title_align = tax.title_alignment(title)
    # Trajectory: recent roles weigh more (the JD cares "have you done it
    # *recently* and at a product company").
    role_aligns = []
    weight_sum = 0.0
    weighted = 0.0
    for i, h in enumerate(history):
        a = tax.title_alignment(h.get("title", ""))
        role_aligns.append(a)
        w = 1.0 / (1.0 + i)  # role 0 (most recent) weight 1, then 1/2, 1/3...
        weighted += a * w
        weight_sum += w
        cclass = tax.company_class(h.get("company", ""))
        if cclass == "services":
            f.n_services_roles += 1
        elif cclass == "product":
            f.n_product_roles += 1
    f.trajectory_align = (weighted / weight_sum) if weight_sum else f.current_title_align
    f.title_align = max([f.current_title_align] + role_aligns) if role_aligns else f.current_title_align

    # shipped-evidence from free text
    ev = 0.0
    for pat, val in tax.SHIPPED_EVIDENCE:
        if re.search(pat, blob):
            ev = max(ev, val)
    f.shipped_evidence = ev

    # services-only career (JD soft-disqualifier)
    f.services_only = (len(history) >= 1 and f.n_services_roles == len(history)
                       and f.n_product_roles == 0)

    # research-only career (JD hard-disqualifier): every title research-flavored
    # AND no shipped/production evidence in any description.
    research_re = re.compile(r"research|phd|postdoc|scientist", re.I)
    titles_all = [title] + [h.get("title", "") for h in history]
    research_titles = sum(1 for t in titles_all if research_re.search(t or ""))
    prod_re = re.compile(r"production|deployed|shipped|launched|users|scale", re.I)
    has_prod = bool(prod_re.search(blob))
    f.research_only = (research_titles >= 1
                       and research_titles >= max(1, len(titles_all) - 0)
                       and not has_prod and f.n_product_roles == 0)

    # job-hopper (JD: switching every ~1.5y to chase titles)
    completed = [h for h in history if not h.get("is_current")]
    if len(completed) >= 3:
        avg_tenure_m = sum((h.get("duration_months", 0) or 0)
                           for h in completed) / len(completed)
        f.job_hopper = avg_tenure_m < 18  # <1.5y average

    # ---- skills (corroborated) -------------------------------------------
    assessment_scores = signals.get("skill_assessment_scores", {}) or {}
    group_best: Dict[str, float] = {}
    n_cv_speech = 0
    n_total = max(1, len(skills))
    n_noise = 0
    for s in skills:
        name = (s.get("name") or "")
        nlow = name.lower().strip()
        if nlow in tax.NOISE_SKILLS:
            n_noise += 1
        if nlow in tax.ADJACENT_NON_IR:
            n_cv_speech += 1
        group = _skill_group_of(name)
        if group:
            trust = _skill_corroboration(s, blob, assessment_scores)
            group_best[group] = max(group_best.get(group, 0.0), trust)
            if group == "core_retrieval_ranking" and trust >= 0.4:
                f.matched_core_skills.append(name)
    f.skill_trust = group_best
    f.noise_ratio = n_noise / n_total
    # CV/speech/robotics PRIMARY (JD disqualifier) = dominates skill list AND
    # no IR/ranking core skills and weak shipped evidence.
    f.cv_speech_robotics_primary = (
        n_cv_speech >= 3 and n_cv_speech / n_total >= 0.30
        and "core_retrieval_ranking" not in group_best
        and f.shipped_evidence < 0.6
    )

    # ---- experience band fit ---------------------------------------------
    lo, hi = jd.ideal_yoe
    if lo <= yoe <= hi:
        f.exp_fit = 1.0
    elif yoe < lo:
        # ramp from floor up to band
        f.exp_fit = max(0.0, (yoe - jd.yoe_hard_floor) / max(0.1, (lo - jd.yoe_hard_floor)))
    else:  # above band
        over = yoe - hi
        f.exp_fit = max(0.35, 1.0 - over / max(1.0, (jd.yoe_soft_ceiling - hi)))
    f.exp_fit = round(min(1.0, max(0.0, f.exp_fit)), 4)

    # ---- location ---------------------------------------------------------
    f.loc_fit, f.loc_note = tax.location_fit(
        f.location, f.country, bool(signals.get("willing_to_relocate")))

    # ---- education / extras ----------------------------------------------
    tier_rank = {"tier_1": 1.0, "tier_2": 0.75, "tier_3": 0.5,
                 "tier_4": 0.3, "unknown": 0.4}
    best_tier_val = 0.0
    for e in edu:
        t = e.get("tier", "unknown")
        if tier_rank.get(t, 0.4) > best_tier_val:
            best_tier_val = tier_rank.get(t, 0.4)
            f.edu_tier_best = t
    gh = signals.get("github_activity_score", -1)
    f.github = gh if gh is not None else -1
    gh_norm = max(0.0, gh / 100.0) if gh and gh > 0 else 0.0
    cert_norm = min(1.0, len(certs) / 4.0)
    f.edu_extra = round(0.5 * best_tier_val + 0.35 * gh_norm + 0.15 * cert_norm, 4)

    # ---- behavioral signals captured (multiplier computed in behavioral.py)
    f.last_active = parse_date(signals.get("last_active_date"))
    f.response_rate = float(signals.get("recruiter_response_rate", 0) or 0)
    f.open_to_work = bool(signals.get("open_to_work_flag"))
    f.notice_days = int(signals.get("notice_period_days", 0) or 0)

    # ---- semantic document -----------------------------------------------
    # Bias the doc toward *what they did* (headline, summary, role descriptions)
    # and away from the raw skills list, so semantic similarity is not gamed by
    # keyword stuffing in the skills array.
    f.doc = " ".join([
        headline, headline,            # headline twice (concise role signal)
        summary,
        descriptions,
        " ".join(s.get("name", "") for s in skills),  # skills once, low weight
    ]).strip()

    return f
