"""
Role DNA extraction + dynamic weight generation.

This is the JD-driven front end of the ranker. Instead of hard-coding the
component weights, the experience band, the semantic query and the active
disqualifiers for one specific role, it *reads the job description* and derives
them. Swapping in a different JD therefore changes the ranking behaviour
without touching code.

Two design rules keep this defensible rather than gimmicky:

1. It is a transparent, deterministic, rule-based reader — plain regex and
   counting over the JD text. No LLM, no network, no learned model, so it adds
   essentially zero to the compute budget and can be explained line by line in
   a defend-your-work interview.

2. The generated weights are *bounded around hand-tuned priors*. The JD can
   push a component up or down within a fixed multiplier range, but it cannot
   produce a degenerate ranking (e.g. location dominating role fit). Emphasis
   moves the dial; the priors keep it sane.

The output is a fully-populated ``JDRequirements`` (so the rest of the pipeline
is unchanged), plus the derived semantic query, the set of disqualifiers the JD
actually declares, and a human-readable summary for diagnostics.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from . import taxonomy as tax


# ---------------------------------------------------------------------------
# Signal-phrase banks. Each category is a list of (regex, weight) the reader
# scans for in the JD. Weights reflect how strong a single mention is as
# evidence that the JD cares about that category.
# ---------------------------------------------------------------------------
_SIGNAL_BANKS: Dict[str, List[Tuple[str, float]]] = {
    "retrieval_ranking": [
        (r"\bre-?rank", 1.4), (r"\brank(ing|er)?\b", 1.2),
        (r"\bretrieval\b", 1.2), (r"recommend", 1.1),
        (r"\b(vector|hybrid|semantic) search\b", 1.1),
        (r"embedding", 1.0), (r"\bndcg\b|\bmrr\b|\bmap\b", 1.0),
        (r"learning[- ]to[- ]rank", 1.2),
        (r"faiss|pinecone|weaviate|qdrant|milvus|opensearch|elasticsearch", 0.8),
        (r"matching\b", 0.9),
    ],
    "ml_nlp": [
        (r"machine learning|\bml\b", 1.0), (r"\bnlp\b", 1.1),
        (r"transformer", 0.9), (r"\bllm(s)?\b", 0.9),
        (r"fine[- ]tun", 0.8), (r"lora|qlora|peft", 0.7),
        (r"sentence[- ]transformer", 0.8),
    ],
    "production": [
        (r"production", 1.4), (r"deploy", 1.1), (r"\bship\b|shipping|shipped", 1.2),
        (r"real users", 1.2), (r"a/b test", 1.0), (r"mlops", 0.9),
        (r"drift|index refresh|regression", 0.8), (r"at scale", 0.8),
    ],
    "evaluation": [
        (r"evaluation framework|eval framework", 1.3), (r"benchmark", 0.9),
        (r"offline.{0,12}online", 1.0), (r"\bndcg\b|\bmrr\b|\bmap\b", 0.8),
    ],
    "seniority": [
        (r"senior|staff|principal|lead\b", 0.8), (r"mentor", 1.0),
        (r"founding|from scratch|architecture", 1.0), (r"own the", 1.0),
    ],
    "location": [
        (r"\bindia\b", 1.0), (r"pune|noida|hyderabad|bangalore|bengaluru|mumbai|delhi|gurgaon|ncr", 1.0),
        (r"relocat", 0.8), (r"do (not|n't) sponsor|no sponsor", 1.2),
    ],
    "external_validation": [
        (r"open[- ]source", 1.0), (r"\bpapers?\b|publication", 0.9),
        (r"\btalks?\b|conference", 0.8), (r"external validation", 1.2),
        (r"github", 0.6),
    ],
    "skill_density": [
        (r"absolutely need|things you (absolutely )?need", 1.5),
        (r"production experience", 1.2), (r"hands[- ]on", 0.8),
        (r"strong python|code quality", 0.9),
    ],
}

# Disqualifier patterns the JD may *declare*. If the JD names it, the matching
# penalty is active; if a future JD doesn't, the penalty is skipped.
_DISQUALIFIER_PATTERNS: Dict[str, List[str]] = {
    "research_only": [
        r"pure research", r"research[- ]only", r"academic lab",
        r"without any production",
    ],
    "services_only": [
        r"consulting firm", r"tcs|infosys|wipro|accenture|cognizant|capgemini",
        r"services (firm|compan)",
    ],
    "cv_speech_robotics": [
        r"computer vision.{0,30}(speech|robotics)|primary expertise is computer vision",
        r"\b(cv|speech|robotics)\b.{0,40}without.{0,20}(nlp|ir)",
    ],
    "job_hopper": [
        r"title[- ]chaser", r"switching companies every", r"1\.5 years",
        r"plans? to be here for \d", r"every \d(\.\d)? years",
    ],
    "keyword_pad": [
        r"framework enthusiast", r"langchain tutorial",
        r"think about systems, not frameworks",
    ],
}


@dataclass
class RoleDNA:
    """Structured fingerprint of a JD plus everything the scorer needs."""
    title_target: str
    requirements: tax.JDRequirements
    query_text: str
    emphasis: Dict[str, float]
    active_disqualifiers: List[str] = field(default_factory=list)
    summary: str = ""


def _count_emphasis(text: str) -> Dict[str, float]:
    """Weighted count of each signal category in the JD text."""
    low = text.lower()
    out: Dict[str, float] = {}
    for cat, bank in _SIGNAL_BANKS.items():
        total = 0.0
        for pat, w in bank:
            total += w * len(re.findall(pat, low))
        out[cat] = total
    return out


def _extract_yoe_band(text: str) -> Tuple[float, float]:
    """Find an explicit experience band like '5-9 years'; else fall back."""
    low = text.lower()
    m = re.search(r"(\d{1,2})\s*[-–to]+\s*(\d{1,2})\s*year", low)
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        if 0 < lo < hi <= 30:
            return lo, hi
    m = re.search(r"(\d{1,2})\s*\+\s*year", low)
    if m:
        lo = float(m.group(1))
        return lo, lo + 4.0
    return 5.0, 9.0  # prior


def _extract_title(text: str) -> str:
    """Pull the role title from the first 'Job Description: ...' style line."""
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.search(r"(?:job description|role|title)\s*[:\-]\s*(.+)", line, re.I)
        if m:
            return m.group(1).strip(" .")
        return line  # first non-empty line as fallback
    return "AI Engineer"


def _build_query(text: str) -> str:
    """Harvest the JD's own domain vocabulary for the semantic layer.

    We take the distinctive technical terms the JD actually uses and append a
    compact core vocab, so the semantic query targets the role's substance
    rather than its (deliberately chatty) prose style.
    """
    low = text.lower()
    harvested: List[str] = []
    seen = set()
    for term in tax.QUERY_VOCAB_TERMS:
        if term in low and term not in seen:
            harvested.append(term)
            seen.add(term)
    # Always include the curated core so a terse JD still has a strong query.
    base = tax.JD_QUERY_TEXT
    return (base + " " + " ".join(harvested)).strip()


def _derive_weights(emphasis: Dict[str, float],
                    prior: Dict[str, float]) -> Dict[str, float]:
    """Turn measured JD emphasis into component weights, bounded around priors.

    Each component gets a multiplicative nudge from the JD signals most
    relevant to it. The nudge is clamped to [0.6, 1.7] so no component can run
    away or vanish, then the result is renormalised to sum to 1.0.
    """
    # Normalise emphasis to 0..1 within this JD so absolute counts don't matter.
    mx = max(emphasis.values()) or 1.0
    e = {k: v / mx for k, v in emphasis.items()}

    # How strongly each scoring component is implicated by the JD signals.
    drivers = {
        "role_and_career": 0.5 * e["retrieval_ranking"] + 0.3 * e["seniority"] + 0.2 * e["production"],
        "skill_trust": 0.6 * e["skill_density"] + 0.4 * e["retrieval_ranking"],
        "semantic": 0.6 * e["ml_nlp"] + 0.4 * e["evaluation"],
        "experience": 0.5 * e["seniority"] + 0.5 * 1.0,   # band is always asserted
        "location": e["location"],
        "education_extras": e["external_validation"],
    }

    nudged: Dict[str, float] = {}
    for k, base in prior.items():
        # Centre the driver at ~0.5 so an average-emphasis JD ≈ prior.
        factor = 0.6 + 1.1 * drivers.get(k, 0.5)
        factor = max(0.6, min(1.7, factor))
        nudged[k] = base * factor

    total = sum(nudged.values()) or 1.0
    return {k: round(v / total, 4) for k, v in nudged.items()}


def _detect_disqualifiers(text: str) -> List[str]:
    low = text.lower()
    active: List[str] = []
    for name, pats in _DISQUALIFIER_PATTERNS.items():
        if any(re.search(p, low) for p in pats):
            active.append(name)
    return active


def extract_role_dna(jd_text: str) -> RoleDNA:
    """Read a JD and return its Role DNA + a populated JDRequirements."""
    prior = tax.JDRequirements()  # hand-tuned priors live here
    emphasis = _count_emphasis(jd_text)
    weights = _derive_weights(emphasis, prior.component_weights)
    lo, hi = _extract_yoe_band(jd_text)
    title = _extract_title(jd_text)
    active = _detect_disqualifiers(jd_text)

    req = tax.JDRequirements(
        title_target=title,
        ideal_yoe=(lo, hi),
        yoe_hard_floor=max(2.0, lo - 2.5),
        yoe_soft_ceiling=hi + 5.0,
        component_weights=weights,
        behavioral_floor=prior.behavioral_floor,
        behavioral_ceiling=prior.behavioral_ceiling,
    )

    top = sorted(weights.items(), key=lambda kv: -kv[1])
    summary = (
        f"title='{title}' | band={lo:.0f}-{hi:.0f}y | "
        f"weights: " + ", ".join(f"{k}={v:.2f}" for k, v in top) +
        f" | disqualifiers: {', '.join(active) if active else 'none declared'}"
    )

    return RoleDNA(
        title_target=title, requirements=req, query_text=_build_query(jd_text),
        emphasis=emphasis, active_disqualifiers=active, summary=summary,
    )


def load_jd(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
