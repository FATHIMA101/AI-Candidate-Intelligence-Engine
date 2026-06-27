"""
ranker.py
=========
Orchestrates the whole pipeline within the compute budget:

  1. Stream candidates (memory-bounded), extract compact features + a doc.
  2. Fit the lexical hybrid (TF-IDF + BM25) on the corpus, score vs the JD.
  3. (Optional) blend a precomputed dense-embedding signal if present.
  4. Run honeypot detection + behavioral multiplier per candidate.
  5. Composite score (JD-weighted blend · disqualifier · behavioral, honeypot
     gated) -> rank -> take top-N.
  6. Generate grounded reasoning for the top-N only (cheap, keeps us in budget).

The heavy O(N) work is vectorised numpy / sparse sklearn; per-candidate Python
work is restricted to feature extraction (one cheap pass) and reasoning for just
the top-N rows.
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional

import numpy as np

from . import behavioral, honeypot, reasoning, role_dna, scoring
from .embeddings import DenseScorer
from .features import CandidateFeatures, extract
from .io_utils import stream_candidates, write_submission
from .semantic import HybridSemanticScorer
from . import taxonomy as tax


def _resolve_jd(candidates_path: str, jd_path: Optional[str]) -> Optional[str]:
    """Find a JD text file: explicit path, else a few sensible defaults."""
    candidates = []
    if jd_path:
        candidates.append(jd_path)
    here = os.path.dirname(os.path.abspath(candidates_path))
    candidates += [
        os.path.join(here, "job_description.txt"),
        os.path.join(here, "data", "job_description.txt"),
        os.path.join(os.getcwd(), "data", "job_description.txt"),
        os.path.join(os.path.dirname(__file__), "..", "..", "data",
                     "job_description.txt"),
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return None


def rank(candidates_path: str, out_path: str, top_n: int = 100,
         use_dense: bool = True, verbose: bool = True,
         jd_path: Optional[str] = None, low_memory: bool = False) -> List[Dict]:
    # ---- Role DNA: derive requirements + weights from the JD --------------
    jd_file = _resolve_jd(candidates_path, jd_path)
    if jd_file:
        dna = role_dna.extract_role_dna(role_dna.load_jd(jd_file))
        jd = dna.requirements
        jd_query = dna.query_text
        active_disq = set(dna.active_disqualifiers)
        if verbose:
            print(f"[role-dna] {dna.summary}")
    else:
        jd = tax.JDRequirements()           # hand-tuned priors as fallback
        jd_query = tax.JD_QUERY_TEXT
        active_disq = None                  # apply all penalties
        if verbose:
            print("[role-dna] no JD file found — using built-in priors")
    t0 = time.time()

    # ---- Pass 1: stream + feature extraction ------------------------------
    feats: List[CandidateFeatures] = []
    docs: List[str] = []
    raw_for_top: Dict[str, dict] = {}  # keep raw only for likely-top to save RAM
    for cand in stream_candidates(candidates_path):
        f = extract(cand, jd)
        hp_score, hp_reasons = honeypot.detect(cand)
        f.honeypot_score, f.honeypot_reasons = hp_score, hp_reasons
        behavioral.compute(f, cand, jd)
        feats.append(f)
        docs.append(f.doc)
    n = len(feats)
    if verbose:
        print(f"[rank] loaded + featurised {n} candidates in {time.time()-t0:.1f}s")

    if n == 0:
        raise SystemExit("No candidates found.")

    # ---- Lexical hybrid semantic scoring ----------------------------------
    t1 = time.time()
    sem = HybridSemanticScorer(low_memory=low_memory)
    if verbose and low_memory:
        print("[rank] low-memory mode: unigrams only, vocab cap halved")
    sem.fit(docs)
    semantic_scores = sem.score(jd_query)
    if verbose:
        print(f"[rank] lexical hybrid scored in {time.time()-t1:.1f}s")

    # ---- Optional dense blend ---------------------------------------------
    if use_dense:
        dense = DenseScorer()
        if dense.available:
            ids = [f.candidate_id for f in feats]
            dvec = dense.score_for_ids(jd_query, ids)
            if dvec is not None:
                # Blend dense with lexical: dense captures meaning beyond exact
                # terms (a Tier-5 who never writes "RAG"); lexical anchors it.
                semantic_scores = (0.55 * dvec + 0.45 * semantic_scores).astype(np.float32)
                if verbose:
                    print("[rank] blended precomputed dense embeddings")
        elif verbose:
            print("[rank] dense layer not present — using lexical hybrid only")

    # ---- Composite scoring -------------------------------------------------
    t2 = time.time()
    scored = [scoring.score(f, float(semantic_scores[i]), jd, active_disq)
              for i, f in enumerate(feats)]
    # Sort best-first; tie-break by candidate_id ascending (validator rule).
    scored.sort(key=lambda s: (-s.final, s.features.candidate_id))
    top = scored[:top_n]
    if verbose:
        print(f"[rank] scored + sorted in {time.time()-t2:.1f}s")

    # ---- Reasoning for top-N ----------------------------------------------
    rows: List[Dict] = []
    for sc in top:
        rows.append({
            "candidate_id": sc.features.candidate_id,
            "rank": 0,  # stamped by writer
            "score": sc.final,
            "reasoning": reasoning.generate(sc),
        })

    write_submission(rows, out_path)
    if verbose:
        elapsed = time.time() - t0
        hp_in_top = sum(1 for sc in top if sc.features.honeypot_score >= 0.6)
        print(f"[rank] wrote {len(rows)} rows to {out_path}")
        print(f"[rank] honeypots in top-{top_n}: {hp_in_top} "
              f"({hp_in_top/max(1,len(top)):.1%})")
        print(f"[rank] TOTAL ranking time: {elapsed:.1f}s "
              f"(budget 300s, {'OK' if elapsed < 300 else 'OVER'})")
    return rows
