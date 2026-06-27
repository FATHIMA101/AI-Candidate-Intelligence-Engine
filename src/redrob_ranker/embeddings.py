"""
embeddings.py
=============
OPTIONAL dense-retrieval enhancement.

The JD asks for embeddings-based retrieval experience, and the strongest
production systems blend a dense bi-encoder signal with the lexical backbone in
``semantic.py``. This module provides that dense signal *when a local
sentence-transformer model is available on disk* — and degrades gracefully to
"lexical only" when it is not, so the core ranker runs anywhere with zero
network access (a hard Stage-3 requirement).

Why precompute? An embedding forward-pass over 100K profiles does not fit the
5-minute ranking budget on CPU. The spec explicitly allows pre-computation to
exceed the window as long as the *ranking step* stays within it. So:

    precompute.py  ->  artifacts/candidate_embeddings.npy   (offline, one-off)
    rank.py        ->  memory-maps that .npy, embeds only the JD (one vector),
                       computes cosine in a fast vectorised pass.

If the artifact is absent, ``DenseScorer.available`` is False and the ranker
uses the lexical hybrid alone. No code path requires the network at rank time.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import numpy as np

# Default compact, CPU-friendly model. Swappable via REDROB_EMB_MODEL.
DEFAULT_MODEL = os.environ.get("REDROB_EMB_MODEL", "BAAI/bge-small-en-v1.5")
EMB_PATH = Path("artifacts/candidate_embeddings.npy")
IDS_PATH = Path("artifacts/candidate_ids.npy")


def _try_import_st():
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        return SentenceTransformer
    except Exception:
        return None


def precompute(docs: List[str], ids: List[str],
               model_name: str = DEFAULT_MODEL,
               out_emb: Path = EMB_PATH, out_ids: Path = IDS_PATH,
               batch_size: int = 256) -> bool:
    """Offline: embed all candidate docs and persist as float16 .npy.

    Returns True on success, False if sentence-transformers / the model are
    unavailable (in which case the project simply runs lexical-only)."""
    ST = _try_import_st()
    if ST is None:
        print("[emb] sentence-transformers not installed; skipping dense precompute.")
        return False
    try:
        model = ST(model_name)
    except Exception as e:
        print(f"[emb] could not load model '{model_name}': {e}")
        return False
    out_emb.parent.mkdir(parents=True, exist_ok=True)
    vecs = model.encode(docs, batch_size=batch_size, show_progress_bar=True,
                        normalize_embeddings=True)
    np.save(out_emb, vecs.astype(np.float16))
    np.save(out_ids, np.array(ids))
    print(f"[emb] wrote {vecs.shape} embeddings to {out_emb}")
    return True


class DenseScorer:
    """Loads precomputed embeddings (read-only) and scores against the JD.

    No network. No GPU. Just one JD forward-pass (if the model is local) and a
    matrix-vector cosine over the memory-mapped candidate matrix.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.available = False
        self._emb: Optional[np.ndarray] = None
        self._ids: Optional[np.ndarray] = None
        self._model = None
        if EMB_PATH.exists() and IDS_PATH.exists():
            try:
                self._emb = np.load(EMB_PATH, mmap_mode="r")
                self._ids = np.load(IDS_PATH, allow_pickle=True)
                ST = _try_import_st()
                if ST is not None:
                    self._model = ST(model_name)
                    self.available = True
            except Exception as e:
                print(f"[emb] dense layer unavailable: {e}")

    def score_for_ids(self, jd_query: str, ordered_ids: List[str]) -> Optional[np.ndarray]:
        """Return a 0..1 dense relevance array aligned to ``ordered_ids``,
        or None if the dense layer is unavailable."""
        if not self.available or self._model is None or self._emb is None:
            return None
        q = self._model.encode([jd_query], normalize_embeddings=True)[0]
        sims = (np.asarray(self._emb, dtype=np.float32) @ q.astype(np.float32))
        id_to_row = {cid: i for i, cid in enumerate(self._ids.tolist())}
        out = np.zeros(len(ordered_ids), dtype=np.float32)
        for k, cid in enumerate(ordered_ids):
            r = id_to_row.get(cid)
            if r is not None:
                out[k] = sims[r]
        lo, hi = float(out.min()), float(out.max())
        if hi - lo > 1e-9:
            out = (out - lo) / (hi - lo)
        return out
