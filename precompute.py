#!/usr/bin/env python3
"""
precompute.py — OFFLINE step. Embeds every candidate profile with a local
sentence-transformer and writes artifacts/candidate_embeddings.npy +
candidate_ids.npy. May exceed the 5-minute window (allowed by the spec); the
ranking step (rank.py) then memory-maps these and stays within budget.

    python precompute.py --candidates ./candidates.jsonl

No-op-with-clear-message if sentence-transformers / a local model are absent.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from redrob_ranker.features import extract          # noqa: E402
from redrob_ranker.io_utils import stream_candidates  # noqa: E402
from redrob_ranker import taxonomy as tax            # noqa: E402
from redrob_ranker.embeddings import precompute      # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    args = ap.parse_args()
    jd = tax.JDRequirements()
    docs, ids = [], []
    for c in stream_candidates(args.candidates):
        f = extract(c, jd)
        docs.append(f.doc)
        ids.append(f.candidate_id)
    print(f"[precompute] embedding {len(docs)} profiles...")
    ok = precompute(docs, ids)
    print("[precompute] done" if ok else "[precompute] skipped (no model available)")


if __name__ == "__main__":
    main()
