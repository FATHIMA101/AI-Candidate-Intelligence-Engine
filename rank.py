#!/usr/bin/env python3
"""
rank.py — single-command entrypoint that turns a candidates file into a
validator-compliant top-100 submission CSV.

    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Handles both .jsonl and .jsonl.gz. CPU-only, no network. If precomputed dense
embeddings exist under ./artifacts they are used automatically; otherwise the
lexical hybrid runs alone.
"""
import argparse
import sys
from pathlib import Path

# allow `python rank.py` from repo root without install
sys.path.insert(0, str(Path(__file__).parent / "src"))
from redrob_ranker.ranker import rank  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Redrob candidate ranker")
    ap.add_argument("--candidates", required=True,
                    help="Path to candidates.jsonl or candidates.jsonl.gz")
    ap.add_argument("--out", default="submission.csv", help="Output CSV path")
    ap.add_argument("--top-n", type=int, default=100, help="How many to rank")
    ap.add_argument("--no-dense", action="store_true",
                    help="Disable the optional dense-embedding blend")
    ap.add_argument("--jd", default=None,
                    help="Path to a job-description .txt. If omitted, looks for "
                         "data/job_description.txt; falls back to built-in priors.")
    ap.add_argument("--low-memory", action="store_true",
                    help="Use unigrams only and a smaller vocabulary to cut "
                         "peak RAM usage by ~60%%. Recommended on machines with "
                         "less than 8 GB free RAM when ranking 100k+ candidates.")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    rank(args.candidates, args.out, top_n=args.top_n,
         use_dense=not args.no_dense, verbose=not args.quiet, jd_path=args.jd,
         low_memory=args.low_memory)


if __name__ == "__main__":
    main()
