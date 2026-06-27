"""
io_utils.py
===========
Memory-bounded IO. The pool is ~100K records / ~465 MB uncompressed, and the
ranking step has a 16 GB ceiling, so we *stream* the JSONL and never hold the
raw dict list in memory longer than one record at a time. The caller extracts a
compact feature record per candidate and lets the raw dict be garbage-collected.

Also: a CSV writer that emits exactly the format the official
``validate_submission.py`` expects, including the non-increasing-score and
candidate_id tie-break invariants.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Tuple


def open_maybe_gzip(path: str) -> io.TextIOBase:
    """Open .jsonl or .jsonl.gz transparently as UTF-8 text."""
    p = Path(path)
    if p.suffix == ".gz" or p.name.endswith(".jsonl.gz"):
        return io.TextIOWrapper(gzip.open(p, "rb"), encoding="utf-8")
    return open(p, "r", encoding="utf-8")


def stream_candidates(path: str) -> Iterator[dict]:
    """Yield one candidate dict at a time.

    Auto-detects the container format: a pretty-printed JSON *array* (the
    bundled ``sample_candidates.json``) is loaded whole; a JSON-Lines file (the
    full ``candidates.jsonl`` / ``.jsonl.gz``) is streamed line-by-line so the
    100K-row pool never has to sit fully decoded in memory."""
    with open_maybe_gzip(path) as f:
        first = f.read(1)
        while first and first.isspace():
            first = f.read(1)
        if first == "[":
            # JSON array: read the remainder and parse once.
            data = json.loads("[" + f.read())
            for obj in data:
                yield obj
            return
        # JSON-Lines: reconstruct the first line we already consumed a char of.
        rest = f.readline()
        line_no = 1
        line = (first + rest).strip()
        while True:
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"[io] skipping malformed line {line_no}: {e}")
            line = f.readline()
            if not line:
                break
            line_no += 1
            line = line.strip()


def write_submission(rows: List[Dict], out_path: str) -> None:
    """Write the top-N submission CSV.

    ``rows`` must already be ranked best-first and carry keys:
    candidate_id, rank, score, reasoning.

    We enforce the two validator invariants here as a final safety net:
      * score is non-increasing as rank increases;
      * on exact score ties, candidate_id is ascending.
    We do this by *re-sorting* on (-score, candidate_id) and re-stamping ranks,
    so the written file is correct regardless of minor upstream drift.
    """
    ordered = sorted(rows, key=lambda r: (-float(r["score"]), r["candidate_id"]))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        for i, r in enumerate(ordered, 1):
            score = round(float(r["score"]), 6)
            reasoning = " ".join(str(r.get("reasoning", "")).split())
            w.writerow([r["candidate_id"], i, f"{score:.6f}", reasoning])


def count_lines(path: str) -> int:
    n = 0
    with open_maybe_gzip(path) as f:
        for line in f:
            if line.strip():
                n += 1
    return n
