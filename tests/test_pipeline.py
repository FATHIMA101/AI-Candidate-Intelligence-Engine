"""
End-to-end tests for the Redrob ranker.

Two things are proven here:

1. The pipeline runs end to end on the bundled 50-candidate sample and
   produces a well-formed, internally consistent ranking.
2. A full 100-row submission produced from a >=100-candidate pool passes
   the *official* validator (validate_submission.py) byte for byte.

The full pool shipped with the challenge is 100k candidates and is not
included in the repo, so the compliance test synthesises a pool of 120
candidates by tiling the public sample with fresh, unique candidate_ids.
That is enough to force the ranker to emit exactly 100 ranked rows and to
exercise every invariant the validator enforces (unique ids, unique ranks
1-100, non-increasing score, ascending id tie-break).
"""

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SAMPLE = ROOT / "data" / "sample_candidates.json"

sys.path.insert(0, str(SRC))

from redrob_ranker.ranker import rank  # noqa: E402
from redrob_ranker import io_utils  # noqa: E402

# Import the official validator as a module so the test fails for exactly the
# same reasons the organisers' check would.
sys.path.insert(0, str(ROOT))
import validate_submission as official  # noqa: E402


def _load_sample():
    with open(SAMPLE) as f:
        return json.load(f)


def _write_pool(rows, path):
    """Write a list of candidate dicts as JSONL (the full-pool format)."""
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _synth_pool(n_min=120):
    """Tile the public sample into a pool of at least n_min unique candidates."""
    base = _load_sample()
    out = []
    i = 1
    while len(out) < n_min:
        for c in base:
            clone = dict(c)
            clone["candidate_id"] = f"CAND_{i:07d}"
            out.append(clone)
            i += 1
            if len(out) >= n_min:
                break
    return out


def test_pipeline_runs_on_sample(tmp_path):
    """The ranker produces a ranking from the 50-candidate sample."""
    out = tmp_path / "sample_submission.csv"
    result = rank(
        candidates_path=str(SAMPLE),
        out_path=str(out),
        top_n=50,            # sample only has 50, so ask for all of them
        use_dense=False,     # lexical backbone is always runnable, no network
        verbose=False,
    )
    assert out.exists()
    assert len(result) == 50

    with open(out, newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["candidate_id", "rank", "score", "reasoning"]
    assert len(rows) == 51  # header + 50

    # Every reasoning string is non-empty and mentions something concrete.
    for r in rows[1:]:
        assert r[3].strip(), "reasoning must never be blank"


def test_scores_are_non_increasing_and_tie_broken(tmp_path):
    """Internal invariants the writer is responsible for."""
    out = tmp_path / "sub.csv"
    rank(str(SAMPLE), str(out), top_n=50, use_dense=False, verbose=False)

    with open(out, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    scores = [float(r["score"]) for r in rows]
    ranks = [int(r["rank"]) for r in rows]
    ids = [r["candidate_id"] for r in rows]

    assert ranks == list(range(1, len(rows) + 1))
    for a, b in zip(scores, scores[1:]):
        assert a >= b, "score must be non-increasing by rank"
    # Where scores tie, candidate_id must ascend.
    for i in range(len(rows) - 1):
        if scores[i] == scores[i + 1]:
            assert ids[i] < ids[i + 1]


def test_full_submission_passes_official_validator(tmp_path):
    """A 100-row submission from a >=100 pool passes the real validator."""
    pool = _synth_pool(120)
    pool_path = tmp_path / "candidates.jsonl"
    _write_pool(pool, pool_path)

    out = tmp_path / "team_redrob.csv"
    result = rank(str(pool_path), str(out), top_n=100, use_dense=False, verbose=False)
    assert len(result) == 100

    errors = official.validate_submission(str(out))
    assert errors == [], "official validator reported:\n" + "\n".join(errors)


def test_streamer_autodetects_array_and_jsonl(tmp_path):
    """io_utils.stream_candidates handles both the sample array and JSONL."""
    array_count = sum(1 for _ in io_utils.stream_candidates(str(SAMPLE)))
    assert array_count == 50

    pool = _synth_pool(30)
    p = tmp_path / "p.jsonl"
    _write_pool(pool, p)
    jsonl_count = sum(1 for _ in io_utils.stream_candidates(str(p)))
    assert jsonl_count == 30


def test_honeypots_are_kept_out_of_the_top(tmp_path):
    """A blatant honeypot should not surface near the top of the ranking."""
    pool = _synth_pool(120)
    # Forge an obvious internal impossibility: claim 40 years at a single role
    # while the rest of the profile is junior. The honeypot gate should sink it.
    trap = dict(_load_sample()[0])
    trap["candidate_id"] = "CAND_9999999"
    hist = json.loads(json.dumps(trap.get("career_history", [])))
    if hist:
        hist[0]["title"] = "Senior AI Engineer"
        hist[0]["start_date"] = "2024-01-01"
        hist[0]["end_date"] = None
        hist[0]["duration_months"] = 480  # 40 years inside a 2-year window
        trap["career_history"] = hist
    pool.append(trap)

    pool_path = tmp_path / "candidates.jsonl"
    _write_pool(pool, pool_path)
    out = tmp_path / "sub.csv"
    rank(str(pool_path), str(out), top_n=100, use_dense=False, verbose=False)

    with open(out, newline="") as f:
        top_ids = [r["candidate_id"] for r in csv.DictReader(f)]
    assert "CAND_9999999" not in top_ids[:50]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
