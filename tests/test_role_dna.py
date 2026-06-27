"""
Tests for the JD-driven front end (role_dna).

These prove the dynamic weight generator is well-behaved: weights are a valid
distribution, the reader is deterministic, the experience band and disqualifiers
are pulled from the JD text, and — the point of the whole exercise — a
*different* JD produces *different* weights. The bounding around priors is also
checked so a degenerate JD can't produce a degenerate ranking.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from redrob_ranker import role_dna  # noqa: E402
from redrob_ranker import taxonomy as tax  # noqa: E402

JD_PATH = ROOT / "data" / "job_description.txt"

OTHER_JD = """
Job Description: Senior Computer Vision Engineer
Location: Remote (US only)
Experience Required: 8-12 years

We build autonomous-driving perception. You will work on object detection,
segmentation, sensor fusion, and robotics. Deep computer vision and 3D
geometry expertise required. Publications at CVPR/ICCV strongly preferred.
This is a research-leaning role; production deployment is secondary.
"""


def test_weights_are_a_valid_distribution():
    dna = role_dna.extract_role_dna(role_dna.load_jd(str(JD_PATH)))
    w = dna.requirements.component_weights
    assert set(w) == set(tax.JDRequirements().component_weights)
    assert abs(sum(w.values()) - 1.0) < 1e-6
    assert all(v > 0 for v in w.values())


def test_extraction_is_deterministic():
    text = role_dna.load_jd(str(JD_PATH))
    a = role_dna.extract_role_dna(text)
    b = role_dna.extract_role_dna(text)
    assert a.requirements.component_weights == b.requirements.component_weights
    assert a.requirements.ideal_yoe == b.requirements.ideal_yoe
    assert a.active_disqualifiers == b.active_disqualifiers


def test_band_and_title_pulled_from_text():
    dna = role_dna.extract_role_dna(role_dna.load_jd(str(JD_PATH)))
    assert dna.requirements.ideal_yoe == (5.0, 9.0)
    assert "AI Engineer" in dna.title_target


def test_declared_disqualifiers_detected():
    dna = role_dna.extract_role_dna(role_dna.load_jd(str(JD_PATH)))
    # The Redrob JD explicitly names all five exclusions.
    for name in ("research_only", "services_only", "cv_speech_robotics",
                 "job_hopper", "keyword_pad"):
        assert name in dna.active_disqualifiers


def test_retrieval_jd_favours_role_over_semantic():
    """For a retrieval-heavy JD, role/career should outweigh raw text similarity."""
    dna = role_dna.extract_role_dna(role_dna.load_jd(str(JD_PATH)))
    w = dna.requirements.component_weights
    assert w["role_and_career"] > w["semantic"]
    assert w["role_and_career"] == max(w.values())


def test_a_different_jd_changes_the_weights_and_band():
    redrob = role_dna.extract_role_dna(role_dna.load_jd(str(JD_PATH)))
    other = role_dna.extract_role_dna(OTHER_JD)
    assert other.requirements.component_weights != redrob.requirements.component_weights
    # The CV role states an 8-12 band.
    assert other.requirements.ideal_yoe == (8.0, 12.0)


def test_weights_stay_bounded_around_priors():
    """No component runs away or collapses, even on a lopsided JD."""
    prior = tax.JDRequirements().component_weights
    other = role_dna.extract_role_dna(OTHER_JD).requirements.component_weights
    for k in prior:
        ratio = other[k] / prior[k]
        assert 0.3 < ratio < 3.0, f"{k} moved too far from prior ({ratio:.2f}x)"


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
