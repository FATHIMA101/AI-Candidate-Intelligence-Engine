#!/usr/bin/env python3
"""
calibrate.py
============
We have no labelled training data (ground truth is hidden), so instead of
fitting weights to self-invented labels we *sanity-check* the hand-set weights
in ``taxonomy.JDRequirements`` against unambiguous archetypes built straight
from the JD's own statements. If the ordering below ever breaks, a weight has
drifted away from what the JD says — this is the guardrail for the additive
scoring model.

Run:  python calibrate.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from redrob_ranker import behavioral, honeypot, scoring   # noqa: E402
from redrob_ranker.features import extract                 # noqa: E402
from redrob_ranker import taxonomy as tax                  # noqa: E402

JD = tax.JDRequirements()


def _signals(**over):
    base = dict(
        profile_completeness_score=90, signup_date="2024-01-01",
        last_active_date="2026-05-20", open_to_work_flag=True,
        profile_views_received_30d=20, applications_submitted_30d=2,
        recruiter_response_rate=0.8, avg_response_time_hours=10,
        skill_assessment_scores={}, connection_count=300,
        endorsements_received=40, notice_period_days=20,
        expected_salary_range_inr_lpa={"min": 20, "max": 40},
        preferred_work_mode="hybrid", willing_to_relocate=True,
        github_activity_score=60, search_appearance_30d=100,
        saved_by_recruiters_30d=5, interview_completion_rate=0.9,
        offer_acceptance_rate=0.7, verified_email=True,
        verified_phone=True, linkedin_connected=True,
    )
    base.update(over)
    return base


def _cand(cid, title, yoe, skills, company="Swiggy", location="Bangalore",
          country="India", summary="", desc="", signals=None, history=None):
    return {
        "candidate_id": cid,
        "profile": {
            "anonymized_name": "Test", "headline": title, "summary": summary,
            "location": location, "country": country, "years_of_experience": yoe,
            "current_title": title, "current_company": company,
            "current_company_size": "1001-5000", "current_industry": "Tech",
        },
        "career_history": history or [{
            "company": company, "title": title, "start_date": "2020-01-01",
            "end_date": None, "duration_months": int(yoe * 12), "is_current": True,
            "industry": "Tech", "company_size": "1001-5000", "description": desc,
        }],
        "education": [{"institution": "IIT", "degree": "B.Tech",
                       "field_of_study": "CS", "start_year": 2014,
                       "end_year": 2018, "tier": "tier_1"}],
        "skills": skills,
        "redrob_signals": signals or _signals(),
    }


def _skill(name, prof="advanced", endo=20, dur=30):
    return {"name": name, "proficiency": prof, "endorsements": endo,
            "duration_months": dur}


def score_of(cand):
    f = extract(cand, JD)
    hp, hr = honeypot.detect(cand)
    f.honeypot_score, f.honeypot_reasons = hp, hr
    behavioral.compute(f, cand, JD)
    # semantic is supplied externally in production; use a neutral 0.5 here so
    # the test isolates the *structured* logic.
    return scoring.score(f, 0.5, JD)


def main():
    # Archetype A: textbook fit — recsys engineer, product co, corroborated.
    A = _cand("CAND_0000001", "Recommendation Systems Engineer", 7.0,
              [_skill("FAISS"), _skill("Embeddings"), _skill("Information Retrieval"),
               _skill("Learning to Rank"), _skill("Python")],
              summary="Built and shipped a recommendation and ranking system in production at scale.",
              desc="Led the search ranking system; deployed embeddings retrieval with FAISS to real users.")

    # Archetype B: keyword-stuffer — Marketing Manager with AI words, no proof.
    B = _cand("CAND_0000002", "Marketing Manager", 7.0,
              [_skill("FAISS", "expert", 0, 0), _skill("Embeddings", "expert", 0, 0),
               _skill("Recommendation Systems", "expert", 0, 0),
               _skill("Pinecone", "expert", 0, 0), _skill("Marketing", "expert", 50, 60)],
              company="Globex Inc",
              summary="Marketing leader. Curious about AI; experimented with ChatGPT.",
              desc="Owned marketing KPIs and campaigns.")

    # Archetype C: services-only career.
    C = _cand("CAND_0000003", "Software Engineer", 7.0,
              [_skill("Python"), _skill("Java"), _skill("Spring Boot")],
              company="Infosys",
              history=[{"company": "Infosys", "title": "Software Engineer",
                        "start_date": "2018-01-01", "end_date": None,
                        "duration_months": 84, "is_current": True, "industry": "IT Services",
                        "company_size": "10001+", "description": "Maintained client Java apps."}])

    # Archetype D: honeypot — impossible tenure.
    D = _cand("CAND_0000004", "ML Engineer", 5.0,
              [_skill("Embeddings"), _skill("FAISS")],
              history=[{"company": "Acme", "title": "ML Engineer",
                        "start_date": "2010-01-01", "end_date": None,
                        "duration_months": 200, "is_current": True, "industry": "Tech",
                        "company_size": "1001-5000", "description": "ML."}])

    # Archetype E: strong on paper but a ghost (dormant + unresponsive).
    E = _cand("CAND_0000005", "Recommendation Systems Engineer", 7.0,
              [_skill("FAISS"), _skill("Embeddings"), _skill("Information Retrieval")],
              summary="Built a recommendation system in production.",
              desc="Shipped ranking models to users.",
              signals=_signals(last_active_date="2025-08-01",
                               recruiter_response_rate=0.03, open_to_work_flag=False))

    sA, sB, sC, sD, sE = (score_of(x) for x in (A, B, C, D, E))
    print(f"A textbook fit          : {sA.final:.3f}  (tier {sA.tier})")
    print(f"B keyword-stuffer       : {sB.final:.3f}  (tier {sB.tier})")
    print(f"C services-only         : {sC.final:.3f}  (tier {sC.tier})")
    print(f"D honeypot              : {sD.final:.3f}  (tier {sD.tier})")
    print(f"E strong-but-ghost      : {sE.final:.3f}  (tier {sE.tier})")

    ok = True
    checks = [
        ("textbook beats keyword-stuffer", sA.final > sB.final + 0.3),
        ("textbook beats services-only", sA.final > sC.final),
        ("honeypot forced to ~0", sD.final <= 0.05),
        ("ghost down-weighted vs identical-on-paper fit", sE.final < sA.final),
        ("keyword-stuffer is low tier", sB.tier <= 1),
    ]
    print("\nCalibration checks:")
    for name, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        ok = ok and passed
    if not ok:
        sys.exit("Calibration FAILED — a weight has drifted from the JD.")
    print("\nAll calibration checks passed.")


if __name__ == "__main__":
    main()
