## Candidate Intelligence Ranking Engine

An explainable, multi-stage system that reads a pool of 100,000 candidate
profiles and returns the 100 best-fit candidates for the "Senior AI Engineer —
Founding Team" role, each with a score and a grounded one-line justification.

The design goal was a ranker that a human reviewer can argue with. Every point a
candidate gains or loses traces back to a named signal in their profile, so the
ranking can be defended line by line rather than treated as a black box. That
matters here because the ground-truth labels are hidden and the challenge is
judged partly on a manual review of the reasoning text.

## The single command

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

That reads `candidates.jsonl` (one JSON profile per line), ranks the pool, and
writes a 100-row `submission.csv` with the columns `candidate\_id, rank, score, reasoning`. On the reference 8-core / 16 GB machine it finishes well inside the
five-minute ranking budget; on a single core it measured 122 seconds against the
full 100k pool at a peak of 2.25 GB of RAM. CPU only, no network, no hosted-LLM
calls at rank time.

The same entry point also reads the bundled 50-profile sample, which is a JSON
array rather than JSONL — the input reader detects the format automatically:

```bash
python rank.py --candidates ./data/sample\_candidates.json --out ./sample.csv --top-n 50
```

## How it ranks

Before any candidate is read, a JD-driven front end turns the job description
into the configuration the rest of the pipeline runs on (see "Role DNA" below).
Then the pipeline streams each profile once and turns it into a small set of
interpretable features, then combines them in four conceptual stages.

The first stage is the core read of the profile: how well the job titles and the
shape of the career match a founding senior IR/ML engineer, and how much each
claimed skill can be trusted. Skill trust is the part that defeats keyword
stuffing — a skill listed at "expert" with no endorsements, no time used, and no
mention in any role description collapses toward a trust of around 0.1, so a
profile that simply lists every fashionable term gains almost nothing from it. A
recency-weighted trajectory term rewards people who are doing relevant work now
rather than years ago.

The second stage is semantic similarity to the job description, used as a
supporting signal rather than the spine of the score. It blends TF-IDF cosine
similarity with a BM25 score over a shared vocabulary, both computed locally with
scikit-learn. The profile text fed to this stage is biased toward the narrative
sections over the raw skills list, which keeps "plain-language" strong
candidates — real fits who do not pepper their profile with buzzwords — from
being penalised for their phrasing.

The third stage is an optional dense-embedding layer. If a local
sentence-transformer model and precomputed embeddings are present, the ranker
blends a dense semantic score in with the lexical one; if they are not, it prints
a notice and proceeds on the lexical backbone alone. Nothing about the result
becomes invalid without it — it is a graceful enhancement, not a dependency, and
it never reaches for the network at rank time. See "Optional dense layer" below.

The fourth stage applies judgement. The blended component score is multiplied by
a disqualifier penalty and a behavioural modifier. Disqualifiers are the patterns
the job description calls out: a research-only profile with no production
evidence, a career spent entirely inside IT-services firms, a CV/speech/robotics
specialist with no NLP or retrieval work, or a job-hopper. The behavioural
modifier reads engagement signals — recency, recruiter response rate, interview
completion, notice period — and can pull down a "perfect on paper" candidate who
never actually responds, while being deliberately unable to turn a non-fit into a
fit.

Finally, a honeypot gate runs independently of the score. It looks only for
internal impossibilities a single role lasting longer than the person's whole
career, tenure that sums to far more than their stated years of experience,
several advanced skills with zero months of use and zero endorsements — and
forces any profile that trips it to the bottom. It is tuned for precision: it
catches the unmistakable traps and leaves the merely-weak profiles to be
out-ranked normally, which is the right trade-off given that even a handful of
false honeypots in the top 100 are cheaper than missing real fits. On the full
pool, zero honeypots reached the top 100.

The candidates are then sorted by final score with candidate\_id as the
tie-break, the top 100 are kept, and a short reasoning line is written for each.

## Role DNA: deriving the weights from the JD

The component weights, the experience band, the semantic query, and which
disqualifiers apply are not hardcoded — they are read from the job description
at startup by `role\_dna.py`. It is a transparent, deterministic, rule-based
reader (plain regex and counting, no LLM, no network), so it adds nothing to
the compute budget and can be explained line by line. It measures how strongly
the JD emphasises each signal category — retrieval and ranking, ML/NLP,
production, evaluation, seniority, location, external validation — and nudges
each component weight up or down from a hand-tuned prior accordingly, then
renormalises so the weights sum to one. The nudge is bounded, so a lopsided JD
can shift emphasis but cannot produce a degenerate ranking (location can't
swamp role fit). On the Redrob JD this pushes role-and-career to about 0.40,
because the description leans heavily on retrieval, ranking and shipping.

The reader also pulls the experience band straight from the text ("5–9 years"),
extracts the role title, and detects which of the JD's "do NOT want" clauses are
actually declared. Only declared disqualifiers are applied as penalties, so the
same code ranks sensibly against a different JD that doesn't share these
exclusions. Point it at any job description with `--jd path/to/jd.txt`; with no
flag it looks for `data/job\_description.txt` and falls back to built-in priors
if none is found. The derived weights and band are printed at the top of every
run so the configuration is visible and auditable.

## The reasoning text

Each line is assembled from the candidate's own data: an opener whose tone tracks
the tier, a fact clause naming the title, years of experience and the specific
core skills that matched, an evidence clause, and an honest-concern clause that
surfaces the single biggest reservation — a disqualifier, a dormancy signal, a
weak response rate, an out-of-band experience level, or a location mismatch. The
clauses are drawn deterministically per candidate so the text varies between
profiles instead of repeating a template, and nothing is asserted that is not
present in the profile.

## Optional dense layer

To enable the dense enhancement, install the extra dependencies and precompute
embeddings once (this step is allowed to exceed the five-minute budget; the
ranking step is not):

```bash
pip install -r requirements-dense.txt
python precompute.py --candidates ./candidates.jsonl
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

`precompute.py` writes `artifacts/candidate\_embeddings.npy` and
`artifacts/candidate\_ids.npy` using a local sentence-transformer. At rank time
the ranker memory-maps those arrays and embeds only the job description, so no
model download or network call happens during ranking. Run `rank.py` with
`--no-dense` to force the lexical-only path.

## Setup

```bash
pip install -r requirements.txt        # core, CPU-only
pip install -e ".\[dev]"                # plus pytest, if you want to run the tests
```

Python 3.10 or newer. The core ranker needs only numpy, scipy, scikit-learn and
PyYAML.

## Tests

```bash
python -m pytest -v
```

The suite runs the pipeline end to end on the public sample, checks the score
and tie-break invariants, confirms the input reader handles both the array and
JSONL formats, checks that an obvious honeypot stays out of the top, and — the
important one — builds a 100-row submission from a synthesised pool and runs it
through the unmodified official `validate\_submission.py`, asserting it reports no
errors.

## Validating a submission

```bash
python validate\_submission.py submission.csv
```

This is the challenge's own validator, included unmodified. The writer re-sorts
and re-stamps ranks before writing, so the structural invariants it checks
(exactly 100 rows, ranks 1–100 each once, non-increasing score, ascending
candidate\_id on ties) hold by construction.

## Layout

```
rank.py                     single-command ranking entry point
precompute.py               offline dense-embedding precompute (optional)
calibrate.py                archetype sanity check for the scorer
validate\_submission.py      the challenge's official validator (unmodified)
submission\_metadata.yaml    portal metadata
src/redrob\_ranker/
  taxonomy.py               skill ontology, title tiers, JD priors
  role\_dna.py               JD reader: derives weights, band, disqualifiers
  io\_utils.py               streaming reader; submission writer
  features.py               profile -> interpretable features
  semantic.py               TF-IDF + BM25 lexical scorer
  embeddings.py             optional dense scorer (graceful fallback)
  honeypot.py               internal-impossibility trap detector
  behavioral.py             engagement modifier
  scoring.py                composite score + tiers
  reasoning.py              grounded justification text
  ranker.py                 orchestration
data/                       public 50-profile sample, schema, job\_description.txt
artifacts/                  example output (submission\_full.csv)
tests/                      end-to-end, validator-compliance, role-dna tests
```

## 

