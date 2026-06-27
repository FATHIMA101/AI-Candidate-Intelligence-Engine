# CPU-only, network-free reproduction of the Redrob ranker.
#
#   docker build -t redrob-ranker .
#   docker run --rm -v "$PWD":/work redrob-ranker \
#       python rank.py --candidates /work/candidates.jsonl --out /work/submission.csv
#
# Mount the directory containing your candidates.jsonl at /work; the submission
# is written back there. The ranking step needs no network.

FROM python:3.11-slim

WORKDIR /app

# Install core dependencies first so the layer caches across code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Project code.
COPY pyproject.toml .
COPY src ./src
COPY rank.py precompute.py calibrate.py validate_submission.py ./
COPY data ./data

RUN pip install --no-cache-dir -e .

# Default: show the CLI help. Override with a rank command at run time.
CMD ["python", "rank.py", "--help"]
