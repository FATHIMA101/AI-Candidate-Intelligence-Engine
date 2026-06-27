"""
taxonomy.py
===========
The domain ontology that turns the job description from a string into a
*structured requirement model*, and gives the ranker the world-knowledge it
needs to read a profile the way a human recruiter would.

Nothing in this file is decorative. Every set / weight here is consumed by
``features.py`` and ``scoring.py`` to produce a decision the system can defend
at the Stage-5 "explain your architecture" interview.

The design thesis (straight from the JD's note to hackathon participants):

    "The right answer is NOT 'find candidates whose skills section contains the
     most AI keywords.' ... A Tier-5 candidate may not use the words 'RAG' or
     'Pinecone'. A candidate with all the AI keywords but whose title is
     'Marketing Manager' is not a fit."

So the ontology is built around *roles and career trajectory first, skill
strings second*, and it explicitly encodes the JD's stated disqualifiers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Tuple


# ---------------------------------------------------------------------------
# 1. Skill universe relevant to "Senior AI Engineer — retrieval / ranking"
# ---------------------------------------------------------------------------
# We group skills by how directly they evidence the JD's *must-haves*. Each
# group carries a weight reflecting the JD's own emphasis. These are matched
# against BOTH the skills array and free text (summary / role descriptions),
# because the strongest candidates describe the work without listing the
# buzzword as a "skill".

# The JD's non-negotiables: embeddings retrieval, vector/hybrid search,
# ranking-evaluation, strong Python.
CORE_RETRIEVAL_RANKING: FrozenSet[str] = frozenset({
    "embeddings", "sentence transformers", "sentence-transformers",
    "information retrieval", "retrieval", "dense retrieval", "semantic search",
    "vector search", "vector database", "hybrid search", "bm25", "okapi bm25",
    "faiss", "pinecone", "weaviate", "qdrant", "milvus", "opensearch",
    "elasticsearch", "annoy", "hnsw", "scann", "nmslib",
    "learning to rank", "ltr", "ranking", "re-ranking", "reranking",
    "recommendation systems", "recommender systems", "recsys",
    "ndcg", "mrr", "mean reciprocal rank", "mean average precision",
    "bge", "e5", "rag", "retrieval augmented generation", "ann",
})

# Core applied-ML / NLP that the role lives in.
CORE_ML_NLP: FrozenSet[str] = frozenset({
    "machine learning", "deep learning", "nlp",
    "natural language processing", "transformers",
    "hugging face transformers", "huggingface", "pytorch", "tensorflow",
    "scikit-learn", "xgboost", "lightgbm", "feature engineering",
    "model deployment", "mlops", "mlflow", "model serving", "bentoml",
    "fine-tuning llms", "lora", "qlora", "peft", "llm", "large language models",
    "embeddings generation", "text classification", "ner",
    "language models", "prompt engineering",
})

# Production / platform engineering that signals "shipper, not researcher".
SUPPORTING_PLATFORM: FrozenSet[str] = frozenset({
    "python", "spark", "airflow", "kafka", "kubernetes", "docker",
    "aws", "gcp", "azure", "distributed systems", "data pipelines",
    "feature store", "redis", "postgresql", "sql", "snowflake", "dbt",
    "ray", "kubeflow", "triton", "onnx", "inference optimization",
    "a/b testing", "experimentation",
})

# JD "nice to have but won't reject you for".
NICE_TO_HAVE: FrozenSet[str] = frozenset({
    "lora", "qlora", "peft", "learning to rank", "xgboost",
    "hr tech", "recruiting tech", "marketplace",
    "distributed systems", "inference optimization", "open source",
})

# Skills that, when they DOMINATE a profile, signal the JD's
# "computer vision / speech / robotics primary, no NLP/IR" disqualifier.
ADJACENT_NON_IR: FrozenSet[str] = frozenset({
    "image classification", "object detection", "image segmentation",
    "opencv", "cnn", "gans", "image generation", "computer vision",
    "speech recognition", "tts", "text to speech", "asr", "wav2vec",
    "robotics", "ros", "slam", "motion planning", "control systems",
})

# Pure-noise / unrelated skills, used to detect keyword-stuffing density
# (a real AI engineer's skill list is not 50% web-frontend + AI buzzwords).
NOISE_SKILLS: FrozenSet[str] = frozenset({
    "photoshop", "tailwind", "figma", "webpack", "redux", "graphql",
    "six sigma", "sap", "content writing", "sales", "marketing",
    "node.js", "spring boot", "angular", ".net", "jquery", "bootstrap",
})

# Map a skill group -> weight (how much listed-and-corroborated membership
# raises the skill-match component). Tuned to JD emphasis.
SKILL_GROUP_WEIGHTS: Dict[str, float] = {
    "core_retrieval_ranking": 1.00,
    "core_ml_nlp": 0.65,
    "supporting_platform": 0.30,
    "nice_to_have": 0.20,
}

SKILL_GROUPS: Dict[str, FrozenSet[str]] = {
    "core_retrieval_ranking": CORE_RETRIEVAL_RANKING,
    "core_ml_nlp": CORE_ML_NLP,
    "supporting_platform": SUPPORTING_PLATFORM,
    "nice_to_have": NICE_TO_HAVE,
}


# ---------------------------------------------------------------------------
# 2. Role / title interpretation (the decisive anti-keyword-stuffer signal)
# ---------------------------------------------------------------------------
# The JD: a Marketing Manager with AI keywords is NOT a fit. The title and the
# career trajectory gate everything else. We score titles on a 0..1 "role
# alignment" scale.

# Regexes are matched case-insensitively against a title string.
_TITLE_TIERS: List[Tuple[float, List[str]]] = [
    # 1.00 — the bullseye: the role is literally about ranking/retrieval/recsys.
    (1.00, [
        r"recommendation", r"\brecsys\b", r"search engineer", r"ranking",
        r"retrieval", r"information retrieval", r"relevance engineer",
    ]),
    # 0.92 — applied ML / ML / NLP engineer (ships models).
    (0.92, [
        r"\bml\b.*engineer", r"machine learning engineer", r"applied (ml|ai)",
        r"\bnlp\b.*engineer", r"ai engineer", r"\bmlops\b",
    ]),
    # 0.80 — data scientist / research engineer (good, but research-leaning
    # gets adjudicated by career, see disqualifiers).
    (0.80, [
        r"data scientist", r"ai research engineer", r"research engineer",
        r"research scientist",
    ]),
    # 0.62 — data / backend / analytics engineering: adjacent, retrieval-capable.
    (0.62, [
        r"data engineer", r"analytics engineer", r"backend engineer",
        r"platform engineer", r"data analyst",
    ]),
    # 0.40 — general software / cloud / devops: tech but off-domain.
    (0.40, [
        r"software engineer", r"full stack", r"backend developer",
        r"cloud engineer", r"devops", r"sde",
    ]),
    # 0.22 — other engineering / dev specialisations.
    (0.22, [
        r"frontend", r"mobile developer", r"\bqa\b", r"java developer",
        r"\.net developer", r"android", r"ios",
    ]),
    # 0.05 — clearly non-technical: the keyword-stuffer trap surface.
    (0.05, [
        r"marketing", r"sales", r"\bhr\b", r"human resources", r"accountant",
        r"customer support", r"operations manager", r"project manager",
        r"business analyst", r"content writer", r"graphic designer",
        r"mechanical engineer", r"civil engineer", r"electrical engineer",
    ]),
]


def title_alignment(title: str) -> float:
    """Return a 0..1 alignment of a single title with the JD's target role."""
    if not title:
        return 0.0
    t = title.lower()
    for score, patterns in _TITLE_TIERS:
        for p in patterns:
            if re.search(p, t):
                return score
    return 0.30  # unknown technical-ish title: mild benefit of the doubt


# Phrases in free text (summary / role descriptions) that *evidence* having
# actually shipped the target systems. Worth more than any skill token because
# they describe outcomes, which keyword-stuffers don't fabricate convincingly.
SHIPPED_EVIDENCE: List[Tuple[str, float]] = [
    (r"recommendation (system|engine)", 1.0),
    (r"(search|ranking|relevance) (system|pipeline|engine|stack)", 1.0),
    (r"(semantic|vector|dense) (search|retrieval)", 1.0),
    (r"(re-?rank|learning to rank|ndcg|mrr|map@)", 0.9),
    (r"(embedding|sentence transformer|bi-encoder|cross-encoder)", 0.8),
    (r"(faiss|pinecone|weaviate|qdrant|milvus|elasticsearch|opensearch)", 0.8),
    (r"(production|deployed|shipped|launched).{0,40}(model|ml|pipeline)", 0.6),
    (r"a/?b test", 0.5),
    (r"(retrieval augmented generation|rag)", 0.7),
    (r"(fine-?tun(e|ed|ing)).{0,20}(llm|model|lora|qlora)", 0.5),
]


# ---------------------------------------------------------------------------
# 3. Employer classification (JD's product-vs-services disqualifier)
# ---------------------------------------------------------------------------
# JD: "People who have ONLY worked at consulting firms (TCS, Infosys, Wipro,
# Accenture, Cognizant, Capgemini, ...) in their entire career" = not a fit.
SERVICES_FIRMS: FrozenSet[str] = frozenset(s.lower() for s in {
    "tcs", "tata consultancy services", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "tech mahindra", "mphasis", "mindtree",
    "ltimindtree", "l&t infotech", "deloitte", "ibm", "dxc", "birlasoft",
})

# Real product companies (esp. India) — strong "shipped at a product company"
# signal that the JD explicitly rewards.
PRODUCT_COMPANIES: FrozenSet[str] = frozenset(s.lower() for s in {
    "swiggy", "zomato", "flipkart", "cred", "razorpay", "meesho", "inmobi",
    "ola", "uber", "google", "meta", "amazon", "microsoft", "netflix",
    "myntra", "phonepe", "paytm", "sharechat", "dream11", "mad street den",
    "freshworks", "postman", "browserstack", "zerodha", "nykaa",
})

# Fictional placeholder employers in this synthetic dataset. Treated as neutral
# (neither services nor a known product brand) so they neither help nor hurt.
NEUTRAL_PLACEHOLDER_COMPANIES: FrozenSet[str] = frozenset(s.lower() for s in {
    "wayne enterprises", "initech", "pied piper", "globex inc", "globex",
    "acme corp", "acme", "dunder mifflin", "hooli", "stark industries",
})


def company_class(name: str) -> str:
    """Classify an employer as 'services' | 'product' | 'neutral'."""
    n = (name or "").strip().lower()
    if n in SERVICES_FIRMS:
        return "services"
    if n in PRODUCT_COMPANIES:
        return "product"
    return "neutral"


# ---------------------------------------------------------------------------
# 4. Location interpretation (JD: Noida / Pune preferred, India strong)
# ---------------------------------------------------------------------------
PREFERRED_HUBS = ("noida", "pune", "hyderabad", "bangalore", "bengaluru",
                  "delhi", "gurgaon", "gurugram", "mumbai", "ncr", "chennai")


def location_fit(location: str, country: str, willing_to_relocate: bool) -> Tuple[float, str]:
    """Return (0..1 location score, short human note)."""
    loc = (location or "").lower()
    ctry = (country or "").lower()
    if any(h in loc for h in PREFERRED_HUBS):
        return 1.0, "in a preferred Indian hub"
    if "india" in ctry:
        return 0.85, "India-based"
    # Abroad
    if willing_to_relocate:
        return 0.55, f"{country}-based but open to relocation"
    return 0.20, f"{country}-based, not flagged for relocation (no visa sponsorship)"


# ---------------------------------------------------------------------------
# 5. The structured JD requirement model
# ---------------------------------------------------------------------------
@dataclass
class JDRequirements:
    """Everything the scorer needs to know about *this* JD, in structured form.

    Centralising it here means swapping in a different JD is a config change,
    not a rewrite — which is exactly the "own the intelligence layer" mandate.
    """
    title_target: str = "Senior AI Engineer (retrieval / ranking / recsys)"
    ideal_yoe: Tuple[float, float] = (5.0, 9.0)
    yoe_hard_floor: float = 2.5      # below this, very unlikely to be senior
    yoe_soft_ceiling: float = 14.0   # far above the band -> mild penalty
    # JD weights for the blended fit score (must sum ~1.0 across components).
    component_weights: Dict[str, float] = field(default_factory=lambda: {
        "role_and_career": 0.34,   # title + trajectory + shipped-evidence
        "skill_trust": 0.24,       # corroborated, JD-relevant skills
        "semantic": 0.18,          # JD<->profile text similarity (the AI layer)
        "experience": 0.10,        # years-of-experience band fit
        "location": 0.08,          # India / relocation
        "education_extras": 0.06,  # tier, certifications, github
    })
    # Behavioural availability is a *multiplier* on top of fit, not an additive
    # component — exactly as the signals doc recommends.
    behavioral_floor: float = 0.45   # worst-case multiplier for a "ghost"
    behavioral_ceiling: float = 1.12  # best-case small boost for hot, responsive


# A compact bag-of-words "query" representing the JD, used by the semantic
# layer. Hand-curated from the JD so the semantic similarity targets the role's
# *substance* rather than its prose style.
JD_QUERY_TEXT = (
    "senior ai engineer machine learning embeddings retrieval ranking "
    "recommendation systems vector search hybrid search faiss pinecone "
    "qdrant milvus elasticsearch opensearch information retrieval semantic "
    "search re-ranking learning to rank ndcg mrr map evaluation framework "
    "nlp transformers sentence transformers fine-tuning llm lora python "
    "production deployment shipped to real users product company "
    "recruiter matching candidate search relevance a/b testing mlops"
)

# Distinctive domain terms the JD-query harvester (role_dna._build_query) looks
# for in a JD so the semantic query reflects the JD's own vocabulary. Built
# from the skill ontology plus a few multi-word phrases.
QUERY_VOCAB_TERMS = sorted({
    t for grp in (CORE_RETRIEVAL_RANKING, CORE_ML_NLP, SUPPORTING_PLATFORM,
                  NICE_TO_HAVE)
    for t in grp
} | {
    "retrieval", "ranking", "re-ranking", "recommendation", "embeddings",
    "vector search", "hybrid search", "semantic search", "learning to rank",
    "ndcg", "mrr", "fine-tuning", "evaluation", "a/b test", "production",
    "mlops", "transformers", "llm",
})
