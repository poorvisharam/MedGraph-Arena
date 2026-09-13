"""
MedGraph Arena — Centralized Configuration
==========================================
All paths, model configs, API settings, and constants in one place.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─── Paths ───────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data" / "documents"
INDEXES_DIR = PROJECT_ROOT / "indexes"
RESULTS_DIR = PROJECT_ROOT / "results"

# Per-source document directories
PUBMED_DIR = DATA_DIR / "pubmed"
GUIDELINES_DIR = DATA_DIR / "guidelines"
FDA_DIR = DATA_DIR / "fda_labels"
WHO_DIR = DATA_DIR / "who_eml"
MEDQUAD_DIR = DATA_DIR / "medquad"

# Per-system index directories
NAIVE_INDEX_DIR = INDEXES_DIR / "naive_rag"
GRAPHRAG_INDEX_DIR = INDEXES_DIR / "graphrag"
LIGHTRAG_INDEX_DIR = INDEXES_DIR / "lightrag"
NODERAG_INDEX_DIR = INDEXES_DIR / "noderag"

# ─── LLM Configuration (Google Gemini Free Tier) ────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

# OpenAI-compatible endpoint for Gemini (used by GraphRAG/LightRAG/NodeRAG)
GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_OPENAI_API_KEY = GEMINI_API_KEY

# ─── Embedding Configuration (Local, Free) ──────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# ─── Naive RAG Settings ─────────────────────────────────────────────
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 5

# ─── Data Fetching ──────────────────────────────────────────────────
PUBMED_QUERIES = [
    "metformin type 2 diabetes",
    "warfarin drug interactions",
    "ACE inhibitors kidney disease",
    "statin cardiovascular prevention",
    "ibuprofen gastrointestinal risk",
    "insulin resistance treatment",
    "aspirin antiplatelet therapy",
    "beta blockers heart failure",
]
PUBMED_MAX_RESULTS_PER_QUERY = 1  # Total ~8 abstracts

FDA_DRUG_NAMES = [
    "metformin", "warfarin", "lisinopril", "atorvastatin",
    "ibuprofen", "insulin", "aspirin", "metoprolol",
    "omeprazole", "amlodipine",
]

# ─── Evaluation ──────────────────────────────────────────────────────
EVAL_DATASET_PATH = PROJECT_ROOT / "eval_dataset.json"
RAGAS_RESULTS_PATH = RESULTS_DIR / "ragas_scores.json"
JUDGE_RESULTS_PATH = RESULTS_DIR / "judge_results.json"
GEVAL_RESULTS_PATH = RESULTS_DIR / "geval_scores.json"
QUERY_OUTPUTS_PATH = RESULTS_DIR / "query_outputs.json"

# ─── Rate Limiting ──────────────────────────────────────────────────
GEMINI_RPM = 15        # Free tier: 15 requests per minute
GEMINI_TPM = 1_000_000  # Free tier: 1M tokens per minute
RETRY_MAX_ATTEMPTS = 5
RETRY_WAIT_SECONDS = 10

# ─── System Names ───────────────────────────────────────────────────
SYSTEM_NAMES = ["naive_rag", "graphrag", "lightrag", "noderag"]

SYSTEM_DISPLAY_NAMES = {
    "naive_rag": "Naive RAG",
    "graphrag": "GraphRAG",
    "lightrag": "LightRAG",
    "noderag": "NodeRAG",
}
