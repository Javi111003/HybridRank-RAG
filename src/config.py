import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


class Config:
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / ".data"
    CHROMA_NORMA_DIR = str(DATA_DIR / "chroma_normas")
    CHROMA_NORMA_COLLECTION = "hybridrank_normas"
    BM25_NORMA_INDEX_DIR = str(DATA_DIR / "bm25_norma_index")

    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

    GENERATOR_PROVIDER = os.getenv("GENERATOR_PROVIDER", "mistral")
    GENERATOR_MODEL = os.getenv("GENERATOR_MODEL", "mistral-small-latest")
    GENERATOR_TEMPERATURE = float(os.getenv("GENERATOR_TEMPERATURE", "0.1"))
    GENERATOR_MAX_TOKENS = int(os.getenv("GENERATOR_MAX_TOKENS", "2048"))

    CONTEXT_MAX_FRAGMENTS = int(os.getenv("CONTEXT_MAX_FRAGMENTS", "8"))
    CONTEXT_MAX_CHARS = int(os.getenv("CONTEXT_MAX_CHARS", "12000"))

    TOP_K = int(os.getenv("TOP_K", "10"))
    CANDIDATE_K = int(os.getenv("CANDIDATE_K", "50"))
    FUSION_STRATEGY = os.getenv("FUSION_STRATEGY", "hybridrank")

    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_DIR = str(DATA_DIR / "logs")


config = Config()
