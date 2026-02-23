import os
import torch
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# --- Directories ---
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DOCS_DIR = PROJECT_ROOT / "docs"
SCRATCH_DIR = PROJECT_ROOT / "scratch"
CACHE_DIR = PROJECT_ROOT / "model_cache"
LOG_DIR = PROJECT_ROOT / "logs"

for d in [DOCS_DIR, SCRATCH_DIR, CACHE_DIR, LOG_DIR]:
    d.mkdir(exist_ok=True)

load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

# --- Cache Redirection (For UV/Pip/HF) ---
os.environ["HF_HOME"] = str(CACHE_DIR / "huggingface")
os.environ["FASTEMBED_CACHE_PATH"] = str(CACHE_DIR / "fastembed")

# --- Hardware & Models ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBED_MODEL_ID = "intfloat/e5-small-v2"
NO_CONTEXT_MSG = "I could not find any legally verified references for this specific query."

# --- Authentication & Hardware Audit ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
DB_URI = os.getenv("POSTGRES_URI")


if not GROQ_API_KEY:
    logger.critical("❌ GROQ_API_KEY is missing from .env!")
if not HF_TOKEN:
    logger.warning("⚠️ HF_TOKEN not found. Gated models may be inaccessible.")
else:
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN
    logger.success("🔑 HuggingFace Token authenticated.")

logger.info(f"🖥️  Hardware Acceleration: {DEVICE.upper()} detected.")

# --- LLM Instances ---
# Primary Advisor (High Reasoning)
llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0, api_key=GROQ_API_KEY, max_tokens=800, max_retries=2)

# Fast Utility Model (Internal Query Expansion)
fast_llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0, api_key=GROQ_API_KEY, max_tokens=1024, max_retries=2)