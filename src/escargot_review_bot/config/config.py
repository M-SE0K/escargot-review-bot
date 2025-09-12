import os
from typing import List

from dotenv import load_dotenv


# Load environment variables from the project root .env
load_dotenv()


# Core repository settings
REPO_PATH = os.getenv("REPO_PATH")
if not REPO_PATH or not os.path.isdir(REPO_PATH):
    raise ValueError(f"REPO_PATH '{REPO_PATH}' is not a valid directory")


# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG").upper()


# Concurrency for /review endpoint
REVIEW_MAX_CONCURRENCY = int(os.getenv("REVIEW_MAX_CONCURRENCY", "1"))


# Review bot settings
DIFF_CONTEXT = int(os.getenv("DIFF_CONTEXT", "10"))
REVIEW_INCLUDE_PATHS: List[str] = [
    p.strip() for p in os.getenv("REVIEW_INCLUDE_PATHS", "src/").split(",") if p.strip()
]


# Ollama / LLM configuration
MODEL_NAME = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.1"))
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
OLLAMA_NUM_BATCH = int(os.getenv("OLLAMA_NUM_BATCH", "256"))
OLLAMA_REPEAT_PENALTY = float(os.getenv("OLLAMA_REPEAT_PENALTY", "1.1"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.8"))
ALIGN_SEARCH_WINDOW = int(os.getenv("ALIGN_SEARCH_WINDOW", "25"))
OLLAMA_TIMEOUT_SECONDS = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "600"))
OLLAMA_MAX_RETRIES = int(os.getenv("OLLAMA_MAX_RETRIES", "2"))
INTER_REQUEST_DELAY_SECONDS = float(os.getenv("INTER_REQUEST_DELAY_SECONDS", "5"))