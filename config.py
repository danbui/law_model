# config.py
from pathlib import Path

# ==========================================================
# PATHS
# ==========================================================
BASE_DIR = Path(__file__).parent
LAW_DIR = BASE_DIR / "lawdata"
QDRANT_PATH = str(BASE_DIR / "qdrant_db")
TFIDF_MODEL_PATH = str(BASE_DIR / "tfidf_model.pkl")

# ==========================================================
# COLLECTION & MODEL SETTINGS
# ==========================================================
COLLECTION_NAME = "vn_law"
DENSE_MODEL_NAME = "bkai-foundation-models/vietnamese-bi-encoder"

# ==========================================================
# BM25 PARAMETERS
# ==========================================================
BM25_K1 = 1.5
BM25_B = 0.75

# ==========================================================
# SEARCH HYPERPARAMETERS
# ==========================================================
# Number of candidates to fetch from each branch before fusion
SPARSE_LIMIT = 50   # Top K candidates from BM25
DENSE_LIMIT = 100   # Top K candidates from Dense Vector


# Final number of results to return after RRF fusion
FUSION_LIMIT = 60

# GEMINI CONFIGURATION
import os
from dotenv import load_dotenv

# Load environment variables from .env file (if exists)
load_dotenv()

# Try to get API KEY from Environment (Local) or Streamlit Secrets (Cloud)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    try:
        import streamlit as st
        # Helps avoid error if run outside streamlit context but st is installed
        if hasattr(st, "secrets"):
            GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        pass

# Fallback/Placeholder
if not GEMINI_API_KEY:
    GEMINI_API_KEY = "" # User must provide this via .env or secrets.toml

GEMINI_MODEL_NAME = "gemini-flash-latest"
