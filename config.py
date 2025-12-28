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
