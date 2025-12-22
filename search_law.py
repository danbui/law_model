from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
import pickle
from pyvi import ViTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer # Required to unpickle
import numpy as np
import sys

# Force UTF-8 for Windows Console
try:
    sys.stdin.reconfigure(encoding='utf-8')
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

# -------------------------
# CONFIG
# -------------------------
QDRANT_PATH = "./qdrant_db"
COLLECTION_NAME = "vn_law"
TOP_K = 5

# -------------------------
# HELPER
# -------------------------
def vi_tokenizer(text):
    return ViTokenizer.tokenize(text).split()

# -------------------------
# LOAD MODEL
# -------------------------
print("Loading Vietnamese bi-encoder...")
dense_model = SentenceTransformer(
    "bkai-foundation-models/vietnamese-bi-encoder"
)
print("Dense model loaded.")

print("Loading Sparse model (TF-IDF)...")
try:
    with open("tfidf_model.pkl", "rb") as f:
        sparse_model = pickle.load(f)
    print("Sparse model loaded.")
except FileNotFoundError:
    print("[ERROR] 'tfidf_model.pkl' not found. Please run 'preprocess_word.py' first.")
    exit(1)


# -------------------------
# CONNECT TO QDRANT
# -------------------------
client = QdrantClient(
    path=QDRANT_PATH
)
print("Connected to Qdrant.")
print("Type Vietnamese query. Type 'exit' to quit.\n")

# -------------------------
# INTERACTIVE LOOP
# -------------------------
try:
    while True:
        query = input("Hybrid Query > ").strip()

        if not query:
            continue

        if query.lower() in ["exit", "quit", "q"]:
            print("\nShutting down. Goodbye 👋")
            break

        # -------------------------
        # EMBED QUERY (DENSE)
        # -------------------------
        # encode returns numpy array by default
        dense_vector = dense_model.encode(
            query,
            normalize_embeddings=True
        ).tolist()

        # -------------------------
        # EMBED QUERY (SPARSE)
        # -------------------------
        sparse_vec_data = sparse_model.transform([query])
        # Extract indices and values from the single row sparse matrix
        indices = sparse_vec_data.indices.tolist()
        values = sparse_vec_data.data.tolist()

        # -------------------------
        # HYBRID SEARCH (RRF)
        # -------------------------
        # Using Qdrant's Query Fusion (available in newer versions)
        # We prefetch candidates from both Dense and Sparse, then Fuse.
        
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                models.Prefetch(
                    query=models.SparseVector(indices=indices, values=values),
                    using="sparse",
                    limit=TOP_K * 2, # Fetch more candidates for fusion
                ),
                models.Prefetch(
                    query=dense_vector,
                    using="dense",
                    limit=TOP_K * 2,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=TOP_K
        )

        # -------------------------
        # DISPLAY RESULTS
        # -------------------------
        print("\n📄 Results:\n")

        if not results.points:
            print("No results found.\n")
            continue

        for idx, point in enumerate(results.points, 1):
            payload = point.payload
            
            # Score logic: RRF scoring is different (rank based).
            print(f"Result {idx}")
            print(f"Score   : {point.score:.4f}") 
            print(f"Doc ID  : {payload.get('doc_id')}")
            article = payload.get('article')
            clause = payload.get('clause')
            
            if article:
                print(f"Article : {article}")
            if clause:    
                print(f"Clause  : {clause}")
                
            print("Text snippet:")
            text = payload.get("text", "")
            # Truncate for display if too long
            print(text[:200] + "..." if len(text) > 200 else text)
            print("-" * 80)

except KeyboardInterrupt:
    print("\n\nProcess interrupted. Stopping.")
