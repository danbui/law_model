# preprocess_word.py
# ==========================================================
# Ingest DOCX luật VN -> chunk -> dense + sparse -> Qdrant hybrid index
# ==========================================================

from pathlib import Path
from docx import Document
import re
import pickle
import sys

import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from qdrant_client.models import (
    PointStruct,
    VectorParams,
    SparseVectorParams,
    Distance,
    SparseIndexParams,
)

from sklearn.feature_extraction.text import TfidfVectorizer
from bm25_util import BM25SparseVectorizer


# ==========================================================
# CONFIG
# ==========================================================
# ==========================================================
# CONFIG
# ==========================================================
import config

BASE_DIR = config.BASE_DIR
LAW_DIR = config.LAW_DIR
QDRANT_PATH = config.QDRANT_PATH
COLLECTION_NAME = config.COLLECTION_NAME
TFIDF_MODEL_PATH = config.TFIDF_MODEL_PATH

# ... (Regex section remains same)

# ... (Functions remain same)

# ==========================================================
# MAIN
# ==========================================================
def main():
    # ...
    
    # 3) Dense embeddings
    print(f"Loading Dense Model ({config.DENSE_MODEL_NAME})...")
    dense_model = SentenceTransformer(config.DENSE_MODEL_NAME)

    texts = [c["text"] for c in all_chunks]
    print(f"Generating dense embeddings for {len(texts)} chunks...")
    dense_embeddings = dense_model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=True
    )
    dense_dim = dense_embeddings.shape[1]

    # 4) Sparse model (BM25)
    print(f"Training Sparse Model (BM25) k1={config.BM25_K1}, b={config.BM25_B}...")
    sparse_model = BM25SparseVectorizer(
        tokenizer=vi_tokenizer, 
        k1=config.BM25_K1, 
        b=config.BM25_B
    )
    
    sparse_matrix = sparse_model.fit_transform(texts)
    # ...
    # vocabulary_ is inside vectorizer
    print(f"Sparse vocabulary size: {len(sparse_model.vectorizer.vocabulary_)}")

    with open(TFIDF_MODEL_PATH, "wb") as f:
        pickle.dump(sparse_model, f)
    print(f"Saved sparse model (BM25) to: {TFIDF_MODEL_PATH}")

    # 5) Recreate Qdrant collection (hybrid)
    print("Recreating Qdrant collection...")
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": VectorParams(size=dense_dim, distance=Distance.COSINE)
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(
                index=SparseIndexParams(on_disk=False)
            )
        }
    )

    # 6) Create payload indexes for reliable filtering
    print("Creating payload indexes (article_num, clause_num, point_id)...")

    def _create_index(field_name: str, schema_type):
        try:
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field_name,
                field_schema=schema_type
            )
        except Exception:
            # ignore if already exists or version differences
            pass

    # Try modern enums first; fallback to strings if needed
    try:
        _create_index("article_num", models.PayloadSchemaType.INTEGER)
        _create_index("clause_num", models.PayloadSchemaType.INTEGER)
        _create_index("point_id", models.PayloadSchemaType.KEYWORD)
    except Exception:
        _create_index("article_num", "integer")
        _create_index("clause_num", "integer")
        _create_index("point_id", "keyword")

    # 7) Prepare points
    print("Preparing points...")
    points = []

    for idx, chunk in enumerate(all_chunks):
        payload = dict(chunk)
        text = payload.pop("text")

        # Dense vector
        dense_vec = dense_embeddings[idx].tolist()

        # Sparse vector (CSR row slice)
        row_start = sparse_matrix.indptr[idx]
        row_end = sparse_matrix.indptr[idx + 1]
        indices = sparse_matrix.indices[row_start:row_end].tolist()
        values = sparse_matrix.data[row_start:row_end].tolist()

        points.append(
            PointStruct(
                id=idx,
                vector={
                    "dense": dense_vec,
                    "sparse": {"indices": indices, "values": values},
                },
                payload={**payload, "text": text}
            )
        )

    # 8) Upsert
    BATCH_SIZE = 64
    print("Upserting to Qdrant (Hybrid)...")
    for i in range(0, len(points), BATCH_SIZE):
        batch = points[i:i + BATCH_SIZE]
        client.upsert(collection_name=COLLECTION_NAME, points=batch)
        print(f"Inserted {i + len(batch)} / {len(points)}")

    print(f"Success! {len(points)} chunks indexed.")
    print("\nNext steps:")
    print("  - Run Streamlit:  streamlit run streamlit_app.py")
    print("  - Or CLI search:  python search_law.py")


if __name__ == "__main__":
    main()
