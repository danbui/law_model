# retrieval.py
# ==========================================================
# Shared retrieval utilities for Streamlit + CLI
# ==========================================================

import re
import pickle
from dataclasses import dataclass
from pathlib import Path

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models


# ==========================================================
# CONFIG
# ==========================================================
BASE_DIR = Path(__file__).parent
QDRANT_PATH = str(BASE_DIR / "qdrant_db")
COLLECTION_NAME = "vn_law"
TFIDF_MODEL_PATH = str(BASE_DIR / "tfidf_model.pkl")


@dataclass
class Resources:
    dense_model: SentenceTransformer
    sparse_model: object
    client: QdrantClient


def load_resources() -> Resources:
    dense_model = SentenceTransformer("bkai-foundation-models/vietnamese-bi-encoder")

    with open(TFIDF_MODEL_PATH, "rb") as f:
        sparse_model = pickle.load(f)

    client = QdrantClient(path=QDRANT_PATH)

    # Ensure payload indexes exist (safe to call repeatedly)
    def _create_index(field_name: str, schema_type):
        try:
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field_name,
                field_schema=schema_type
            )
        except Exception:
            pass

    try:
        _create_index("article_num", models.PayloadSchemaType.INTEGER)
        _create_index("clause_num", models.PayloadSchemaType.INTEGER)
        _create_index("point_id", models.PayloadSchemaType.KEYWORD)
    except Exception:
        _create_index("article_num", "integer")
        _create_index("clause_num", "integer")
        _create_index("point_id", "keyword")

    return Resources(dense_model=dense_model, sparse_model=sparse_model, client=client)


def parse_filter_hints(prompt: str):
    """
    Extract filter hints from Vietnamese query:
      - Điều 10 / Đ 10
      - Khoản 2 / K 2
      - Điểm a / a)
    """
    article_m = re.search(r"(?:điều|đ)\s*(\d+)", prompt, re.IGNORECASE)
    clause_m  = re.search(r"(?:khoản|k)\s*(\d+)", prompt, re.IGNORECASE)
    point_m   = re.search(r"(?:điểm)\s*([a-z])\b|([a-z])\)", prompt, re.IGNORECASE)

    article_num = int(article_m.group(1)) if article_m else None
    clause_num = int(clause_m.group(1)) if clause_m else None

    point_id = None
    if point_m:
        point_id = (point_m.group(1) or point_m.group(2)).lower()

    return article_num, clause_num, point_id


def build_qdrant_filter(prompt: str):
    article_num, clause_num, point_id = parse_filter_hints(prompt)

    must = []
    if article_num is not None:
        must.append(models.FieldCondition(
            key="article_num",
            match=models.MatchValue(value=article_num)
        ))
    if clause_num is not None:
        must.append(models.FieldCondition(
            key="clause_num",
            match=models.MatchValue(value=clause_num)
        ))
    if point_id is not None:
        must.append(models.FieldCondition(
            key="point_id",
            match=models.MatchValue(value=point_id)
        ))

    return models.Filter(must=must) if must else None


def hybrid_search(prompt: str, top_k: int, r: Resources):
    q_filter = build_qdrant_filter(prompt)

    # Dense
    q_dense = r.dense_model.encode(prompt, normalize_embeddings=True).tolist()

    # Sparse
    q_sparse = r.sparse_model.transform([prompt])
    indices = q_sparse.indices.tolist()
    values = q_sparse.data.tolist()

    results = r.client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(
                query=models.SparseVector(indices=indices, values=values),
                using="sparse",
                filter=q_filter,
                limit=top_k * 2,
            ),
            models.Prefetch(
                query=q_dense,
                using="dense",
                filter=q_filter,
                limit=top_k * 2,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=top_k
    )

    return results, q_filter
