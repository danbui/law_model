import sys
# Force UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

import config
from retrieval import load_resources, build_qdrant_filter
from qdrant_client import models

def print_hit(rank, hit, source):
    # Extract metadata
    article = hit.payload.get('article_num', 'N/A')
    clause = hit.payload.get('clause_num', '-')
    text = hit.payload.get('text', '').replace('\n', ' ')[:100]
    score = hit.score
    print(f"   #{rank} | Score: {score:.4f} | Điều {article} (Khoản {clause}) | \"{text}...\"")

def run_debug(query):
    print(f"\nAnalyzing Query: \"{query}\"")
    print("-" * 60)
    
    print("Loading resources...")
    r = load_resources()
    
    # 1. Prepare Vectors
    print("Encoding vectors...")
    # Dense
    q_dense = r.dense_model.encode(query, normalize_embeddings=True).tolist()
    
    # Sparse
    if hasattr(r.sparse_model, "transform_query"):
        q_sparse_matrix = r.sparse_model.transform_query([query])
    else:
        q_sparse_matrix = r.sparse_model.transform([query])
    
    q_indices = q_sparse_matrix.indices.tolist()
    q_values = q_sparse_matrix.data.tolist()
    
    # Filter (if any hints in query)
    q_filter = build_qdrant_filter(query)

    # 2. Execute DENSE Search
    print("\n[A] DENSE RANKING (Top 5 - Semantic)")
    dense_hits = r.client.query_points(
        collection_name=config.COLLECTION_NAME,
        query=q_dense,
        using="dense",
        query_filter=q_filter,
        limit=5
    ).points
    for i, hit in enumerate(dense_hits):
        print_hit(i+1, hit, "Dense")

    # 3. Execute SPARSE Search
    print("\n[B] SPARSE RANKING (Top 5 - BM25/Keyword)")
    sparse_hits = r.client.query_points(
        collection_name=config.COLLECTION_NAME,
        query=models.SparseVector(indices=q_indices, values=q_values),
        using="sparse",
        query_filter=q_filter,
        limit=5
    ).points
    for i, hit in enumerate(sparse_hits):
        print_hit(i+1, hit, "Sparse")

    # 4. Execute FUSION Search (Hybrid)
    print("\n[C] FINAL FUSION RANKING (Top 5 - RRF)")
    # We use query_points with Prefetch for RRF
    fusion_hits = r.client.query_points(
        collection_name=config.COLLECTION_NAME,
        prefetch=[
            models.Prefetch(
                query=models.SparseVector(indices=q_indices, values=q_values),
                using="sparse",
                filter=q_filter,
                limit=config.SPARSE_LIMIT,
            ),
            models.Prefetch(
                query=q_dense,
                using="dense",
                filter=q_filter,
                limit=config.DENSE_LIMIT,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=5
    ).points
    
    for i, hit in enumerate(fusion_hits):
        print_hit(i+1, hit, "Hybrid")

if __name__ == "__main__":
    query_text = "đối tượng nào đóng bảo hiểm y tế"
    if len(sys.argv) > 1:
        query_text = sys.argv[1]
    run_debug(query_text)
