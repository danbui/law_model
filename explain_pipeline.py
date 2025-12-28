import config
from preprocess_word import chunk_vietnamese_law
from retrieval import load_resources, parse_filter_hints
from qdrant_client import models
import numpy as np

def simulate_ingestion():
    print("\n" + "="*50)
    print("PHẦN 1: MÔ PHỎNG INGESTION (Xử lý dữ liệu đầu vào)")
    print("="*50)

    # 1. Sample Text
    raw_text = """Điều 1. Phạm vi điều chỉnh
    Luật này quy định về đối tượng, mức đóng, trách nhiệm và phương thức đóng bảo hiểm y tế; thẻ bảo hiểm y tế; phạm vi được hưởng bảo hiểm y tế; tổ chức khám bệnh, chữa bệnh bảo hiểm y tế; thanh toán chi phí khám bệnh, chữa bệnh bảo hiểm y tế; quỹ bảo hiểm y tế; quyền và trách nhiệm của các bên liên quan đến bảo hiểm y tế."""
    print(f"\n[Input Text DOXC]:\n{raw_text.strip()}")

    # 2. Chunking
    print(f"\n[Step 1 - Chunking]: Cắt nhỏ văn bản theo Điều luật...")
    chunks = chunk_vietnamese_law(raw_text, "DOC_DEMO")
    
    sample_chunk_text = ""
    for i, chunk in enumerate(chunks):
        print(f"  > Chunk {i+1}:")
        print(f"    - Metadata: Article={chunk['article_num']}, Clause={chunk.get('clause_num')}")
        print(f"    - Content (để vector hóa): \"{chunk['text'][:100]}...\"")
        sample_chunk_text = chunk['text']
    
    # 3. Embedding
    print(f"\n[Step 2 - Embedding]: Chuyển đổi Chunk thành Vector")
    print("  Đang load models, vui lòng đợi...")
    r = load_resources()
    
    # Dense
    print(f"\n  A. Dense Embedding (Semantic - Hiểu Nghĩa):")
    print(f"  - Model: {config.DENSE_MODEL_NAME}")
    dense_vec = r.dense_model.encode(sample_chunk_text, normalize_embeddings=True)
    print(f"  - Kết quả: Vector {len(dense_vec)} chiều (float32)")
    print(f"  - Sample giá trị: {dense_vec[:5]}...")

    # Sparse
    print(f"\n  B. Sparse Embedding (Lexical - Từ khóa BM25):")
    print(f"  - Model: BM25 (Custom)")
    # Note: sparse model transform returns a list of matrices if input is list
    sparse_vec_matrix = r.sparse_model.transform([sample_chunk_text])
    # Extract first row
    sparse_indices = sparse_vec_matrix.indices
    sparse_values = sparse_vec_matrix.data
    
    print(f"  - Kết quả: Vector Thưa (chỉ lưu các từ có ý nghĩa)")
    print(f"  - Số lượng Token quan trọng: {sparse_vec_matrix.nnz}")
    print(f"  - Token Indices (ID của từ): {sparse_indices[:5]}...")
    print(f"  - Weights (Điểm quan trọng của từ): {sparse_values[:5]}...")
    
    return r

def simulate_search(r):
    print("\n" + "="*50)
    print("PHẦN 2: MÔ PHỎNG RETRIEVAL (Tìm kiếm & Xếp hạng)")
    print("="*50)

    query = "đối tượng đóng bảo hiểm y tế"
    print(f"\n[User Query]: \"{query}\"")

    # 1. Query Vectorization
    print(f"\n[Step 1 - Vectorization (Xử lý câu hỏi)]:")
    
    # Dense Query
    q_dense = r.dense_model.encode(query, normalize_embeddings=True).tolist()
    print(f"  > Dense Query Vector: {q_dense[:3]}... (Dim: {len(q_dense)})")

    # Sparse Query
    # Use transform_query logic
    if hasattr(r.sparse_model, "transform_query"):
        q_sparse_matrix = r.sparse_model.transform_query([query])
    else:
        q_sparse_matrix = r.sparse_model.transform([query])
        
    q_indices = q_sparse_matrix.indices.tolist()
    q_values = q_sparse_matrix.data.tolist()
    
    print(f"  > Sparse Query Vector (Các từ khóa tìm kiếm):")
    print(f"    - Indices: {q_indices}")
    print(f"    - Values: {q_values}")

    # 2. Individual Searches (Simulation)
    print(f"\n[Step 2 - Individual Search (Tìm riêng lẻ 2 nhánh)]:")
    
    # A. Dense Search (Semantic)
    print(f"\n  A. Dense Search (Tìm theo Ý nghĩa - Cosine Similarity):")
    dense_hits = r.client.search(
        collection_name=config.COLLECTION_NAME,
        query_vector=("dense", q_dense),
        limit=3
    )
    for i, hit in enumerate(dense_hits):
        print(f"    {i+1}. [Score: {hit.score:.4f}] {hit.payload['text'].splitlines()[0][:60]}...")

    # B. Sparse Search (Keyword)
    print(f"\n  B. Sparse Search (Tìm theo Từ khóa khớp lệnh - BM25):")
    sparse_hits = r.client.search(
        collection_name=config.COLLECTION_NAME,
        query_vector=models.NamedSparseVector(
            name="sparse",
            vector=models.SparseVector(
                indices=q_indices,
                values=q_values
            )
        ),
        limit=3
    )
    for i, hit in enumerate(sparse_hits):
        print(f"    {i+1}. [Score: {hit.score:.4f}] {hit.payload['text'].splitlines()[0][:60]}...")

    # 3. Fusion (RRF)
    print(f"\n[Step 3 - Fusion & Ranking (Trộn kết quả RRF)]:")
    print("  > Thuật toán Reciprocal Rank Fusion (RRF) sẽ lấy thứ hạng của 2 nhánh trên để tính điểm mới.")
    print(f"  > Cấu hình: Sparse Limit={config.SPARSE_LIMIT}, Dense Limit={config.DENSE_LIMIT}")
    
    prefetch_sparse = models.Prefetch(
        query=models.SparseVector(indices=q_indices, values=q_values),
        using="sparse",
        limit=config.SPARSE_LIMIT
    )
    prefetch_dense = models.Prefetch(
        query=q_dense,
        using="dense",
        limit=config.DENSE_LIMIT
    )
    
    hybrid_results = r.client.query_points(
        collection_name=config.COLLECTION_NAME,
        prefetch=[prefetch_sparse, prefetch_dense],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=3
    )
    
    print(f"\n  > KẾT QUẢ CUỐI CÙNG (Hiển thị cho User):")
    for i, point in enumerate(hybrid_results.points):
        # Note: Fusion score is usually generic, not meaningful absolute value
        print(f"    Top {i+1} [RRF Score: {point.score:.4f}]")
        print(f"       - {point.payload['text'].splitlines()[0]}...")
        print(f"       - (Trích đoạn: {point.payload['text'].splitlines()[-1][:50]}...)")

if __name__ == "__main__":
    r = simulate_ingestion()
    simulate_search(r)
