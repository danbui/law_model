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

from bm25_util import BM25SparseVectorizer
import config

# ==========================================================
# CONFIG
# ==========================================================
BASE_DIR = config.BASE_DIR
LAW_DIR = config.LAW_DIR
QDRANT_PATH = config.QDRANT_PATH
COLLECTION_NAME = config.COLLECTION_NAME
TFIDF_MODEL_PATH = config.TFIDF_MODEL_PATH


# ==========================================================
# REGEX PATTERNS
# ==========================================================
# Pattern to identify "Điều X. <Title>"
# Case insensitive, handles "Điều 1." or "Điều 10."
# Group 1: Article Number
# Group 2: Article Title/Content start
ARTICLE_RE = re.compile(r"^\s*Điều\s+(\d+)\.?", re.IGNORECASE)

# Pattern to identify "Khoản Y. <Content>"
CLAUSE_RE = re.compile(r"^\s*Khoản\s+(\d+)\.?", re.IGNORECASE)

# Pattern to identify "Điểm a <Content>" (Optional, usually inside Clause)
POINT_RE = re.compile(r"^\s*([a-zđ])\)\s+", re.IGNORECASE)


# ==========================================================
# HELPER FUNCTIONS
# ==========================================================
def extract_text_from_docx(docx_path):
    """Read all text from a .docx file."""
    doc = Document(docx_path)
    full_text = []
    for para in doc.paragraphs:
        if para.text.strip():
            full_text.append(para.text.strip())
    return "\n".join(full_text)

def chunk_law_text(full_text, doc_filename):
    """
    Split text into chunks based on 'Điều'.
    Returns a list of dicts: 
    [
      {
        "doc_id": "filename.docx",
        "article_num": 1,
        "clause_num": None, 
        "text": "Điều 1. Phạm vi điều chỉnh...",
        ...
      }, ...
    ]
    """
    lines = full_text.split('\n')
    chunks = []
    
    current_article = None
    current_text_buffer = []
    
    # Metadata for current context
    current_meta = {
        "doc_id": doc_filename,
        "chapter": "",
        "chapter_title": "",
        "article": "",     # e.g. "Điều 1"
        "article_title": ""
    }

    def flush_buffer():
        if current_meta["article"] and current_text_buffer:
            # Join lines
            content = "\n".join(current_text_buffer)
            # Try to extract Article Number integer
            art_num_match = re.search(r"(\d+)", current_meta["article"])
            art_num_int = int(art_num_match.group(1)) if art_num_match else 0
            
            # TODO: Advanced - Split by Clause (Khoản) if needed. 
            # For now, we treat one Article as one Chunk to keep context.
            # If Article is too long, we can sub-chunk.
            
            chunk_obj = {
                "doc_id": current_meta["doc_id"],
                "article": current_meta["article"],   # "Điều 1"
                "article_num": art_num_int,
                "text": content,
                "chapter": current_meta["chapter"],
                "chapter_title": current_meta["chapter_title"]
            }
            chunks.append(chunk_obj)

    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check for Chapter (Chương)
        if line.lower().startswith("chương "):
            # If we were building an article, flush it
            flush_buffer()
            current_text_buffer = []
            current_meta["article"] = ""
            
            # Parse chapter
            # e.g. "Chương I. NHỮNG QUY ĐỊNH CHUNG"
            parts = line.split('.', 1)
            current_meta["chapter"] = parts[0].strip()
            current_meta["chapter_title"] = parts[1].strip() if len(parts) > 1 else ""
            continue
            
        # Check for Article (Điều)
        match_art = ARTICLE_RE.match(line)
        if match_art:
            # Flush previous article
            flush_buffer()
            current_text_buffer = []
            
            # Start new article
            current_meta["article"] = f"Điều {match_art.group(1)}"
            current_text_buffer.append(line)
        else:
            # Just content line
            if current_meta["article"]:
                current_text_buffer.append(line)
            else:
                # Content before any Article (Preamble / Căn cứ pháp lý)
                # We can ignore or attach to a "Preamble" chunk
                pass

    # Flush last one
    flush_buffer()
    
    return chunks


# ==========================================================
# MAIN
# ==========================================================
def main():
    print("Initializing...")
    
    # 0) Setup Qdrant
    # For local disk storage
    client = QdrantClient(path=QDRANT_PATH)
    
    # 1) Load documents
    if not LAW_DIR.exists():
        print(f"Error: Directory '{LAW_DIR}' not found.")
        sys.exit(1)
        
    docx_files = list(LAW_DIR.glob("*.docx"))
    if not docx_files:
        print(f"No .docx files found in {LAW_DIR}")
        sys.exit(1)
        
    print(f"Found {len(docx_files)} documents.")
    
    all_chunks = []
    
    # 2) Process each file
    from pyvi import ViTokenizer
    
    for fpath in docx_files:
        print(f"Processing: {fpath.name}")
        text = extract_text_from_docx(fpath)
        file_chunks = chunk_law_text(text, fpath.name)
        all_chunks.extend(file_chunks)
        
    print(f"Total chunks generated: {len(all_chunks)}")
    
    if not all_chunks:
        print("No chunks to process. Exiting.")
        return

    # Prepare tokenizer for BM25
    def vi_tokenizer(text):
        return ViTokenizer.tokenize(text).split()

    # 3) Dense embeddings
    print(f"Loading Dense Model ({config.DENSE_MODEL_NAME})...")
    dense_model = SentenceTransformer(config.DENSE_MODEL_NAME)

    texts = [c["text"] for c in all_chunks]
    print(f"Generating dense embeddings for {len(texts)} chunks...")
    try:
        dense_embeddings = dense_model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=True
        )
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return

    dense_dim = dense_embeddings.shape[1]

    # 4) Sparse model (BM25)
    print(f"Training Sparse Model (BM25) k1={config.BM25_K1}, b={config.BM25_B}...")
    sparse_model = BM25SparseVectorizer(
        tokenizer=vi_tokenizer, 
        k1=config.BM25_K1, 
        b=config.BM25_B
    )
    
    sparse_matrix = sparse_model.fit_transform(texts)
    
    # Save BM25 model
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
        text_content = payload.pop("text") # Payload text

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
                payload={**payload, "text": text_content}
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
