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
# "Điều 1." or "Điều 1 (sửa đổi)"
ARTICLE_RE = re.compile(r"^\s*Điều\s+(\d+)", re.IGNORECASE)

# "Khoản 1." or "1." (Start of line)
CLAUSE_RE = re.compile(r"^(?:Khoản\s+)?(\d+)\.", re.IGNORECASE)

# "a)" or "đ)" (Start of line)
POINT_RE = re.compile(r"^([a-zđ])\)", re.IGNORECASE)

# "Phụ lục I" or "Phụ lục 1"
APPENDIX_RE = re.compile(r"^\s*Phụ\s+lục\s+([IVX0-9]+|số\s+\d+)", re.IGNORECASE)


# ==========================================================
# HELPER FUNCTIONS
# ==========================================================
def extract_text_from_docx(docx_path):
    """Read all text from a .docx file."""
    doc = Document(docx_path)
    full_text = []
    for para in doc.paragraphs:
        # Normalize whitespace
        text = " ".join(para.text.split())
        if text:
            full_text.append(text)
    return "\n".join(full_text)

def chunk_law_text(full_text, doc_filename):
    """
    Split text into granular chunks:
    1. Appendix (Phụ lục)
    2. Article (Điều)
       -> Clause (Khoản)
          -> Point (Điểm)
    """
    lines = full_text.split('\n')
    chunks = []
    
    # Global context
    current_chapter = ""
    current_chapter_title = ""
    
    # Hierarchy State
    # We use a stack or explicit state variables
    current_appendix = None # If not None, we are in Appendix mode
    
    current_article = None
    current_article_lines = []
    
    def flush_article_buffer(art_header, art_lines, chapter, chapter_title):
        """
        Process a raw Article block and sub-chunk it into Clauses/Points.
        """
        if not art_lines: return
        
        # 1. Base Article Info
        art_match = ARTICLE_RE.search(art_header)
        art_num = int(art_match.group(1)) if art_match else 0
        
        # 2. Iterate lines to find Clauses
        # Any text before the first Clause is "Clause 0" (Intro)
        
        clauses = []
        current_clause_header = "" # e.g., "1." or "Khoản 1."
        current_clause_lines = []
        current_clause_num = 0
        
        def flush_clause(c_num, c_header, c_lines):
            if not c_lines: return
            
            # Sub-process for Points (Điểm)
            # Hierarchy: Article > Clause > Point
            
            points = []
            current_point_char = ""
            current_point_lines = []
            
            # Content before first Point is "Point Intro"
            point_intro_lines = []
            
            for cline in c_lines:
                match_p = POINT_RE.match(cline)
                if match_p:
                    # Found a new point
                    # Flush previous point or intro
                    if current_point_char:
                        points.append((current_point_char, current_point_lines))
                    else:
                        if current_point_lines:
                            point_intro_lines.extend(current_point_lines)
                    
                    current_point_char = match_p.group(1).lower()
                    current_point_lines = [cline]
                else:
                    current_point_lines.append(cline)
            
            # Flush last point
            if current_point_char:
                points.append((current_point_char, current_point_lines))
            else:
                # No points found, everything is intro
                point_intro_lines.extend(current_point_lines)
            
            # ----- CREATE CHUNKS -----
            
            # 1. Base Clause Chunk (contains intro text + full text mostly?)
            # Strategy: If we have points, should we create a chunk for the clause intro?
            # Yes, if it's significant.
            
            # For simplicity & semantic search: 
            # Output 1 chunk for the Clause Intro (only if points exist), 
            # and 1 chunk per Point.
            # If no points, 1 chunk for whole Clause.
            
            full_clause_text = "\n".join(c_lines)
            
            # Context string used for retrieval (Breadcrumb)
            breadcrumb = f"{art_header}"
            if c_header:
                breadcrumb += f" {c_header}"
            
            # Chunk 1: The Clause Itself (or Intro if points exist)
            if point_intro_lines:
                text_content = "\n".join(point_intro_lines)
                final_text = f"{breadcrumb}\n{text_content}"
                
                chunks.append({
                    "doc_id": doc_filename,
                    "chapter": chapter,
                    "chapter_title": chapter_title,
                    "article": art_header,
                    "article_num": art_num,
                    "clause_num": c_num,
                    "point_id": None,
                    "text": final_text
                })

            # Chunk 2..N: Points
            for p_char, p_lines in points:
                p_text = "\n".join(p_lines)
                # Recurse breadcrumb
                # "Điều 1. ... Khoản 1. ... Điểm a) ..."
                full_p_text = f"{breadcrumb} Điểm {p_char})\n{p_text}"
                
                chunks.append({
                    "doc_id": doc_filename,
                    "chapter": chapter,
                    "chapter_title": chapter_title,
                    "article": art_header,
                    "article_num": art_num,
                    "clause_num": c_num,
                    "point_id": p_char,
                    "text": full_p_text
                })
        
        # Scan lines in Article
        for line in art_lines:
            match_c = CLAUSE_RE.match(line)
            if match_c:
                # Flush previous
                flush_clause(current_clause_num, current_clause_header, current_clause_lines)
                
                # New clause
                current_clause_header = f"Khoản {match_c.group(1)}"
                current_clause_lines = [line]
                current_clause_num = int(match_c.group(1))
            else:
                current_clause_lines.append(line)
        
        # Flush last clause
        flush_clause(current_clause_num, current_clause_header, current_clause_lines)


    # MAIN LOOP
    for line in lines:
        line = line.strip()
        if not line: continue
        
        # 1. Check Appendix
        match_appendix = APPENDIX_RE.match(line)
        if match_appendix:
            # Switch to Appendix mode
            # Flush pending Article
            if current_article:
                flush_article_buffer(current_article, current_article_lines, current_chapter, current_chapter_title)
                current_article = None
                current_article_lines = []
            
            current_appendix = line # "Phụ lục I"
            chunks.append({
                "doc_id": doc_filename,
                "chapter": "",
                "chapter_title": "",
                "article": current_appendix, # Mapping Appendix to 'Article' field for search compat
                "article_num": 0,
                "text": line # We define Appendix as a single chunk? Or lines?
                # Probably need to buffer appendix content too.
                # For now, let's treat Appendix header as a start.
            })
            continue

        if current_appendix:
            # In appendix mode, we just verify if we hit typical headers?
            # Or just append to the last chunk (simple approach for Apdx)
            chunks[-1]["text"] += f"\n{line}"
            continue

        # 2. Check Chapter
        if line.lower().startswith("chương "):
            if current_article:
                flush_article_buffer(current_article, current_article_lines, current_chapter, current_chapter_title)
                current_article = None
                current_article_lines = []
            
            parts = line.split('.', 1)
            current_chapter = parts[0].strip()
            current_chapter_title = parts[1].strip() if len(parts) > 1 else ""
            continue

        # 3. Check Article
        match_art = ARTICLE_RE.match(line)
        if match_art:
            if current_article:
                flush_article_buffer(current_article, current_article_lines, current_chapter, current_chapter_title)
            
            current_article = line
            current_article_lines = [line]
        else:
            if current_article:
                current_article_lines.append(line)
            else:
                # Preamble text? Ignored or added to a preamble chunk?
                pass
                
    # Final flush
    if current_article:
        flush_article_buffer(current_article, current_article_lines, current_chapter, current_chapter_title)
    
    return chunks


# ==========================================================
# TOKENIZER
# ==========================================================
from bm25_util import vi_tokenizer

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
