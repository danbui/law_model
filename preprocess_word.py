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
from pyvi import ViTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer


# ==========================================================
# CONFIG
# ==========================================================
BASE_DIR = Path(__file__).parent
LAW_DIR = BASE_DIR / "lawdata"
QDRANT_PATH = str(BASE_DIR / "qdrant_db")
COLLECTION_NAME = "vn_law"
TFIDF_MODEL_PATH = str(BASE_DIR / "tfidf_model.pkl")


# ==========================================================
# REGEX (Vietnamese Law Structure)
# ==========================================================
# Notes:
# - Các regex này phụ thuộc format DOCX của bạn. Nếu văn bản có format khác, cần chỉnh.
ARTICLE_RE  = r"(Điều\s+\d+\.\s+[^\n]+)"
CLAUSE_RE   = r"((?:Khoản\s+)?\d+\.)"
POINT_RE    = r"([a-z]\))"
CHAPTER_RE  = r"(Chương\s+[IVXLCDM]+)\s*\n([A-ZÀ-Ỹ\s]+)"
APPENDIX_RE = r"(PHỤ LỤC\s+[IVXLCDM]+)"

ROW_RE = r"(\d+)\.\s+(.+?)\s{2,}([A-Z]\d{2}\.\d+)"


# ==========================================================
# DOCX LOADER
# ==========================================================
def load_docx_text(file_path: Path) -> str:
    doc = Document(file_path)
    return "\n".join(
        p.text.strip()
        for p in doc.paragraphs
        if p.text and p.text.strip()
    )


# ==========================================================
# NORMALIZE METADATA HELPERS
# ==========================================================
def extract_article_num(article_title: str):
    # article_title thường: "Điều 10. Tiêu đề ..."
    m = re.search(r"Điều\s+(\d+)\.", article_title, flags=re.IGNORECASE)
    return int(m.group(1)) if m else None

def extract_clause_num(clause_title: str):
    # clause_title thường là "1." hoặc "Khoản 1."
    m = re.search(r"(\d+)\.", clause_title)
    return int(m.group(1)) if m else None

def extract_point_id(point_title: str):
    # point_title thường là "a)" -> lưu "a"
    m = re.search(r"([a-z])\)", point_title, flags=re.IGNORECASE)
    return m.group(1).lower() if m else None


# ==========================================================
# CHUNKING FUNCTION
# ==========================================================
def chunk_vietnamese_law(text: str, doc_id: str):
    chunks = []

    # ---------- Split main body & appendix ----------
    parts = re.split(APPENDIX_RE, text)
    main_text = parts[0]
    appendix_parts = parts[1:]  # [appendix_id, appendix_body, appendix_id2, appendix_body2, ...]

    # ---------- Process main body ----------
    if re.search(CHAPTER_RE, main_text):
        chapter_blocks = re.split(CHAPTER_RE, main_text)
    else:
        # No chapter structure detected
        chapter_blocks = [None, "NO_CHAPTER", "", main_text]

    # chapter_blocks layout when split with capture groups:
    # [prefix, chapter_id, chapter_title, chapter_body, chapter_id2, chapter_title2, chapter_body2, ...]
    for c in range(1, len(chapter_blocks), 3):
        chapter_id = (chapter_blocks[c] or "").strip()
        chapter_title = (chapter_blocks[c + 1] or "").strip()
        chapter_body = (chapter_blocks[c + 2] or "").strip()

        articles = re.split(ARTICLE_RE, chapter_body)

        # articles layout: [before, article_title, article_body, article_title2, article_body2, ...]
        for i in range(1, len(articles), 2):
            article_title = articles[i].strip()
            article_body = (articles[i + 1] if i + 1 < len(articles) else "").strip()

            article_num = extract_article_num(article_title)

            # If no clause structure -> keep as one chunk
            if not re.search(CLAUSE_RE, article_body):
                chunks.append({
                    "doc_id": doc_id,
                    "section": "BODY",
                    "chapter": chapter_id,
                    "chapter_title": chapter_title,
                    "article": article_title,
                    "article_num": article_num,
                    "clause": None,
                    "clause_num": None,
                    "point": None,
                    "point_id": None,
                    "text": f"{article_title}\n\n{article_body}".strip()
                })
                continue

            clauses = re.split(CLAUSE_RE, article_body)

            # clauses layout: [before, clause_title, clause_body, clause_title2, clause_body2, ...]
            for j in range(1, len(clauses), 2):
                clause_title = clauses[j].strip()
                clause_body = (clauses[j + 1] if j + 1 < len(clauses) else "").strip()

                clause_num = extract_clause_num(clause_title)

                # If no point structure -> chunk at clause level
                if not re.search(POINT_RE, clause_body):
                    chunks.append({
                        "doc_id": doc_id,
                        "section": "BODY",
                        "chapter": chapter_id,
                        "chapter_title": chapter_title,
                        "article": article_title,
                        "article_num": article_num,
                        "clause": clause_title,
                        "clause_num": clause_num,
                        "point": None,
                        "point_id": None,
                        "text": f"{article_title}\n{clause_title}\n\n{clause_body}".strip()
                    })
                    continue

                points = re.split(POINT_RE, clause_body)

                # points layout: [before, point_title, point_body, point_title2, point_body2, ...]
                for k in range(1, len(points), 2):
                    point_title = points[k].strip()
                    point_body = (points[k + 1] if k + 1 < len(points) else "").strip()

                    point_id = extract_point_id(point_title)

                    chunks.append({
                        "doc_id": doc_id,
                        "section": "BODY",
                        "chapter": chapter_id,
                        "chapter_title": chapter_title,
                        "article": article_title,
                        "article_num": article_num,
                        "clause": clause_title,
                        "clause_num": clause_num,
                        "point": point_title,
                        "point_id": point_id,
                        "text": (
                            f"{article_title}\n"
                            f"{clause_title} {point_title}\n\n"
                            f"{point_body}"
                        ).strip()
                    })

    # ---------- Process appendix ----------
    for i in range(0, len(appendix_parts), 2):
        if i + 1 >= len(appendix_parts):
            break
        appendix_id = appendix_parts[i].strip()
        appendix_body = appendix_parts[i + 1].strip()

        rows = re.findall(ROW_RE, appendix_body)

        for stt, name, icd in rows:
            chunks.append({
                "doc_id": doc_id,
                "section": "APPENDIX",
                "appendix": appendix_id,
                "row_id": int(stt),
                "entity": name.strip(),
                "icd10": icd.strip(),
                # normalized fields (optional, keep None)
                "article_num": None,
                "clause_num": None,
                "point_id": None,
                "text": (
                    f"{appendix_id}\n"
                    f"STT {stt}\n"
                    f"Tên: {name}\n"
                    f"Mã ICD-10: {icd}"
                ).strip()
            })

    return chunks


# ==========================================================
# TOKENIZER HELPER (for TF-IDF)
# ==========================================================
def vi_tokenizer(text: str):
    return ViTokenizer.tokenize(text).split()


# ==========================================================
# MAIN
# ==========================================================
def main():
    # Force UTF-8 for Windows Console
    try:
        sys.stdin.reconfigure(encoding="utf-8")
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    # 1) Load + chunk all DOCX files
    all_chunks = []
    print("Loading documents...")

    if not LAW_DIR.exists():
        print(f"[ERROR] Folder not found: {LAW_DIR}")
        return

    docx_files = list(LAW_DIR.glob("*.docx"))
    if not docx_files:
        print(f"[ERROR] No .docx files found in: {LAW_DIR}")
        return

    for docx_file in docx_files:
        doc_id = docx_file.stem
        raw_text = load_docx_text(docx_file)
        chunks = chunk_vietnamese_law(raw_text, doc_id)
        all_chunks.extend(chunks)
        print(f"DONE {doc_id}: {len(chunks)} chunks")

    print(f"Total chunks: {len(all_chunks)}")
    if not all_chunks:
        print("No chunks found. Exiting.")
        return

    # 2) Connect to Qdrant
    print("Connecting to Qdrant...")
    try:
        client = QdrantClient(path=QDRANT_PATH)
        client.get_collections()
    except Exception as e:
        print(f"\n[ERROR] Could not connect to Qdrant DB: {e}")
        print("Please stop any running Streamlit app or other python scripts using the DB.")
        return

    # 3) Dense embeddings
    print("Loading Dense Model (bkai-foundation-models/vietnamese-bi-encoder)...")
    dense_model = SentenceTransformer("bkai-foundation-models/vietnamese-bi-encoder")

    texts = [c["text"] for c in all_chunks]
    print(f"Generating dense embeddings for {len(texts)} chunks...")
    dense_embeddings = dense_model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=True
    )
    dense_dim = dense_embeddings.shape[1]

    # 4) Sparse model (TF-IDF)
    print("Training Sparse Model (TF-IDF with pyvi)...")
    sparse_model = TfidfVectorizer(
        tokenizer=vi_tokenizer,
        token_pattern=None,  # Use tokenizer only
        lowercase=True,
        min_df=1
    )
    sparse_matrix = sparse_model.fit_transform(texts)
    print(f"Sparse vocabulary size: {len(sparse_model.vocabulary_)}")

    with open(TFIDF_MODEL_PATH, "wb") as f:
        pickle.dump(sparse_model, f)
    print(f"Saved sparse model to: {TFIDF_MODEL_PATH}")

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
