# ==========================================================
# 1. IMPORTS
# ==========================================================
from pathlib import Path
from docx import Document
import re
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance


# ==========================================================
# 2. REGEX PATTERNS (Vietnamese Law Structure)
# ==========================================================
ARTICLE_RE  = r"(Điều\s+\d+\.\s+[^\n]+)"
CLAUSE_RE   = r"((?:Khoản\s+)?\d+\.)"
POINT_RE    = r"([a-z]\))"
CHAPTER_RE  = r"(Chương\s+[IVXLCDM]+)\s*\n([A-ZÀ-Ỹ\s]+)"
APPENDIX_RE = r"(PHỤ LỤC\s+[IVXLCDM]+)"

ROW_RE = r"(\d+)\.\s+(.+?)\s{2,}([A-Z]\d{2}\.\d+)"


# ==========================================================
# 3. DOCX LOADER
# ==========================================================
def load_docx_text(file_path: Path) -> str:
    doc = Document(file_path)
    return "\n".join(
        p.text.strip()
        for p in doc.paragraphs
        if p.text.strip()
    )


# ==========================================================
# 4. CHUNKING FUNCTION
# ==========================================================
def chunk_vietnamese_law(text: str, doc_id: str):
    chunks = []

    # ---------- Split main body & appendix ----------
    parts = re.split(APPENDIX_RE, text)
    main_text = parts[0]
    appendix_parts = parts[1:]

    # ---------- Process main body ----------
    if re.search(CHAPTER_RE, main_text):
        chapter_blocks = re.split(CHAPTER_RE, main_text)
    else:
        chapter_blocks = [None, "NO_CHAPTER", "", main_text]

    for c in range(1, len(chapter_blocks), 3):
        chapter_id = chapter_blocks[c].strip()
        chapter_title = chapter_blocks[c + 1].strip()
        chapter_body = chapter_blocks[c + 2].strip()

        articles = re.split(ARTICLE_RE, chapter_body)

        for i in range(1, len(articles), 2):
            article_title = articles[i].strip()
            article_body = articles[i + 1].strip()

            if not re.search(CLAUSE_RE, article_body):
                chunks.append({
                    "doc_id": doc_id,
                    "section": "BODY",
                    "chapter": chapter_id,
                    "chapter_title": chapter_title,
                    "article": article_title,
                    "clause": None,
                    "point": None,
                    "text": f"{article_title}\n\n{article_body}"
                })
                continue

            clauses = re.split(CLAUSE_RE, article_body)

            for j in range(1, len(clauses), 2):
                clause_title = clauses[j].strip()
                clause_body = clauses[j + 1].strip()

                if not re.search(POINT_RE, clause_body):
                    chunks.append({
                        "doc_id": doc_id,
                        "section": "BODY",
                        "chapter": chapter_id,
                        "chapter_title": chapter_title,
                        "article": article_title,
                        "clause": clause_title,
                        "point": None,
                        "text": f"{article_title}\n{clause_title}\n\n{clause_body}"
                    })
                    continue

                points = re.split(POINT_RE, clause_body)

                for k in range(1, len(points), 2):
                    point_title = points[k].strip()
                    point_body = points[k + 1].strip()

                    chunks.append({
                        "doc_id": doc_id,
                        "section": "BODY",
                        "chapter": chapter_id,
                        "chapter_title": chapter_title,
                        "article": article_title,
                        "clause": clause_title,
                        "point": point_title,
                        "text": (
                            f"{article_title}\n"
                            f"{clause_title} {point_title}\n\n"
                            f"{point_body}"
                        )
                    })

    # ---------- Process appendix ----------
    for i in range(0, len(appendix_parts), 2):
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
                "text": (
                    f"{appendix_id}\n"
                    f"STT {stt}\n"
                    f"Tên: {name}\n"
                    f"Mã ICD-10: {icd}"
                )
            })

    return chunks


# ==========================================================
# 5. LOAD & CHUNK ALL DOCX FILES
# ==========================================================
LAW_DIR = Path(__file__).parent / "lawdata"
all_chunks = []

for docx_file in LAW_DIR.glob("*.docx"):
    doc_id = docx_file.stem
    raw_text = load_docx_text(docx_file)
    chunks = chunk_vietnamese_law(raw_text, doc_id)
    all_chunks.extend(chunks)
    print(f"✅ {doc_id}: {len(chunks)} chunks")

print(f"📦 Total chunks: {len(all_chunks)}")


# ==========================================================
# 6. EMBEDDING MODEL
# ==========================================================
model = SentenceTransformer(
    "bkai-foundation-models/vietnamese-bi-encoder"
)

texts = [c["text"] for c in all_chunks]

embeddings = model.encode(
    texts,
    normalize_embeddings=True,
    batch_size=32,
    show_progress_bar=True
)

dim = embeddings.shape[1]


# ==========================================================
# 7. QDRANT SETUP
# ==========================================================
client = QdrantClient(
    url="http://localhost:6333"
)

client.recreate_collection(
    collection_name="vn_law",
    vectors_config=VectorParams(
        size=dim,
        distance=Distance.COSINE
    )
)


# ==========================================================
# 8. STORE INTO QDRANT
# ==========================================================
points = []

for idx, chunk in enumerate(all_chunks):
    payload = chunk.copy()
    text = payload.pop("text")

    points.append(
        PointStruct(
            id=idx,
            vector=embeddings[idx].tolist(),
            payload={
                **payload,
                "text": text
            }
        )
    )

BATCH_SIZE = 64  # or 100, safe on Windows

for i in range(0, len(points), BATCH_SIZE):
    batch = points[i:i + BATCH_SIZE]
    client.upsert(
        collection_name="vn_law",
        points=batch
    )
    print(f"✅ Inserted {i + len(batch)} / {len(points)}")


print(f"🚀 Stored {len(points)} chunks into Qdrant (vn_law)")
