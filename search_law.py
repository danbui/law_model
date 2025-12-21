from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

# -------------------------
# CONFIG
# -------------------------
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "vn_law"
TOP_K = 5

# -------------------------
# LOAD MODEL
# -------------------------
print("Loading Vietnamese bi-encoder...")
model = SentenceTransformer(
    "bkai-foundation-models/vietnamese-bi-encoder"
)
print("Model loaded.\n")

# -------------------------
# CONNECT TO QDRANT
# -------------------------
client = QdrantClient(
    host=QDRANT_HOST,
    port=QDRANT_PORT
)

print("Connected to Qdrant.")
print("Type Vietnamese query. Type 'exit' to quit.\n")

# -------------------------
# INTERACTIVE LOOP
# -------------------------
try:
    while True:
        query = input("🔍 Query > ").strip()

        if not query:
            continue

        if query.lower() in ["exit", "quit", "q"]:
            print("\nShutting down. Goodbye 👋")
            break

        # -------------------------
        # EMBED QUERY
        # -------------------------
        query_vector = model.encode(
            query,
            normalize_embeddings=True
        ).tolist()

        # -------------------------
        # SEARCH (NEW API)
        # -------------------------
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=None,
            query=query_vector,
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

            print(f"Result {idx}")
            print(f"Score   : {point.score:.4f}")
            print(f"Doc ID  : {payload.get('doc_id')}")
            print(f"Article : {payload.get('article')}")
            print(f"Clause  : {payload.get('clause')}")
            print("Text:")
            print(payload.get("text"))
            print("-" * 80)

except KeyboardInterrupt:
    print("\n\nProcess interrupted. Semantic search stopped.")
