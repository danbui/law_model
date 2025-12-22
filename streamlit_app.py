import streamlit as st
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
import pickle
from pyvi import ViTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer # Required for pickle
import numpy as np

# -------------------------
# CONFIG
# -------------------------
QDRANT_PATH = "./qdrant_db"
COLLECTION_NAME = "vn_law"

# -------------------------
# HELPER (Must be defined globally for pickle sometimes, but mostly safe locally if properly imported)
# -------------------------
def vi_tokenizer(text):
    return ViTokenizer.tokenize(text).split()

# -------------------------
# LOAD RESOURCES
# -------------------------
@st.cache_resource
def load_dense_model():
    return SentenceTransformer("bkai-foundation-models/vietnamese-bi-encoder")

@st.cache_resource
def load_sparse_model():
    try:
        with open("tfidf_model.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

@st.cache_resource
def get_qdrant_client():
    return QdrantClient(path=QDRANT_PATH)

# -------------------------
# UI LAYOUT
# -------------------------
st.set_page_config(page_title="Vietnamese Law Hybrid Search", page_icon="⚖️")

st.title("⚖️ Vietnamese Law Hybrid Search")
st.caption("Ask me anything about Vietnamese Law. Using Hybrid Search (Dense + Sparse).")

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Number of results", min_value=1, max_value=20, value=5)

    st.markdown("---")
    if st.button("Xóa lịch sử chat"):
        st.session_state.messages = [{"role": "assistant", "content": "Xin chào! Tôi có thể giúp gì cho bạn về pháp luật Việt Nam hôm nay?"}]
        st.rerun()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Xin chào! Tôi có thể giúp gì cho bạn về pháp luật Việt Nam hôm nay?"}]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Load models early (or lazy load inside search)
dense_model = load_dense_model()
sparse_model = load_sparse_model()
client = get_qdrant_client()

if not sparse_model:
    st.error("⚠️ Sparse Model not found! Please run data ingestion first (`python preprocess_word.py`).")

# React to user input
if prompt := st.chat_input("Nhập câu hỏi của bạn ở đây..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    if not sparse_model:
        st.stop()

    # Search logic
    try:
        # -------------------------
        # HYBRID EMBEDDING
        # -------------------------
        # 1. Dense
        query_dense = dense_model.encode(prompt, normalize_embeddings=True).tolist()

        # 2. Sparse
        query_sparse_data = sparse_model.transform([prompt])
        indices = query_sparse_data.indices.tolist()
        values = query_sparse_data.data.tolist()

        # -------------------------
        # SEARCH
        # -------------------------
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                models.Prefetch(
                    query=models.SparseVector(indices=indices, values=values),
                    using="sparse",
                    limit=top_k * 2,
                ),
                models.Prefetch(
                    query=query_dense,
                    using="dense",
                    limit=top_k * 2,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k
        )

        # -------------------------
        # FORMAT RESULTS
        # -------------------------
        if not results.points:
            response_content = "Không tìm thấy kết quả phù hợp."
        else:
            response_content = f"**Tìm thấy {len(results.points)} kết quả (Hybrid RRF):**\n\n"
            for idx, point in enumerate(results.points, 1):
                payload = point.payload
                score = point.score # RRF score is not cosine similarity, it's rank based reciprocal sum
                
                article = payload.get('article', 'N/A')
                clause = payload.get('clause', 'N/A')
                content = payload.get("text", "N/A")
                
                response_content += f"### {idx}. Điều {article}, Khoản {clause}\n" # Removed "Score" as RRF scores are confusing (e.g. 1.0, 0.5)
                # response_content += f"*(Score: {score:.4f})*\n" 
                response_content += f"> {content}\n\n"
                response_content += "---\n"

    except Exception as e:
        response_content = f"Đã xảy ra lỗi: {e}\n\nHãy kiểm tra log."

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response_content)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response_content})
