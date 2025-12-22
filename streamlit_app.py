import streamlit as st
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

# -------------------------
# CONFIG
# -------------------------
# QDRANT_HOST = "localhost"  <-- REMOVED
# QDRANT_PORT = 6333         <-- REMOVED
QDRANT_PATH = "./qdrant_db"
COLLECTION_NAME = "vn_law"

# -------------------------
# LOAD RESOURCES
# -------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("bkai-foundation-models/vietnamese-bi-encoder")

@st.cache_resource
def get_qdrant_client():
    return QdrantClient(path=QDRANT_PATH)

# -------------------------
# UI LAYOUT
# -------------------------
st.set_page_config(page_title="Vietnamese Law Search", page_icon="⚖️")

st.title("⚖️ Vietnamese Law Search")
st.caption("Ask me anything about Vietnamese Law.")

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Number of results", min_value=1, max_value=20, value=3)

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

# React to user input
if prompt := st.chat_input("Nhập câu hỏi của bạn ở đây..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Search logic
    try:
        model = load_model()
        client = get_qdrant_client()

        # Embed query
        query_vector = model.encode(prompt, normalize_embeddings=True).tolist()

        # Search in Qdrant
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=top_k
        )

        # Format results
        if not results.points:
            response_content = "Không tìm thấy kết quả phù hợp."
        else:
            response_content = f"**Tìm thấy {len(results.points)} kết quả liên quan:**\n\n"
            for idx, point in enumerate(results.points, 1):
                payload = point.payload
                score = point.score
                
                # Accurately referencing the content
                article = payload.get('article', 'N/A')
                clause = payload.get('clause', 'N/A')
                content = payload.get("text", "N/A")
                
                response_content += f"### {idx}. Điều {article}, Khoản {clause} (Độ khớp: {score:.2f})\n"
                response_content += f"> {content}\n\n"
                response_content += "---\n"

    except Exception as e:
        response_content = f"Đã xảy ra lỗi: {e}\n\nHãy đảm bảo Qdrant DB đã được khởi tạo."

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response_content)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response_content})


