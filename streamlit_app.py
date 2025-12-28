import streamlit as st

from retrieval import load_resources, hybrid_search, parse_filter_hints





# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="Vietnamese Law Hybrid Search", page_icon="⚖️")

st.title("⚖️ Vietnamese Law Hybrid Search")
st.caption("Hỏi đáp luật Việt Nam (Hybrid Search: Dense + Sparse + RRF).")


# -------------------------
# LOAD RESOURCES
# -------------------------
@st.cache_resource
def get_resources():
    return load_resources()

try:
    r = get_resources()
except FileNotFoundError:
    st.error("⚠️ Không tìm thấy `tfidf_model.pkl`. Hãy chạy ingest trước: `python preprocess_word.py`.")
    st.stop()
except Exception as e:
    # Qdrant locked or other issues
    msg = str(e)
    if "already accessed" in msg or "locked" in msg.lower():
        st.error(
            "⚠️ Qdrant Local đang bị khóa bởi tiến trình khác.\n\n"
            "👉 Hãy tắt các tab/app đang dùng Qdrant hoặc vào **Manage App → Reboot App**."
        )
        st.stop()
    st.error(f"⚠️ Lỗi load resources: {e}")
    st.stop()


# -------------------------
# SIDEBAR
# -------------------------
import config

# ...

# -------------------------
# SIDEBAR
# -------------------------
with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Số kết quả", min_value=1, max_value=config.FUSION_LIMIT, value=10)

    show_score = st.checkbox("Hiện score", value=False)
    show_full_default = st.checkbox("Mở sẵn nội dung đầy đủ", value=False)

    st.markdown("---")
    if st.button("Xóa lịch sử chat"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Xin chào! Tôi có thể giúp gì cho bạn về pháp luật Việt Nam hôm nay?"}
        ]
        st.rerun()


# -------------------------
# INIT CHAT HISTORY
# -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Xin chào! Tôi có thể giúp gì cho bạn về pháp luật Việt Nam hôm nay?"}
    ]


# -------------------------
# RENDER HISTORY
# -------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# -------------------------
# CHAT INPUT
# -------------------------
prompt = st.chat_input("Nhập câu hỏi của bạn ở đây...")
if prompt:
    # user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # search
    try:
        results, q_filter = hybrid_search(prompt, top_k, r)

        article_num, clause_num, point_id = parse_filter_hints(prompt)
        filter_badge = []
        if article_num is not None:
            filter_badge.append(f"Điều {article_num}")
        if clause_num is not None:
            filter_badge.append(f"Khoản {clause_num}")
        if point_id is not None:
            filter_badge.append(f"Điểm {point_id}")

        # build assistant response (markdown)
        if not results.points:
            response_md = "Không tìm thấy kết quả phù hợp."
            if filter_badge:
                response_md += "\n\n*(Đã áp dụng lọc: " + ", ".join(filter_badge) + ")*"
        else:
            response_md = f"**Tìm thấy {len(results.points)} kết quả.**"
            if filter_badge:
                response_md += "\n\n*(Đã áp dụng lọc: " + ", ".join(filter_badge) + ")*"

            # Build the full markdown response
            full_response_md = response_md  # Start with the summary line

            for i, p in enumerate(results.points, 1):
                payload = p.payload or {}

                doc_id = payload.get("doc_id", "N/A")
                chapter = payload.get("chapter", "")
                chapter_title = payload.get("chapter_title", "")

                article = payload.get("article", "N/A")
                clause = payload.get("clause", None)
                point = payload.get("point", None)
                text = payload.get("text", "")

                # Title construction
                title_parts = [article]
                if clause:
                    title_parts.append(f"Khoản {clause}".replace("Khoản Khoản", "Khoản").strip())
                if point:
                    title_parts.append(f"Điểm {point}".replace("Điểm Điểm", "Điểm").strip())
                title = " • ".join([t for t in title_parts if t])

                # Build markdown for this item
                item_md = f"\n\n---\n\n### {i}. {title}\n"
                
                meta_line = " | ".join([x for x in [doc_id, (chapter + " " + chapter_title).strip()] if x.strip()])
                if meta_line:
                    item_md += f"_{meta_line}_\n\n"
                
                if show_score:
                    item_md += f"*(Score: {p.score:.4f})*\n\n"

                snippet = text[:350] + ("..." if len(text) > 350 else "")
                
                # Use HTML details/summary for collapsible content if possible, 
                # or just blockquote the snippet. Streamlit markdown supports HTML roughly.
                # Let's try <details> for a cleaner look that persists if Streamlit supports it (it usually allows basic HTML).
                # But to be safe and consistent with the previous design:
                item_md += f"> {snippet}\n"
                
                if show_full_default:
                    item_md += f"\n{text}\n"
                else:
                    # We can't use st.expander in a saved string easily. 
                    # We'll just leave it as snippet in history, or maybe add a Note.
                    # For now, let's just show the snippet.
                    pass

                full_response_md += item_md

            # Render the full response once
            with st.chat_message("assistant"):
                st.markdown(full_response_md)

            # Save to history
            st.session_state.messages.append({"role": "assistant", "content": full_response_md})
            # st.stop() remove stop to allow normal flow if needed, but here it's fine.
            st.stop()

    except Exception as e:
        response_md = f"Đã xảy ra lỗi: {e}\n\nHãy kiểm tra log."
        with st.chat_message("assistant"):
            st.markdown(response_md)
        st.session_state.messages.append({"role": "assistant", "content": response_md})
        st.stop()

    # If no results path reached here (rare), render response_md
    with st.chat_message("assistant"):
        st.markdown(response_md)
    st.session_state.messages.append({"role": "assistant", "content": response_md})
