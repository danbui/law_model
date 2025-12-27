# streamlit_app.py
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
with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Số kết quả", min_value=1, max_value=20, value=5)

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

            # Render as rich UI in assistant message area
            with st.chat_message("assistant"):
                st.markdown(response_md)

                for i, p in enumerate(results.points, 1):
                    payload = p.payload or {}

                    doc_id = payload.get("doc_id", "N/A")
                    chapter = payload.get("chapter", "")
                    chapter_title = payload.get("chapter_title", "")

                    article = payload.get("article", "N/A")   # đã là "Điều 10. ..."
                    clause = payload.get("clause", None)      # "1." hoặc "Khoản 1."
                    point = payload.get("point", None)        # "a)"
                    text = payload.get("text", "")

                    # Title line
                    title_parts = [article]
                    if clause:
                        title_parts.append(f"Khoản {clause}".replace("Khoản Khoản", "Khoản").strip())
                    if point:
                        title_parts.append(f"Điểm {point}".replace("Điểm Điểm", "Điểm").strip())
                    title = " • ".join([t for t in title_parts if t])

                    st.markdown(f"### {i}. {title}")

                    meta_line = " | ".join([x for x in [doc_id, (chapter + " " + chapter_title).strip()] if x.strip()])
                    if meta_line:
                        st.caption(meta_line)

                    if show_score:
                        st.caption(f"score: {p.score:.4f}")

                    # snippet
                    snippet = text[:350] + ("..." if len(text) > 350 else "")
                    if snippet:
                        st.markdown(f"> {snippet}")

                    # full content
                    if show_full_default:
                        st.markdown(text)
                    else:
                        with st.expander("Xem đầy đủ"):
                            st.markdown(text)

                    st.divider()

            # also store a compact text version in history (so rerun still shows something)
            # (Keep it short to avoid re-render duplication)
            st.session_state.messages.append({"role": "assistant", "content": response_md})
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
