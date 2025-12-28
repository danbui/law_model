# 🇻🇳 Vietnamese Law RAG with Gemini & Hybrid Search

Hệ thống Tìm kiếm & Giải đáp pháp luật thông minh sử dụng kỹ thuật **RAG (Retrieval-Augmented Generation)** kết hợp với **Hybrid Search** (BM25 + Semantic Search).

Sử dụng:
- **Ngôn ngữ**: Python 3.9+
- **LLM**: Google Gemini 1.5 Flash
- **Database**: Qdrant (Local Embedded Mode - không cần Docker)
- **Framework**: Streamlit, Sentence-Transformers, PyVi

---

## 🚀 Tính năng nổi bật

1. **Tìm kiếm lai (Hybrid Search)**: 
   - Kết hợp giữa từ khóa chính xác (BM25) và ngữ nghĩa ngữ cảnh (Dense Vector).
   - Sử dụng thuật toán **RRF (Reciprocal Rank Fusion)** để trộm điểm số và đưa ra kết quả tốt nhất.
2. **AI Tổng hợp câu trả lời**:
   - Tích hợp **Google Gemini** để đọc các văn bản luật tìm được và trả lời câu hỏi người dùng một cách tự nhiên, có trích dẫn nguồn.
3. **Xử lý dữ liệu sâu (Granular Chunking)**:
   - Tự động tách văn bản luật chi tiết đến cấp: **Điều** -> **Khoản** -> **Điểm** -> **Phụ lục**.
   - Giúp tìm kiếm chính xác vào từng tiểu mục nhỏ nhất.

---

## 🛠️ Cài đặt

### 1. Clone Source Code
```bash
git clone https://github.com/your-username/law_model.git
cd law_model
```

### 2. Thiết lập môi trường ảo (Khuyên dùng)
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 3. Cài đặt thư viện
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Cấu hình API Keys (.env)
Tạo file `.env` tại thư mục gốc và thêm API Key Gemini của bạn vào:
```ini
GEMINI_API_KEY=AIzaSy...YourKeyHere...
```

---

## 🏃‍♂️ Hướng dẫn sử dụng

### Bước 1: Chuẩn bị dữ liệu
- Copy các file luật (`.docx`) mới nhất vào thư mục `lawdata/`.
- Hệ thống sẽ tự động đọc tất cả các file trong thư mục này.

### Bước 2: Đánh chỉ mục (Indexing)
Chạy lệnh sau để xử lý văn bản và lưu vào Database:
```bash
python preprocess_word.py
```
*Lưu ý: Quá trình này có thể mất vài phút tùy vào số lượng văn bản, do phải tạo Vector Embedding.*

### Bước 3: Chạy ứng dụng web
Khởi chạy giao diện Chatbot:
```bash
streamlit run streamlit_app.py
```
Truy cập vào đường dẫn hiển thị trên terminal (thường là `http://localhost:8501`).

---

## 📂 Cấu trúc dự án

- **`streamlit_app.py`**: Ứng dụng Web Chatbot chính.
- **`preprocess_word.py`**: Script xử lý dữ liệu đầu vào (Chunking + Embedding).
- **`retrieval.py`**: Logic tìm kiếm cốt lõi (Hybrid Search + RRF).
- **`generation.py`**: Module kết nối với Gemini AI.
- **`config.py`**: File cấu hình hệ thống (đường dẫn, tham số search).
- **`bm25_util.py`**: Thư viện hỗ trợ tính toán BM25 (được tách riêng để fix lỗi pickle).
- **`lawdata/`**: Thư mục chứa file luật nguồn (.docx).
- **`qdrant_db/`**: Thư mục chứa cơ sở dữ liệu Vector (tự sinh ra).

---

## ☁️ Triển khai trên Streamlit Cloud

1. Push code lên GitHub (không bao gồm file `.env`).
2. Kết nối repo với Streamlit Cloud.
3. Trong phần **Settings -> Secrets**, thêm cấu hình:
   ```toml
   GEMINI_API_KEY = "AIzaSy...YourKeyHere..."
   ```
4. Deploy và Reboot App.
