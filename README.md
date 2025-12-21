YÊU CẦU HỆ THỐNG

Python 3.9 trở lên

Docker Desktop

Git

---

CLONE SOURCE CODE

git clone https://gitlab.com/tranvodat2709/law_model.git

cd law-model

---

TẠO VÀ KÍCH HOẠT VIRTUAL ENV

Windows:
python -m venv .venv
.venv\Scripts\activate

macOS / Linux:
python3 -m venv .venv
source .venv/bin/activate

---

CÀI ĐẶT THƯ VIỆN

pip install --upgrade pip
pip install -r requirements.txt

---

CHẠY QDRANT (Vector database)
docker run -d -p 6333:6333 --name qdrant qdrant/qdrant

---

INDEX DỮ LIỆU LUẬT
python preprocess_word.py

Chức năng: Tách luật theo Chương / Điều / Khoản / Điểm / Phụ lục
Sinh embedding tiếng Việt
Lưu dữ liệu vào Qdrant (collection: vn_law)

---

CHẠY SEMANTIC SEARCH

python search_law.py

Nhập câu hỏi (Query) tiếng Việt và nhấn Enter
Thoát chương trình: exit
