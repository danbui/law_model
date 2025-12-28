import sys
# Force UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

from pyvi import ViTokenizer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# ==============================================================================
# 1. SETUP DATA (Dữ liệu giả lập)
# ==============================================================================
doc_text = "Điều 12. Đối tượng đóng bảo hiểm y tế. Người lao động làm việc theo hợp đồng lao động."
query_text = "đối tượng nào đóng bảo hiểm y tế"

print(f"1. DATA INPUT")
print(f"   - Document: '{doc_text}'")
print(f"   - Query:    '{query_text}'")

# ==============================================================================
# 2. TOKENIZATION (Tách từ tiếng Việt)
# ==============================================================================
print(f"\n2. TOKENIZATION (Tách từ)")
# Dùng pyvi giống trong hệ thống thật
doc_tokens = ViTokenizer.tokenize(doc_text).split()
query_tokens = ViTokenizer.tokenize(query_text).split()

print(f"   - Doc Tokens:   {doc_tokens}")
print(f"   - Query Tokens: {query_tokens}")
# Note: 'bảo hiểm y tế' -> 'bảo_hiểm_y_tế' (được nối lại)

# ==============================================================================
# 3. SPARSE EMBEDDING (BM25 - Từ khóa)
# ==============================================================================
print(f"\n3. SPARSE VECTOR (BM25 - Từ khóa)")
# Giả lập BM25 đơn giản để hiển thị trọng số
vocab = list(set(doc_tokens + query_tokens))
vocab.sort()
print(f"   - Vocabulary (Tập từ vựng): {vocab}")

# Tính TF (Tần suất) giả định
print(f"   - Tính toán trọng số (Giả lập):")
for word in vocab:
    tf_doc = doc_tokens.count(word)
    tf_query = query_tokens.count(word)
    
    # IDF giả định (Từ hiếm 'bảo_hiểm_y_tế' điểm cao hơn từ phổ biến 'làm')
    idf = 1.0
    if "bảo_hiểm" in word or "đối_tượng" in word:
        idf = 2.5 # Quan trọng
    elif "hợp_đồng" in word:
        idf = 2.0
    else:
        idf = 0.5 # Từ thường
        
    # BM25 weight ~ TF * IDF (simplified)
    weight = tf_doc * idf
    
    if weight > 0:
        print(f"     + Từ '{word:<15}': TF={tf_doc} | IDF={idf} -> Weight={weight}")

# ==============================================================================
# 4. DENSE EMBEDDING (Dense - Ngữ nghĩa)
# ==============================================================================
print(f"\n4. DENSE VECTOR (Ngữ nghĩa)")
print(f"   - Hệ thống đưa text vào model 'vietnamese-bi-encoder'.")
print(f"   - Kết quả là một vector 768 chiều (dãy số thực).")
# Tạo vector ngẫu nhiên giả lập
mock_dense = np.random.rand(5).tolist() 
print(f"   - Vector mô phỏng (5 chiều đầu): [{mock_dense[0]:.4f}, {mock_dense[1]:.4f}, ...]")
print(f"   => Máy tính sẽ dùng vector này để so sánh góc (Cosine) với vector câu hỏi.")

# ==============================================================================
# 5. RRF FUSION (Trộn kết quả)
# ==============================================================================
print(f"\n5. RRF FUSION (Tính điểm xếp hạng)")
print(f"   Giả sử ta tìm được 'Điều 12' ở cả 2 nhánh với thứ hạng như sau:")
rank_sparse = 1  # Tìm thấy đầu tiên bên từ khóa
rank_dense = 5   # Tìm thấy thứ 5 bên ngữ nghĩa (vì dense có thể tìm thấy cái khác khớp hơn về nghĩa rộng)

k = 60 # Hằng số RRF
rrf_score = (1 / (k + rank_sparse)) + (1 / (k + rank_dense))

print(f"   - Rank Sparse: {rank_sparse}")
print(f"   - Rank Dense:  {rank_dense}")
print(f"   - Công thức: Score = 1/(60 + {rank_sparse}) + 1/(60 + {rank_dense})")
print(f"   - RRF Score: {rrf_score:.6f}")
print(f"   => Điểm này càng cao thì văn bản càng được xếp lên trên cùng.")
