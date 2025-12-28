import google.generativeai as genai
import config

def initialize_gemini():
    """Initializes the Gemini client with the API key from config."""
    if not config.GEMINI_API_KEY:
        print("Warning: GEMINI_API_KEY not found in config.")
        return None
    
    try:
        genai.configure(api_key=config.GEMINI_API_KEY)
        return True
    except Exception as e:
        print(f"Error configuring Gemini: {e}")
        return None

def generate_answer(query, retrieved_points):
    """
    Generates an answer using Gemini based on retrieved context.
    
    Args:
        query (str): The user's question.
        retrieved_points (list): List of ScoredPoint objects from Qdrant.
        
    Returns:
        str: The generated answer or error message.
    """
    if not retrieved_points:
        return "Không tìm thấy tài liệu phù hợp để trả lời."

    # 1. Prepare Context
    context_text = ""
    for i, point in enumerate(retrieved_points):
        article = point.payload.get('article_num', '?')
        clause = point.payload.get('clause_num')
        text = point.payload.get('text', '').strip()
        
        ref = f"Điều {article}"
        if clause:
            ref += f" Khoản {clause}"
            
        context_text += f"\n--- Tài liệu {i+1} ({ref}) ---\n{text}\n"

    # 2. Construct Prompt
    # Vietnamese Prompt for better cultural alignment
    prompt = f"""Bạn là một Trợ lý Luật sư AI thông minh và chính xác.
Nhiệm vụ của bạn là trả lời câu hỏi của người dùng DỰA TRÊN các đoạn văn bản luật được cung cấp dưới đây.

LƯU Ý QUAN TRỌNG:
1. Chỉ sử dụng thông tin trong phần [Tài liệu tham khảo]. Tuyệt đối không bịa đặt.
2. Nếu tài liệu không chứa câu trả lời, hãy nói "Xin lỗi, văn bản pháp luật hiện có không đề cập rõ vấn đề này."
3. Khi trả lời, hãy trích dẫn cụ thể (ví dụ: "Theo Điều 12...").
4. Trình bày ngắn gọn, súc tích, dễ hiểu.

===== [Tài liệu tham khảo] =====
{context_text}

===== [Câu hỏi] =====
{query}

===== [Câu trả lời] =====
"""

    # 3. Call Gemini
    try:
        model = genai.GenerativeModel(config.GEMINI_MODEL_NAME)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Lỗi khi gọi Gemini: {str(e)}"
