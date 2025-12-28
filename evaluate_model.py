import json
import sys
import numpy as np
from retrieval import load_resources, hybrid_search, parse_filter_hints
import config

# Configuration
TOP_K_EVAL = 10  # Check if correct answer is in top 10

def calculate_metrics(results, expected_articles):
    """
    Calculate metrics for a single query.
    - Hit: Is any expected article in results?
    - Rank: Rank of the FIRST relevant result (1-based).
    """
    found_ranks = []
    
    # Extract article numbers from results
    # Each result is a ScoredPoint. payload['article_num'] is what we check.
    for rank, hit in enumerate(results, 1):
        actual_article = hit.payload.get('article_num')
        if actual_article in expected_articles:
            found_ranks.append(rank)
            
    if not found_ranks:
        return 0.0, 0.0  # Recall=0, MRR=0
        
    # Recall@K (Binary: Did we find it?)
    recall = 1.0
    
    # MRR (1 / Rank of first correct answer)
    first_rank = found_ranks[0]
    mrr = 1.0 / first_rank
    
    return recall, mrr

def run_evaluation():
    print("="*60)
    print(f"BẮT ĐẦU ĐÁNH GIÁ MÔ HÌNH (Benchmark)")
    print(f"Top K: {TOP_K_EVAL}")
    print("="*60)
    
    # 1. Load Data
    try:
        with open("test_dataset.json", "r", encoding="utf-8") as f:
            test_cases = json.load(f)
    except FileNotFoundError:
        print("Error: 'test_dataset.json' not found.")
        return

    # 2. Load Resources (Model + DB)
    print("Loading Models & Database...")
    r = load_resources()
    
    total_recall = 0
    total_mrr = 0
    n = len(test_cases)
    
    print(f"\nEvaluating {n} test cases...\n")
    
    # 3. Validation Loop
    for i, case in enumerate(test_cases):
        query = case['query']
        expected = case['expected_articles']
        desc = case.get('description', '')
        
        print(f"Test #{i+1}: \"{query}\"")
        print(f"   Target: Điều {expected}")
        
        # SEARCH
        # hybrid_search returns (QueryResponse, filter_used)
        response, _ = hybrid_search(query, top_k=TOP_K_EVAL, r=r)
        results = response.points
        
        # METRICS
        recall, mrr = calculate_metrics(results, expected)
        
        # LOGGING
        status = "PASSED" if recall > 0 else "FAILED"
        print(f"   [{status}] Rank found: {int(1/mrr) if mrr > 0 else '-'} | MRR: {mrr:.4f}")
        
        # Print top 3 retrieved for debug
        top_3_found = [h.payload.get('article_num') for h in results[:3]]
        print(f"   Top 3 retrieved: {[f'Điều {x}' for x in top_3_found]}")
        print("-" * 30)
        
        total_recall += recall
        total_mrr += mrr

    # 4. Final Score
    avg_recall = (total_recall / n) * 100
    avg_mrr = total_mrr / n
    
    print("\n" + "="*60)
    print("KẾT QUẢ ĐÁNH GIÁ TỔNG HỢP")
    print("="*60)
    print(f"1. Recall@{TOP_K_EVAL} (Độ phủ): {avg_recall:.2f}%")
    print(f"   (Tỷ lệ câu hỏi tìm thấy ít nhất 1 kết quả đúng trong Top {TOP_K_EVAL})")
    print(f"\n2. MRR (Thứ hạng trung bình): {avg_mrr:.4f}")
    if avg_mrr > 0:
        print(f"   => Trung bình câu trả lời đúng nằm ở vị trí thứ: ~{1/avg_mrr:.1f}")
    else:
        print("   => Không tìm thấy kết quả nào.")
    print("="*60)

if __name__ == "__main__":
    # Force UTF-8 output
    sys.stdout.reconfigure(encoding='utf-8')
    run_evaluation()
