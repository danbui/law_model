# search_law.py
import sys

from retrieval import load_resources, hybrid_search





TOP_K = 5


def main():
    # Force UTF-8 for Windows Console
    try:
        sys.stdin.reconfigure(encoding="utf-8")
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print("Loading resources...")
    try:
        r = load_resources()
    except FileNotFoundError:
        print("[ERROR] tfidf_model.pkl not found. Please run: python preprocess_word.py")
        return
    except Exception as e:
        print(f"[ERROR] Could not load resources: {e}")
        return

    print("Ready. Type Vietnamese query. Type 'exit' to quit.\n")

    while True:
        query = input("Hybrid Query > ").strip()
        if not query:
            continue
        if query.lower() in ["exit", "quit", "q"]:
            print("\nShutting down. Goodbye 👋")
            break

        try:
            results, q_filter = hybrid_search(query, TOP_K, r)
            print("\n📄 Results:\n")

            if not results.points:
                print("No results found.\n")
                continue

            for idx, point in enumerate(results.points, 1):
                payload = point.payload or {}

                doc_id = payload.get("doc_id", "N/A")
                chapter = payload.get("chapter", "")
                chapter_title = payload.get("chapter_title", "")

                article = payload.get("article", "N/A")
                clause = payload.get("clause", None)
                point_meta = payload.get("point", None)
                text = payload.get("text", "")

                print(f"Result {idx}")
                print(f"Score   : {point.score:.4f}")
                print(f"Doc ID  : {doc_id}")
                if chapter or chapter_title:
                    print(f"Chapter : {(chapter + ' ' + chapter_title).strip()}")

                print(f"Article : {article}")
                if clause:
                    print(f"Clause  : {clause}")
                if point_meta:
                    print(f"Point   : {point_meta}")

                print("Text snippet:")
                snippet = text[:250] + ("..." if len(text) > 250 else "")
                print(snippet)
                print("-" * 80)

            print()

        except KeyboardInterrupt:
            print("\n\nProcess interrupted. Stopping.")
            break
        except Exception as e:
            print(f"[ERROR] {e}\n")


if __name__ == "__main__":
    main()
