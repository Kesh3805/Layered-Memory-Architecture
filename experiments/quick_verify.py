"""Quick arm differentiation check — verify retrieval_info flags are correct."""
import requests

BASE = "http://localhost:8000"
QUERY = "What is pgvector and how do I install it?"


def main():
    cid = requests.post(f"{BASE}/conversations", json={"title": "verify"}, timeout=10).json()["id"]

    arms = [
        ("vector_only",           {"hybrid_search": False, "reranker": False}),
        ("hybrid_only",           {"hybrid_search": True,  "reranker": False}),
        ("hybrid_plus_reranker",  {"hybrid_search": True,  "reranker": True}),
    ]

    print(f"\nQuery: {QUERY}\n")
    print(f"{'Arm':<25} {'hybrid':<8} {'reranker':<10} {'avg_sim':<10} {'num_docs'}")
    print("-" * 65)

    for name, cfg in arms:
        requests.post(f"{BASE}/experiments/config", json=cfg, timeout=5)
        r = requests.post(
            f"{BASE}/chat",
            json={"user_query": QUERY, "conversation_id": cid},
            timeout=90,
        )
        if r.status_code != 200:
            print(f"{name:<25} HTTP {r.status_code}")
            continue
        ri = r.json().get("retrieval_info", {})
        print(
            f"{name:<25} {str(ri.get('hybrid_search','?')):<8} "
            f"{str(ri.get('reranker','?')):<10} "
            f"{ri.get('rag_avg_similarity', 0):.4f}   "
            f"{ri.get('num_docs', 0)}"
        )

    requests.post(f"{BASE}/experiments/reset", timeout=5)
    requests.delete(f"{BASE}/conversations/{cid}", timeout=5)
    print("\nDone — config reset.")


if __name__ == "__main__":
    main()
