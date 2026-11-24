import json
from rag.hybrid_retriever import HybridGraphVectorRetriever
from rag.synthesizer import HybridRAGPipeline

BENCHMARK = [
    {"q": "How does RPI4-SORT-01 actuate the stepper gate?", "expected": ["DOC-03"]},
    {"q": "What protects LTE modem MDM-77 during low battery voltage?", "expected": ["DOC-02", "DOC-01"]},
    {"q": "Which PLC connects to both Gateway GW-ZAHLE-01 and DRV-TB6600?", "expected": ["DOC-01", "DOC-03"]},
]

def evaluate_retrieval(retriever: HybridGraphVectorRetriever, k: int = 2) -> dict:
    metrics = {"vector_recall@2": 0.0, "graph_recall@2": 0.0, "hybrid_rrf_recall@2": 0.0}
    for item in BENCHMARK:
        out = retriever.retrieve(item["q"], top_k=k)
        exp = set(item["expected"])
        metrics["vector_recall@2"] += len(exp & set(out["vector_ranking"][:k])) / len(exp)
        metrics["graph_recall@2"] += len(exp & set(out["graph_ranking"][:k])) / len(exp)
        metrics["hybrid_rrf_recall@2"] += len(exp & {r["doc_id"] for r in out["results"]}) / len(exp)
    return {m: round(v / len(BENCHMARK), 3) for m, v in metrics.items()}

if __name__ == "__main__":
    ret = HybridGraphVectorRetriever()
    scores = evaluate_retrieval(ret)
    live = HybridRAGPipeline(ret).answer("If FIB-BEKAA-NORTH degrades and battery voltage drops to 22.0V, how does telemetry stay online?")
    print(json.dumps({"benchmark_metrics": scores, "multi_hop_demo": live}, indent=2))
