import numpy as np
from rag.hybrid_retriever import HybridGraphVectorRetriever

def _fake_embed(texts: list[str]) -> np.ndarray:
    vecs = []
    for t in texts:
        t_l = t.lower()
        vecs.append([float("modbus" in t_l), float("battery" in t_l or "mppt" in t_l), float("yolov8" in t_l or "stepper" in t_l)])
    return np.array(vecs, dtype=np.float32) + 0.01

def test_hybrid_rrf_multi_hop():
    ret = HybridGraphVectorRetriever(embed_fn=_fake_embed)
    out = ret.retrieve("How does RPI4-SORT-01 control stepper gate DRV-TB6600?", top_k=2)
    assert out["results"][0]["doc_id"] == "DOC-03"
    assert any("DRV-TB6600" in p for p in out["graph_paths"])
