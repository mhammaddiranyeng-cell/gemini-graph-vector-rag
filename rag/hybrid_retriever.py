import json, os, re, socket
from typing import Callable, Optional
import networkx as nx
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

_orig = socket.getaddrinfo
socket.getaddrinfo = lambda *a, **kw: [r for r in _orig(*a, **kw) if r[0] == socket.AF_INET] or _orig(*a, **kw)
load_dotenv(override=True)
class HybridGraphVectorRetriever:
    def __init__(self, corpus_path: str = "data/engineering_corpus.json", embed_fn: Optional[Callable[[list[str]], np.ndarray]] = None):
        with open(corpus_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.docs = {d["doc_id"]: d for d in raw["documents"]}
        self.doc_ids = list(self.docs.keys())
        self.graph = nx.DiGraph()
        for t in raw["triples"]:
            self.graph.add_edge(t["head"], t["tail"], relation=t["relation"], doc_id=t["doc_id"])
        self.embed_fn = embed_fn or self._gemini_embed
        self.doc_vecs = self._norm(self.embed_fn([f"{d['title']}: {d['text']}" for d in raw["documents"]]))
    @staticmethod
    def _norm(mat: np.ndarray) -> np.ndarray:
        return mat / np.clip(np.linalg.norm(mat, axis=1, keepdims=True), 1e-9, None)

    def _gemini_embed(self, texts: list[str]) -> np.ndarray:
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"), http_options=types.HttpOptions(api_version="v1beta"))
        res = client.models.embed_content(model="gemini-embedding-001", contents=texts)
        return np.array([e.values for e in res.embeddings], dtype=np.float32)

    def vector_search(self, query: str) -> list[str]:
        q_vec = self._norm(self.embed_fn([query]))[0]
        return [self.doc_ids[i] for i in np.argsort(-(self.doc_vecs @ q_vec))]
    def graph_search(self, query: str) -> tuple[list[str], list[str]]:
        q_toks = {t for t in re.findall(r"[a-z0-9-]+", query.lower()) if len(t) > 3}
        seeds = [n for n in self.graph.nodes if q_toks & set(re.findall(r"[a-z0-9-]+", n.lower()))]
        scores, paths = {}, []
        for s in seeds:
            for u, v, d in self.graph.in_edges(s, data=True):
                scores[d["doc_id"]] = scores.get(d["doc_id"], 0.0) + 1.0
                paths.append(f"{u} --[{d['relation']}]--> {v} ({d['doc_id']})")
            for tgt, dist in nx.single_source_shortest_path_length(self.graph, s, cutoff=2).items():
                for u, v, d in self.graph.out_edges(tgt, data=True):
                    scores[d["doc_id"]] = scores.get(d["doc_id"], 0.0) + 1.0 / (dist + 1)
                    paths.append(f"{u} --[{d['relation']}]--> {v} ({d['doc_id']})")
        return sorted(self.doc_ids, key=lambda x: scores.get(x, 0.0), reverse=True), list(dict.fromkeys(paths))
    def retrieve(self, query: str, top_k: int = 3, rrf_k: int = 60) -> dict:
        vr = self.vector_search(query)
        gr, paths = self.graph_search(query)
        fused = {d: round(1.0 / (rrf_k + vr.index(d) + 1) + 1.0 / (rrf_k + gr.index(d) + 1), 6) for d in self.doc_ids}
        top = sorted(fused, key=fused.get, reverse=True)[:top_k]
        return {"vector_ranking": vr, "graph_ranking": gr, "graph_paths": paths, "results": [{**self.docs[d], "rrf_score": fused[d]} for d in top]}
