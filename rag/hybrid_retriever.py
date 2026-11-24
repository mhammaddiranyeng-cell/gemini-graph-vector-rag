import json, os, socket
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
        texts = [f"{d['title']}: {d['text']}" for d in raw["documents"]]
        self.doc_vecs = self._norm(self.embed_fn(texts))

    @staticmethod
    def _norm(mat: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        return mat / np.clip(norms, 1e-9, None)

    def _gemini_embed(self, texts: list[str]) -> np.ndarray:
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"), http_options=types.HttpOptions(api_version="v1beta"))
        res = client.models.embed_content(model="gemini-embedding-001", contents=texts)
        return np.array([e.values for e in res.embeddings], dtype=np.float32)
    def vector_search(self, query: str) -> list[str]:
        q_vec = self._norm(self.embed_fn([query]))[0]
        scores = self.doc_vecs @ q_vec
        order = np.argsort(-scores)
        return [self.doc_ids[i] for i in order]

    def graph_search(self, query: str) -> tuple[list[str], list[str]]:
        q_low = query.lower()
        seeds = [n for n in self.graph.nodes if any(tok in n.lower() for tok in q_low.split() if len(tok) > 3)]
        doc_scores: dict[str, float] = {}
        paths: list[str] = []
        for s in seeds:
            lengths = nx.single_source_shortest_path_length(self.graph, s, cutoff=2)
            for target, dist in lengths.items():
                for u, v, data in self.graph.out_edges(target, data=True):
                    doc_scores[data["doc_id"]] = doc_scores.get(data["doc_id"], 0.0) + 1.0 / (dist + 1)
                    paths.append(f"{u} --[{data['relation']}]--> {v} ({data['doc_id']})")
        ranked = sorted(self.doc_ids, key=lambda d: doc_scores.get(d, 0.0), reverse=True)
        return ranked, list(dict.fromkeys(paths))

    def retrieve(self, query: str, top_k: int = 3, rrf_k: int = 60) -> dict:
        vec_rank = self.vector_search(query)
        grp_rank, paths = self.graph_search(query)
        fused = {}
        for d in self.doc_ids:
            rv = vec_rank.index(d) + 1
            rg = grp_rank.index(d) + 1
            fused[d] = round(1.0 / (rrf_k + rv) + 1.0 / (rrf_k + rg), 6)
        top_ids = sorted(fused.keys(), key=lambda d: fused[d], reverse=True)[:top_k]
        return {
            "vector_ranking": vec_rank,
            "graph_ranking": grp_rank,
            "graph_paths": paths,
            "results": [{**self.docs[d], "rrf_score": fused[d]} for d in top_ids],
        }
