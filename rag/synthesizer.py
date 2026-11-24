import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError
from pydantic import BaseModel, Field
from rag.hybrid_retriever import HybridGraphVectorRetriever

load_dotenv(override=True)

class CitedClaim(BaseModel):
    claim: str
    doc_id: str

class GroundedAnswer(BaseModel):
    answer: str
    cited_claims: list[CitedClaim] = Field(min_length=1)
    graph_paths_used: list[str]
class HybridRAGPipeline:
    def __init__(self, retriever: HybridGraphVectorRetriever):
        self.retriever = retriever
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"), http_options=types.HttpOptions(api_version="v1beta", timeout=45000))
        self.models = list(dict.fromkeys([os.getenv("GEMINI_MODEL", "gemini-3-flash-preview"), "gemini-2.5-flash"]))

    def answer(self, query: str) -> dict:
        ctx = self.retriever.retrieve(query, top_k=3)
        docs_txt = "\n".join(f"[{r['doc_id']}] {r['title']}: {r['text']}" for r in ctx["results"])
        paths_txt = "\n".join(ctx["graph_paths"]) or "None"
        prompt = f"Answer strictly using the retrieved context and graph paths.\n\nContext:\n{docs_txt}\n\nGraph Paths:\n{paths_txt}\n\nQuestion: {query}"
        cfg = types.GenerateContentConfig(response_mime_type="application/json", response_schema=GroundedAnswer, temperature=0.1)
        for m in self.models:
            try:
                res = self.client.models.generate_content(model=m, contents=prompt, config=cfg)
                parsed = GroundedAnswer.model_validate_json(res.text)
                return {"retrieval": ctx, "grounded_synthesis": parsed.model_dump()}
            except APIError as e:
                if getattr(e, "code", None) in {404, 429, 500, 503, 504}:
                    continue
                raise
        raise RuntimeError("All Gemini models failed")
