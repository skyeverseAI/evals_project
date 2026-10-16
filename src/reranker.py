from sentence_transformers import CrossEncoder
from src.retriever import load_store, TOP_K
from langsmith import traceable

# small, fast, CPU-friendly reranker. Downloads once (~80MB) on first run.
# CROSS_ENCODER1 = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CROSS_ENCODER = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
FETCH_K = 10  # how many the bi-encoder brings back (over-retrieve)

class RerankingRetriever:
    def __init__(self, fetch_k=FETCH_K, top_k=TOP_K):
        self.store = load_store()
        self.reranker = CrossEncoder(
            CROSS_ENCODER,
            max_length=512,    
            )
        self.fetch_k = fetch_k   # how many the bi-encoder brings back (over-retrieve)
        self.top_k = top_k       # how many survive after reranking

    @traceable(run_type="retriever", name="RerankingRetriever")
    def invoke(self, query):
        # 1. OVER-RETRIEVE: fast bi-encoder, deliberately more than we need
        candidates = self.store.similarity_search(query, k=self.fetch_k)

        # 2. RERANK: score each (query, chunk) pair TOGETHER with the cross-encoder
        pairs = [(query, doc.page_content) for doc in candidates]
        scores = self.reranker.predict(pairs)

        # 3. SORT by score, keep the best top_k
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[: self.top_k]]