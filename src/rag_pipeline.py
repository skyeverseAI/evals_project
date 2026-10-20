from src.retriever import TOP_K
from src.reranker import RerankingRetriever, FETCH_K
from src.generator import generate
# from langsmith import traceable 



class RagPipeline:
    def __init__(self, fetch_k=FETCH_K, top_k=TOP_K):
        # one retriever instance — loads the store + reranker model once
        self.retriever = RerankingRetriever(fetch_k=fetch_k, top_k=top_k)

    # @traceable(run_type="chain", name="RagPipeline")
    def invoke(self, query: str) -> dict:
        # 1. RETRIEVE: over-fetch then rerank down to top_k Documents
        docs = self.retriever.invoke(query)

        # 2. UNPACK: generator wants list[str], the triad wants the same strings
        context = [doc.page_content for doc in docs]

        # 3. GENERATE: grounded answer from the retrieved context
        answer = generate(query, context)

        # return all three legs of the triad so the eval harness can score them
        return {
            "query": query,
            "context": context,
            "answer": answer,
        }


# quick manual smoke test: python -m src.rag_pipeline
if __name__ == "__main__":
    
    rag = RagPipeline()
    result = rag.invoke("What is Nitish Sir's phone number??")
    print("QUERY:  ", result["query"], '\n')
    print("ANSWER: ", result["answer"], '\n\n')
    print("\nCONTEXT CHUNKS:", '\n')
    for i, chunk in enumerate(result["context"]):
        print(f"  [{i}] {chunk[:120]}...")