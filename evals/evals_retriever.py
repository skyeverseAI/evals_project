from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric, ContextualPrecisionMetric
from deepeval.models import OpenRouterModel

from src.retriever import (
    build_retriever,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K,
)
from src.reranker import CROSS_ENCODER
from evals.harness import load_goldens, summarize_by_metric, print_summary

load_dotenv()

GOLDEN_PATH = "goldens/retriever_goldens.json"
JUDGE_MODEL = "openai/gpt-4.1-mini"  # OpenRouter's name for the model
THRESHOLD = 0.7


def retriever_config(retriever):
    # read the config off the INJECTED retriever, so the run is tagged with what
    # was actually measured (reranker vs base), not with module constants
    if hasattr(retriever, "fetch_k"):                       # RerankingRetriever
        return {
            "retriever": "reranker",
            "top_k": retriever.top_k,
            "fetch_k": retriever.fetch_k,
            "reranker_model": CROSS_ENCODER,
        }
    top_k = getattr(retriever, "search_kwargs", {}).get("k", TOP_K)   # base VectorStoreRetriever
    return {"retriever": f"base_k{top_k}", "top_k": top_k}


def run(retriever):
    # judge LLM, called through OpenRouter (reads OPENROUTER_API_KEY from .env)
    judge = OpenRouterModel(model=JUDGE_MODEL, temperature=0)

    # 1. LOAD the golden set --- the fixed, human-authored truth
    goldens = load_goldens(GOLDEN_PATH)

    # 2. RUN THE INJECTED RETRIEVER on each question to fill retrieval_context,
    #    then build one test case per golden.
    test_cases = []
    for g in goldens:
        retrieved = retriever.invoke(g["query"])
        retrieval_context = [doc.page_content for doc in retrieved]

        test_cases.append(
            LLMTestCase(
                input=g["query"],
                expected_output=g["ideal_answer"],
                retrieval_context=retrieval_context,
                actual_output="(generator not evaluated in this run)",
            )
        )

    # 3. THE METRICS --- recall (did we miss?) and precision (ranked well?)
    metrics = [
        ContextualRecallMetric(threshold=THRESHOLD, model=judge, include_reason=True),
        ContextualPrecisionMetric(threshold=THRESHOLD, model=judge, include_reason=True),
    ]

    # 4. EVALUATE --- every metric on every case, batched + parallel, printed report.
    #    hyperparameters travel with the run so the report is tagged with the config.
    result = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        hyperparameters={
            **retriever_config(retriever),
            "embedding_model": EMBEDDING_MODEL,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "judge_model": JUDGE_MODEL,
            "golden_set": GOLDEN_PATH,
        },
    )
    return summarize_by_metric(result)


def run_local():
    """Standalone convenience: build the retriever, then run."""
    return run(build_retriever())


if __name__ == "__main__":
    print_summary("retriever", run_local())
