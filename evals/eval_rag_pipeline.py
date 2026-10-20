from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.models import OpenRouterModel
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
)

from src.rag_pipeline import RagPipeline
from src.retriever import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL
from src.reranker import CROSS_ENCODER
from src.generator import llm
from evals.harness import load_goldens, summarize_by_metric, print_summary

load_dotenv()

GOLDEN_PATH = "goldens/faithfulness_goldens.json"   # reuse the queries
JUDGE_MODEL = "openai/gpt-4o-mini"  # OpenRouter's name for the model
THRESHOLD = 0.7


def run(rag):
    # judge LLM, called through OpenRouter (reads OPENROUTER_API_KEY from .env)
    judge = OpenRouterModel(model=JUDGE_MODEL, temperature=0)

    # 1. LOAD goldens (question + reference answer + labels; context comes from the pipeline)
    goldens = load_goldens(GOLDEN_PATH)

    # 2. RUN THE INJECTED PIPELINE per query, build a test case from LIVE output
    test_cases = []
    for g in goldens:
        result = rag.invoke(g["question"])          # retrieve → rerank → generate

        test_cases.append(
            LLMTestCase(
                input=g["question"],
                actual_output=result["answer"],       # what the generator produced (RAG Pipeline)
                retrieval_context=result["context"],  # what the RETRIEVER returned (RAG Pipeline)
                expected_output=g["expected_answer"],
                additional_metadata={"id": g["id"], "type": g["type"],
                                     "language": g["language"]},
            )
        )

    # 3. THE THREE TRIAD METRICS
    metrics = [
        ContextualRelevancyMetric(threshold=THRESHOLD, model=judge, include_reason=True),
        FaithfulnessMetric(threshold=THRESHOLD, model=judge, include_reason=True),
        AnswerRelevancyMetric(threshold=THRESHOLD, model=judge, include_reason=True),
    ]

    # 4. EVALUATE
    result = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        hyperparameters={
            # retrieval
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "embedding_model": EMBEDDING_MODEL,
            "fetch_k": rag.retriever.fetch_k,      # read from the live object, not the constant
            "top_k": rag.retriever.top_k,
            "reranker": CROSS_ENCODER,
            # generation
            "generator_model": llm.model_name,
            "generator_temperature": llm.temperature,
            # eval setup
            "judge_model": JUDGE_MODEL,
            "golden_set": GOLDEN_PATH,
        },
    )
    return summarize_by_metric(result)


def run_local():
    """Standalone convenience: build the pipeline, then run."""
    return run(RagPipeline())


if __name__ == "__main__":
    print_summary("rag_pipeline", run_local())
