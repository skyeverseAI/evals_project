"""
evals/eval_generator.py
=======================
Component-level evaluation of the GENERATOR, in isolation.

Faithfulness: of the claims in the generated answer, how many are supported
by the context it was given? (Did the generator make things up?)

ISOLATION: we feed the generator the GOLDEN context (the known-good chunks
from the faithfulness dataset), NOT the retriever's output. So a low score
is purely the generator's fault — the context was already correct.

    python -m evals.eval_generator
"""

import json
import re

from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.models import OpenRouterModel

from src.generator import generate, llm   # your generator: generate(query, context) -> answer
from src.retriever import CHUNK_SIZE, CHUNK_OVERLAP
from evals.harness import load_goldens, summarize_by_metric, print_summary

load_dotenv()

CHUNKS_PATH = "chunks_dump.json"
GOLDEN_PATH = "goldens/faithfulness_goldens.json"
JUDGE_MODEL = "openai/gpt-4o-mini"  # OpenRouter's name for the model
THRESHOLD = 0.7


def load_chunk_text():
    # {lecture: {chunk_no: text}} --- the text the goldens' chunk pointers refer to
    with open(CHUNKS_PATH) as f:
        return {
            lec["lecture"]: {c["chunk"]: c["text"] for c in lec["chunks"]}
            for lec in json.load(f)
        }


def golden_context(g, chunk_text):
    # "lecture 1, chunks 11-12" / "lecture 11, chunk 31" -> [text of 11, text of 12]
    m = re.search(r"chunks? (\d+)(?:-(\d+))?", g["context"])
    start, end = int(m.group(1)), int(m.group(2) or m.group(1))
    return [chunk_text[g["lecture"]][n] for n in range(start, end + 1)]


def run():
    # judge LLM, called through OpenRouter (reads OPENROUTER_API_KEY from .env)
    judge = OpenRouterModel(model=JUDGE_MODEL, temperature=0)

    # 1. LOAD the faithfulness golden set, and the chunk text its pointers refer to
    goldens = load_goldens(GOLDEN_PATH)
    chunk_text = load_chunk_text()

    # 2. RUN THE GENERATOR on the GOLDEN context (isolation), build one test case each
    test_cases = []
    for g in goldens:
        context = golden_context(g, chunk_text)   # known-good context (list of chunk strings)
        answer = generate(g["question"], context)  # RUN the generator -> actual_output

        test_cases.append(
            LLMTestCase(
                input=g["question"],
                actual_output=answer,             # the generated answer we're judging
                retrieval_context=context,        # faithfulness checks the answer against THIS
                expected_output=g["expected_answer"],
                additional_metadata={"id": g["id"], "type": g["type"], "language": g["language"]},
            )
        )

    # 3. THE METRICS --- decompose actual_output into claims, attribute each to context
    metrics = [
        FaithfulnessMetric(
            threshold=THRESHOLD,
            model=judge,
            include_reason=True,   # prints WHY each score --- shows which claims were unsupported
        ),
        AnswerRelevancyMetric(
            threshold=THRESHOLD,
            model=judge,
            include_reason=True,
        ),
    ]

    # 4. EVALUATE --- runs the metrics on every case, prints a report
    result = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        hyperparameters={
            "generator_model": llm.model_name,
            "generator_temperature": llm.temperature,
            "judge_model": JUDGE_MODEL,
            "golden_set": GOLDEN_PATH,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
        },
    )
    return summarize_by_metric(result)


if __name__ == "__main__":
    print_summary("generator", run())
