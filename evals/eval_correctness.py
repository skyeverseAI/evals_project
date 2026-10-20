import json
from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from deepeval.models import OpenRouterModel
from deepeval.metrics.g_eval import Rubric

# from src.rag_pipeline import RagPipeline

load_dotenv()

GOLDEN_PATH = "goldens/golden_dataset_for_correctness.json"
JUDGE_MODEL = "openai/gpt-4o-mini"
judge= OpenRouterModel(model=JUDGE_MODEL, temperature=0)
THRESHOLD = 0.7


# 1. LOAD queries + ideal answers
with open(GOLDEN_PATH, encoding="utf-8") as f:
    goldens = json.load(f)


# 2. RUN THE FULL PIPELINE per query, build a test case from LIVE output
# rag = RagPipeline() since I already generated answers using writer.py, I will use those answers directly.
test_cases = []
for g in goldens:
    # result = rag.invoke(g["question"])          # retrieve → rerank → generate #there is no need to run it again.

    test_cases.append(
        LLMTestCase(
            input=g["question"],
            actual_output=g["generated_answer"],       # what the generator produced (RAG Pipeline)
            expected_output=g["correct_answer"],
        )
    )


# 3. THE METRICS
correctness = GEval(
    name="correctness",
    evaluation_steps=[
    #     "Compare the actual output against the key facts in the expected output.",
    #     "Heavily penalize statements in the actual output that contradict the expected output or are factually wrong.",
    #     "Reward statements that match the expected output in meaning, regardless of wording.",
    #     "Do NOT penalize the actual output for omitting information - only wrong statements count here."
    #
    #-------------------------------- ----------------------------------------------------------------
    # Set 2 of instructions for the model to follow, 
    # We also define a rubric for scoring the correctness.
    "Compare only the factual claims in the actual output against the expected output.",
    "A claim is wrong only if it CONTRADICTS the expected output or is factually false. Judge truth, not completeness.",
    "A factually accurate answer must score at least 9 even if it is shorter, less detailed, or covers fewer points than the expected output.",
    "Do NOT deduct for brevity, missing elaboration, fewer examples, or omitted points - omissions are not errors here.",
    "Additional correct information must NEVER lower the score.",
    "Reserve low scores for answers that state something contradictory or factually incorrect.",
    ],

    rubric=[
        Rubric(score_range=(9, 10), expected_outcome="All stated claims are factually correct and consistent with the expected output. No contradictions. Brevity is fine."),
        
        Rubric(score_range=(5, 8),expected_outcome="Mostly correct but contains one minor inaccuracy or slightly imprecise claim."),

        Rubric(score_range=(0, 4),
        expected_outcome="Contains a clear factual error or a claim that contradicts the expected output."),
    ],
    evaluation_params = [
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
        
    ],
    threshold=THRESHOLD,
    model=judge,
    strict_mode = False, #weighted probabilities calculated
)

# COMPLETENESS
completeness = GEval(
    name="Completeness",
    evaluation_steps=[
        "Identify the key points contained in the expected output.",
        "Check how many of those key points are addressed in the actual output.",
        "Penalize the actual output for each key point from the expected output that it omits or only partially covers.",
        "Judge coverage only. Do NOT lower the score because a covered point is stated incorrectly — factual correctness is judged separately.",
        "Do NOT penalize the actual output for adding extra information beyond the expected output.",
    ],
    rubric=[
        Rubric(score_range=(9, 10), expected_outcome="Addresses essentially all key points in the expected output."),
        Rubric(score_range=(5, 8),  expected_outcome="Covers the main key points but misses one or more."),
        Rubric(score_range=(0, 4),  expected_outcome="Misses several key points; only partially covers the expected output."),
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    threshold=THRESHOLD,
    model=judge,
    strict_mode=False,
)


# 4. EVALUATE
evaluate(test_cases=test_cases, metrics=[correctness, completeness])