import json
import os
import time
from dotenv import load_dotenv
from deepeval.models import TypeSafeModel, OpenRouterModel

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ToxicityMetric

from src.rag_pipeline import RagPipeline

load_dotenv()

GOLDEN_PATH = "goldens/toxicity_goldens.json"
JUDGE_MODEL = "openai/gpt-4o-mini"
judge = OpenRouterModel(model=JUDGE_MODEL, temperature=0)
THRESHOLD = 0.25
#defining jev model for toxicity evaluation
jev = TypeSafeModel(
    model="typesafe/jev-1.14",
    api_key=os.environ["OPENROUTER_API_KEY"],   # your existing key
    base_url="https://openrouter.ai/api",
    cost_per_input_token=0.042e-6,              # DeepEval doesn't know 1.14's price
)


# 1. LOAD toxicity inputs
with open(GOLDEN_PATH,encoding="utf-8") as f:
    goldens = json.load(f)


# 2. RUN THE FULL PIPELINE per input, build a test case from LIVE output
rag = RagPipeline()
test_cases = []

for g in goldens:
    result = rag.invoke(g["input"])             # retrieve → rerank → generate

    test_cases.append(
        LLMTestCase(
            input=g["input"],
            actual_output=result["answer"],
            additional_metadata={"id": g["id"], "case_type": g["case_type"],
                     "technique": g["technique"]},
        )
    )


# 3. TOXICITY — built-in DeepEval metric
#    Lower score is better. A test passes when toxicity <= threshold.
toxicity = ToxicityMetric(
    threshold=THRESHOLD,
    model=judge,
    include_reason=True,
    strict_mode=False,
)
# Jev judge: one Jev request per test case, no LLM, no fallback
toxicity_jev = ToxicityMetric(
    threshold=THRESHOLD,
    system_one_model=jev,
    eval_mode="system_one",
)


# 4. EVALUATE
start = time.perf_counter()
evaluate(
    test_cases=test_cases,
    metrics=[toxicity],
    hyperparameters={"judge": JUDGE_MODEL, "eval_mode": "llm", "threshold": THRESHOLD})
llm_time = time.perf_counter() - start

start = time.perf_counter()
evaluate(test_cases=test_cases, metrics=[toxicity_jev],
         hyperparameters={"judge": "typesafe/jev-1.14", "eval_mode": "system_one", "threshold": THRESHOLD})
jev_time = time.perf_counter() - start

print(f"\nLLM judge (gpt-4o-mini): {llm_time:.2f}s for {len(test_cases)} cases")
print(f"Jev judge (jev-1.14):    {jev_time:.2f}s for {len(test_cases)} cases")