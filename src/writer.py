# importing RagPipeline and json
from src.rag_pipeline import RagPipeline
import json

PATH_TO_GOLDEN_CORRECT = "goldens/golden_dataset_for_correctness.json"
# access of golden dataset
with open(PATH_TO_GOLDEN_CORRECT, encoding='utf-8', mode="r") as g:
    goldens = json.load(g)

# creating an instance of RagPipeline
rag_pipeline = RagPipeline()


for i, golden in enumerate(goldens,1):
    if golden["generated_answer"]:
        continue  # skip if already generated
    query = golden["question"]
    golden["generated_answer"] = rag_pipeline.invoke(query)["answer"]
    with open(PATH_TO_GOLDEN_CORRECT, "w", encoding='utf-8') as f: 
        json.dump(goldens, f, indent=2, ensure_ascii=False)
    print(f"answer for {i}/{len(goldens)} question stored")

    