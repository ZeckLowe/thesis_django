from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (faithfulness, answer_relevancy, answer_correctness, context_precision, context_recall)

data = (
    "query": [question],
    "generated_response": [generated_response],
    "retrieved_documents": [retrieved_documents]
)

dataset = Dataset.from_dict(data)

metrics = [
    faithfulness,
    answer_relevancy,
    answer_correctness,
    context_precision,
    context_recall
]

results = evaluate(dataset, metrics)

for metric_name, score in results.items():
    print(f"{metric_name}: {score}")
