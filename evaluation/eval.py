"""
RAGAS Evaluation Harness
Run from the project root: python -m evaluation.eval

Requirements:
    pip install ragas datasets

golden_dataset.json must be populated with real Q&A pairs before running.
"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

from llm.rag import ask


def load_golden_dataset(path="evaluation/golden_dataset.json"):
    with open(path) as f:
        data = json.load(f)
    print(f"[INFO] Loaded {len(data)} questions from golden dataset.")
    return data


def run_pipeline(golden_data):
    """Runs the full RAG pipeline on every question in the golden dataset."""
    rows = []

    for i, item in enumerate(golden_data, start=1):
        question = item["question"]
        ground_truth = item["ground_truth"]

        print(f"[{i}/{len(golden_data)}] {question}")
        response = ask(question)

        rows.append({
            "question": question,
            "answer": response["answer"],
            "contexts": [s["text"] for s in response["sources"]],
            "ground_truth": ground_truth,
        })

    return rows


def main():
    print("=== EnterpriseRAG — RAGAS Evaluation ===\n")

    golden_data = load_golden_dataset()
    rows = run_pipeline(golden_data)

    dataset = Dataset.from_list(rows)

    print("\n[INFO] Running RAGAS metrics (faithfulness, answer_relevancy, context_precision, context_recall)...")
    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    print("\n=== Results ===")
    print(results)

    output_path = "evaluation/results.csv"
    results.to_pandas().to_csv(output_path, index=False)
    print(f"\n[INFO] Results saved to {output_path}")


if __name__ == "__main__":
    main()
