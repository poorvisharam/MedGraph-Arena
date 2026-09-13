"""
Query Runner — Run all evaluation questions through ALL 4 RAG systems.
"""

import json
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    EVAL_DATASET_PATH,
    QUERY_OUTPUTS_PATH,
    RESULTS_DIR,
    SYSTEM_DISPLAY_NAMES,
    GEMINI_RPM,
)


def load_eval_dataset(path: Path = EVAL_DATASET_PATH) -> list[dict]:
    """Load evaluation questions from JSON."""
    with open(path, "r") as f:
        return json.load(f)


def run_queries(systems: dict, eval_dataset: list[dict] | None = None) -> list[dict]:
    """
    Run each evaluation question through all 4 systems.
    Returns list of result dicts with per-system answers.
    """
    if eval_dataset is None:
        eval_dataset = load_eval_dataset()

    print("=" * 60)
    print("MedGraph Arena — Query Runner")
    print(f"  Questions: {len(eval_dataset)}")
    print(f"  Systems:   {len(systems)}")
    print("=" * 60)

    results = []
    total_queries = len(eval_dataset) * len(systems)
    query_count = 0

    for i, qa in enumerate(eval_dataset):
        question = qa["question"]
        print(f"\n❓ [{i+1}/{len(eval_dataset)}] {question[:80]}...")

        result = {
            "id": qa["id"],
            "question": question,
            "ground_truth": qa["ground_truth"],
            "question_type": qa["question_type"],
            "sources_needed": qa.get("sources_needed", []),
            "systems": {},
        }

        for sys_name, system in systems.items():
            display_name = SYSTEM_DISPLAY_NAMES.get(sys_name, sys_name)
            query_count += 1
            print(f"  → [{display_name}] Querying... ({query_count}/{total_queries})")

            try:
                output = system.query(question)
                result["systems"][sys_name] = output
                print(f"    ✓ {output['latency_ms']:.0f}ms — {output['answer'][:80]}...")
            except Exception as e:
                print(f"    ⚠ Error: {e}")
                result["systems"][sys_name] = {
                    "answer": f"[Error: {e}]",
                    "contexts": [],
                    "latency_ms": 0,
                }
            time.sleep(2)

        results.append(result)

        # Incremental save so no progress is ever lost
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(QUERY_OUTPUTS_PATH, "w") as f:
            json.dump(results, f, indent=2)
        print(f"    💾 Saved progress ({i+1}/{len(eval_dataset)}) to {QUERY_OUTPUTS_PATH.name}")

    print(f"\n💾 All results saved successfully to {QUERY_OUTPUTS_PATH}")
    return results


if __name__ == "__main__":
    from systems.naive_rag import NaiveRAG
    from systems.graph_rag import GraphRAGSystem
    from systems.light_rag import LightRAGSystem
    from systems.node_rag import NodeRAGSystem

    print("\n📦 Loading RAG systems for evaluation...")
    systems = {
        "naive_rag": NaiveRAG(),
        "graphrag": GraphRAGSystem(),
        "lightrag": LightRAGSystem(),
        "noderag": NodeRAGSystem(),
    }
    run_queries(systems)

