"""
Update GraphRAG answers in results/query_outputs.json using --method local.
Preserves all Naive RAG, LightRAG, and NodeRAG outputs already computed.
"""

import json
import time
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from systems.graph_rag import GraphRAGSystem
from config import QUERY_OUTPUTS_PATH

def main():
    print("=" * 60)
    print("Updating GraphRAG outputs to Local Search (--method local)")
    print(f"Loading outputs from: {QUERY_OUTPUTS_PATH}")
    print("=" * 60)

    with open(QUERY_OUTPUTS_PATH, "r") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} evaluation questions.")
    graphrag = GraphRAGSystem()

    for i, item in enumerate(data):
        qid = item["id"]
        question = item["question"]
        print(f"\n[{i+1}/{len(data)}] ({qid}) {question[:70]}...")
        
        try:
            output = graphrag.query(question, method="local")
            item["systems"]["graphrag"] = output
            print(f"  ✓ {output['latency_ms']}ms — {output['answer'][:90]}...")
        except Exception as e:
            print(f"  ⚠ Error: {e}")
            item["systems"]["graphrag"] = {
                "answer": f"[Error: {e}]",
                "contexts": [],
                "latency_ms": 0,
            }

        # Save progress incrementally
        with open(QUERY_OUTPUTS_PATH, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  💾 Saved progress ({i+1}/{len(data)}) to {QUERY_OUTPUTS_PATH.name}")

        time.sleep(2)

    print(f"\n🎉 Successfully updated all 25 GraphRAG responses to Local Search in {QUERY_OUTPUTS_PATH}")

if __name__ == "__main__":
    main()
