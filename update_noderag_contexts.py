"""
Update NodeRAG contexts in results/query_outputs.json using searcher.search().
Preserves all existing answers.
"""

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from systems.node_rag import NodeRAGSystem
from NodeRAG import NodeSearch
from config import QUERY_OUTPUTS_PATH

def main():
    print("=" * 60)
    print("Populating NodeRAG retrieved contexts in query_outputs.json")
    print("=" * 60)

    with open(QUERY_OUTPUTS_PATH, "r") as f:
        data = json.load(f)

    nr = NodeRAGSystem()
    searcher = NodeSearch(nr.noderag_config)

    for i, item in enumerate(data):
        qid = item["id"]
        question = item["question"]
        print(f"[{i+1}/{len(data)}] ({qid}) Extracting contexts for: {question[:60]}...")
        
        try:
            retrieval = searcher.search(question)
            contexts = []
            if hasattr(retrieval, "retrieved_list") and retrieval.retrieved_list:
                contexts = [
                    item_c[0] if isinstance(item_c, (tuple, list)) else str(item_c)
                    for item_c in retrieval.retrieved_list
                ]
            elif hasattr(retrieval, "unstructured_prompt") and retrieval.unstructured_prompt:
                contexts = [retrieval.unstructured_prompt]

            if contexts:
                item["systems"]["noderag"]["contexts"] = contexts[:10]
                print(f"  ✓ Extracted {len(item['systems']['noderag']['contexts'])} contexts")
            else:
                print("  ⚠ No contexts found")
        except Exception as e:
            print(f"  ⚠ Error: {e}")

        # Save incrementally
        with open(QUERY_OUTPUTS_PATH, "w") as f:
            json.dump(data, f, indent=2)

    print(f"\n🎉 Successfully populated NodeRAG contexts in {QUERY_OUTPUTS_PATH}")

if __name__ == "__main__":
    main()
