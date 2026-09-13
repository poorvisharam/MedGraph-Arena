"""
RAGAS Evaluation — [DEPRECATED & COMMENTED OUT]

NOTE ON EVALUATION METHODOLOGY:
RAGAS evaluation has been commented out and disabled in favor of G-Eval / LLM-as-Judge:
1. G-Eval is a newer, state-of-the-art evaluation framework based on form-filling Chain-of-Thought (CoT)
   with explicit criteria and task-specific rubrics.
2. RAGAS also uses an LLM behind the scenes for synthetic statement decomposition and claim extraction,
   which is substantially slower, highly brittle to API rate limits, and significantly more expensive.
"""

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
from config import RAGAS_RESULTS_PATH

def run_ragas_eval():
    """
    [DISABLED] RAGAS evaluation commented out.
    G-Eval is a newer method. RAGAS also uses an LLM behind the scenes and is expensive.
    """
    print("=" * 60)
    print("MedGraph Arena — RAGAS Evaluation [SKIPPED / DEPRECATED]")
    print("=" * 60)
    print("  ℹ RAGAS evaluation is disabled in favor of G-Eval / LLM-as-Judge.")
    print("  ℹ Reason: G-Eval is a newer, more robust method. RAGAS also relies heavily on LLM")
    print("    decompositions behind the scenes, making it brittle, slow, and expensive.")
    print("=" * 60)
    return {}

# ==============================================================================
# ORIGINAL RAGAS CODE PRESERVED BELOW FOR REFERENCE (COMMENTED OUT)
# ==============================================================================
# import os
# import time
# import pandas as pd
# from datasets import Dataset
# from config import (
#     QUERY_OUTPUTS_PATH,
#     GEMINI_API_KEY,
#     GEMINI_MODEL,
#     SYSTEM_NAMES,
# )
#
# os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
#
# def _original_ragas_eval():
#     from ragas import evaluate
#     from ragas.metrics import faithfulness, context_precision, context_recall
#     from langchain_google_genai import ChatGoogleGenerativeAI
#
#     if not QUERY_OUTPUTS_PATH.exists():
#         return
#
#     with open(QUERY_OUTPUTS_PATH, "r") as f:
#         query_results = json.load(f)
#
#     llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL)
#     metrics = [faithfulness, context_precision, context_recall]
#     all_scores = {}
#
#     for sys_name in SYSTEM_NAMES:
#         dataset_dict = {
#             "question": [],
#             "answer": [],
#             "contexts": [],
#             "ground_truth": [],
#         }
#         for item in query_results:
#             dataset_dict["question"].append(item["question"])
#             dataset_dict["answer"].append(item["systems"].get(sys_name, {}).get("answer", ""))
#             contexts = item["systems"].get(sys_name, {}).get("contexts", [])
#             dataset_dict["contexts"].append([str(c) for c in contexts] if contexts else [""])
#             dataset_dict["ground_truth"].append(item["ground_truth"])
#
#         dataset = Dataset.from_dict(dataset_dict)
#         result = evaluate(dataset, metrics=metrics, llm=llm, raise_exceptions=False)
#         scores_df = result.to_pandas()
#         sys_scores = {
#             "faithfulness": float(scores_df["faithfulness"].mean()) if "faithfulness" in scores_df else 0.0,
#             "context_precision": float(scores_df["context_precision"].mean()) if "context_precision" in scores_df else 0.0,
#             "context_recall": float(scores_df["context_recall"].mean()) if "context_recall" in scores_df else 0.0,
#         }
#         all_scores[sys_name] = sys_scores
#
#     with open(RAGAS_RESULTS_PATH, "w") as f:
#         json.dump(all_scores, f, indent=2)
#     return all_scores

if __name__ == "__main__":
    run_ragas_eval()
