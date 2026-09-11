"""
RAGAS Evaluation — Calculate RAGAS metrics for all 4 systems.
"""

import json
from pathlib import Path
import os
import time

import pandas as pd
from datasets import Dataset

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    QUERY_OUTPUTS_PATH,
    RAGAS_RESULTS_PATH,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    SYSTEM_NAMES,
)

# Setup Gemini for Ragas
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

def run_ragas_eval():
    """
    Run RAGAS evaluation on the output of query_runner.py.
    """
    print("=" * 60)
    print("MedGraph Arena — RAGAS Evaluation")
    print("=" * 60)

    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            context_precision,
            context_recall,
        )
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError:
        print("  ⚠ Please install ragas and langchain-google-genai")
        return

    if not QUERY_OUTPUTS_PATH.exists():
        print(f"  ⚠ Query outputs not found at {QUERY_OUTPUTS_PATH}")
        return

    with open(QUERY_OUTPUTS_PATH, "r") as f:
        query_results = json.load(f)

    # Initialize Gemini models for RAGAS
    try:
        llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL)
    except Exception as e:
        print(f"  ⚠ Failed to initialize Gemini for Ragas: {e}")
        return

    metrics = [
        faithfulness,
        context_precision,
        context_recall,
    ]

    all_scores = {}

    for sys_name in SYSTEM_NAMES:
        print(f"\n📊 Evaluating {sys_name}...")
        
        # Prepare dataset for this system
        dataset_dict = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": [],
        }

        for item in query_results:
            dataset_dict["question"].append(item["question"])
            dataset_dict["answer"].append(item["systems"].get(sys_name, {}).get("answer", ""))
            # Ensure contexts is a list of strings
            contexts = item["systems"].get(sys_name, {}).get("contexts", [])
            dataset_dict["contexts"].append([str(c) for c in contexts] if contexts else [""])
            dataset_dict["ground_truth"].append(item["ground_truth"])

        dataset = Dataset.from_dict(dataset_dict)

        # Run evaluation
        try:
            # We do this in smaller batches to avoid rate limits
            result = evaluate(
                dataset,
                metrics=metrics,
                llm=llm,
                raise_exceptions=False,
            )
            
            # Convert result to dict
            scores_df = result.to_pandas()
            
            # Calculate averages
            sys_scores = {
                "faithfulness": float(scores_df["faithfulness"].mean()) if "faithfulness" in scores_df else 0.0,
                "context_precision": float(scores_df["context_precision"].mean()) if "context_precision" in scores_df else 0.0,
                "context_recall": float(scores_df["context_recall"].mean()) if "context_recall" in scores_df else 0.0,
            }
            
            all_scores[sys_name] = sys_scores
            
            print(f"    ✓ Faithfulness:      {sys_scores['faithfulness']:.3f}")
            print(f"    ✓ Context Precision: {sys_scores['context_precision']:.3f}")
            print(f"    ✓ Context Recall:    {sys_scores['context_recall']:.3f}")
            
        except Exception as e:
            print(f"  ⚠ RAGAS evaluation failed for {sys_name}: {e}")
            all_scores[sys_name] = {
                "faithfulness": 0.0,
                "context_precision": 0.0,
                "context_recall": 0.0,
            }
    # Save results
    with open(RAGAS_RESULTS_PATH, "w") as f:
        json.dump(all_scores, f, indent=2)
    print(f"\n💾 RAGAS results saved to {RAGAS_RESULTS_PATH}")

    return all_scores

if __name__ == "__main__":
    run_ragas_eval()
