"""
LLM-as-Judge — Pairwise tournament between the 4 systems.
"""

import json
import itertools
import time
from pathlib import Path

from openai import OpenAI
import pydantic

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    QUERY_OUTPUTS_PATH,
    JUDGE_RESULTS_PATH,
    GEMINI_OPENAI_API_KEY,
    GEMINI_OPENAI_BASE_URL,
    GEMINI_MODEL,
    SYSTEM_NAMES,
)

class JudgeVerdict(pydantic.BaseModel):
    winner: str # 'A', 'B', or 'Tie'
    reasoning: str
    confidence: int # 1 to 5

def run_llm_judge():
    """Run pairwise tournament using LLM as judge."""
    print("=" * 60)
    print("MedGraph Arena — LLM-as-Judge Tournament")
    print("=" * 60)

    if not QUERY_OUTPUTS_PATH.exists():
        print(f"  ⚠ Query outputs not found at {QUERY_OUTPUTS_PATH}")
        return

    with open(QUERY_OUTPUTS_PATH, "r") as f:
        query_results = json.load(f)

    llm = OpenAI(
        api_key=GEMINI_OPENAI_API_KEY,
        base_url=GEMINI_OPENAI_BASE_URL,
    )

    # Generate all unique pairs of systems (6 pairs for 4 systems)
    matchups = list(itertools.combinations(SYSTEM_NAMES, 2))
    print(f"  Evaluating {len(matchups)} matchups...")

    results = []
    
    # Track wins/losses for Elo calculation
    stats = {sys: {"wins": 0, "losses": 0, "ties": 0} for sys in SYSTEM_NAMES}

    for item in query_results:
        q_id = item["id"]
        question = item["question"]
        ground_truth = item["ground_truth"]
        q_type = item["question_type"]

        print(f"\n❓ Question: {question[:60]}...")

        for sys_a, sys_b in matchups:
            ans_a = item["systems"].get(sys_a, {}).get("answer", "")
            ans_b = item["systems"].get(sys_b, {}).get("answer", "")

            # Skip if either is missing or error
            if not ans_a or not ans_b or ans_a.startswith("[Error") or ans_b.startswith("[Error"):
                continue

            # Judge System A vs System B
            verdict = _judge_pair(llm, question, ground_truth, ans_a, ans_b)
            
            # Position swap for bias mitigation (B vs A)
            verdict_swapped = _judge_pair(llm, question, ground_truth, ans_b, ans_a)

            # Determine final winner
            winner = "Tie"
            if verdict.winner == 'A' and verdict_swapped.winner == 'B':
                winner = sys_a
                stats[sys_a]["wins"] += 1
                stats[sys_b]["losses"] += 1
            elif verdict.winner == 'B' and verdict_swapped.winner == 'A':
                winner = sys_b
                stats[sys_b]["wins"] += 1
                stats[sys_a]["losses"] += 1
            else:
                stats[sys_a]["ties"] += 1
                stats[sys_b]["ties"] += 1

            print(f"  {sys_a} vs {sys_b} -> Winner: {winner}")

            results.append({
                "question_id": q_id,
                "question_type": q_type,
                "system_a": sys_a,
                "system_b": sys_b,
                "winner": winner,
                "reasoning_1": verdict.reasoning,
                "reasoning_2": verdict_swapped.reasoning,
            })
            
            time.sleep(2) # Rate limit mitigation

    # Save results
    with open(JUDGE_RESULTS_PATH, "w") as f:
        json.dump({"matchups": results, "stats": stats}, f, indent=2)
    print(f"\n💾 Judge results saved to {JUDGE_RESULTS_PATH}")
    
    print("\n🏆 Final Stats:")
    for sys, s in stats.items():
        print(f"  {sys:10s}: {s['wins']}W - {s['losses']}L - {s['ties']}T")

def _judge_pair(llm, question, ground_truth, answer_a, answer_b) -> JudgeVerdict:
    prompt = (
        "You are an expert medical judge evaluating two AI systems. "
        "Compare their answers to the question based on the ground truth.\n\n"
        f"Question: {question}\n\n"
        f"Ground Truth: {ground_truth}\n\n"
        f"Answer A:\n{answer_a}\n\n"
        f"Answer B:\n{answer_b}\n\n"
        "Evaluate which answer is more accurate, comprehensive, and helpful. "
        "Return your verdict as a JSON object with 'winner' ('A', 'B', or 'Tie'), "
        "'reasoning' (brief explanation), and 'confidence' (1-5)."
    )

    try:
        response = llm.chat.completions.create(
            model=GEMINI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        result_dict = json.loads(response.choices[0].message.content)
        return JudgeVerdict(**result_dict)
    except Exception as e:
        print(f"  ⚠ Judge error: {e}")
        return JudgeVerdict(winner="Tie", reasoning=f"Error: {e}", confidence=1)

if __name__ == "__main__":
    run_llm_judge()
