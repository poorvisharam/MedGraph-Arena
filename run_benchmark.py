"""
End-to-End Orchestrator for MedGraph Arena Benchmark.
"""

import sys
import subprocess
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))

def run_script(script_name):
    print(f"\n🚀 Running {script_name}...")
    try:
        subprocess.run([sys.executable, script_name], check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running {script_name}. Aborting.")
        sys.exit(1)

def main():
    print("*" * 60)
    print("🏁 Starting MedGraph Arena Benchmark Pipeline")
    print("*" * 60)

    # 1. Fetch data
    run_script("data/fetch_all.py")

    # 2. Ingest
    run_script("ingest.py")

    # 3. Query
    run_script("query_runner.py")

    # 4. Evaluate with G-Eval (6 Criteria)
    run_script("geval_eval.py")

    # 5. Evaluate with LLM Judge
    run_script("llm_judge.py")

    print("*" * 60)
    print("✅ Benchmark Complete!")
    print("📊 Run `streamlit run app.py` to view the dashboard.")
    print("*" * 60)

if __name__ == "__main__":
    main()
