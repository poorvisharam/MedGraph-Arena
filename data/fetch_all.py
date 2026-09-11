"""
Orchestrator: Fetch all documents from all 5 sources.
Run this to populate the data/documents/ directory.
"""

from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.fetch_pubmed import fetch_pubmed_abstracts
from data.fetch_guidelines import fetch_guidelines
from data.fetch_fda_labels import fetch_fda_labels
from data.fetch_who_eml import fetch_who_eml
from data.fetch_medquad import fetch_medquad


def fetch_all() -> dict[str, list[Path]]:
    """
    Fetch documents from all 5 sources.
    Returns a dict mapping source name to list of saved file paths.
    """
    print("=" * 60)
    print("MedGraph Arena — Document Fetcher")
    print("=" * 60)

    results = {}
    total = 0

    # 1. PubMed abstracts
    print("\n📄 [1/5] Fetching PubMed abstracts...")
    start = time.time()
    try:
        results["pubmed"] = fetch_pubmed_abstracts()
    except Exception as e:
        print(f"  ⚠ PubMed fetch failed: {e}")
        results["pubmed"] = []
    elapsed = time.time() - start
    total += len(results["pubmed"])
    print(f"  ⏱ {elapsed:.1f}s")

    # 2. Clinical guidelines
    print("\n📋 [2/5] Fetching clinical guidelines...")
    start = time.time()
    try:
        results["guidelines"] = fetch_guidelines()
    except Exception as e:
        print(f"  ⚠ Guidelines fetch failed: {e}")
        results["guidelines"] = []
    elapsed = time.time() - start
    total += len(results["guidelines"])
    print(f"  ⏱ {elapsed:.1f}s")

    # 3. FDA drug labels
    print("\n💊 [3/5] Fetching FDA drug labels...")
    start = time.time()
    try:
        results["fda_labels"] = fetch_fda_labels()
    except Exception as e:
        print(f"  ⚠ FDA fetch failed: {e}")
        results["fda_labels"] = []
    elapsed = time.time() - start
    total += len(results["fda_labels"])
    print(f"  ⏱ {elapsed:.1f}s")

    # 4. WHO Essential Medicines List
    print("\n🌍 [4/5] Creating WHO Essential Medicines document...")
    start = time.time()
    try:
        results["who_eml"] = fetch_who_eml()
    except Exception as e:
        print(f"  ⚠ WHO EML fetch failed: {e}")
        results["who_eml"] = []
    elapsed = time.time() - start
    total += len(results["who_eml"])
    print(f"  ⏱ {elapsed:.1f}s")

    # 5. MedQuAD Q&A pairs
    print("\n❓ [5/5] Creating MedQuAD Q&A documents...")
    start = time.time()
    try:
        results["medquad"] = fetch_medquad()
    except Exception as e:
        print(f"  ⚠ MedQuAD fetch failed: {e}")
        results["medquad"] = []
    elapsed = time.time() - start
    total += len(results["medquad"])
    print(f"  ⏱ {elapsed:.1f}s")

    # Summary
    print("\n" + "=" * 60)
    print("📊 Fetch Summary")
    print("=" * 60)
    for source, files in results.items():
        print(f"  {source:15s}: {len(files):3d} documents")
    print(f"  {'TOTAL':15s}: {total:3d} documents")
    print("=" * 60)

    return results


if __name__ == "__main__":
    fetch_all()
