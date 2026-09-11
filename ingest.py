"""
Unified Ingestion — Index documents into ALL 4 RAG systems.
"""

import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_DIR, SYSTEM_NAMES, SYSTEM_DISPLAY_NAMES

from systems.naive_rag import NaiveRAG
from systems.graph_rag import GraphRAGSystem
from systems.light_rag import LightRAGSystem
from systems.node_rag import NodeRAGSystem


def load_documents(data_dir: Path = DATA_DIR) -> tuple[list[str], list[dict]]:
    """
    Load all documents from data/documents/ directory.
    Returns: (list of document texts, list of metadata dicts)
    """
    documents = []
    metadatas = []

    for subdir in sorted(data_dir.iterdir()):
        if not subdir.is_dir():
            continue
        source_name = subdir.name
        for filepath in sorted(subdir.glob("*.txt")):
            text = filepath.read_text(encoding="utf-8")
            documents.append(text)
            metadatas.append({
                "source": source_name,
                "filename": filepath.name,
                "filepath": str(filepath),
            })

    return documents, metadatas


def ingest_all(systems: dict | None = None):
    """
    Ingest documents into all provided systems.
    If systems dict is None, initializes all 4 systems.
    """
    print("=" * 60)
    print("MedGraph Arena — Document Ingestion")
    print("=" * 60)

    # Load documents
    print("\n📄 Loading documents...")
    documents, metadatas = load_documents()
    print(f"  Loaded {len(documents)} documents from {len(set(m['source'] for m in metadatas))} sources")

    if not documents:
        print("  ⚠ No documents found! Run `python data/fetch_all.py` first.")
        return {}

    # Initialize systems if not provided
    if systems is None:
        print("\n🔧 Initializing RAG systems...")
        systems = {
            "naive_rag": NaiveRAG(),
            "graphrag": GraphRAGSystem(),
            "lightrag": LightRAGSystem(),
            "noderag": NodeRAGSystem(),
        }

    # Ingest into each system
    timings = {}
    for name, system in systems.items():
        display_name = SYSTEM_DISPLAY_NAMES.get(name, name)
        print(f"\n📥 [{display_name}] Starting ingestion...")
        start = time.time()

        try:
            system.ingest(documents, metadatas)
        except Exception as e:
            print(f"  ⚠ [{display_name}] Ingestion failed: {e}")

        elapsed = time.time() - start
        timings[name] = round(elapsed, 1)
        print(f"  ⏱ [{display_name}] Completed in {elapsed:.1f}s")

    # Summary
    print("\n" + "=" * 60)
    print("📊 Ingestion Summary")
    print("=" * 60)
    for name, elapsed in timings.items():
        display_name = SYSTEM_DISPLAY_NAMES.get(name, name)
        print(f"  {display_name:15s}: {elapsed:8.1f}s")
    print("=" * 60)

    return systems


if __name__ == "__main__":
    ingest_all()
