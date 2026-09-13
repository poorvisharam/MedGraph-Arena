"""
NodeRAG — Wrapper around NodeRAG (heterogeneous graph nodes).
State-of-the-art multi-hop reasoning using concepts, relations, and semantic units.
"""

import time
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    NODERAG_INDEX_DIR,
    GEMINI_API_KEY,
    GEMINI_MODEL,
)


class NodeRAGSystem:
    """
    Wrapper around NodeRAG for heterogeneous graph-based retrieval.
    Traverses concept nodes, relation nodes, and semantic unit text chunks.
    """

    def __init__(
        self,
        cache_dir: str | Path = NODERAG_INDEX_DIR,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.noderag = None
        self.noderag_config = None
        self._init_noderag()

    def _init_noderag(self):
        """Initialize NodeRAG instance with Gemini Free Tier / Flash configuration."""
        try:
            from NodeRAG import NodeConfig, NodeRag

            config_dict = {
                "config": {
                    "main_folder": str(self.cache_dir),
                    "language": "English",
                    "chunk_size": 1200,
                    "dim": 3072,
                },
                "model_config": {
                    "model_name": GEMINI_MODEL,
                    "api_keys": GEMINI_API_KEY,
                    "service_provider": "gemini",
                },
                "embedding_config": {
                    "embedding_model_name": "gemini-embedding-001",
                    "api_keys": GEMINI_API_KEY,
                    "service_provider": "gemini_embedding",
                },
            }
            self.noderag_config = NodeConfig(config_dict)
            self.noderag = NodeRag(self.noderag_config, web_ui=False)
            print("  [NodeRAG] Initialized successfully")
        except Exception as e:
            print(f"  [NodeRAG] ⚠ Init error: {e}")
            self.noderag = None

    def ingest(self, documents: list[str], metadatas: list[dict] | None = None):
        """
        Index documents into NodeRAG's heterogeneous graph.
        """
        print(f"  [NodeRAG] Ingesting {len(documents)} documents...")

        if self.noderag is None:
            raise RuntimeError("NodeRAG is not initialized properly.")

        input_dir = self.cache_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)

        for i, doc in enumerate(documents):
            filepath = input_dir / f"doc_{i:03d}.txt"
            filepath.write_text(doc, encoding="utf-8")

        self.noderag.run()
        print("  [NodeRAG] Indexing complete ✓")

    def query(self, question: str) -> dict:
        """
        Query NodeRAG using heterogeneous graph traversal.
        Returns: { answer, contexts, latency_ms }
        """
        start_time = time.time()

        if self.noderag_config is None:
            return {
                "answer": "[NodeRAG Error: System not configured]",
                "contexts": [],
                "latency_ms": round((time.time() - start_time) * 1000, 1),
            }

        try:
            from NodeRAG import NodeSearch

            strict_question = (
                f"{question}\n\n[STRICT INSTRUCTION: Answer based ONLY on the retrieved graph context. "
                "Do NOT use any pre-trained external knowledge or clinical assumptions. If the context does not contain "
                "enough information, state: 'The provided context does not contain sufficient information to answer this question.']"
            )
            searcher = NodeSearch(self.noderag_config)
            retrieval = searcher.search(question)
            answer = searcher.answer(strict_question)

            contexts = []
            if hasattr(retrieval, "retrieved_list") and retrieval.retrieved_list:
                contexts = [
                    item[0] if isinstance(item, (tuple, list)) else str(item)
                    for item in retrieval.retrieved_list
                ]
            elif hasattr(retrieval, "unstructured_prompt") and retrieval.unstructured_prompt:
                contexts = [retrieval.unstructured_prompt]

            latency_ms = (time.time() - start_time) * 1000
            return {
                "answer": str(answer).strip(),
                "contexts": contexts[:10],
                "latency_ms": round(latency_ms, 1),
            }
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            print(f"  [NodeRAG] Query error: {e}")
            return {
                "answer": f"[NodeRAG Error: {e}]",
                "contexts": [],
                "latency_ms": round(latency_ms, 1),
            }
