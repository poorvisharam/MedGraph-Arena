"""
NodeRAG — Wrapper around NodeRAG (heterogeneous graph nodes).
State-of-the-art multi-hop reasoning using concepts, relations, and semantic units.
"""

import time
from pathlib import Path

from openai import OpenAI

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    NODERAG_INDEX_DIR,
    GEMINI_OPENAI_BASE_URL,
    GEMINI_OPENAI_API_KEY,
    GEMINI_MODEL,
    EMBEDDING_MODEL,
)


class NodeRAGSystem:
    """
    Wrapper around NodeRAG for heterogeneous graph-based retrieval.
    Uses concept nodes, relation nodes, and semantic unit nodes.
    """

    def __init__(
        self,
        cache_dir: str | Path = NODERAG_INDEX_DIR,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.noderag = None
        self._init_noderag()

        # Fallback LLM
        self.llm = OpenAI(
            api_key=GEMINI_OPENAI_API_KEY,
            base_url=GEMINI_OPENAI_BASE_URL,
        )
        self._documents = []

    def _init_noderag(self):
        """Initialize NodeRAG instance."""
        try:
            from NodeRAG import NodeConfig, NodeRag

            config_dict = {
                'config': {
                    'main_folder': str(self.cache_dir),
                    'language': 'English',
                    'chunk_size': 1200
                },
                'model_config': {
                    'model_name': GEMINI_MODEL,
                    'api_keys': GEMINI_OPENAI_API_KEY,
                    'base_url': GEMINI_OPENAI_BASE_URL,
                    'service_provider': 'openai'
                },
                'embedding_config': {
                    'embedding_model_name': 'text-embedding-004',
                    'api_keys': GEMINI_OPENAI_API_KEY,
                    'base_url': GEMINI_OPENAI_BASE_URL,
                    'service_provider': 'gemini_embedding'
                }
            }
            self.noderag_config = NodeConfig(config_dict)
            self.noderag = NodeRag(self.noderag_config, web_ui=True)
            print("  [NodeRAG] Initialized successfully")
        except ImportError:
            print("  [NodeRAG] ⚠ NodeRAG not installed — will use fallback mode")
            self.noderag = None
        except Exception as e:
            print(f"  [NodeRAG] ⚠ Init error: {e} — will use fallback mode")
            self.noderag = None

    def ingest(self, documents: list[str], metadatas: list[dict] | None = None):
        """
        Index documents into NodeRAG's heterogeneous graph.
        """
        print(f"  [NodeRAG] Ingesting {len(documents)} documents...")
        self._documents = documents

        if self.noderag is None:
            print("  [NodeRAG] Using fallback mode — documents stored in memory")
            return

        try:
            # NodeRAG expects documents to be written to files
            input_dir = self.cache_dir / "input"
            input_dir.mkdir(parents=True, exist_ok=True)

            for i, doc in enumerate(documents):
                filepath = input_dir / f"doc_{i:03d}.txt"
                filepath.write_text(doc, encoding="utf-8")

            self.noderag.run()
            print(f"  [NodeRAG] Indexing complete ✓")
        except Exception as e:
            print(f"  [NodeRAG] ⚠ Indexing error: {e}")
            print("  [NodeRAG] Falling back to in-memory store")

    def query(self, question: str) -> dict:
        """
        Query NodeRAG using heterogeneous graph traversal.
        Returns: { answer, contexts, latency_ms }
        """
        start_time = time.time()

        if self.noderag is not None:
            try:
                from NodeRAG import NodeSearch
                searcher = NodeSearch(self.noderag_config)
                result = searcher.answer(question)
                latency_ms = (time.time() - start_time) * 1000
                return {
                    "answer": str(result),
                    "contexts": ["[NodeRAG heterogeneous graph retrieval]"],
                    "latency_ms": round(latency_ms, 1),
                }
            except Exception as e:
                print(f"  [NodeRAG] Query error: {e}, using fallback")

        return self._fallback_query(question, start_time)

    def _fallback_query(self, question: str, start_time: float) -> dict:
        """
        Fallback: Simulates NodeRAG's multi-hop reasoning via prompt engineering.
        NodeRAG's key innovation is decomposing queries into concept/relation/semantic traversals.
        """
        context_text = "\n\n---\n\n".join(self._documents[:10])

        try:
            response = self.llm.chat.completions.create(
                model=GEMINI_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a medical knowledge assistant using advanced multi-hop graph reasoning. "
                            "Think step-by-step through the knowledge graph:\n"
                            "1. CONCEPTS: Identify the key medical concepts in the question\n"
                            "2. RELATIONS: Trace relationships between concepts (drug→effect, condition→treatment, etc.)\n"
                            "3. SEMANTIC UNITS: Find coherent semantic units that bridge multiple concepts\n"
                            "4. SYNTHESIS: Combine multi-hop paths into a comprehensive answer\n\n"
                            "Show your reasoning path. Cite specific relationships you traversed."
                        ),
                    },
                    {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {question}"},
                ],
                temperature=0.1,
                max_tokens=1024,
            )
            answer = response.choices[0].message.content.strip()
        except Exception as e:
            answer = f"[NodeRAG fallback error: {e}]"

        contexts = [doc[:200] + "..." for doc in self._documents[:5]]
        latency_ms = (time.time() - start_time) * 1000

        return {
            "answer": answer,
            "contexts": contexts,
            "latency_ms": round(latency_ms, 1),
        }
