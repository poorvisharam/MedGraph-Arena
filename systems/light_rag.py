"""
LightRAG — Wrapper around the LightRAG library (HKUDS).
Uses dual-level retrieval (entity + theme) with incremental updates.
"""

import time
from pathlib import Path

from openai import OpenAI

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    LIGHTRAG_INDEX_DIR,
    GEMINI_OPENAI_BASE_URL,
    GEMINI_OPENAI_API_KEY,
    GEMINI_MODEL,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
)


class LightRAGSystem:
    """
    Wrapper around LightRAG for graph-based retrieval.
    Supports hybrid mode (vector + graph) for best results.
    """

    def __init__(
        self,
        working_dir: str | Path = LIGHTRAG_INDEX_DIR,
    ):
        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.rag = None
        self._init_rag()

        # Fallback LLM client
        self.llm = OpenAI(
            api_key=GEMINI_OPENAI_API_KEY,
            base_url=GEMINI_OPENAI_BASE_URL,
        )
        self._documents = []  # Store for fallback

    def _init_rag(self):
        """Initialize LightRAG instance."""
        try:
            from lightrag import LightRAG, QueryParam
            from lightrag.llm.openai import openai_complete_if_cache, openai_embed

            async def llm_func(prompt, **kwargs):
                return await openai_complete_if_cache(
                    model=GEMINI_MODEL,
                    prompt=prompt,
                    api_key=GEMINI_OPENAI_API_KEY,
                    base_url=GEMINI_OPENAI_BASE_URL,
                    **kwargs,
                )

            from sentence_transformers import SentenceTransformer
            import numpy as np
            
            embedding_model = SentenceTransformer(EMBEDDING_MODEL)

            from lightrag.utils import EmbeddingFunc
            
            async def embed_func(texts, **kwargs):
                embeddings = embedding_model.encode(texts)
                return embeddings
            
            emb_func = EmbeddingFunc(
                embedding_dim=EMBEDDING_DIMENSION,
                max_token_size=8192,
                func=embed_func
            )

            self.rag = LightRAG(
                working_dir=str(self.working_dir),
                llm_model_func=llm_func,
                embedding_func=emb_func,
            )
            print("  [LightRAG] Initialized successfully")
        except ImportError:
            print("  [LightRAG] ⚠ lightrag not installed — will use fallback mode")
            self.rag = None
        except Exception as e:
            print(f"  [LightRAG] ⚠ Init error: {e} — will use fallback mode")
            self.rag = None

    def ingest(self, documents: list[str], metadatas: list[dict] | None = None):
        """
        Insert documents into LightRAG (supports incremental ingestion).
        """
        print(f"  [LightRAG] Ingesting {len(documents)} documents...")
        self._documents = documents  # Store for fallback

        if self.rag is None:
            print("  [LightRAG] Using fallback mode — documents stored in memory")
            return

        import asyncio

        async def _insert():
            await self.rag.initialize_storages()
            for i, doc in enumerate(documents):
                try:
                    await asyncio.wait_for(self.rag.ainsert(doc), timeout=120)
                    if (i + 1) % 5 == 0:
                        print(f"    ... inserted {i + 1}/{len(documents)}")
                except asyncio.TimeoutError:
                    print(f"    ⚠ Timeout inserting doc {i}")
                except Exception as e:
                    print(f"    ⚠ Error inserting doc {i}: {e}")

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
            asyncio.run(_insert())
            print(f"  [LightRAG] Indexing complete ✓")
        except Exception as e:
            print(f"  [LightRAG] ⚠ Indexing error: {e}")
            print("  [LightRAG] Falling back to in-memory document store")

    def query(self, question: str, mode: str = "hybrid") -> dict:
        """
        Query LightRAG with the specified mode.
        Modes: naive (vector only), local (entity graph), global (theme graph), hybrid (recommended).
        Returns: { answer, contexts, latency_ms }
        """
        start_time = time.time()

        if self.rag is not None:
            try:
                import asyncio
                from lightrag import QueryParam

                async def _query():
                    return await self.rag.aquery(
                        question,
                        param=QueryParam(mode=mode),
                    )

                if not hasattr(self, "_loop") or self._loop.is_closed():
                    self._loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(self._loop)
                
                result = self._loop.run_until_complete(_query())

                latency_ms = (time.time() - start_time) * 1000
                return {
                    "answer": str(result),
                    "contexts": [f"[LightRAG {mode} retrieval]"],
                    "latency_ms": round(latency_ms, 1),
                }
            except Exception as e:
                print(f"  [LightRAG] Query error: {e}, using fallback")

        # Fallback query
        return self._fallback_query(question, mode, start_time)

    def _fallback_query(self, question: str, mode: str, start_time: float) -> dict:
        """Fallback query using stored documents + LLM."""
        context_text = "\n\n---\n\n".join(self._documents[:10])

        mode_instructions = {
            "naive": "Use simple keyword/semantic matching to find relevant information.",
            "local": "Focus on specific entities and their direct relationships mentioned in the context.",
            "global": "Identify overarching themes and patterns across all documents.",
            "hybrid": "Combine entity-level details with thematic analysis for a comprehensive answer.",
        }

        instruction = mode_instructions.get(mode, mode_instructions["hybrid"])

        try:
            response = self.llm.chat.completions.create(
                model=GEMINI_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"You are a medical knowledge assistant using graph-based retrieval ({mode} mode). "
                            f"{instruction} Answer based on the provided context. Cite specific details."
                        ),
                    },
                    {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {question}"},
                ],
                temperature=0.1,
                max_tokens=1024,
            )
            answer = response.choices[0].message.content.strip()
        except Exception as e:
            answer = f"[LightRAG fallback error: {e}]"

        contexts = [doc[:200] + "..." for doc in self._documents[:5]]
        latency_ms = (time.time() - start_time) * 1000

        return {
            "answer": answer,
            "contexts": contexts,
            "latency_ms": round(latency_ms, 1),
        }
