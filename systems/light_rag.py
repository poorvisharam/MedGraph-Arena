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

        if self.rag is None:
            return {
                "answer": "[LightRAG Error: System not initialized]",
                "contexts": [],
                "latency_ms": round((time.time() - start_time) * 1000, 1),
            }

        try:
            import asyncio
            from lightrag import QueryParam

            if not hasattr(self, "_loop") or self._loop.is_closed():
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)

            strict_prompt = (
                "You are a strict medical evaluation assistant. Answer the question based ONLY and EXCLUSIVELY on the provided context. "
                "Do NOT use any pre-trained external knowledge, clinical assumptions, or unstated facts. "
                "If the context does not contain enough information to answer the question, you MUST explicitly state: "
                "'The provided context does not contain sufficient information to answer this question.' "
                "Do NOT extrapolate or infer beyond what is directly stated."
            )

            async def _run_query():
                await self.rag.initialize_storages()
                # 1. Retrieve the actual context from the knowledge graph
                raw_context = await self.rag.aquery(
                    question,
                    param=QueryParam(mode=mode, only_need_context=True),
                )
                # 2. Generate the grounded answer
                answer = await self.rag.aquery(
                    question,
                    param=QueryParam(mode=mode, user_prompt=strict_prompt),
                )
                return str(answer), str(raw_context)

            answer, raw_context = self._loop.run_until_complete(_run_query())

            # Split context text into readable sections/chunks for evaluation
            context_blocks = [
                block.strip() for block in raw_context.split("\n\n") if block.strip()
            ]
            if not context_blocks:
                context_blocks = [raw_context]

            latency_ms = (time.time() - start_time) * 1000
            return {
                "answer": answer.strip(),
                "contexts": context_blocks[:10],
                "latency_ms": round(latency_ms, 1),
            }
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            print(f"  [LightRAG] Query error: {e}")
            return {
                "answer": f"[LightRAG Error: {e}]",
                "contexts": [],
                "latency_ms": round(latency_ms, 1),
            }

