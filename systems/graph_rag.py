"""
Microsoft GraphRAG — Wrapper around the graphrag CLI.
Uses Leiden community detection + hierarchical summarization.
"""

import json
import os
import subprocess
import time
from pathlib import Path

from openai import OpenAI

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    GRAPHRAG_INDEX_DIR,
    GEMINI_OPENAI_BASE_URL,
    GEMINI_OPENAI_API_KEY,
    GEMINI_MODEL,
    EMBEDDING_MODEL,
)


class GraphRAGSystem:
    """
    Wrapper around Microsoft's GraphRAG CLI.
    Handles initialization, indexing, and querying.
    """

    def __init__(
        self,
        workspace_dir: str | Path = GRAPHRAG_INDEX_DIR,
    ):
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.input_dir = self.workspace_dir / "input"
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self._initialized = False

        # LLM client for fallback querying
        self.llm = OpenAI(
            api_key=GEMINI_OPENAI_API_KEY,
            base_url=GEMINI_OPENAI_BASE_URL,
        )

    def _init_workspace(self):
        """Initialize GraphRAG workspace with settings."""
        if self._initialized:
            return

        # Create settings.yaml for GraphRAG
        settings = {
            "llm": {
                "api_key": "${GRAPHRAG_API_KEY}",
                "type": "openai_chat",
                "model": GEMINI_MODEL,
                "api_base": GEMINI_OPENAI_BASE_URL,
                "api_version": None,
                "max_tokens": 4096,
                "temperature": 0.0,
                "top_p": 1.0,
                "request_timeout": 120,
            },
            "embeddings": {
                "llm": {
                    "api_key": "${GRAPHRAG_API_KEY}",
                    "type": "openai_embedding",
                    "model": "text-embedding-004",
                    "api_base": GEMINI_OPENAI_BASE_URL,
                }
            },
            "input": {
                "type": "file",
                "file_type": "text",
                "base_dir": "input",
            },
            "chunks": {
                "size": 1200,
                "overlap": 100,
            },
            "cache": {
                "type": "file",
                "base_dir": "cache",
            },
            "reporting": {
                "type": "file",
                "base_dir": "output",
            },
            "storage": {
                "type": "file",
                "base_dir": "output",
            },
        }

        import yaml
        settings_path = self.workspace_dir / "settings.yaml"
        with open(settings_path, "w") as f:
            yaml.dump(settings, f, default_flow_style=False)

        # Set environment variable for API key
        os.environ["GRAPHRAG_API_KEY"] = GEMINI_OPENAI_API_KEY

        # Run graphrag init
        try:
            result = subprocess.run(
                ["graphrag", "init"],
                cwd=str(self.workspace_dir),
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode == 0:
                print("  [GraphRAG] Workspace initialized")
            else:
                print(f"  [GraphRAG] Init warning: {result.stderr[:200]}")
        except FileNotFoundError:
            print("  [GraphRAG] ⚠ graphrag CLI not found — will use API-based fallback")
        except Exception as e:
            print(f"  [GraphRAG] ⚠ Init error: {e}")

        self._initialized = True

    def ingest(self, documents: list[str], metadatas: list[dict] | None = None):
        """
        Copy documents into GraphRAG input directory and run indexing.
        """
        print(f"  [GraphRAG] Ingesting {len(documents)} documents...")
        self._init_workspace()

        # Write documents to input directory
        for i, doc in enumerate(documents):
            source = ""
            if metadatas and i < len(metadatas):
                source = metadatas[i].get("source", "")
            filepath = self.input_dir / f"doc_{i:03d}_{source}.txt"
            filepath.write_text(doc, encoding="utf-8")

        # Run graphrag index
        print("  [GraphRAG] Running indexing (this may take a while)...")
        try:
            result = subprocess.run(
                ["graphrag", "index"],
                cwd=str(self.workspace_dir),
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
                env={**os.environ, "GRAPHRAG_API_KEY": GEMINI_OPENAI_API_KEY},
            )
            if result.returncode == 0:
                print("  [GraphRAG] Indexing complete ✓")
            else:
                print(f"  [GraphRAG] Indexing error: {result.stderr[:500]}")
                print("  [GraphRAG] Will use LLM-based fallback for queries")
        except FileNotFoundError:
            print("  [GraphRAG] ⚠ graphrag CLI not available — storing docs for fallback mode")
        except subprocess.TimeoutExpired:
            print("  [GraphRAG] ⚠ Indexing timed out — will use fallback mode")
        except Exception as e:
            print(f"  [GraphRAG] ⚠ Indexing failed: {e}")

    def query(self, question: str, method: str = "global") -> dict:
        """
        Query GraphRAG using CLI or fallback to direct LLM query with stored documents.
        method: 'global' for theme-level synthesis, 'local' for entity-centric retrieval.
        Returns: { answer, contexts, latency_ms }
        """
        start_time = time.time()

        # Try CLI query first
        try:
            result = subprocess.run(
                ["graphrag", "query", "--method", method, "--query", question],
                cwd=str(self.workspace_dir),
                capture_output=True,
                text=True,
                timeout=120,
                env={**os.environ, "GRAPHRAG_API_KEY": GEMINI_OPENAI_API_KEY},
            )
            if result.returncode == 0 and result.stdout.strip():
                answer = result.stdout.strip()
                latency_ms = (time.time() - start_time) * 1000
                return {
                    "answer": answer,
                    "contexts": [f"[GraphRAG {method} search]"],
                    "latency_ms": round(latency_ms, 1),
                }
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
            pass

        # Fallback: Use stored documents with LLM
        return self._fallback_query(question, method, start_time)

    def _fallback_query(self, question: str, method: str, start_time: float) -> dict:
        """
        Fallback query using stored documents + LLM when CLI is unavailable.
        Simulates GraphRAG's global vs local approach via prompt engineering.
        """
        # Read all documents from input directory
        docs = []
        if self.input_dir.exists():
            for f in sorted(self.input_dir.glob("*.txt")):
                docs.append(f.read_text(encoding="utf-8"))

        context_text = "\n\n---\n\n".join(docs[:10])  # Limit context

        if method == "global":
            system_prompt = (
                "You are a medical knowledge assistant performing GLOBAL analysis. "
                "Synthesize themes, patterns, and overarching insights across ALL provided documents. "
                "Focus on high-level connections between topics, common themes, and cross-document relationships. "
                "Provide a comprehensive synthesis, not just individual facts."
            )
        else:
            system_prompt = (
                "You are a medical knowledge assistant performing LOCAL entity-focused retrieval. "
                "Find specific entities, facts, and relationships mentioned in the context. "
                "Be precise and cite specific details. Focus on the exact entities asked about."
            )

        try:
            response = self.llm.chat.completions.create(
                model=GEMINI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {question}"},
                ],
                temperature=0.1,
                max_tokens=1024,
            )
            answer = response.choices[0].message.content.strip()
        except Exception as e:
            answer = f"[GraphRAG fallback error: {e}]"

        contexts = [doc[:200] + "..." for doc in docs[:5]]
        latency_ms = (time.time() - start_time) * 1000

        return {
            "answer": answer,
            "contexts": contexts,
            "latency_ms": round(latency_ms, 1),
        }
