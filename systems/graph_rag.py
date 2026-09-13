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
        self.output_dir = self.workspace_dir / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)
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

        # Set environment variable for API key
        os.environ["GRAPHRAG_API_KEY"] = GEMINI_OPENAI_API_KEY

        # Run graphrag init if settings don't exist yet
        settings_path = self.workspace_dir / "settings.yaml"
        if not settings_path.exists():
            try:
                subprocess.run(
                    [sys.executable, "-m", "graphrag", "init", "--root", str(self.workspace_dir)],
                    cwd=str(self.workspace_dir),
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
            except Exception:
                pass

        # Write our custom settings.yaml configured for Gemini Free/Tier 1
        settings = {
            "concurrent_requests": 10,
            "async_mode": "threaded",
            "completion_models": {
                "default_completion_model": {
                    "model_provider": "openai",
                    "type": "litellm",
                    "model": GEMINI_MODEL,
                    "api_base": GEMINI_OPENAI_BASE_URL,
                    "api_key": GEMINI_OPENAI_API_KEY,
                    "api_version": None,
                    "max_tokens": 4096,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "request_timeout": 180,
                }
            },
            "embedding_models": {
                "default_embedding_model": {
                    "model_provider": "openai",
                    "type": "litellm",
                    "model": "gemini-embedding-001",
                    "api_base": GEMINI_OPENAI_BASE_URL,
                    "api_key": GEMINI_OPENAI_API_KEY,
                }
            },
            "input": {
                "type": "text",
            },
            "input_storage": {
                "type": "file",
                "base_dir": "input",
            },
            "output_storage": {
                "type": "file",
                "base_dir": "output",
            },
            "chunks": {
                "size": 1200,
                "overlap": 100,
            },
            "cache": {
                "type": "json",
                "base_dir": "cache",
            },
            "reporting": {
                "type": "file",
                "base_dir": "output",
            },
        }

        import yaml
        with open(settings_path, "w") as f:
            yaml.dump(settings, f, default_flow_style=False)

        print("  [GraphRAG] Workspace settings configured with gemini-embedding-001 & concurrency 10")
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
                [sys.executable, "-m", "graphrag", "index", "--root", str(self.workspace_dir)],
                cwd=str(self.workspace_dir),
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
                env={**os.environ, "GRAPHRAG_API_KEY": GEMINI_OPENAI_API_KEY},
            )
            if result.returncode == 0:
                print("  [GraphRAG] Indexing complete ✓")
            else:
                print(f"  [GraphRAG] Indexing error: {result.stderr[:500]}")
        except subprocess.TimeoutExpired:
            print("  [GraphRAG] ⚠ Indexing timed out")
        except Exception as e:
            print(f"  [GraphRAG] ⚠ Indexing failed: {e}")

    def query(self, question: str, method: str = "local") -> dict:
        """
        Query GraphRAG using official CLI.
        method: 'local' for entity-centric retrieval, 'global' for theme-level synthesis.
        Returns: { answer, contexts, latency_ms }
        """
        start_time = time.time()

        strict_question = (
            f"{question}\n\n[STRICT INSTRUCTION: Answer based ONLY on the retrieved graph context. "
            "Do NOT use any pre-trained external knowledge. If the context does not contain enough information, "
            "state: 'The provided context does not contain sufficient information to answer this question.']"
        )

        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "graphrag",
                    "query",
                    "--root",
                    str(self.workspace_dir),
                    "--method",
                    method,
                    strict_question,
                ],
                cwd=str(self.workspace_dir),
                capture_output=True,
                text=True,
                timeout=180,
                env={**os.environ, "GRAPHRAG_API_KEY": GEMINI_OPENAI_API_KEY},
            )
            latency_ms = (time.time() - start_time) * 1000

            if result.returncode == 0 and result.stdout.strip():
                stdout = result.stdout.strip()
                answer = stdout
                for prefix in [
                    "SUCCESS: Global Search Response:\n",
                    "SUCCESS: Local Search Response:\n",
                    "SUCCESS:",
                ]:
                    if answer.startswith(prefix):
                        answer = answer[len(prefix):].strip()

                contexts = self._extract_retrieved_contexts()
                return {
                    "answer": answer,
                    "contexts": contexts,
                    "latency_ms": round(latency_ms, 1),
                }
            else:
                err_msg = result.stderr.strip() or result.stdout.strip() or "CLI query failed"
                print(f"  [GraphRAG] Query error: {err_msg[:200]}")
                return {
                    "answer": f"[GraphRAG Error: {err_msg[:300]}]",
                    "contexts": [],
                    "latency_ms": round(latency_ms, 1),
                }
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            print(f"  [GraphRAG] Query error: {e}")
            return {
                "answer": f"[GraphRAG Error: {e}]",
                "contexts": [],
                "latency_ms": round(latency_ms, 1),
            }

    def _extract_retrieved_contexts(self) -> list[str]:
        """Extract summaries or text units from GraphRAG parquet tables."""
        contexts = []
        try:
            import pandas as pd
            reports_path = self.output_dir / "community_reports.parquet"
            if not reports_path.exists():
                reports_path = self.output_dir / "create_final_community_reports.parquet"
            text_units_path = self.output_dir / "text_units.parquet"
            if not text_units_path.exists():
                text_units_path = self.output_dir / "create_final_text_units.parquet"

            if reports_path.exists():
                df = pd.read_parquet(reports_path)
                if "summary" in df.columns:
                    contexts.extend(df["summary"].dropna().tolist()[:5])
                elif "full_content" in df.columns:
                    contexts.extend(df["full_content"].dropna().tolist()[:5])
            if not contexts and text_units_path.exists():
                df = pd.read_parquet(text_units_path)
                if "text" in df.columns:
                    contexts.extend(df["text"].dropna().tolist()[:5])
        except Exception as e:
            print(f"  [GraphRAG] Context extraction note: {e}")
        return contexts if contexts else ["[GraphRAG community reports]"]

