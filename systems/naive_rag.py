"""
Naive RAG — ChromaDB vector search + LLM baseline.
The simplest RAG architecture: chunk → embed → retrieve → synthesize.
"""

import time
from pathlib import Path

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from openai import OpenAI

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    NAIVE_INDEX_DIR,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K,
    GEMINI_OPENAI_BASE_URL,
    GEMINI_OPENAI_API_KEY,
    GEMINI_MODEL,
)


class NaiveRAG:
    """
    Baseline RAG system using ChromaDB for vector search
    and Gemini for answer synthesis.
    """

    def __init__(
        self,
        persist_dir: str | Path = NAIVE_INDEX_DIR,
        embedding_model: str = EMBEDDING_MODEL,
        collection_name: str = "medgraph_naive",
    ):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name

        # Load embedding model (local, free)
        print(f"  [NaiveRAG] Loading embedding model: {embedding_model}")
        self.embedder = SentenceTransformer(embedding_model)

        # ChromaDB client with persistence
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.persist_dir)
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # Text splitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        # LLM client (Gemini via OpenAI-compatible endpoint)
        self.llm = OpenAI(
            api_key=GEMINI_OPENAI_API_KEY,
            base_url=GEMINI_OPENAI_BASE_URL,
        )

    def ingest(self, documents: list[str], metadatas: list[dict] | None = None):
        """
        Chunk documents, embed, and store in ChromaDB.
        """
        print(f"  [NaiveRAG] Ingesting {len(documents)} documents...")
        all_chunks = []
        all_metas = []
        all_ids = []

        for i, doc in enumerate(documents):
            chunks = self.splitter.split_text(doc)
            for j, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                meta = {"doc_index": i, "chunk_index": j}
                if metadatas and i < len(metadatas):
                    meta.update(metadatas[i])
                all_metas.append(meta)
                all_ids.append(f"doc{i}_chunk{j}")

        # Embed all chunks
        print(f"  [NaiveRAG] Embedding {len(all_chunks)} chunks...")
        embeddings = self.embedder.encode(all_chunks, show_progress_bar=True)

        # Upsert into ChromaDB (in batches to handle large corpora)
        batch_size = 100
        for start in range(0, len(all_chunks), batch_size):
            end = min(start + batch_size, len(all_chunks))
            self.collection.upsert(
                ids=all_ids[start:end],
                documents=all_chunks[start:end],
                embeddings=embeddings[start:end].tolist(),
                metadatas=all_metas[start:end],
            )

        print(f"  [NaiveRAG] Indexed {len(all_chunks)} chunks into ChromaDB")

    def query(self, question: str, top_k: int = TOP_K) -> dict:
        """
        Retrieve top-k chunks and synthesize an answer.
        Returns: { answer, contexts, latency_ms }
        """
        start_time = time.time()

        # Embed the question
        q_embedding = self.embedder.encode([question])[0].tolist()

        # Retrieve top-k similar chunks
        results = self.collection.query(
            query_embeddings=[q_embedding],
            n_results=top_k,
        )

        contexts = results.get("documents", [[]])[0]
        context_text = "\n\n---\n\n".join(contexts)

        # Synthesize answer with LLM
        prompt = (
            "You are a strict medical evaluation assistant. Answer the question based ONLY and EXCLUSIVELY on the provided context. "
            "Do NOT use any pre-trained external knowledge, clinical assumptions, or unstated facts. "
            "If the context does not contain enough information to answer the question, you MUST explicitly state: "
            "'The provided context does not contain sufficient information to answer this question.' "
            "Do NOT extrapolate or infer beyond what is directly stated. Cite specific details from the context.\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

        try:
            response = self.llm.chat.completions.create(
                model=GEMINI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1024,
            )
            answer = response.choices[0].message.content.strip()
        except Exception as e:
            answer = f"[Error generating answer: {e}]"

        latency_ms = (time.time() - start_time) * 1000

        return {
            "answer": answer,
            "contexts": contexts,
            "latency_ms": round(latency_ms, 1),
        }
