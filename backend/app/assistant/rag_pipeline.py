"""
BoxOfficeIQ - RAG Knowledge Retrieval Pipeline
Indexes project markdown documentation from docs/ directory into a vector index (FAISS + SentenceTransformers)
with an integrated TF-IDF fallback when vector libraries or GPU/pre-trained weights are not present.
"""

from __future__ import annotations

import os
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
DOCS_DIR = ROOT_DIR / "docs"
INDEX_DIR = ROOT_DIR / "ml" / "models" / "faiss_index"


class DocumentChunk:
    def __init__(self, content: str, source: str, chunk_id: int):
        self.content = content
        self.source = source
        self.chunk_id = chunk_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "source": self.source,
            "chunk_id": self.chunk_id,
        }


class RAGPipeline:
    """
    RAG Pipeline that splits docs/ into 500-char chunks (50 overlap),
    creates embeddings via sentence-transformers/all-MiniLM-L6-v2, and indexes with FAISS.
    Includes TF-IDF fallback for environments without faiss-cpu installed.
    """

    def __init__(
        self,
        docs_dir: Optional[Path] = None,
        index_dir: Optional[Path] = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        self.docs_dir = Path(docs_dir) if docs_dir else DOCS_DIR
        self.index_dir = Path(index_dir) if index_dir else INDEX_DIR
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.chunks: List[DocumentChunk] = []
        self._faiss_index = None
        self._embedder = None
        self._use_vector = False

        self._load_and_chunk_documents()
        self._initialize_index()

    def _load_and_chunk_documents(self):
        """Loads all markdown files in docs/ and creates overlapping chunks."""
        if not self.docs_dir.exists():
            logger.warning(f"Docs directory not found at {self.docs_dir}")
            return

        chunk_counter = 0
        for md_file in sorted(self.docs_dir.glob("*.md")):
            try:
                text = md_file.read_text(encoding="utf-8")
                # Clean multiple empty lines
                text = re.sub(r"\n{3,}", "\n\n", text).strip()
                if not text:
                    continue

                # Sliding window chunking
                start = 0
                while start < len(text):
                    end = min(start + self.chunk_size, len(text))
                    chunk_text = text[start:end].strip()
                    if chunk_text:
                        self.chunks.append(
                            DocumentChunk(
                                content=chunk_text,
                                source=md_file.name,
                                chunk_id=chunk_counter,
                            )
                        )
                        chunk_counter += 1
                    if end == len(text):
                        break
                    start += self.chunk_size - self.chunk_overlap
            except Exception as e:
                logger.error(f"Error processing doc {md_file.name}: {e}")

        logger.info(f"Loaded {len(self.chunks)} document chunks from {self.docs_dir}")

    def _initialize_index(self):
        """Attempts to build or load FAISS index using SentenceTransformers."""
        if not self.chunks:
            return

        try:
            from sentence_transformers import SentenceTransformer
            import faiss
            import numpy as np

            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            self._embedder = SentenceTransformer(model_name)
            dim = self._embedder.get_sentence_embedding_dimension()

            index_file = self.index_dir / "index.faiss"
            meta_file = self.index_dir / "metadata.json"

            if index_file.exists() and meta_file.exists():
                try:
                    self._faiss_index = faiss.read_index(str(index_file))
                    self._use_vector = True
                    logger.info("Loaded pre-existing FAISS vector index.")
                    return
                except Exception as e:
                    logger.warning(f"Could not read existing FAISS index: {e}. Rebuilding...")

            # Build FAISS index
            texts = [c.content for c in self.chunks]
            embeddings = self._embedder.encode(texts, show_progress_bar=False, normalize_embeddings=True)
            embeddings = np.array(embeddings, dtype=np.float32)

            index = faiss.IndexFlatIP(dim)
            index.add(embeddings)
            self._faiss_index = index
            self._use_vector = True

            # Save to disk
            self.index_dir.mkdir(parents=True, exist_ok=True)
            faiss.write_index(index, str(index_file))
            with open(meta_file, "w", encoding="utf-8") as f:
                json.dump([c.to_dict() for c in self.chunks], f, indent=2)

            logger.info(f"Successfully built and saved FAISS index to {self.index_dir}")
        except Exception as e:
            logger.info(f"Vector search indexing skipped ({e}). Falling back to TF-IDF lexical search.")
            self._use_vector = False

    def retrieve(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieves top-k most relevant document chunks for the query.
        Returns list of dicts with content, source, and score.
        """
        if not self.chunks:
            return []

        if self._use_vector and self._faiss_index is not None and self._embedder is not None:
            try:
                import numpy as np

                q_emb = self._embedder.encode([query], normalize_embeddings=True)
                q_emb = np.array(q_emb, dtype=np.float32)
                scores, indices = self._faiss_index.search(q_emb, k)

                results = []
                for score, idx in zip(scores[0], indices[0]):
                    if idx >= 0 and idx < len(self.chunks):
                        c = self.chunks[idx]
                        results.append({
                            "content": c.content,
                            "source": c.source,
                            "chunk_id": c.chunk_id,
                            "score": float(score),
                        })
                if results:
                    return results
            except Exception as e:
                logger.warning(f"Vector search query failed ({e}), using lexical fallback.")

        # Lexical keyword / token overlap fallback
        q_tokens = set(re.findall(r"\w+", query.lower()))
        scored_chunks: List[Tuple[float, DocumentChunk]] = []

        for c in self.chunks:
            c_tokens = set(re.findall(r"\w+", c.content.lower()))
            overlap = len(q_tokens.intersection(c_tokens))
            if overlap > 0:
                score = overlap / (len(q_tokens) + 0.1)
                scored_chunks.append((score, c))

        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        top_k = scored_chunks[:k]

        return [
            {
                "content": c.content,
                "source": c.source,
                "chunk_id": c.chunk_id,
                "score": float(score),
            }
            for score, c in top_k
        ]
