import os
import re
import pickle
import numpy as np
import faiss
import fitz  # PyMuPDF
from dotenv import load_dotenv
from openai import OpenAI
from typing import List, Dict, Any, Optional

load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-4o-mini")
PDF_PATH = os.getenv("PDF_PATH", "ntp_en_annual_report_2025.pdf")
INDEX_PATH = "faiss_index.bin"
CHUNKS_PATH = "chunks_metadata.pkl"

class RAGEngine:
    def __init__(self, pdf_path: str = PDF_PATH):
        self.pdf_path = pdf_path
        self.client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
        self.chunks: List[Dict[str, Any]] = []
        self.index: Optional[faiss.IndexFlatIP] = None
        self.dimension: int = 1536
        self.is_initialized: bool = False

    @staticmethod
    def fix_ligatures(text: str) -> str:
        """Fixes PDF ligature issues and common artifacts."""
        return (
            text.replace("ﬀ", "ff")
            .replace("ﬁ", "fi")
            .replace("ﬂ", "fl")
            .replace("ﬃ", "ffi")
            .replace("ﬄ", "ffl")
            .replace("/f_", "f")
        )

    def extract_text_from_pdf(self) -> List[Dict[str, Any]]:
        """Extracts text page by page with ligature cleanup."""
        if not os.path.exists(self.pdf_path):
            raise FileNotFoundError(f"PDF file not found at: {self.pdf_path}")

        doc = fitz.open(self.pdf_path)
        pages_data = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            cleaned_text = " ".join(text.split())
            cleaned_text = self.fix_ligatures(cleaned_text)
            if cleaned_text.strip():
                pages_data.append({
                    "page_number": page_num + 1,
                    "text": cleaned_text
                })

        return pages_data

    def chunk_document(self, pages_data: List[Dict[str, Any]], max_chunk_chars: int = 650, overlap_chars: int = 100) -> List[Dict[str, Any]]:
        """
        Creates semantic chunks respecting sentence boundaries and page sources.
        """
        chunks = []
        chunk_id = 0

        for page in pages_data:
            page_num = page["page_number"]
            text = page["text"]
            words = text.split()

            current = []
            current_len = 0

            for w in words:
                current.append(w)
                current_len += len(w) + 1

                # If word ends with a period or chunk size exceeded
                if (w.endswith('.') and current_len >= 300) or current_len >= max_chunk_chars:
                    chunk_text = " ".join(current).strip()
                    if chunk_text:
                        chunks.append({
                            "id": chunk_id,
                            "page": page_num,
                            "text": chunk_text
                        })
                        chunk_id += 1

                    # Keep overlap words if any
                    overlap_words = []
                    overlap_len = 0
                    for rev_w in reversed(current):
                        if overlap_len + len(rev_w) < overlap_chars:
                            overlap_words.insert(0, rev_w)
                            overlap_len += len(rev_w) + 1
                        else:
                            break
                    current = overlap_words
                    current_len = sum(len(x) + 1 for x in current)

            if current:
                chunk_text = " ".join(current).strip()
                if chunk_text:
                    chunks.append({
                        "id": chunk_id,
                        "page": page_num,
                        "text": chunk_text
                    })
                    chunk_id += 1

        return chunks

    def get_embeddings(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """Computes normalized embeddings in batches using OpenAI."""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=batch
            )
            embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(embeddings)

        embeddings_np = np.array(all_embeddings, dtype=np.float32)
        # Normalize for cosine similarity via Inner Product (IndexFlatIP)
        faiss.normalize_L2(embeddings_np)
        return embeddings_np

    def build_or_load_index(self, force_rebuild: bool = False):
        """Builds FAISS index or loads from cached files."""
        if not force_rebuild and os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH):
            print(f"Loading cached FAISS index from {INDEX_PATH}...")
            self.index = faiss.read_index(INDEX_PATH)
            with open(CHUNKS_PATH, "rb") as f:
                self.chunks = pickle.load(f)
            self.dimension = self.index.d
            self.is_initialized = True
            print(f"Loaded {len(self.chunks)} chunks into index.")
            return

        print(f"Indexing PDF from scratch: {self.pdf_path}...")
        pages_data = self.extract_text_from_pdf()
        self.chunks = self.chunk_document(pages_data)
        print(f"Generated {len(self.chunks)} chunks from {len(pages_data)} pages.")

        texts = [chunk["text"] for chunk in self.chunks]
        embeddings = self.get_embeddings(texts)

        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)

        # Save to disk for fast startup
        faiss.write_index(self.index, INDEX_PATH)
        with open(CHUNKS_PATH, "wb") as f:
            pickle.dump(self.chunks, f)

        self.is_initialized = True
        print(f"FAISS index built and saved with {self.index.ntotal} vectors.")

    def retrieve(self, query: str, top_k: int = 3, api_key: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieves top_k relevant chunks with cosine similarity scores."""
        if not self.is_initialized or self.index is None:
            self.build_or_load_index()

        client = OpenAI(api_key=api_key) if api_key else self.client
        if not client or not client.api_key:
            raise ValueError("No OpenAI API key found. Please provide an API key in the sidebar or set OPENAI_API_KEY in .env")

        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[query]
        )
        query_emb = np.array([response.data[0].embedding], dtype=np.float32)
        faiss.normalize_L2(query_emb)

        scores, indices = self.index.search(query_emb, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.chunks):
                chunk = self.chunks[idx].copy()
                chunk["score"] = float(score)
                results.append(chunk)

        return results

    def generate_answer(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]] = None,
        top_k: int = 3,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.2,
        api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Retrieves context and generates an augmented conversational response.
        """
        client = OpenAI(api_key=api_key) if api_key else self.client
        if not client or not client.api_key:
            raise ValueError("No OpenAI API key found. Please provide an API key in the sidebar or set OPENAI_API_KEY in .env")

        retrieved_chunks = self.retrieve(query, top_k=top_k, api_key=api_key)

        # Build context block
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            context_parts.append(
                f"[Source {i} | Page {chunk['page']}]\n{chunk['text']}"
            )
        context_str = "\n\n".join(context_parts)

        system_prompt = (
            "You are the official National Transformation Program (NTP) AI Assistant. "
            "Your task is to provide accurate, comprehensive, and well-structured answers to user questions "
            "based strictly on the provided National Transformation Program Annual Report 2025 context.\n\n"
            "Guidelines:\n"
            "1. Base your answer on the provided context below.\n"
            "2. Whenever citing facts, metrics, or achievements, reference the page number or Source where possible.\n"
            "3. If the context does not contain enough information to answer completely, state what is known from the context and clarify the limitation honestly.\n"
            "4. Maintain a professional, clear, and executive tone.\n"
            "5. Use markdown formatting (bullet points, bold text) for readability.\n\n"
            f"--- DOCUMENT CONTEXT ---\n{context_str}\n------------------------"
        )

        messages = [{"role": "system", "content": system_prompt}]

        if history:
            # Include recent conversation turns (excluding previous system prompt)
            for msg in history[-6:]:  # Last 3 rounds of conversation
                if msg.get("role") in ["user", "assistant"]:
                    messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": query})

        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature
        )

        answer = completion.choices[0].message.content

        return {
            "answer": answer,
            "sources": retrieved_chunks,
            "query": query,
            "model": model
        }

# Global singleton instance
rag_engine = RAGEngine()
