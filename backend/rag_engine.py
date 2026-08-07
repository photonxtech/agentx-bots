import os
import io
import re
import json
import numpy as np
import fitz  # PyMuPDF

import chromadb
from groq import Groq

# --- Hybrid OCR tuning knobs ---
# Below this many characters, we don't trust the extracted/OCR'd text and
# escalate to the next tier (native text -> Tesseract -> Vision LLM).
MIN_TRUSTED_TEXT_CHARS = 20
# Tesseract's own per-word confidence (0-100). Below this average, treat the
# OCR pass as unreliable even if it returned some text.
MIN_TESSERACT_CONFIDENCE = 60
# DPI used when rasterizing PDF pages that have no extractable text layer.
SCAN_RASTER_DPI = 200

VISION_TRANSCRIBE_PROMPT = (
    "Extract all readable text from this image exactly as it appears. "
    "If the image contains diagrams, forms, or tables, represent their structure in markdown/text as clearly as possible. "
    "If the image contains primarily visual content without text, describe the visual content in detail."
)


class DocumentParser:
    """Parses various document and image formats and extracts text page-by-page."""
    @staticmethod
    def parse(file_path: str, ext: str, groq_api_key: str = None) -> list[dict]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        ext = ext.lower()
        if ext == ".pdf":
            return DocumentParser._parse_pdf(file_path, groq_api_key)
        elif ext == ".docx":
            return DocumentParser._parse_docx(file_path)
        elif ext == ".doc":
            return DocumentParser._parse_doc_fallback(file_path)
        elif ext == ".txt":
            return DocumentParser._parse_txt(file_path)
        elif ext in {".png", ".jpg", ".jpeg", ".webp"}:
            return DocumentParser._parse_image(file_path, ext, groq_api_key)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

    @staticmethod
    def _parse_pdf(pdf_path: str, groq_api_key: str = None) -> list[dict]:
        doc = fitz.open(pdf_path)
        pages = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()

            # Native text layer missing or near-empty -> this page is likely a
            # scan/photo, not born-digital text. Rasterize it and run it
            # through the same OCR tiers used for standalone images.
            if len(text.strip()) < MIN_TRUSTED_TEXT_CHARS:
                pix = page.get_pixmap(dpi=SCAN_RASTER_DPI)
                image_bytes = pix.tobytes("png")
                ocr_text, source = DocumentParser._ocr_image_bytes(
                    image_bytes, mime_type="image/png", groq_api_key=groq_api_key
                )
                if ocr_text.strip():
                    text = ocr_text
                    print(f"[OCR] Page {page_num + 1}: no text layer, recovered via {source}")

            pages.append({
                "page_number": page_num + 1,
                "text": text
            })
        doc.close()
        return pages

    @staticmethod
    def _parse_docx(docx_path: str) -> list[dict]:
        import docx
        doc = docx.Document(docx_path)
        full_text = []
        for para in doc.paragraphs:
            if para.text.strip():
                full_text.append(para.text)
        
        # Extract table contents too
        for table in doc.tables:
            for row in table.rows:
                row_text = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if row_text:
                    full_text.append(" | ".join(row_text))
                    
        text = "\n".join(full_text)
        return [{"page_number": 1, "text": text}]

    @staticmethod
    def _parse_doc_fallback(doc_path: str) -> list[dict]:
        # Try as docx first in case it's actually a docx file misnamed as .doc
        try:
            return DocumentParser._parse_docx(doc_path)
        except Exception:
            pass
            
        # Fallback: Extract readable ASCII/Unicode text strings from binary file
        try:
            with open(doc_path, "rb") as f:
                data = f.read()
            # Find all consecutive printable character blocks (minimum 4 chars)
            import string
            printable_chars = set(string.printable.encode('ascii'))
            text_blocks = []
            current_block = []
            for byte in data:
                if byte in printable_chars:
                    current_block.append(chr(byte))
                else:
                    if len(current_block) >= 4:
                        text_blocks.append("".join(current_block))
                    current_block = []
            if len(current_block) >= 4:
                text_blocks.append("".join(current_block))
            
            # Filter and clean up extracted lines
            clean_text = "\n".join([line.strip() for line in text_blocks if len(line.strip()) > 10])
            return [{"page_number": 1, "text": clean_text}]
        except Exception as e:
            raise ValueError(f"Failed to parse legacy .doc file. Please convert it to .docx. Error: {str(e)}")

    @staticmethod
    def _parse_txt(txt_path: str) -> list[dict]:
        for encoding in ("utf-8", "latin-1", "utf-16"):
            try:
                with open(txt_path, "r", encoding=encoding) as f:
                    text = f.read()
                return [{"page_number": 1, "text": text}]
            except UnicodeDecodeError:
                continue
        raise ValueError("Could not decode plain text file with utf-8, latin-1, or utf-16 encodings.")

    @staticmethod
    def _parse_image(image_path: str, ext: str, groq_api_key: str) -> list[dict]:
        mime_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp"
        }
        mime_type = mime_types.get(ext, "image/jpeg")

        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()

        extracted_text, source = DocumentParser._ocr_image_bytes(
            image_bytes, mime_type=mime_type, groq_api_key=groq_api_key
        )
        if not extracted_text.strip() and not groq_api_key:
            raise ValueError(
                "Tesseract found no readable text and no Groq API key is configured "
                "to fall back to vision-LLM OCR."
            )
        print(f"[OCR] {os.path.basename(image_path)}: extracted via {source}")
        return [{"page_number": 1, "text": extracted_text}]

    # --- Hybrid OCR tiers ---------------------------------------------------
    # Tier 1: Tesseract (free, fast, deterministic, literal transcription).
    # Tier 2: Vision LLM (Groq/Qwen) — used only when Tesseract fails or is
    #         low-confidence, e.g. handwriting, diagrams, tilted/noisy scans.

    @staticmethod
    def _ocr_image_bytes(image_bytes: bytes, mime_type: str, groq_api_key: str = None) -> tuple[str, str]:
        """Run the hybrid OCR pipeline on raw image bytes.
        Returns (text, source) where source is 'tesseract' or 'vision_llm'.
        """
        tesseract_text, confidence = DocumentParser._ocr_with_tesseract(image_bytes)

        tesseract_ok = (
            len(tesseract_text.strip()) >= MIN_TRUSTED_TEXT_CHARS
            and confidence >= MIN_TESSERACT_CONFIDENCE
        )
        if tesseract_ok:
            return tesseract_text, "tesseract"

        if not groq_api_key:
            # No fallback available — return whatever Tesseract managed, even if weak.
            return tesseract_text, "tesseract_low_confidence_no_fallback"

        try:
            vision_text = DocumentParser._ocr_with_vision_llm(image_bytes, mime_type, groq_api_key)
            return vision_text, "vision_llm"
        except Exception as e:
            print(f"[OCR WARNING] Vision LLM fallback failed: {e}")
            return tesseract_text, "tesseract_low_confidence_fallback_failed"

    @staticmethod
    def _ocr_with_tesseract(image_bytes: bytes) -> tuple[str, float]:
        """Tier 1 OCR. Returns (text, average_confidence_0_to_100)."""
        try:
            import pytesseract
            from PIL import Image
        except ImportError:
            print("[OCR WARNING] pytesseract/Pillow not installed — skipping Tesseract tier.")
            return "", 0.0

        try:
            image = Image.open(io.BytesIO(image_bytes))
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            words, confidences = [], []
            for word, conf in zip(data.get("text", []), data.get("conf", [])):
                if word.strip():
                    words.append(word)
                    conf_val = float(conf)
                    if conf_val >= 0:  # -1 means "no confidence available" for that token
                        confidences.append(conf_val)
            text = " ".join(words)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            return text, avg_confidence
        except Exception as e:
            print(f"[OCR WARNING] Tesseract failed: {e}")
            return "", 0.0

    @staticmethod
    def _ocr_with_vision_llm(image_bytes: bytes, mime_type: str, groq_api_key: str) -> str:
        """Tier 2 OCR fallback via Groq-hosted vision LLM."""
        import base64

        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        client = Groq(api_key=groq_api_key)

        response = client.chat.completions.create(
            model="qwen/qwen3.6-27b",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": VISION_TRANSCRIBE_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            temperature=0,
            max_tokens=2048
        )
        return response.choices[0].message.content


class StructureChunker:
    """Chunks text using an overlapping sliding window across concatenated document pages, preserving page context and preventing boundary splits."""
    def __init__(self, max_chunk_size: int = 2500, min_chunk_size: int = 800):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size

    def split_text(self, text: str) -> list[str]:
        chunks = []
        if not text.strip():
            return []
            
        start = 0
        overlap = self.min_chunk_size # Use min_chunk_size as overlap size
        
        while start < len(text):
            end = start + self.max_chunk_size
            if end >= len(text):
                chunks.append(text[start:])
                break
                
            boundary = -1
            # Search backwards up to 300 characters for double newline
            for search_idx in range(end, max(start, end - 300), -1):
                if text[search_idx:search_idx+2] == "\n\n":
                    boundary = search_idx + 2
                    break
            if boundary == -1:
                # Search backwards for single newline
                for search_idx in range(end, max(start, end - 300), -1):
                    if text[search_idx] == "\n":
                        boundary = search_idx + 1
                        break
            if boundary == -1:
                # Search backwards for sentence punctuation
                for search_idx in range(end, max(start, end - 300), -1):
                    if text[search_idx] in ".?!":
                        boundary = search_idx + 1
                        break
                        
            if boundary != -1:
                chunk = text[start:boundary]
                chunks.append(chunk)
                start = boundary - overlap
            else:
                chunks.append(text[start:end])
                start = end - overlap
                
        return [c.strip() for c in chunks if c.strip()]

    def split_documents(self, documents: list[dict], source_name: str) -> list[dict]:
        # 1. Concatenate all pages with page markers
        full_text = ""
        for doc in documents:
            page_num = doc["page_number"]
            full_text += f"\n--- PAGE_START:{page_num} ---\n{doc['text']}"
            
        # 2. Split by explicit section divider headers (====== divider lines)
        # Matches: ===== followed by ==, followed by N. Title, followed by ===== and ==
        pattern = r"(={10,}\s*\n==\s*\n\d+\.\s+[^\n]+\n={10,}\s*\n==)"
        parts = re.split(pattern, full_text)
        
        split_texts = []
        
        # Intro/preamble content before the first header
        intro_content = parts[0]
        if intro_content.strip():
            split_texts.extend(self.split_text(intro_content))
            
        # Process each section
        # parts[i] will contain the header, parts[i+1] will contain the section contents
        for i in range(1, len(parts), 2):
            header = parts[i]
            content = parts[i+1]
            section_text = f"{header}\n{content}"
            
            if len(section_text) <= self.max_chunk_size:
                split_texts.append(section_text)
            else:
                # If too large, perform sliding window chunking within this section
                split_texts.extend(self.split_text(section_text))
        
        # 3. Build chunks and map page numbers with cleaning & deduplication
        chunks = []
        seen_texts = set()
        import unicodedata
        
        for i, text in enumerate(split_texts):
            # Find all page numbers in this chunk
            page_matches = re.findall(r"--- PAGE_START:(\d+) ---", text)
            if page_matches:
                page_num = int(page_matches[0])
            else:
                page_num = 1
                
            # Clean up the page markers from the text so LLM/User doesn't see them
            clean_text = re.sub(r"\n?--- PAGE_START:\d+ ---\n?", "\n", text).strip()
            
            # Normalize whitespace and Unicode
            normalized_text = unicodedata.normalize("NFKC", clean_text)
            normalized_text = re.sub(r"\s+", " ", normalized_text).strip()
            
            # Skip empty or duplicate chunks
            if not normalized_text or normalized_text in seen_texts:
                continue
            seen_texts.add(normalized_text)
            
            chunks.append({
                "id": f"{source_name}_chunk_{i}",
                "text": clean_text,
                "metadata": {
                    "source": source_name,
                    "page_number": page_num,
                    "chunk_index": i
                }
            })
        return chunks

class LocalEmbeddingClient:
    """Generates text embeddings locally using BAAI/bge-small-en-v1.5 model."""
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(LocalEmbeddingClient, cls).__new__(cls)
            cls._instance._model = None
        return cls._instance

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model_name = model_name

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def get_embedding(self, text: str) -> list[float]:
        if not text.strip():
            return [0.0] * 384
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def get_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        cleaned_texts = [t if t.strip() else " " for t in texts]
        
        # Standard encoding for small inputs to avoid subprocess creation overhead
        if len(cleaned_texts) < 50:
            embeddings = self.model.encode(
                cleaned_texts,
                batch_size=32,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            return embeddings.tolist()

        try:
            # Parallelize embedding generation across all available CPU cores
            pool = self.model.start_multi_process_pool()
            try:
                embeddings = self.model.encode_multi_process(
                    cleaned_texts,
                    pool,
                    batch_size=128
                )
                # Normalize embedding results
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1.0, norms)
                embeddings = embeddings / norms
            finally:
                self.model.stop_multi_process_pool(pool)
        except Exception as e:
            print(f"[INGEST WARNING] Multi-process encoding failed, falling back: {e}")
            embeddings = self.model.encode(
                cleaned_texts,
                batch_size=128,
                normalize_embeddings=True,
                show_progress_bar=False
            )
        return embeddings.tolist()


class GroqClient:
    """LLM answer generation using Groq API (llama-3.3-70b-versatile).
    Free tier: ~14,400 requests/day at very high speed.
    """
    DEFAULT_MODEL = "llama-3.3-70b-versatile"

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self.client = Groq(api_key=self.api_key)

    def generate_answer(self, prompt: str, model: str = None, system_instruction: str = None) -> str:
        model = model or self.DEFAULT_MODEL
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            max_tokens=2048,
        )
        # Store token usage for caller to read
        self.last_usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
            "model": model,
            "provider": "groq"
        }
        return response.choices[0].message.content

    def generate_json(
        self,
        prompt: str,
        model: str,
        system_instruction: str,
        max_tokens: int = 1200,
    ) -> dict:
        """Run a deterministic JSON-only evaluation call on an explicit model."""
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        self.last_usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
            "model": model,
            "provider": "groq",
        }
        content = (response.choices[0].message.content or "").strip()
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise ValueError("Evaluation model returned a JSON value other than an object.")
        return parsed

class VectorStore:
    """Wrapper around ChromaDB for index management and similarity searching."""
    def __init__(self, persist_directory: str | None = None):
        if persist_directory is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            persist_directory = os.path.join(base_dir, "chroma_db")
        os.makedirs(persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_directory)
        
    def get_collection(self, collection_name: str = "rag_collection"):
        # We specify cosine similarity for distance metric.
        # This translates distance to: 1 - cosine_similarity.
        return self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_documents(self, collection_name: str, chunks: list[dict], embeddings: list[list[float]]):
        collection = self.get_collection(collection_name)
        
        ids = [c["id"] for c in chunks]
        documents = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        
        # Batch insert to avoid size limit issues
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            collection.add(
                ids=ids[i:i+batch_size],
                embeddings=embeddings[i:i+batch_size],
                documents=documents[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size]
            )

    def query(self, collection_name: str, query_str: str, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        # 1. Dense Vector Query (ANN Search)
        collection = self.get_collection(collection_name)
        if collection.count() == 0:
            return []
            
        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(10, collection.count())
            )
        except Exception as e:
            # Handle dimension mismatch (e.g. switching between 768 and 384 dimensional embeddings)
            if "dimension" in str(e).lower() or "dimensionality" in str(e).lower():
                try:
                    self.delete_collection(collection_name)
                except Exception:
                    pass
                return []
            raise e
            
        dense_results = []
        if results["ids"] and results["ids"][0]:
            for idx in range(len(results["ids"][0])):
                distance = results["distances"][0][idx]
                dense_results.append({
                    "id": results["ids"][0][idx],
                    "text": results["documents"][0][idx],
                    "metadata": results["metadatas"][0][idx],
                    "similarity": 1.0 - distance
                })
                
        # 2. Sparse Keyword Query
        # Fetch all indexed documents in memory for sparse TF-IDF / overlap matching
        all_data = collection.get()
        all_docs = []
        if all_data and "ids" in all_data and all_data["ids"]:
            for idx in range(len(all_data["ids"])):
                all_docs.append({
                    "id": all_data["ids"][idx],
                    "text": all_data["documents"][idx],
                    "metadata": all_data["metadatas"][idx],
                    "similarity": 0.0
                })
                
        sparse_results = self.sparse_retrieve(query_str, all_docs, top_k=10)
        
        # 3. Reciprocal Rank Fusion (RRF) Merge
        hybrid_results = self.rrf_merge(dense_results, sparse_results, top_n=10)
        
        # 4. Reranking (Lexical Jaccard + Semantic similarity)
        reranked_results = self.rerank_documents(query_str, hybrid_results, top_n=top_k)
        
        # 5. Add Source Confidence Scoring
        for doc in reranked_results:
            score = doc.get("rerank_score", doc.get("similarity", 0.0))
            if score >= 0.55:
                doc["confidence"] = "HIGH"
            elif score >= 0.35:
                doc["confidence"] = "MEDIUM"
            else:
                doc["confidence"] = "LOW"
                
        return reranked_results

    def sparse_retrieve(self, query_str: str, documents: list[dict], top_k: int = 10) -> list[dict]:
        query_words = set(re.findall(r"\w+", query_str.lower()))
        if not query_words:
            return []
            
        scored_docs = []
        for doc in documents:
            doc_words = re.findall(r"\w+", doc["text"].lower())
            if not doc_words:
                continue
            intersection = query_words.intersection(doc_words)
            if intersection:
                # Basic overlap score normalized by document length log
                score = len(intersection) / (np.log(len(doc_words) + 2))
                scored_docs.append((score, doc))
                
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        # Return ranked documents with calculated keyword similarity proxy
        results = []
        for score, doc in scored_docs[:top_k]:
            doc["similarity"] = min(1.0, score * 0.15) # Map sparse score to [0,1] similarity
            results.append(doc)
        return results

    def rrf_merge(self, dense: list[dict], sparse: list[dict], k: int = 60, top_n: int = 10) -> list[dict]:
        scores = {}
        doc_map = {}
        
        # Add dense candidates rank scores
        for rank, doc in enumerate(dense):
            doc_id = doc["id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (k + rank + 1))
            doc_map[doc_id] = doc
            
        # Add sparse candidates rank scores
        for rank, doc in enumerate(sparse):
            doc_id = doc["id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (k + rank + 1))
            if doc_id not in doc_map:
                doc_map[doc_id] = doc
                
        merged = []
        for doc_id, rrf_score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            doc = doc_map[doc_id]
            doc["rrf_score"] = rrf_score
            merged.append(doc)
        return merged[:top_n]

    def rerank_documents(self, query_str: str, documents: list[dict], top_n: int = 3) -> list[dict]:
        query_words = set(re.findall(r"\w+", query_str.lower()))
        if not query_words:
            return documents[:top_n]
            
        reranked = []
        for doc in documents:
            # Sentence Jaccard similarity reranking
            sentences = re.split(r"(?<=[.!?])\s+", doc["text"])
            best_sentence_score = 0.0
            
            for sent in sentences:
                sent_words = set(re.findall(r"\w+", sent.lower()))
                if not sent_words:
                    continue
                intersection = len(query_words.intersection(sent_words))
                union = len(query_words.union(sent_words))
                jaccard = intersection / union if union > 0 else 0.0
                best_sentence_score = max(best_sentence_score, jaccard)
                
            # Combined score: 60% sentence Jaccard + 40% dense/sparse base similarity
            base_score = doc.get("similarity", 0.0)
            rerank_score = (0.6 * best_sentence_score) + (0.4 * base_score)
            doc["rerank_score"] = rerank_score
            # doc["similarity"] is kept as base semantic similarity to prevent over-filtering
            reranked.append(doc)
            
        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_n]

    def list_collections(self):
        return self.client.list_collections()

    def delete_collection(self, collection_name: str):
        try:
            self.client.delete_collection(name=collection_name)
        except Exception:
            pass
        
    def get_collection_count(self, collection_name: str) -> int:
        collection = self.get_collection(collection_name)
        return collection.count()

    def get_all_documents(self, collection_name: str) -> dict:
        collection = self.get_collection(collection_name)
        return collection.get()