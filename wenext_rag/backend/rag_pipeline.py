"""
rag_pipeline.py
The 3-stage RAG pipeline, adapted from the prompt templates in PROMPT TEMPLATES.docx
(LangChain multi-query + LlamaIndex choice-select + LlamaIndex tree-summarize),
running on a local Ollama model for generation and local sentence-transformers for embeddings.

Pipeline:
  1. Multi-Query Rewrite  -> turns the user's question into 3 phrasings to widen recall
  2. Retrieve             -> embeds each phrasing, pulls top-k chunks per phrasing from ChromaDB, dedupes
  3. Choice Select        -> asks the LLM to rank/filter which retrieved chunks are actually relevant
  4. Tree Summarize       -> asks the LLM to synthesize a final answer from the selected chunks only
"""

import re
import json
from typing import List, Dict

import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "wenext_ai_features"
EMBED_MODEL = "all-MiniLM-L6-v2"
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_HOST = "http://localhost:11434"
DEFAULT_MODEL = "llama3.1:8b"   # used when the caller doesn't pick a model

TOP_K_PER_QUERY = 4


# ---------- Prompts (adapted, not copied verbatim, from the swipe-file templates) ----------

MULTI_QUERY_PROMPT = """You are an AI assistant helping to improve document retrieval.
Given a single user question, generate {n} different versions of it that capture the same
intent using different phrasing or focus, to help retrieve relevant documents from a vector database.

Provide these alternative questions separated by newlines, with no numbering and no extra commentary.

Original question: {question}"""

CHOICE_SELECT_PROMPT = """A list of numbered document excerpts is shown below, each with a short summary.
Given the question, return the numbers of the excerpts that are relevant to answering it,
in order of relevance, along with a relevance score from 1-10.

Only include excerpts that are actually useful for answering the question. If none are relevant, return nothing.

Respond ONLY in this exact format, one per line, no other text:
Excerpt: <number>, Relevance: <score>

Excerpts:
{excerpts}

Question: {question}
Answer:"""

TREE_SUMMARIZE_PROMPT = """You are the WeNext AI Features Guide assistant. Answer the user's question
using ONLY the context excerpts provided below. Do not use outside knowledge and do not guess.
If the context does not contain enough information to answer, say so plainly instead of making
something up. Keep the answer clear and practical, in plain language, and mention which module(s)
of WeNext the answer relates to.

Context excerpts:
{context}

Question: {question}
Answer:"""


class WeNextRAG:
    def __init__(self):
        self.embed_model = SentenceTransformer(EMBED_MODEL)
        self.client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.collection = self.client.get_collection(COLLECTION_NAME)
        self.llm = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")

    # ---- Stage 1: Multi-Query Rewrite ----
    def rewrite_query(self, question: str, model: str, n: int = 3) -> List[str]:
        prompt = MULTI_QUERY_PROMPT.format(n=n, question=question)
        resp = self.llm.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        lines = [l.strip() for l in resp.choices[0].message.content.split("\n") if l.strip()]
        queries = [question] + lines[:n]
        return queries

    # ---- Stage 2: Retrieve ----
    def retrieve(self, queries: List[str], top_k: int = TOP_K_PER_QUERY) -> List[Dict]:
        seen_ids = set()
        results = []
        for q in queries:
            emb = self.embed_model.encode([q]).tolist()
            hits = self.collection.query(query_embeddings=emb, n_results=top_k)
            for i, doc_id in enumerate(hits["ids"][0]):
                if doc_id in seen_ids:
                    continue
                seen_ids.add(doc_id)
                results.append(
                    {
                        "id": doc_id,
                        "title": hits["metadatas"][0][i]["title"],
                        "text": hits["documents"][0][i],
                        "distance": hits["distances"][0][i],
                    }
                )
        return results

    # ---- Stage 3: Choice Select ----
    def choice_select(self, question: str, candidates: List[Dict], model: str) -> List[Dict]:
        if not candidates:
            return []
        excerpts = "\n\n".join(
            f"({i+1}) [{c['title']}]\n{c['text'][:500]}" for i, c in enumerate(candidates)
        )
        prompt = CHOICE_SELECT_PROMPT.format(excerpts=excerpts, question=question)
        resp = self.llm.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        raw = resp.choices[0].message.content

        selected = []
        for line in raw.split("\n"):
            m = re.search(r"Excerpt:\s*(\d+),\s*Relevance:\s*(\d+)", line)
            if m:
                idx = int(m.group(1)) - 1
                score = int(m.group(2))
                if 0 <= idx < len(candidates):
                    c = dict(candidates[idx])
                    c["relevance"] = score
                    selected.append(c)

        selected.sort(key=lambda c: c["relevance"], reverse=True)
        return selected if selected else candidates[:3]   # fallback: don't return empty context

    # ---- Stage 4: Tree Summarize ----
    def synthesize(self, question: str, selected: List[Dict], model: str) -> str:
        context = "\n\n".join(f"[{c['title']}]\n{c['text']}" for c in selected)
        prompt = TREE_SUMMARIZE_PROMPT.format(context=context, question=question)
        resp = self.llm.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return resp.choices[0].message.content

    # ---- Full pipeline ----
    def answer(self, question: str, model: str = DEFAULT_MODEL) -> Dict:
        # Query rewriting and reranking need reliable instruction-following to be useful;
        # small/weak models asked to do this step themselves tend to misrank and silently
        # drop the correct chunk before the answering model ever sees it. So retrieval
        # support always runs on DEFAULT_MODEL, and the user-selected model is only used
        # for the final synthesis step that's actually being compared.
        queries = self.rewrite_query(question, DEFAULT_MODEL)
        candidates = self.retrieve(queries)
        selected = self.choice_select(question, candidates, DEFAULT_MODEL)
        answer_text = self.synthesize(question, selected, model)
        return {
            "question": question,
            "model": model,
            "rewritten_queries": queries,
            "candidates_retrieved": len(candidates),
            "sources": [
                {"title": c["title"], "relevance": c.get("relevance"), "text": c["text"]}
                for c in selected
            ],
            "answer": answer_text,
        }


if __name__ == "__main__":
    rag = WeNextRAG()
    result = rag.answer("How does the WhatsApp AI agent know what to say to customers?")
    print(json.dumps(result, indent=2))
