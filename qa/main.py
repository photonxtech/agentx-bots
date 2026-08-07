import os
import re
import uuid
import json
import time
import io
import base64
import asyncio
import sqlite3
from typing import Optional, List
from dotenv import load_dotenv

load_dotenv()

import fitz  # PyMuPDF
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from groq import Groq, RateLimitError

# DeepEval Custom LLM Imports & Metrics
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    GEval
)
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

app = FastAPI(title="QA Generator with Vision Model & Split DeepEval via Groq")

groq_api_key = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=groq_api_key) if groq_api_key else None
DB_FILE = "qa_sessions.db"

# Model Configuration
VISION_MODEL = os.getenv("VISION_MODEL", "qwen/qwen3.6-27b") # Groq Vision model
GEVAL_JUDGE_MODEL = os.getenv("GEVAL_JUDGE_MODEL", "llama-3.3-70b-versatile")
RAG_JUDGE_MODEL = os.getenv("RAG_JUDGE_MODEL", "openai/gpt-oss-120b")

EVAL_CONCURRENCY_LIMIT = int(os.getenv("GROQ_EVAL_CONCURRENCY", "2"))
_eval_semaphore = asyncio.Semaphore(EVAL_CONCURRENCY_LIMIT)

_RETRY_AFTER_RE = re.compile(r"try again in ([\d.]+)(ms|s)", re.IGNORECASE)

# --- Database Setup ---
def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            title TEXT,
            filename TEXT,
            sample_json TEXT,
            source_doc TEXT,
            generated_qa TEXT,
            deepeval_score REAL,
            deepeval_details TEXT,
            is_golden INTEGER DEFAULT 0,
            status TEXT DEFAULT 'Pending',
            test_case_count INTEGER DEFAULT 20
        )
    """)
    cursor.execute("PRAGMA table_info(sessions)")
    existing_columns = {row[1] for row in cursor.fetchall()}
    if "test_case_count" not in existing_columns:
        cursor.execute("ALTER TABLE sessions ADD COLUMN test_case_count INTEGER DEFAULT 20")
    conn.commit()
    conn.close()

init_db()

def get_db_connection():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

TEST_CASE_DEPTH_INSTRUCTIONS = {
    10: (
        "Generate exactly 10 test cases covering ONLY the primary, explicitly "
        "stated concepts in the document. One test case per major concept — "
        "do not include sub-topics, edge cases, or minor details."
    ),
    20: (
        "Generate exactly 20 test cases covering the primary concepts AND "
        "their directly related sub-topics or details explicitly mentioned "
        "in the document."
    ),
    30: (
        "Generate exactly 30 test cases with keen observation: cover primary "
        "concepts, sub-topics, edge cases, boundary/negative scenarios "
        "(e.g. what happens if a step is skipped or a value is missing), and "
        "implicit relationships or cross-references between different parts "
        "of the document. Prioritize thoroughness and nuance over repetition."
    ),
}
ALLOWED_TEST_CASE_COUNTS = (10, 20, 30)

# --- Custom Groq Evaluator ---
class GroqEvaluatorLLM(DeepEvalBaseLLM):
    def __init__(self, model_name="openai/gpt-oss-120b"):
        self.model_name = model_name
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:
        truncated_prompt = prompt[:4000]
        max_retries = 5
        base_delay = 3.0

        for attempt in range(max_retries):
            try:
                chat_completion = self.client.chat.completions.create(
                    messages=[{"role": "user", "content": truncated_prompt}],
                    model=self.model_name,
                    temperature=0.0
                )
                return chat_completion.choices[0].message.content
            except Exception as e:
                is_rate_limit = "rate_limit_exceeded" in str(e).lower() or isinstance(e, RateLimitError)
                if is_rate_limit and attempt < max_retries - 1:
                    sleep_time = self._resolve_retry_delay(str(e), attempt, base_delay)
                    print(f"[DeepEval] Rate limit hit on {self.model_name}. Retrying in {sleep_time:.1f}s...")
                    time.sleep(sleep_time)
                    continue
                raise e

    @staticmethod
    def _resolve_retry_delay(error_text: str, attempt: int, base_delay: float) -> float:
        match = _RETRY_AFTER_RE.search(error_text)
        if match:
            value, unit = float(match.group(1)), match.group(2).lower()
            suggested = value / 1000.0 if unit == "ms" else value
            return suggested + 0.5
        return base_delay * (attempt + 1)

    async def a_generate(self, prompt: str) -> str:
        async with _eval_semaphore:
            return await asyncio.to_thread(self.generate, prompt)

    def get_model_name(self) -> str:
        return self.model_name

# --- Vision Analysis Helper ---
def analyze_page_image_with_vision(image_bytes: bytes) -> str:
    """Passes page rendering image to Groq Multimodal Vision model to describe visual content."""
    if not groq_client:
        return ""
    
    try:
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        response = groq_client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": (
                                "Examine this document page image. Describe all visual components, "
                                "including diagrams, flowcharts, UI mockups, infographics, visual tables, "
                                "and embedded image annotations in explicit detail."
                            )
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            temperature=0.1
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[Vision Analysis Failed]: {e}")
        return ""

def extract_document_text(file_bytes: bytes, filename: str) -> str:
    text_content = ""
    try:
        if filename.lower().endswith(".pdf"):
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            for i, page in enumerate(doc, start=1):
                # 1. Standard text extraction
                extracted_text = page.get_text().strip()
                
                # 2. Render page to image for Vision model
                pix = page.get_pixmap(dpi=150)
                img_bytes = pix.tobytes("png")
                
                visual_description = analyze_page_image_with_vision(img_bytes)
                
                text_content += f"[PAGE {i}]\n"
                if extracted_text:
                    text_content += f"--- EXTRACTED TEXT ---\n{extracted_text}\n"
                if visual_description:
                    text_content += f"--- VISUAL & DIAGRAM DESCRIPTION ---\n{visual_description}\n"
                text_content += "\n"
        else:
            text_content = file_bytes.decode("utf-8", errors="ignore")
    except Exception as e:
        print(f"Text extraction error: {e}")
    return text_content.strip()


def chunk_document(text: str, chunk_size: int = 400, overlap: int = 40, max_chunks: int = 3) -> List[str]:
    text = text.strip()
    if not text:
        return ["No context provided."]

    chunks = []
    start = 0
    step = max(chunk_size - overlap, 1)
    while start < len(text) and len(chunks) < max_chunks:
        chunk = text[start:start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += step

    return chunks if chunks else [text[:chunk_size]]


def build_reference_answer(eval_llm: "GroqEvaluatorLLM", context_chunks: List[str]) -> Optional[str]:
    try:
        context_text = "\n---\n".join(context_chunks)[:4000]
        prompt = (
            "Extract 3 to 5 key factual statements directly present in the text below. "
            "Return them as a plain numbered list, one statement per line, with no extra commentary.\n\n"
            f"TEXT:\n{context_text}"
        )
        result = eval_llm.generate(prompt)
        return result.strip() if result else None
    except Exception as e:
        print(f"[DeepEval] Reference answer generation failed: {e}")
        return None


async def run_deepeval_evaluation(doc_text: str, generated_json: dict, sample_json: str = "") -> dict:
    if not os.getenv("GROQ_API_KEY"):
        raise HTTPException(status_code=500, detail="GROQ_API_KEY missing for evaluation.")

    try:
        rag_eval_llm = GroqEvaluatorLLM(model_name=RAG_JUDGE_MODEL)
        geval_eval_llm = GroqEvaluatorLLM(model_name=GEVAL_JUDGE_MODEL)

        context_chunks = chunk_document(doc_text) if doc_text else ["No context provided."]

        input_prompt = (
            "Generate structured QA test cases extracted strictly from the provided "
            "source document, adhering to the target JSON schema."
        )
        actual_output_str = json.dumps(generated_json)

        used_independent_reference = False
        expected_output = sample_json.strip() if sample_json and sample_json.strip() else None
        if not expected_output:
            expected_output = await asyncio.to_thread(build_reference_answer, rag_eval_llm, context_chunks)
            used_independent_reference = True
        if not expected_output:
            expected_output = actual_output_str
            used_independent_reference = False

        test_case = LLMTestCase(
            input=input_prompt,
            actual_output=actual_output_str,
            retrieval_context=context_chunks,
            expected_output=expected_output
        )

        faithfulness = FaithfulnessMetric(threshold=0.7, model=rag_eval_llm)
        answer_relevancy = AnswerRelevancyMetric(threshold=0.7, model=rag_eval_llm)
        contextual_relevancy = ContextualRelevancyMetric(threshold=0.7, model=rag_eval_llm)
        contextual_precision = ContextualPrecisionMetric(threshold=0.7, model=rag_eval_llm)
        contextual_recall = ContextualRecallMetric(threshold=0.7, model=rag_eval_llm)
        g_eval = GEval(
            name="Schema Structure & Quality",
            criteria=(
                "Step 1: Check that actual_output is valid JSON matching target schema fields. "
                "Step 2: Check that no extra/undocumented fields are present. "
                "Step 3: Check ground_truth values against retrieval_context facts. "
                "Step 4: Penalize score proportionally to unsupported ground_truth values."
            ),
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.RETRIEVAL_CONTEXT],
            model=geval_eval_llm,
            strict_mode=False
        )

        await asyncio.gather(
            faithfulness.a_measure(test_case),
            answer_relevancy.a_measure(test_case),
            contextual_relevancy.a_measure(test_case),
            contextual_precision.a_measure(test_case),
            contextual_recall.a_measure(test_case),
            g_eval.a_measure(test_case),
        )

        faith_score = round(float(faithfulness.score), 2)
        ans_rel_score = round(float(answer_relevancy.score), 2)
        ctx_rel_score = round(float(contextual_relevancy.score), 2)
        ctx_prec_score = round(float(contextual_precision.score), 2)
        ctx_rec_score = round(float(contextual_recall.score), 2)
        geval_score = round(float(g_eval.score), 2)

        scores = [faith_score, ans_rel_score, ctx_rel_score, ctx_prec_score, ctx_rec_score, geval_score]
        overall = round(sum(scores) / len(scores), 2)

        all_passed = (
            faithfulness.is_successful() and
            answer_relevancy.is_successful() and
            contextual_relevancy.is_successful() and
            contextual_precision.is_successful() and
            contextual_recall.is_successful() and
            g_eval.is_successful()
        )

        return {
            "overall_score": overall,
            "faithfulness": faith_score,
            "answer_relevancy": ans_rel_score,
            "contextual_relevancy": ctx_rel_score,
            "contextual_precision": ctx_prec_score,
            "contextual_recall": ctx_rec_score,
            "g_eval": geval_score,
            "passed": all_passed,
            "used_independent_reference": used_independent_reference,
            "retrieval_chunk_count": len(context_chunks),
            "reason": (
                f"Faithfulness: {faithfulness.reason} | "
                f"Answer Rel: {answer_relevancy.reason} | "
                f"Context Rel: {contextual_relevancy.reason} | "
                f"Context Prec: {contextual_precision.reason} | "
                f"Context Rec: {contextual_recall.reason} | "
                f"G-Eval: {g_eval.reason}"
            )
        }
    except Exception as e:
        print(f"DeepEval Execution Error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"DeepEval evaluation failed via Groq: {str(e)}"
        )

# --- FastAPI Endpoints ---

@app.get("/api/sessions")
async def list_sessions():
    conn = get_db_connection()
    rows = conn.execute("SELECT session_id, title, status, is_golden FROM sessions ORDER BY session_id DESC").fetchall()
    conn.close()
    return [{"id": r["session_id"], "title": r["title"], "status": r["status"], "is_golden": bool(r["is_golden"])} for r in rows]


@app.post("/api/sessions")
async def create_session():
    new_id = str(uuid.uuid4())
    conn = get_db_connection()
    count = conn.execute('SELECT COUNT(*) FROM sessions').fetchone()[0] + 1
    title = f"Session {count}"
    conn.execute(
        "INSERT INTO sessions (session_id, title, sample_json, status, is_golden) VALUES (?, ?, ?, ?, ?)",
        (new_id, title, "", "Pending", 0)
    )
    conn.commit()
    conn.close()
    return {"id": new_id, "title": title}


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    conn = get_db_connection()
    row = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Session not found")

    return {
        "session_id": row["session_id"],
        "title": row["title"],
        "filename": row["filename"],
        "sample_json": row["sample_json"],
        "has_document": bool(row["source_doc"]),
        "status": row["status"],
        "is_golden": bool(row["is_golden"]),
        "generated_data": json.loads(row["generated_qa"]) if row["generated_qa"] else None,
        "deepeval_score": row["deepeval_score"],
        "deepeval_details": json.loads(row["deepeval_details"]) if row["deepeval_details"] else None,
        "test_case_count": row["test_case_count"] if row["test_case_count"] else 20
    }


@app.post("/api/sessions/{session_id}/generate")
async def generate_qa_testcases(
    session_id: str,
    file: Optional[UploadFile] = File(None),
    sample_json: str = Form(...),
    test_case_count: int = Form(20)
):
    if test_case_count not in ALLOWED_TEST_CASE_COUNTS:
        raise HTTPException(
            status_code=400,
            detail=f"test_case_count must be one of {ALLOWED_TEST_CASE_COUNTS}"
        )

    conn = get_db_connection()
    row = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()

    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Session not found")

    filename = row["filename"]
    doc_text = row["source_doc"] or ""

    if file and file.filename:
        filename = file.filename
        content = await file.read()
        doc_text = extract_document_text(content, filename)

    if not groq_client:
        conn.close()
        raise HTTPException(status_code=500, detail="GROQ_API_KEY environment variable missing.")

    truncated_context = doc_text[:8000] if doc_text else "No document context uploaded."

    depth_instruction = TEST_CASE_DEPTH_INSTRUCTIONS[test_case_count]

    has_page_markers = "[PAGE " in truncated_context
    page_instruction = (
        "The document context contains [PAGE n] markers showing where each page starts. "
        "For every test case, add a field named \"page_no\" set to the integer page number "
        "the underlying fact was drawn from."
        if has_page_markers else
        "The document has no page markers (non-PDF source). Omit any page_no field, "
        "or set it to null if your schema requires the key to be present."
    )

    system_prompt = (
        "You are an expert QA Automation Engineer. Generate QA test cases "
        "extracted strictly from the provided text and visual diagram descriptions in the source document, "
        "adhering to the target JSON schema."
    )

    user_prompt = f"""DOCUMENT CONTEXT:
{truncated_context}

TARGET SAMPLE JSON SCHEMA:
{sample_json if sample_json else "{}"}

GENERATION DEPTH INSTRUCTION:
{depth_instruction}

PAGE NUMBER INSTRUCTION:
{page_instruction}
"""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2
        )
        generated_json = json.loads(response.choices[0].message.content)
    except Exception as e:
        conn.close()
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

    await asyncio.sleep(1)

    eval_results = await run_deepeval_evaluation(doc_text, generated_json, sample_json)

    conn.execute(
        """UPDATE sessions 
           SET filename = ?, sample_json = ?, source_doc = ?, generated_qa = ?, 
               deepeval_score = ?, deepeval_details = ?, is_golden = 0, status = 'Pending',
               test_case_count = ?
           WHERE session_id = ?""",
        (filename, sample_json, doc_text, json.dumps(generated_json),
         eval_results["overall_score"], json.dumps(eval_results), test_case_count, session_id)
    )
    conn.commit()
    conn.close()

    actual_count = None
    if isinstance(generated_json, list):
        actual_count = len(generated_json)
    elif isinstance(generated_json, dict):
        for v in generated_json.values():
            if isinstance(v, list):
                actual_count = len(v)
                break

    return {
        "generated_qa": generated_json,
        "deepeval": eval_results,
        "status": "Pending",
        "is_golden": False,
        "filename": filename,
        "requested_count": test_case_count,
        "actual_count": actual_count
    }


@app.post("/api/sessions/{session_id}/approve")
async def approve_golden_dataset(session_id: str):
    conn = get_db_connection()
    conn.execute("UPDATE sessions SET is_golden = 1, status = 'Approved' WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()
    return {"message": "Saved to Golden Dataset", "status": "Approved", "is_golden": True}


@app.get("/api/golden-dataset/export")
async def export_golden_dataset():
    conn = get_db_connection()
    rows = conn.execute("SELECT * FROM sessions WHERE is_golden = 1").fetchall()
    conn.close()

    return [{
        "session_id": r["session_id"],
        "filename": r["filename"],
        "qa_data": json.loads(r["generated_qa"]) if r["generated_qa"] else {},
        "deepeval_score": r["deepeval_score"],
        "status": r["status"]
    } for r in rows]

app.mount("/", StaticFiles(directory="static", html=True), name="static")