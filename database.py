"""
MySQL-backed persistence layer.

Tables:
  requests         — every HR request submitted
  trace_logs       — step-by-step agent execution log
  approval_history — structured audit trail of every approve/reject decision
  llm_traces       — LLM token usage and latency per request
"""

import json
import os
import uuid
from datetime import datetime

import mysql.connector
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------

def _get_conn(with_db: bool = True):
    cfg = dict(
        host=os.environ.get("MYSQL_HOST", "127.0.0.1"),
        port=int(os.environ.get("MYSQL_PORT", "3306")),
        user=os.environ.get("MYSQL_USER", "root"),
        password=os.environ.get("MYSQL_PASSWORD", ""),
    )
    if with_db:
        cfg["database"] = os.environ.get("MYSQL_DATABASE", "hr_agent")
    return mysql.connector.connect(**cfg)


# ---------------------------------------------------------------------------
# Init — create database + all tables
# ---------------------------------------------------------------------------

def init_db():
    db_name = os.environ.get("MYSQL_DATABASE", "hr_agent")

    # Create database if it doesn't exist
    conn = _get_conn(with_db=False)
    c = conn.cursor()
    c.execute(f"CREATE DATABASE IF NOT EXISTS `{db_name}`")
    conn.commit()
    conn.close()

    conn = _get_conn()
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS requests (
            id           VARCHAR(20)  PRIMARY KEY,
            request_text TEXT,
            status       VARCHAR(50),
            tool_name    VARCHAR(100),
            tool_args    TEXT,
            result       TEXT,
            source_channel VARCHAR(50),
            source_user    VARCHAR(50),
            created_at   VARCHAR(50),
            updated_at   VARCHAR(50)
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS trace_logs (
            id          INT AUTO_INCREMENT PRIMARY KEY,
            request_id  VARCHAR(20),
            step_type   VARCHAR(100),
            content     TEXT,
            timestamp   VARCHAR(50),
            INDEX idx_request_id (request_id)
        )
    """)

    # Structured approval audit trail
    c.execute("""
        CREATE TABLE IF NOT EXISTS approval_history (
            id              INT AUTO_INCREMENT PRIMARY KEY,
            request_id      VARCHAR(20),
            decision        VARCHAR(20),
            actor_name      VARCHAR(100),
            actor_slack_id  VARCHAR(50),
            source          VARCHAR(50),
            request_text    TEXT,
            tool_name       VARCHAR(100),
            tool_args       TEXT,
            result          TEXT,
            decided_at      VARCHAR(50),
            INDEX idx_request_id (request_id),
            INDEX idx_decided_at (decided_at)
        )
    """)

    # LLM token usage and latency per request
    c.execute("""
        CREATE TABLE IF NOT EXISTS llm_traces (
            id                INT AUTO_INCREMENT PRIMARY KEY,
            request_id        VARCHAR(20),
            model             VARCHAR(100),
            prompt_tokens     INT DEFAULT 0,
            completion_tokens INT DEFAULT 0,
            total_tokens      INT DEFAULT 0,
            latency_ms        INT DEFAULT 0,
            tool_called       VARCHAR(100),
            created_at        VARCHAR(50),
            INDEX idx_request_id (request_id),
            INDEX idx_created_at (created_at)
        )
    """)

    conn.commit()
    conn.close()
    print(f"[DB] Connected to MySQL — database: {db_name}")


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------

def create_request(request_text: str) -> str:
    request_id = str(uuid.uuid4())[:8].upper()
    conn = _get_conn()
    c = conn.cursor()
    now = datetime.now().isoformat()
    c.execute(
        "INSERT INTO requests (id, request_text, status, created_at, updated_at) VALUES (%s, %s, 'processing', %s, %s)",
        (request_id, request_text, now, now),
    )
    conn.commit()
    conn.close()
    return request_id


def set_pending_approval(request_id: str, tool_name: str, tool_args: dict):
    conn = _get_conn()
    c = conn.cursor()
    now = datetime.now().isoformat()
    c.execute(
        "UPDATE requests SET status='pending_approval', tool_name=%s, tool_args=%s, updated_at=%s WHERE id=%s",
        (tool_name, json.dumps(tool_args), now, request_id),
    )
    conn.commit()
    conn.close()


def update_request_status(request_id: str, status: str, result: str = None):
    conn = _get_conn()
    c = conn.cursor()
    now = datetime.now().isoformat()
    if result is not None:
        c.execute(
            "UPDATE requests SET status=%s, result=%s, updated_at=%s WHERE id=%s",
            (status, result, now, request_id),
        )
    else:
        c.execute(
            "UPDATE requests SET status=%s, updated_at=%s WHERE id=%s",
            (status, now, request_id),
        )
    conn.commit()
    conn.close()


def set_request_source(request_id: str, source_channel: str, source_user: str):
    conn = _get_conn()
    c = conn.cursor()
    c.execute(
        "UPDATE requests SET source_channel=%s, source_user=%s WHERE id=%s",
        (source_channel, source_user, request_id),
    )
    conn.commit()
    conn.close()


def get_request(request_id: str):
    conn = _get_conn()
    c = conn.cursor()
    c.execute("SELECT * FROM requests WHERE id=%s", (request_id,))
    row = c.fetchone()
    conn.close()
    return row


def get_request_source(request_id: str) -> tuple:
    conn = _get_conn()
    c = conn.cursor()
    c.execute("SELECT source_channel, source_user FROM requests WHERE id=%s", (request_id,))
    row = c.fetchone()
    conn.close()
    return row if row else (None, None)


def get_pending_approvals():
    conn = _get_conn()
    c = conn.cursor()
    c.execute(
        "SELECT id, request_text, tool_name, tool_args, created_at FROM requests WHERE status='pending_approval' ORDER BY created_at DESC"
    )
    rows = c.fetchall()
    conn.close()
    return rows


def get_all_requests():
    conn = _get_conn()
    c = conn.cursor()
    c.execute(
        "SELECT id, request_text, status, tool_name, result, created_at FROM requests ORDER BY created_at DESC"
    )
    rows = c.fetchall()
    conn.close()
    return rows


# ---------------------------------------------------------------------------
# Trace logs
# ---------------------------------------------------------------------------

def add_trace_log(request_id: str, step_type: str, content: str):
    conn = _get_conn()
    c = conn.cursor()
    now = datetime.now().isoformat()
    c.execute(
        "INSERT INTO trace_logs (request_id, step_type, content, timestamp) VALUES (%s, %s, %s, %s)",
        (request_id, step_type, content, now),
    )
    conn.commit()
    conn.close()


def get_trace_logs(request_id: str = None):
    conn = _get_conn()
    c = conn.cursor()
    if request_id:
        c.execute(
            "SELECT request_id, step_type, content, timestamp FROM trace_logs WHERE request_id=%s ORDER BY id ASC",
            (request_id,),
        )
    else:
        c.execute(
            "SELECT request_id, step_type, content, timestamp FROM trace_logs ORDER BY id DESC LIMIT 200"
        )
    rows = c.fetchall()
    conn.close()
    return rows


# ---------------------------------------------------------------------------
# Approval history  (structured audit trail)
# ---------------------------------------------------------------------------

def add_approval_record(
    request_id: str,
    decision: str,
    actor_name: str,
    actor_slack_id: str,
    source: str = "slack",
    result: str = None,
):
    """Record every approve/reject decision with full context for audit."""
    req = get_request(request_id)
    request_text = req[1] if req else ""
    tool_name    = req[3] if req else ""
    tool_args    = req[4] if req else ""

    conn = _get_conn()
    c = conn.cursor()
    now = datetime.now().isoformat()
    c.execute(
        """INSERT INTO approval_history
           (request_id, decision, actor_name, actor_slack_id, source,
            request_text, tool_name, tool_args, result, decided_at)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
        (request_id, decision, actor_name, actor_slack_id, source,
         request_text, tool_name, tool_args, result, now),
    )
    conn.commit()
    conn.close()


def get_approval_history(limit: int = 100):
    conn = _get_conn()
    c = conn.cursor()
    c.execute(
        """SELECT request_id, decision, actor_name, source,
                  request_text, tool_name, tool_args, result, decided_at
           FROM approval_history ORDER BY decided_at DESC LIMIT %s""",
        (limit,),
    )
    rows = c.fetchall()
    conn.close()
    return rows


# ---------------------------------------------------------------------------
# LLM traces  (token usage + latency)
# ---------------------------------------------------------------------------

def add_llm_trace(
    request_id: str,
    model: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int = 0,
    latency_ms: int = 0,
    tool_called: str = None,
):
    conn = _get_conn()
    c = conn.cursor()
    now = datetime.now().isoformat()
    c.execute(
        """INSERT INTO llm_traces
           (request_id, model, prompt_tokens, completion_tokens,
            total_tokens, latency_ms, tool_called, created_at)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
        (request_id, model, prompt_tokens, completion_tokens,
         total_tokens, latency_ms, tool_called, now),
    )
    conn.commit()
    conn.close()


def get_llm_traces(request_id: str = None, limit: int = 100):
    conn = _get_conn()
    c = conn.cursor()
    if request_id:
        c.execute(
            """SELECT request_id, model, prompt_tokens, completion_tokens,
                      total_tokens, latency_ms, tool_called, created_at
               FROM llm_traces WHERE request_id=%s ORDER BY id ASC""",
            (request_id,),
        )
    else:
        c.execute(
            """SELECT request_id, model, prompt_tokens, completion_tokens,
                      total_tokens, latency_ms, tool_called, created_at
               FROM llm_traces ORDER BY id DESC LIMIT %s""",
            (limit,),
        )
    rows = c.fetchall()
    conn.close()
    return rows


def get_token_summary():
    """Total tokens used across all requests — for the dashboard."""
    conn = _get_conn()
    c = conn.cursor()
    c.execute(
        "SELECT SUM(prompt_tokens), SUM(completion_tokens), SUM(total_tokens), COUNT(*) FROM llm_traces"
    )
    row = c.fetchone()
    conn.close()
    return {
        "prompt_tokens":     row[0] or 0,
        "completion_tokens": row[1] or 0,
        "total_tokens":      row[2] or 0,
        "llm_calls":         row[3] or 0,
    }
