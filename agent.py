"""
LangChain-powered HR workflow agent.

Flow:
  1. LLM analyzes the HR request and decides which tool to call.
  2. Instead of executing the tool immediately, we save a "pending approval"
     record to the database so an HR manager can review it in Streamlit.
  3. After approval, execute_approved_tool() runs the actual action.

LangChain components used:
  - ChatOpenAI (or swap for ChatAnthropic) with bind_tools() for function calling
  - LangChain @tool decorator for defining HR actions
  - BaseCallbackHandler for trace logging every LLM/tool event
"""

import json
import os
import time
from typing import Any

from dotenv import load_dotenv
load_dotenv()

# ── LangSmith tracing ────────────────────────────────────────────────────────
# Set LANGCHAIN_TRACING_V2=true in .env to enable.
# Every LLM call will appear in your LangSmith dashboard automatically.
if os.environ.get("LANGCHAIN_TRACING_V2", "").lower() == "true":
    os.environ.setdefault("LANGCHAIN_PROJECT", os.environ.get("LANGCHAIN_PROJECT", "hr-workflow-agent"))

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

import database as db
import hr_tools

# ---------------------------------------------------------------------------
# LangChain Tool definitions
# ---------------------------------------------------------------------------

@tool
def process_leave_request(
    employee_name: str, leave_type: str, start_date: str, end_date: str
) -> str:
    """
    Process an employee leave request in the HR system.

    Args:
        employee_name: The employee's full name or partial name (e.g. "Alice", "Alice Johnson").
        leave_type:    Type of leave — annual, sick, or personal.
        start_date:    Leave start date in YYYY-MM-DD format.
        end_date:      Leave end date in YYYY-MM-DD format.
    """
    result = hr_tools.process_leave_request(employee_name, leave_type, start_date, end_date)
    return json.dumps(result, indent=2)


@tool
def lookup_employee(query: str) -> str:
    """
    Look up an employee's details by their employee ID or full/partial name.

    Args:
        query: Employee ID (e.g. E001) or employee name (e.g. "Alice").
    """
    result = hr_tools.lookup_employee(query)
    return json.dumps(result, indent=2)


TOOLS = [process_leave_request, lookup_employee]
TOOLS_MAP = {t.name: t for t in TOOLS}

# ---------------------------------------------------------------------------
# LangChain Callback — writes every agent event to the trace_logs table
# ---------------------------------------------------------------------------

class TraceLogger(BaseCallbackHandler):
    def __init__(self, request_id: str, model: str = "gpt-4o-mini"):
        self.request_id  = request_id
        self.model       = model
        self._llm_start  = None   # timestamp for latency calculation
        self._tool_called = None  # track which tool was called

    def on_llm_start(self, serialized, prompts, **kwargs):
        self._llm_start = time.time()
        db.add_trace_log(self.request_id, "llm_start", f"LLM call started — model: {self.model}")

    def on_llm_end(self, response, **kwargs):
        latency_ms = int((time.time() - self._llm_start) * 1000) if self._llm_start else 0

        # Extract token usage from OpenAI response
        usage = {}
        try:
            usage = response.llm_output.get("token_usage", {})
        except Exception:
            pass

        prompt_tokens     = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens      = usage.get("total_tokens", 0)

        db.add_trace_log(
            self.request_id, "llm_end",
            f"LLM done — tokens: {total_tokens} (prompt: {prompt_tokens}, completion: {completion_tokens}) | latency: {latency_ms}ms"
        )

        # Store in llm_traces table for analytics
        db.add_llm_trace(
            request_id=self.request_id,
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            tool_called=self._tool_called,
        )

    def on_llm_error(self, error, **kwargs):
        db.add_trace_log(self.request_id, "llm_error", f"LLM error: {str(error)}")

    def on_tool_start(self, serialized, input_str, **kwargs):
        name = serialized.get("name", "unknown")
        self._tool_called = name
        db.add_trace_log(self.request_id, "tool_start", f"Tool '{name}' called with: {input_str[:200]}")

    def on_tool_end(self, output, **kwargs):
        db.add_trace_log(self.request_id, "tool_end", f"Tool result: {str(output)[:300]}")

    def on_tool_error(self, error, **kwargs):
        db.add_trace_log(self.request_id, "tool_error", f"Tool error: {str(error)}")


# ---------------------------------------------------------------------------
# System prompt — built dynamically so today's date is always injected
# ---------------------------------------------------------------------------

def _build_system_prompt() -> str:
    from datetime import datetime
    now        = datetime.now()
    today_str  = now.strftime("%Y-%m-%d")           # e.g. 2026-07-11
    day_name   = now.strftime("%A")                  # e.g. Saturday
    # Pre-compute the next few weekday dates for reliable resolution
    from datetime import timedelta
    weekdays   = {}
    for i in range(1, 8):
        d = now + timedelta(days=i)
        weekdays[d.strftime("%A")] = d.strftime("%Y-%m-%d")

    weekday_hint = ", ".join([f"{k} = {v}" for k, v in weekdays.items()])

    return f"""You are an HR assistant for a company. Your job is to process HR requests submitted by employees via Slack.

You have two tools available:
- process_leave_request: Use when someone wants to apply for leave.
- lookup_employee: Use when someone wants to look up employee information.

IMPORTANT — Today's date and date resolution:
- Today is {day_name}, {today_str}.
- Always resolve relative day names to exact YYYY-MM-DD dates using today as the reference.
- Upcoming weekday dates from today: {weekday_hint}
- Always output dates in YYYY-MM-DD format.

IMPORTANT — Multi-day date resolution rules:
- "Monday and Tuesday" → start_date = Monday's date, end_date = Tuesday's date
- "Monday to Wednesday" → start_date = Monday's date, end_date = Wednesday's date
- "Monday through Friday" → start_date = Monday's date, end_date = Friday's date
- "just Monday" or "only Monday" → start_date = Monday's date, end_date = Monday's date (same day)
- "tomorrow" → start_date = tomorrow, end_date = tomorrow
- "next week" → start_date = next Monday, end_date = next Friday
- When two separate days are mentioned, ALWAYS use the first as start_date and the second as end_date.
- NEVER use the same date for both start and end when two different days are mentioned.

IMPORTANT — Employee identity:
- The employee's Slack name will always be provided at the start of the message as: [Employee: <name>]
- Use that name directly as the employee_name parameter — never ask for an employee ID.
- If the request is about themselves, use their name from the context.

When you receive a request:
1. Extract the employee name from [Employee: <name>] at the start.
2. Understand what action is needed.
3. Carefully resolve ALL date references to exact YYYY-MM-DD using the weekday dates above.
4. Call the appropriate tool using the employee's name.

Be concise and professional."""


# ---------------------------------------------------------------------------
# Main agent functions
# ---------------------------------------------------------------------------

def process_request(request_text: str, user_name: str = None) -> str:
    """
    Analyze the HR request with LangChain, determine the required tool,
    and save it as pending_approval.  Returns the request_id.

    user_name: the Slack display name of the person who sent the message.
               Injected as context so the LLM knows who is requesting.
    """
    request_id = db.create_request(request_text)
    db.add_trace_log(request_id, "user_input", f"Request from {user_name or 'unknown'}: {request_text}")

    # Prefix the message with the user's name so the LLM can use it
    if user_name:
        full_message = f"[Employee: {user_name}]\n{request_text}"
    else:
        full_message = request_text

    try:
        model_name = "gpt-4o-mini"
        logger = TraceLogger(request_id, model=model_name)
        llm = ChatOpenAI(model=model_name, temperature=0, callbacks=[logger])
        llm_with_tools = llm.bind_tools(TOOLS)

        db.add_trace_log(request_id, "agent_thinking", f"LangChain agent analyzing request for {user_name or 'user'}...")

        messages = [
            SystemMessage(content=_build_system_prompt()),
            HumanMessage(content=full_message),
        ]
        response = llm_with_tools.invoke(messages)

        if response.tool_calls:
            tool_call = response.tool_calls[0]
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            db.add_trace_log(
                request_id,
                "tool_identified",
                f"Action identified: {tool_name} | Parameters: {json.dumps(tool_args)}",
            )
            db.add_trace_log(
                request_id,
                "awaiting_approval",
                "Paused — waiting for HR manager approval before executing.",
            )
            db.set_pending_approval(request_id, tool_name, tool_args)

        else:
            # LLM responded without a tool call (e.g. clarification needed)
            content = response.content or "No action needed."
            db.add_trace_log(request_id, "llm_direct_response", content)
            db.update_request_status(request_id, "completed", content)

    except Exception as e:
        db.add_trace_log(request_id, "error", f"Agent error: {str(e)}")
        db.update_request_status(request_id, "failed", str(e))

    return request_id


def execute_approved_tool(request_id: str) -> str:
    """
    Called after an HR manager approves a request.
    Runs the previously identified tool and stores the result.
    """
    request = db.get_request(request_id)
    if not request:
        return "Request not found."

    # Schema: id, request_text, status, tool_name, tool_args, result, created_at, updated_at
    tool_name = request[3]
    tool_args_str = request[4]

    db.add_trace_log(request_id, "execution_start", f"Executing '{tool_name}' after approval...")

    try:
        tool_args = json.loads(tool_args_str) if tool_args_str else {}
        tool_fn = TOOLS_MAP.get(tool_name)

        if not tool_fn:
            msg = f"Tool '{tool_name}' is not registered."
            db.add_trace_log(request_id, "error", msg)
            db.update_request_status(request_id, "failed", msg)
            return msg

        result_str = tool_fn.invoke(tool_args)

        db.add_trace_log(request_id, "tool_result", f"Result: {result_str[:400]}")
        db.add_trace_log(request_id, "completed", "Request completed successfully.")
        db.update_request_status(request_id, "completed", result_str)

        return result_str

    except Exception as e:
        msg = f"Execution error: {str(e)}"
        db.add_trace_log(request_id, "error", msg)
        db.update_request_status(request_id, "failed", msg)
        return msg
