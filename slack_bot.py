"""
HR Slack Bot — workflow

Flow:
  1. Employee DMs the "Employe" bot directly in Slack
  2. Bot processes request via LangChain agent
  3. Bot DMs the designated approver with Approve/Reject buttons
     (card shows: employee, reason, leave type, days, dates)
  4. Approver clicks Approve or Reject in their DM
  5. Result posted to #testing_copilot_rag with full details
  6. Employee also gets a DM reply with the decision

Run with:
    python slack_bot.py
"""

import json
import os
import re
import ssl
import sys

import certifi
from dotenv import load_dotenv

# Fix macOS Python SSL certificate verification
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

load_dotenv()

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import agent as hr_agent
import database as db

db.init_db()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

app = App(
    token=os.environ["SLACK_BOT_TOKEN"],
    signing_secret=os.environ["SLACK_SIGNING_SECRET"],
)

RESULTS_CHANNEL   = os.environ["SLACK_RESULTS_CHANNEL"]    # #testing_copilot_rag channel ID
APPROVER_USER_ID  = os.environ["SLACK_APPROVER_USER_ID"]   # single approver's Slack User ID


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _open_dm(user_id: str) -> str:
    """Open a DM channel with any user and return the channel ID."""
    resp = app.client.conversations_open(users=[user_id])
    return resp["channel"]["id"]


def _get_user_display(user_id: str) -> str:
    """Return a human-readable name for a Slack user ID."""
    try:
        info = app.client.users_info(user=user_id)
        profile = info["user"]["profile"]
        return profile.get("real_name") or profile.get("display_name") or user_id
    except Exception:
        return user_id


def _build_approval_dm(
    request_id: str,
    reason: str,
    requester_id: str,
    tool_name: str,
    tool_args: dict,
) -> list:
    """
    Block Kit card sent to the approver.
    Shows: who requested, reason, action type, and all parameters.
    """
    tool_label = tool_name.replace("_", " ").title()
    requester_name = _get_user_display(requester_id)

    # Build a clean parameter block
    param_fields = []
    field_labels = {
        "employee_id": "Employee ID",
        "leave_type":  "Leave Type",
        "start_date":  "Start Date",
        "end_date":    "End Date",
        "query":       "Query",
    }
    for k, v in tool_args.items():
        label = field_labels.get(k, k.replace("_", " ").title())
        param_fields.append({"type": "mrkdwn", "text": f"*{label}*\n`{v}`"})

    # Calculate days if leave request
    days_text = ""
    if "start_date" in tool_args and "end_date" in tool_args:
        try:
            from datetime import datetime
            start = datetime.strptime(tool_args["start_date"], "%Y-%m-%d")
            end   = datetime.strptime(tool_args["end_date"],   "%Y-%m-%d")
            days  = (end - start).days + 1
            days_text = f"  •  *{days} day{'s' if days != 1 else ''}*"
        except Exception:
            pass

    blocks = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": "HR Approval Request", "emoji": True},
        },
        {"type": "divider"},
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Requested by*\n{requester_name}"},
                {"type": "mrkdwn", "text": f"*Action*\n{tool_label}{days_text}"},
            ],
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Reason / Request*\n_{reason}_",
            },
        },
    ]

    if param_fields:
        # Slack allows max 10 fields per section, split into rows of 2
        for i in range(0, len(param_fields), 2):
            blocks.append({
                "type": "section",
                "fields": param_fields[i : i + 2],
            })

    blocks += [
        {"type": "divider"},
        {
            "type": "actions",
            "block_id": f"approval_block_{request_id}",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Approve", "emoji": True},
                    "style": "primary",
                    "action_id": "slack_approve",
                    "value": json.dumps({"request_id": request_id, "requester_id": requester_id}),
                    "confirm": {
                        "title": {"type": "plain_text", "text": "Confirm Approval"},
                        "text": {
                            "type": "mrkdwn",
                            "text": f"Approve *{tool_label}* request from *{requester_name}*?",
                        },
                        "confirm": {"type": "plain_text", "text": "Yes, Approve"},
                        "deny":    {"type": "plain_text", "text": "Cancel"},
                    },
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Reject", "emoji": True},
                    "style": "danger",
                    "action_id": "slack_reject",
                    "value": json.dumps({"request_id": request_id, "requester_id": requester_id}),
                },
            ],
        },
    ]
    return blocks


def _post_to_results_channel(
    request_id: str,
    reason: str,
    requester_id: str,
    tool_name: str,
    tool_args: dict,
    approved: bool,
    actor_id: str,
    result_str: str = None,
):
    """
    Post a detailed summary to #testing_copilot_rag after approve/reject.
    Shows: employee, reason, leave type, days, dates, decision, reference ID.
    """
    requester_name = _get_user_display(requester_id)
    actor_name     = _get_user_display(actor_id)
    tool_label     = tool_name.replace("_", " ").title()

    status_icon  = ":white_check_mark:" if approved else ":x:"
    status_label = "APPROVED" if approved else "REJECTED"

    # Parse days from tool_args
    days_line = ""
    if "start_date" in tool_args and "end_date" in tool_args:
        try:
            from datetime import datetime
            start = datetime.strptime(tool_args["start_date"], "%Y-%m-%d")
            end   = datetime.strptime(tool_args["end_date"],   "%Y-%m-%d")
            days  = (end - start).days + 1
            days_line = f"\n>*Days Requested:* {days} day{'s' if days != 1 else ''} ({tool_args['start_date']} → {tool_args['end_date']})"
        except Exception:
            pass

    # Build parameter summary
    param_lines = ""
    field_labels = {
        "employee_id": "Employee ID",
        "leave_type":  "Leave Type",
        "start_date":  "Start Date",
        "end_date":    "End Date",
    }
    for k, v in tool_args.items():
        if k not in ("start_date", "end_date"):  # dates shown in days_line
            label = field_labels.get(k, k.replace("_", " ").title())
            param_lines += f"\n>*{label}:* {v}"

    # Parse result for extra info (reference ID, remaining balance)
    result_extras = ""
    if approved and result_str:
        try:
            rd = json.loads(result_str)
            if rd.get("reference_id"):
                result_extras += f"\n>*Reference ID:* `{rd['reference_id']}`"
            if rd.get("remaining_balance") is not None:
                result_extras += f"\n>*Remaining Leave Balance:* {rd['remaining_balance']} days"
            if rd.get("success") is False:
                result_extras += f"\n>:warning: *Note:* {rd.get('error', 'Processing error')}"
        except Exception:
            pass

    blocks = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"{status_icon} HR Request {status_label} — {request_id}",
                "emoji": True,
            },
        },
        {"type": "divider"},
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Employee*\n{requester_name}"},
                {"type": "mrkdwn", "text": f"*Action*\n{tool_label}"},
                {"type": "mrkdwn", "text": f"*Decision*\n{status_label} by {actor_name}"},
                {"type": "mrkdwn", "text": f"*Request ID*\n`{request_id}`"},
            ],
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f"*Reason:* _{reason}_"
                    f"{param_lines}"
                    f"{days_line}"
                    f"{result_extras}"
                ),
            },
        },
        {"type": "divider"},
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f"Processed by HR Bot  •  Request submitted by <@{requester_id}>",
                }
            ],
        },
    ]

    app.client.chat_postMessage(
        channel=RESULTS_CHANNEL,
        text=f"HR Request {status_label} — {request_id}",
        blocks=blocks,
    )


# ---------------------------------------------------------------------------
# DM handler — employee messages the Employe bot directly
# ---------------------------------------------------------------------------

@app.event("message")
def handle_dm(message, say):
    """
    Handle direct messages sent to the bot.
    Only fires for im (DM) channel type — ignores channel messages.
    """
    # Ignore bots, edits, deletions
    if message.get("subtype") or message.get("bot_id"):
        return

    # Only handle DMs (channel_type = im)
    if message.get("channel_type") != "im":
        return

    user_id = message.get("user", "")
    text    = message.get("text", "").strip()

    if not text:
        user_name = _get_user_display(user_id)
        say(
            f"Hi {user_name}! Tell me your HR request and I'll process it.\n"
            "Example: _'I need annual leave from 2026-07-20 to 2026-07-22'_"
        )
        return

    user_name = _get_user_display(user_id)
    say(f"Got it {user_name}! Analyzing your request...")

    request_id = hr_agent.process_request(text, user_name=user_name)
    db.set_request_source(request_id, message.get("channel"), user_id)

    req    = db.get_request(request_id)
    status = req[2] if req else "unknown"

    if status == "pending_approval":
        tool_name = req[3]
        tool_args = json.loads(req[4]) if req[4] else {}

        # DM the approver
        dm_channel = _open_dm(APPROVER_USER_ID)
        blocks = _build_approval_dm(request_id, text, user_id, tool_name, tool_args)
        app.client.chat_postMessage(
            channel=dm_channel,
            text=f"HR Approval Request — {request_id}",
            blocks=blocks,
        )

        db.add_trace_log(request_id, "slack_dm_sent", f"Approval DM sent to approver {APPROVER_USER_ID}")

        say(
            f"Your request `{request_id}` has been sent to the HR manager for approval.\n"
            "You'll receive an update once a decision is made."
        )

    elif status == "completed":
        # No approval needed (e.g. employee lookup)
        result = req[5] or "Done."
        try:
            rd = json.loads(result)
            lines = "\n".join([f"• *{k.replace('_',' ').title()}*: {v}" for k, v in rd.items()])
            say(f"Here's the result for `{request_id}`:\n{lines}")
        except Exception:
            say(f"Result: {result}")

    else:
        say(f"Request `{request_id}` status: *{status}*. Please contact HR for details.")


@app.event("app_mention")
def handle_mention(event, say):
    """Handle @bot mentions in channels."""
    raw_text = event.get("text", "")
    text     = re.sub(r"<@[A-Z0-9]+>", "", raw_text).strip()
    user_id  = event.get("user", "")
    channel  = event.get("channel", "")

    if not text:
        user_name = _get_user_display(user_id)
        say(
            f"Hi {user_name}! DM me directly or mention me with your request.\n"
            "Example: `@HRBot I need annual leave from 2026-07-20 to 2026-07-22`"
        )
        return

    user_name = _get_user_display(user_id)
    say(f"Got it {user_name}! Analyzing...")

    request_id = hr_agent.process_request(text, user_name=user_name)
    db.set_request_source(request_id, channel, user_id)

    req    = db.get_request(request_id)
    status = req[2] if req else "unknown"

    if status == "pending_approval":
        tool_name = req[3]
        tool_args = json.loads(req[4]) if req[4] else {}

        dm_channel = _open_dm(APPROVER_USER_ID)
        blocks = _build_approval_dm(request_id, text, user_id, tool_name, tool_args)
        app.client.chat_postMessage(
            channel=dm_channel,
            text=f"HR Approval Request — {request_id}",
            blocks=blocks,
        )

        db.add_trace_log(request_id, "slack_dm_sent", f"Approval DM sent to approver {APPROVER_USER_ID}")
        say(f"<@{user_id}> Your request `{request_id}` is pending approval. You'll be notified here.")

    elif status == "completed":
        result = req[5] or "Done."
        try:
            rd = json.loads(result)
            lines = "\n".join([f"• *{k}*: {v}" for k, v in rd.items()])
            say(f"<@{user_id}> Result for `{request_id}`:\n{lines}")
        except Exception:
            say(f"<@{user_id}> {result}")


# ---------------------------------------------------------------------------
# Approval button handlers (fired from approver's DM)
# ---------------------------------------------------------------------------

@app.action("slack_approve")
def handle_approve(ack, body, client):
    ack()

    payload       = json.loads(body["actions"][0]["value"])
    request_id    = payload["request_id"]
    requester_id  = payload["requester_id"]
    approver_id   = body["user"]["id"]
    approver_name = _get_user_display(approver_id)
    dm_channel    = body["channel"]["id"]
    msg_ts        = body["message"]["ts"]

    db.add_trace_log(request_id, "approval_decision", f"Approved by {approver_name} via Slack DM")

    result_str = hr_agent.execute_approved_tool(request_id)

    # Get stored request details for the result post
    req       = db.get_request(request_id)
    reason    = req[1] if req else ""
    tool_name = req[3] if req else "unknown"
    tool_args = json.loads(req[4]) if req and req[4] else {}

    # Write structured audit record
    db.add_approval_record(
        request_id=request_id,
        decision="approved",
        actor_name=approver_name,
        actor_slack_id=approver_id,
        source="slack",
        result=result_str,
    )

    # Update the approver's DM card — replace buttons with confirmation
    try:
        result_data  = json.loads(result_str)
        result_brief = "Processed successfully" if result_data.get("success") is not False else f"Error: {result_data.get('error', '')}"
    except Exception:
        result_brief = "Executed"

    client.chat_update(
        channel=dm_channel,
        ts=msg_ts,
        text=f"Approved — {request_id}",
        blocks=[
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f":white_check_mark: *You approved request `{request_id}`*\n_{result_brief}_",
                },
            }
        ],
    )

    # Post full result to #testing_copilot_rag
    _post_to_results_channel(
        request_id=request_id,
        reason=reason,
        requester_id=requester_id,
        tool_name=tool_name,
        tool_args=tool_args,
        approved=True,
        actor_id=approver_id,
        result_str=result_str,
    )

    # Also DM the employee with the decision
    try:
        emp_dm = _open_dm(requester_id)
        client.chat_postMessage(
            channel=emp_dm,
            text=(
                f":white_check_mark: Your HR request `{request_id}` has been *approved* by {approver_name}.\n"
                f"The result has been posted to the team channel."
            ),
        )
    except Exception:
        pass


@app.action("slack_reject")
def handle_reject(ack, body, client):
    ack()

    payload       = json.loads(body["actions"][0]["value"])
    request_id    = payload["request_id"]
    requester_id  = payload["requester_id"]
    rejector_id   = body["user"]["id"]
    rejector_name = _get_user_display(rejector_id)
    dm_channel    = body["channel"]["id"]
    msg_ts        = body["message"]["ts"]

    db.add_trace_log(request_id, "approval_decision", f"Rejected by {rejector_name} via Slack DM")
    db.update_request_status(request_id, "rejected", f"Rejected by {rejector_name}")

    req       = db.get_request(request_id)
    reason    = req[1] if req else ""
    tool_name = req[3] if req else "unknown"
    tool_args = json.loads(req[4]) if req and req[4] else {}

    # Write structured audit record
    db.add_approval_record(
        request_id=request_id,
        decision="rejected",
        actor_name=rejector_name,
        actor_slack_id=rejector_id,
        source="slack",
    )

    # Update the approver's DM card
    client.chat_update(
        channel=dm_channel,
        ts=msg_ts,
        text=f"Rejected — {request_id}",
        blocks=[
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f":x: *You rejected request `{request_id}`*",
                },
            }
        ],
    )

    # Post rejection to #testing_copilot_rag
    _post_to_results_channel(
        request_id=request_id,
        reason=reason,
        requester_id=requester_id,
        tool_name=tool_name,
        tool_args=tool_args,
        approved=False,
        actor_id=rejector_id,
    )

    # DM the employee with the rejection
    try:
        emp_dm = _open_dm(requester_id)
        client.chat_postMessage(
            channel=emp_dm,
            text=(
                f":x: Your HR request `{request_id}` has been *rejected* by {rejector_name}.\n"
                "Please contact HR if you have questions."
            ),
        )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("HR Slack Bot starting...")
    print(f"Results channel : {RESULTS_CHANNEL}")
    print(f"Approver user ID: {APPROVER_USER_ID}")
    handler = SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"])
    handler.start()
