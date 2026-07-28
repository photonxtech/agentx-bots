"""
HR Workflow Agent — Streamlit UI

Pages:
  1. Submit Request  — natural language HR requests
  2. Pending Approvals — HR manager approval queue
  3. Request History  — all requests with status
  4. Trace Logs       — full LangChain execution trace
"""

import json
import time

import streamlit as st

import database as db

db.init_db()

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="HR Workflow Agent",
    page_icon="HR",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("HR Workflow Agent")
st.sidebar.caption("Powered by LangChain + Streamlit")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    ["Submit Request", "Pending Approvals", "Request History", "Trace Logs"],
)

pending_count = len(db.get_pending_approvals())
if pending_count:
    st.sidebar.warning(f"{pending_count} approval(s) waiting")

st.sidebar.divider()
st.sidebar.markdown("**Test Employee IDs**")
st.sidebar.markdown("- `E001` Alice Johnson (Eng)\n- `E002` Bob Smith (Mktg)\n- `E003` Carol Davis (HR)")

# ---------------------------------------------------------------------------
# Page: Submit Request
# ---------------------------------------------------------------------------

if page == "Submit Request":
    st.title("Submit HR Request")
    st.markdown("Describe the HR action in plain English. The agent will analyze it, identify the right action, and route it for manager approval.")

    col_form, col_info = st.columns([3, 1])

    with col_form:
        EXAMPLES = [
            "Employee E001 wants annual leave from 2026-07-20 to 2026-07-22",
            "Apply sick leave for E002 from 2026-07-14 to 2026-07-15",
            "Look up information for employee E003",
            "Find details for Alice Johnson",
            "E001 needs 5 days personal leave starting 2026-08-01",
        ]

        st.markdown("**Quick examples — click to load:**")
        cols = st.columns(2)
        for i, ex in enumerate(EXAMPLES):
            if cols[i % 2].button(ex, key=f"ex_{i}", use_container_width=True):
                st.session_state["prefill"] = ex

        st.divider()

        default_text = st.session_state.pop("prefill", "")
        request_text = st.text_area(
            "HR Request",
            value=default_text,
            placeholder="e.g. 'Employee E001 wants 3 days annual leave from 2026-07-20 to 2026-07-22'",
            height=100,
        )

        if st.button("Submit to Agent", type="primary", disabled=not request_text.strip()):
            with st.spinner("LangChain agent analyzing..."):
                import agent
                request_id = agent.process_request(request_text.strip())

            req = db.get_request(request_id)
            status = req[2] if req else "unknown"

            if status == "pending_approval":
                st.success(f"Request `{request_id}` created — waiting for approval.")
                st.info("Go to **Pending Approvals** to review and approve.")
            elif status == "completed":
                st.success(f"Request `{request_id}` completed (no approval needed).")
                result = req[5]
                if result:
                    try:
                        st.json(json.loads(result))
                    except Exception:
                        st.write(result)
            else:
                st.error(f"Request `{request_id}` status: **{status}**. Check Trace Logs for details.")

    with col_info:
        st.markdown("**How it works**")
        st.markdown("""
1. You submit a plain-English request
2. LangChain LLM identifies the action & parameters
3. Request is paused for HR manager approval
4. Manager approves/rejects in the dashboard
5. On approval, action executes in the HR system
6. Full trace is logged for audit
""")
        st.divider()
        st.markdown("**Available actions**")
        st.markdown("- Process Leave Request\n- Lookup Employee")

# ---------------------------------------------------------------------------
# Page: Pending Approvals
# ---------------------------------------------------------------------------

elif page == "Pending Approvals":
    st.title("Pending Approvals")

    col_h, col_r = st.columns([5, 1])
    with col_r:
        if st.button("Refresh", use_container_width=True):
            st.rerun()

    pending = db.get_pending_approvals()

    if not pending:
        st.info("No pending approvals at this time.")
    else:
        st.markdown(f"**{len(pending)} request(s) awaiting your decision**")
        st.divider()

        for req_id, req_text, tool_name, tool_args_str, created_at in pending:
            tool_args = json.loads(tool_args_str) if tool_args_str else {}

            with st.container(border=True):
                col_details, col_action = st.columns([3, 1])

                with col_details:
                    st.markdown(f"**Request ID:** `{req_id}`")
                    st.markdown(f"**Original Request:**  {req_text}")
                    st.markdown(f"**Action to execute:** `{tool_name}`")
                    st.markdown("**Parameters:**")
                    # Render parameters as a clean table
                    for k, v in tool_args.items():
                        st.markdown(f"  - `{k}`: **{v}**")
                    st.caption(f"Submitted: {created_at.replace('T', ' ')[:19]}")

                with col_action:
                    st.markdown("**Decision**")

                    if st.button("Approve", key=f"approve_{req_id}", type="primary", use_container_width=True):
                        db.add_trace_log(req_id, "approval_decision", "Approved by HR manager via dashboard")
                        with st.spinner("Executing action..."):
                            import agent
                            result_str = agent.execute_approved_tool(req_id)
                        st.success("Approved & executed!")
                        try:
                            st.json(json.loads(result_str))
                        except Exception:
                            st.write(result_str)
                        time.sleep(1.5)
                        st.rerun()

                    if st.button("Reject", key=f"reject_{req_id}", use_container_width=True):
                        db.add_trace_log(req_id, "approval_decision", "Rejected by HR manager via dashboard")
                        db.update_request_status(req_id, "rejected", "Rejected by HR manager")
                        st.warning("Request rejected.")
                        time.sleep(1)
                        st.rerun()

# ---------------------------------------------------------------------------
# Page: Request History
# ---------------------------------------------------------------------------

elif page == "Request History":
    st.title("Request History")

    col_h, col_r = st.columns([5, 1])
    with col_r:
        if st.button("Refresh", use_container_width=True):
            st.rerun()

    requests = db.get_all_requests()

    if not requests:
        st.info("No requests yet. Submit one from the 'Submit Request' page.")
    else:
        STATUS_ICON = {
            "pending_approval": "PENDING",
            "completed": "DONE",
            "rejected": "REJECTED",
            "failed": "FAILED",
            "processing": "PROCESSING",
        }
        STATUS_COLOR = {
            "pending_approval": "orange",
            "completed": "green",
            "rejected": "red",
            "failed": "red",
            "processing": "blue",
        }

        for req_id, req_text, status, tool_name, result, created_at in requests:
            icon = STATUS_ICON.get(status, status.upper())
            color = STATUS_COLOR.get(status, "gray")

            with st.expander(f"[{icon}]  {req_id} — {req_text[:60]}{'...' if len(req_text) > 60 else ''}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Request:** {req_text}")
                    st.markdown(f"**Status:** :{color}[{status.replace('_', ' ').title()}]")
                    st.markdown(f"**Tool:** {tool_name or 'N/A'}")
                    st.markdown(f"**Submitted:** {created_at.replace('T', ' ')[:19]}")
                with col2:
                    if result:
                        st.markdown("**Result:**")
                        try:
                            st.json(json.loads(result))
                        except Exception:
                            st.code(result)

                if st.button(f"View trace logs for {req_id}", key=f"view_trace_{req_id}"):
                    st.session_state["trace_filter"] = req_id

# ---------------------------------------------------------------------------
# Page: Trace Logs
# ---------------------------------------------------------------------------

elif page == "Trace Logs":
    st.title("Trace Logs")
    st.markdown("Full audit trail of every LangChain agent step, tool call, approval decision, and result.")

    col_filter, col_r = st.columns([4, 1])
    with col_filter:
        prefill_id = st.session_state.pop("trace_filter", "")
        filter_id = st.text_input("Filter by Request ID (leave blank for all)", value=prefill_id)
    with col_r:
        st.markdown("")
        st.markdown("")
        if st.button("Refresh"):
            st.rerun()

    logs = db.get_trace_logs(filter_id.strip() if filter_id.strip() else None)

    STEP_ICON = {
        "user_input":          "USER",
        "agent_thinking":      "THINK",
        "llm_start":           "LLM",
        "llm_end":             "LLM",
        "llm_error":           "ERROR",
        "tool_identified":     "TOOL",
        "awaiting_approval":   "WAIT",
        "approval_decision":   "APPROVAL",
        "execution_start":     "RUN",
        "tool_start":          "TOOL",
        "tool_end":            "TOOL",
        "tool_result":         "RESULT",
        "tool_error":          "ERROR",
        "completed":           "DONE",
        "error":               "ERROR",
        "llm_direct_response": "LLM",
    }

    if not logs:
        st.info("No trace logs found.")
    else:
        st.markdown(f"Showing **{len(logs)}** log entries")
        st.divider()

        for req_id, step_type, content, timestamp in logs:
            icon = STEP_ICON.get(step_type, "LOG")
            time_str = timestamp[11:19] if len(timestamp) >= 19 else timestamp

            cols = st.columns([0.7, 1, 5, 1])
            cols[0].markdown(f"`{icon}`")
            cols[1].markdown(f"`{req_id}`")
            cols[2].markdown(content)
            cols[3].caption(time_str)
