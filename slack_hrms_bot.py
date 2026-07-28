"""
HRMS Slack AI Agent — Full Employee Self-Service

True agentic loop: LLM reasons → picks tools → chains calls → natural language reply.

Tools covered (24 total):
  Profile & Team      : get_my_profile, get_my_team, get_special_events
  Leaves              : get_leave_balance, get_my_leaves, apply_leave, cancel_leave
  Attendance          : get_attendance_summary, get_swipe_info, checkin_checkout
  Payslip             : get_payslips
  Work From Home      : get_wfh_list, apply_wfh
  Claims              : get_claim_types, get_my_claims, submit_claim
  Notifications       : get_notifications
  Public Holidays     : get_public_holidays
  My Shift            : get_my_shift
  Benefits            : get_my_benefits
  Travel Requests     : get_my_travel_requests
  Regularization      : get_regularizations
  Grievance           : get_my_grievances, raise_grievance

Run with:
    python3 slack_hrms_bot.py
"""

import json
import logging
import os
import re
import ssl
import certifi
import traceback
from datetime import datetime, timedelta
from pathlib import Path

import requests
from dotenv import load_dotenv

ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())
load_dotenv()

# ---------------------------------------------------------------------------
# Logging setup — writes to console AND hrms_bot.log
# ---------------------------------------------------------------------------

LOG_FILE = Path(__file__).parent / "hrms_bot.log"

_fmt = logging.Formatter(
    fmt="%(asctime)s  %(levelname)-8s  %(name)s  |  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

_file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
_file_handler.setFormatter(_fmt)

_console_handler = logging.StreamHandler()
_console_handler.setFormatter(_fmt)

logging.basicConfig(level=logging.INFO, handlers=[_file_handler, _console_handler])

log = logging.getLogger("hrms_bot")

# ---------------------------------------------------------------------------
# MongoDB — conversation history (one document per Slack user)
# ---------------------------------------------------------------------------

from pymongo import MongoClient, ASCENDING
from pymongo.errors import PyMongoError

_MONGO_URL = "mongodb+srv://abhizgn1026:5nVJkTYWN6d0q3my@cluster0.wzp4ld5.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

try:
    _mongo_client = MongoClient(_MONGO_URL, serverSelectionTimeoutMS=5000)
    _mongo_client.admin.command("ping")          # verify connection at startup
    _conversations = _mongo_client["hrms_bot"]["conversations"]
    _conversations.create_index([("slack_user_id", ASCENDING)], unique=True)
    log.info("MongoDB connected — hrms_bot.conversations")
except Exception as _e:
    _conversations = None
    log.error("MongoDB connection failed: %s — conversation saving disabled", _e)


def save_message(slack_user_id: str, role: str, text: str, employee_name: str = "", emp_id: str = ""):
    """
    Append one message to the user's conversation document.
    Document shape:
      {
        slack_user_id, employee_name, emp_id,
        messages: [{role, text, timestamp}, ...],
        created_at, last_updated
      }
    """
    if _conversations is None:
        return
    try:
        now = datetime.utcnow()
        _conversations.update_one(
            {"slack_user_id": slack_user_id},
            {
                "$push": {
                    "messages": {
                        "role":      role,       # "user" or "bot"
                        "text":      text,
                        "timestamp": now,
                    }
                },
                "$set":  {"last_updated": now, "employee_name": employee_name, "emp_id": emp_id},
                "$setOnInsert": {"slack_user_id": slack_user_id, "created_at": now},
            },
            upsert=True,
        )
    except PyMongoError as e:
        log.error("MongoDB save_message failed: %s", e)

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from langchain_core.tools import tool

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# HRMS_BASE_URL = "https://photonxtech.stellarhrm.com/api"
HRMS_BASE_URL = "http://localhost:9091/api"
HRMS_URL_HDR  = "photonxtech.stellarhrm.com"
SESSIONS_FILE = Path(__file__).parent / "hrms_sessions.json"

app = App(
    token=os.environ["SLACK_BOT_TOKEN"],
    signing_secret=os.environ["SLACK_SIGNING_SECRET"],
)

# ---------------------------------------------------------------------------
# Session store
# ---------------------------------------------------------------------------

def _load_sessions() -> dict:
    if SESSIONS_FILE.exists():
        with open(SESSIONS_FILE) as f:
            return json.load(f)
    return {}

def _save_sessions(sessions: dict):
    with open(SESSIONS_FILE, "w") as f:
        json.dump(sessions, f, indent=2)

def _get_session(uid: str) -> dict | None:
    return _load_sessions().get(uid)

def _set_session(uid: str, token: str, emp_id: str, name: str):
    s = _load_sessions()
    s[uid] = {"token": token, "emp_id": emp_id, "name": name}
    _save_sessions(s)

def _clear_session(uid: str):
    s = _load_sessions()
    s.pop(uid, None)
    _save_sessions(s)

# ---------------------------------------------------------------------------
# HRMS HTTP layer
# ---------------------------------------------------------------------------

def _api(method: str, path: str, token: str = None, **kwargs):
    """Returns (ok: bool, data: any).  ok=True when HTTP 2xx."""
    headers = {"url": HRMS_URL_HDR, "Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    url = f"{HRMS_BASE_URL}{path}"
    log.info("API  %s %s", method, path)
    try:
        resp = requests.request(method, url, headers=headers, timeout=15, **kwargs)
        try:
            data = resp.json()
        except Exception:
            data = {"message": resp.text[:300]}
        ok = resp.status_code < 300
        if ok:
            log.info("API  %s %s → %s OK", method, path, resp.status_code)
        else:
            log.warning("API  %s %s → %s  body=%s", method, path, resp.status_code,
                        str(data)[:200])
        return ok, data
    except Exception as e:
        log.error("API  %s %s → EXCEPTION: %s", method, path, e, exc_info=True)
        return False, {"message": str(e)}


def hrms_login(username: str, password: str):
    log.info("LOGIN attempt  username=%s", username)
    ok, data = _api("POST", "/Employee/login", json={"username": username, "password": password})
    if ok:
        token   = data.get("Token") or data.get("token") or ""
        results = data.get("results") or {}
        emp_id  = str(results.get("emp_id", ""))
        fname   = results.get("fname", username)
        lname   = results.get("lname", "")
        name    = f"{fname} {lname}".strip() if lname else fname
        log.info("LOGIN success  username=%s  emp_id=%s  name=%s", username, emp_id, name)
        return True, {"token": token, "emp_id": emp_id, "name": name}
    msg = data.get("message") or "Invalid credentials"
    log.warning("LOGIN failed  username=%s  reason=%s", username, msg)
    return False, {"message": msg}

# ---------------------------------------------------------------------------
# LangChain Tools factory
# All tools are closures bound to the user's JWT token for this request.
# ---------------------------------------------------------------------------

def _rows(data) -> list:
    """Normalise API responses — extract rows regardless of wrapper shape."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("rows") or data.get("data") or data.get("results") or []
    return []

def _ok_or_err(ok: bool, data, empty_msg: str = "No data found.") -> str:
    if not ok:
        return f"Error: {data.get('message', 'Request failed')}"
    rows = _rows(data)
    if not rows:
        return empty_msg
    return None  # caller should continue processing


def create_hrms_tools(token: str) -> list:

    # ── PROFILE & TEAM ──────────────────────────────────────────────────────

    @tool
    def get_my_profile() -> str:
        """Get the logged-in employee's full profile: name, designation, department, branch, email, phone, joining date."""
        ok, data = _api("GET", "/Employee/profile", token=token)
        if not ok:
            return f"Error: {data.get('message', 'Could not fetch profile')}"
        d = data if isinstance(data, dict) else ({} if not data else data[0])
        return json.dumps({
            "name":        f"{d.get('fname','')} {d.get('lname','')}".strip(),
            "employee_id": d.get("emp_id"),
            "email":       d.get("email_id") or d.get("email"),
            "phone":       d.get("phone") or d.get("mobile"),
            "designation": d.get("Designation", {}).get("designation_name") if isinstance(d.get("Designation"), dict) else d.get("designation"),
            "department":  d.get("Department", {}).get("department_name") if isinstance(d.get("Department"), dict) else d.get("department"),
            "branch":      d.get("Branch", {}).get("branch_name") if isinstance(d.get("Branch"), dict) else d.get("branch"),
            "date_of_joining": (d.get("date_of_joining") or "")[:10],
            "role":        d.get("role"),
        }, indent=2)

    @tool
    def get_my_team() -> str:
        """Get the list of employees in the logged-in user's team (direct reports)."""
        ok, data = _api("GET", "/Employee/myTeam", token=token)
        if not ok:
            return f"Error: {data.get('message', 'Could not fetch team')}"
        rows = _rows(data)
        if not rows:
            return "No team members found."
        result = [{"name": f"{r.get('fname','')} {r.get('lname','')}".strip(),
                   "designation": r.get("Designation", {}).get("designation_name", "") if isinstance(r.get("Designation"), dict) else "",
                   "email": r.get("email_id") or r.get("email", "")} for r in rows]
        return json.dumps(result, indent=2)

    @tool
    def get_special_events() -> str:
        """Get upcoming birthdays and work anniversaries for team members this week/month."""
        ok, data = _api("GET", "/Employee/special_events", token=token)
        if not ok:
            return f"Error: {data.get('message', 'Could not fetch events')}"
        return json.dumps(data, indent=2, default=str)

    # ── LEAVE ───────────────────────────────────────────────────────────────

    @tool
    def get_leave_balance() -> str:
        """
        Get the employee's current leave balance for all leave types.
        Returns leave_id, leave_type, available days, used days, total days.
        Always call this before apply_leave to get the correct leave_id.
        """
        ok, data = _api("GET", "/Leave/leave-balance", token=token)
        if not ok:
            return f"Error: {data.get('message', 'Could not fetch leave balance')}"
        records = _rows(data)
        if not records:
            return "No leave allocations found."
        result = []
        for r in records:
            lt = r.get("LeaveType", {})
            result.append({
                "leave_id":  r.get("leave_id"),
                "leave_type": lt.get("leave_type_name", r.get("leave_type", "Unknown")),
                "available": r.get("available_leaves_count", 0),
                "used":      r.get("utilized_leaves_count", 0),
                "total":     r.get("total_leaves_count", 0),
            })
        return json.dumps(result, indent=2)

    @tool
    def get_my_leaves(status: str = "") -> str:
        """
        Get the employee's leave request history.
        Args:
            status: Optional — "Inprogress", "Approved", "Rejected", "Cancelled". Empty = all.
        """
        params = {"leave_status": status} if status else {}
        ok, data = _api("GET", "/Leave/list", token=token, params=params)
        if not ok:
            return f"Error: {data.get('message', 'Could not fetch leaves')}"
        rows = _rows(data)
        if not rows:
            return "No leave requests found."
        result = []
        for r in rows[:10]:
            result.append({
                "leave_request_id": r.get("leave_request_id"),
                "type":   (r.get("LeaveType") or {}).get("leave_type_name", r.get("leave_type", "")),
                "from":   (r.get("from_date") or "")[:10],
                "to":     (r.get("to_date") or "")[:10],
                "days":   r.get("requested_days"),
                "status": r.get("leave_status"),
                "reason": r.get("reason", ""),
            })
        return json.dumps(result, indent=2)

    @tool
    def apply_leave(leave_id: int, from_date: str, to_date: str, reason: str) -> str:
        """
        Apply for leave. MUST call get_leave_balance first to get the correct leave_id.
        Args:
            leave_id:  leave_id from get_leave_balance for the requested leave type.
            from_date: YYYY-MM-DD
            to_date:   YYYY-MM-DD
            reason:    Reason for leave.
        """
        ok, data = _api("POST", "/Leave/create", token=token,
                        json={"leave_id": leave_id, "from_date": from_date, "to_date": to_date, "reason": reason})
        if ok:
            return json.dumps({"success": True, "status": "Pending Approval", "from_date": from_date, "to_date": to_date})
        return json.dumps({"success": False, "error": data.get("error") or data.get("message", "Failed")})

    @tool
    def cancel_leave(leave_request_id: int) -> str:
        """
        Cancel a pending leave request.
        Args:
            leave_request_id: The leave_request_id from get_my_leaves.
        """
        ok, data = _api("PATCH", f"/Leave/update/{leave_request_id}", token=token,
                        json={"leave_status": "Cancelled"})
        if ok:
            return json.dumps({"success": True, "message": "Leave cancelled successfully."})
        return json.dumps({"success": False, "error": data.get("message", "Could not cancel leave")})

    # ── ATTENDANCE ──────────────────────────────────────────────────────────

    @tool
    def get_attendance_summary() -> str:
        """Get the employee's attendance summary for the current month (present, absent, late, half-day counts)."""
        now = datetime.now()
        ok, data = _api("GET", "/Attendance/summary", token=token,
                        params={"month": now.month, "year": now.year})
        if not ok:
            return f"Error: {data.get('message', 'Could not fetch attendance')}"
        d = data
        if isinstance(d, list):
            d = d[0] if d else {}
        elif isinstance(d, dict):
            inner = d.get("data") or d.get("rows")
            if isinstance(inner, list):
                d = inner[0] if inner else d
            elif isinstance(inner, dict):
                d = inner
        return json.dumps(d, indent=2, default=str)

    @tool
    def get_swipe_info() -> str:
        """Get the employee's check-in and check-out swipe records for today or recent days."""
        ok, data = _api("GET", "/Attendance/swipeinfo", token=token)
        if not ok:
            return f"Error: {data.get('message', 'Could not fetch swipe info')}"
        rows = _rows(data)
        if not rows:
            return "No swipe records found."
        result = []
        for r in rows[:5]:
            result.append({
                "date":       (r.get("date") or r.get("attendance_date") or "")[:10],
                "check_in":   r.get("check_in") or r.get("in_time"),
                "check_out":  r.get("check_out") or r.get("out_time"),
                "work_hours": r.get("work_hours") or r.get("total_hours"),
                "status":     r.get("status") or r.get("attendance_status"),
            })
        return json.dumps(result, indent=2)

    @tool
    def checkin_checkout(action: str, work_from: str = "office") -> str:
        """
        Mark attendance check-in or check-out.
        Args:
            action:    "in" to check in, "out" to check out.
            work_from: "office" or "home". Default is "office".
        """
        ok, data = _api("POST", "/Attendance/checkinout", token=token,
                        json={"type": action, "work_from": work_from})
        if ok:
            label = "Checked in" if action == "in" else "Checked out"
            return json.dumps({"success": True, "message": f"{label} successfully from {work_from}."})
        return json.dumps({"success": False, "error": data.get("message", "Failed to mark attendance")})

    # ── PAYSLIP ─────────────────────────────────────────────────────────────

    @tool
    def get_payslips() -> str:
        """Get the employee's recent payslips with month, year, net salary, and status."""
        ok, data = _api("GET", "/Payslip/list", token=token)
        if not ok:
            return f"Error: {data.get('message', 'Could not fetch payslips')}"
        rows = _rows(data)
        if not rows:
            return "No payslips found."
        result = []
        for r in rows[:5]:
            result.append({
                "month":      r.get("month") or r.get("payroll_month", ""),
                "year":       r.get("year") or r.get("payroll_year", ""),
                "net_salary": r.get("net_salary") or r.get("net_pay") or r.get("total_earnings"),
                "gross":      r.get("gross_salary") or r.get("gross_pay"),
                "status":     r.get("status") or r.get("payslip_status", ""),
            })
        return json.dumps(result, indent=2)

    # ── WORK FROM HOME ──────────────────────────────────────────────────────

    @tool
    def get_wfh_list() -> str:
        """Get the employee's Work From Home / On-Duty request history."""
        ok, data = _api("GET", "/OnDuty/list", token=token)
        if not ok:
            return f"Error: {data.get('message', 'Could not fetch WFH list')}"
        rows = _rows(data)
        if not rows:
            return "No WFH/On-Duty requests found."
        result = []
        for r in rows[:10]:
            result.append({
                "from":   (r.get("from_date") or "")[:10],
                "to":     (r.get("to_date") or "")[:10],
                "days":   r.get("requested_days"),
                "reason": r.get("reason", ""),
                "status": r.get("status") or r.get("wfh_status", ""),
            })
        return json.dumps(result, indent=2)

    @tool
    def apply_wfh(from_date: str, to_date: str, reason: str) -> str:
        """
        Apply for Work From Home (WFH) or On-Duty.
        Args:
            from_date: YYYY-MM-DD
            to_date:   YYYY-MM-DD
            reason:    Reason for WFH.
        """
        ok, data = _api("POST", "/OnDuty/create", token=token,
                        json={"from_date": from_date, "to_date": to_date, "reason": reason})
        if ok:
            return json.dumps({"success": True, "message": "WFH request submitted. Pending approval.", "from_date": from_date, "to_date": to_date})
        return json.dumps({"success": False, "error": data.get("message", "Failed to submit WFH request")})

    # ── CLAIMS ──────────────────────────────────────────────────────────────

    @tool
    def get_claim_types() -> str:
        """Get all available claim types with their IDs. Call this before submit_claim to find the correct claim_type_id."""
        ok, data = _api("GET", "/ClaimType/list", token=token)
        if not ok:
            return f"Error: {data.get('message', 'Could not fetch claim types')}"
        rows = _rows(data)
        if not rows:
            return "No claim types found."
        return json.dumps([{"claim_type_id": r.get("claim_type_id"), "name": r.get("claim_type_name")} for r in rows], indent=2)

    @tool
    def get_my_claims() -> str:
        """Get the employee's expense claim history with amount and status."""
        ok, data = _api("GET", "/Claim/list", token=token)
        if not ok:
            return f"Error: {data.get('message', 'Could not fetch claims')}"
        rows = _rows(data)
        if not rows:
            return "No claims found."
        result = []
        for r in rows[:10]:
            result.append({
                "claim_id":   r.get("claim_id"),
                "type":       (r.get("ClaimType") or {}).get("claim_type_name", r.get("claim_type", "")),
                "amount":     r.get("amount"),
                "status":     r.get("status"),
                "date":       (r.get("claimed_date") or "")[:10],
                "comments":   r.get("comments", ""),
            })
        return json.dumps(result, indent=2)

    @tool
    def submit_claim(claim_type_id: int, amount: float, comments: str = "") -> str:
        """
        Submit an expense claim. MUST call get_claim_types first to get the correct claim_type_id.
        Args:
            claim_type_id: From get_claim_types.
            amount:        Amount in the company currency.
            comments:      Optional description/reason.
        """
        ok, data = _api("POST", "/Claim/create", token=token,
                        json={"claim_type_id": claim_type_id, "amount": amount, "comments": comments})
        if ok:
            return json.dumps({"success": True, "message": f"Claim of {amount} submitted successfully. Status: Pending."})
        return json.dumps({"success": False, "error": data.get("message", "Failed to submit claim")})

    # ── NOTIFICATIONS ───────────────────────────────────────────────────────

    @tool
    def get_notifications() -> str:
        """Get the employee's recent unread notifications from the HRMS system."""
        ok, data = _api("GET", "/Notification/list", token=token)
        if not ok:
            return f"Error: {data.get('message', 'Could not fetch notifications')}"
        rows = _rows(data)
        if not rows:
            return "No notifications found."
        result = []
        for r in rows[:10]:
            result.append({
                "message":  r.get("message") or r.get("notification_message", ""),
                "type":     r.get("type") or r.get("notification_type", ""),
                "is_read":  r.get("is_read", False),
                "date":     (r.get("createdAt") or r.get("created_at") or "")[:10],
            })
        return json.dumps(result, indent=2)

    # ── PUBLIC HOLIDAYS ─────────────────────────────────────────────────────

    @tool
    def get_public_holidays(year: int = 0, month: int = 0) -> str:
        """
        Get public holidays for the company.

        Args:
            year:  The year to fetch holidays for (e.g. 2026). Pass 0 to use current year.
            month: The starting month (1-12). Pass 0 to fetch the full year.

        If the result is empty, try calling again with a different year (e.g. the previous year).
        """
        now = datetime.now()
        use_year  = year  if year  > 0 else now.year
        use_month = month if month > 0 else 1
        params = {
            "from_year":  use_year,
            "from_month": f"{use_month:02d}",
            "to_year":    use_year,
            "to_month":   "12",
        }
        ok, data = _api("GET", "/PublicHoliday/v2/list", token=token, params=params)
        if not ok:
            return f"Error: {data.get('message', 'Could not fetch holidays')}"
        rows = _rows(data)
        if not rows:
            return f"No holidays found for {use_year}. Try calling again with a different year."
        result = []
        for r in rows:
            result.append({
                "name": r.get("holiday_name") or r.get("name", ""),
                "date": (r.get("holiday_date") or r.get("date") or "")[:10],
            })
        return json.dumps(result, indent=2)

    # ── SHIFT ───────────────────────────────────────────────────────────────

    @tool
    def get_my_shift() -> str:
        """Get the employee's current assigned shift details (shift name, timings, days)."""
        ok, data = _api("GET", "/Shift/employees-shifts", token=token)
        if not ok:
            return f"Error: {data.get('message', 'Could not fetch shift info')}"
        rows = _rows(data)
        if not rows:
            return "No shift information found."
        r = rows[0] if rows else {}
        shift = r.get("Shift") or r
        return json.dumps({
            "shift_name":  shift.get("shift_name", ""),
            "start_time":  shift.get("start_time", ""),
            "end_time":    shift.get("end_time", ""),
            "working_days": shift.get("working_days", ""),
            "grace_time":  shift.get("grace_time", ""),
        }, indent=2)

    # ── BENEFITS ────────────────────────────────────────────────────────────

    @tool
    def get_my_benefits() -> str:
        """Get the employee's enrolled benefit plans (health insurance, etc.)."""
        ok, data = _api("GET", "/EmployeeBenefit/list", token=token)
        if not ok:
            return f"Error: {data.get('message', 'Could not fetch benefits')}"
        rows = _rows(data)
        if not rows:
            return "No benefit plans found."
        result = []
        for r in rows:
            bp = r.get("BenefitPlan") or {}
            result.append({
                "plan":       bp.get("plan_name", r.get("plan_name", "")),
                "type":       bp.get("benefit_type", r.get("benefit_type", "")),
                "coverage":   bp.get("coverage_amount") or r.get("coverage_amount"),
                "status":     r.get("status", ""),
                "start_date": (r.get("start_date") or "")[:10],
            })
        return json.dumps(result, indent=2)

    # ── TRAVEL REQUESTS ─────────────────────────────────────────────────────

    @tool
    def get_my_travel_requests() -> str:
        """Get the employee's travel requests with destination, dates, and approval status."""
        ok, data = _api("GET", "/TravelRequest/list", token=token)
        if not ok:
            return f"Error: {data.get('message', 'Could not fetch travel requests')}"
        rows = _rows(data)
        if not rows:
            return "No travel requests found."
        result = []
        for r in rows[:10]:
            result.append({
                "destination": r.get("destination", ""),
                "purpose":     r.get("purpose", ""),
                "from_date":   (r.get("from_date") or "")[:10],
                "to_date":     (r.get("to_date") or "")[:10],
                "status":      r.get("status", ""),
            })
        return json.dumps(result, indent=2)

    # ── REGULARIZATION ──────────────────────────────────────────────────────

    @tool
    def get_regularizations() -> str:
        """Get the employee's attendance regularization requests (missed punch corrections)."""
        ok, data = _api("GET", "/Regularization/list", token=token)
        if not ok:
            return f"Error: {data.get('message', 'Could not fetch regularizations')}"
        rows = _rows(data)
        if not rows:
            return "No regularization requests found."
        result = []
        for r in rows[:10]:
            result.append({
                "date":       (r.get("attendance_date") or r.get("date") or "")[:10],
                "in_time":    r.get("in_time", ""),
                "out_time":   r.get("out_time", ""),
                "reason":     r.get("reason", ""),
                "status":     r.get("status", ""),
            })
        return json.dumps(result, indent=2)

    # ── GRIEVANCE ───────────────────────────────────────────────────────────

    @tool
    def get_my_grievances() -> str:
        """Get the employee's submitted grievances and their resolution status."""
        ok, data = _api("GET", "/Grievance/list", token=token)
        if not ok:
            return f"Error: {data.get('message', 'Could not fetch grievances')}"
        rows = _rows(data)
        if not rows:
            return "No grievances found."
        result = []
        for r in rows[:10]:
            result.append({
                "grievance_id": r.get("grievance_id"),
                "title":        r.get("title", ""),
                "category":     r.get("category", ""),
                "priority":     r.get("priority", ""),
                "status":       r.get("status", ""),
                "date":         (r.get("createdAt") or "")[:10],
            })
        return json.dumps(result, indent=2)

    @tool
    def raise_grievance(title: str, description: str, category: str = "General", priority: str = "Medium") -> str:
        """
        Raise a grievance or complaint.
        Args:
            title:       Short title of the grievance.
            description: Full description of the issue.
            category:    e.g. "Harassment", "Payroll", "Policy", "Work Environment", "General".
            priority:    "Low", "Medium", or "High".
        """
        ok, data = _api("POST", "/Grievance/create", token=token,
                        json={"title": title, "description": description, "category": category, "priority": priority})
        if ok:
            return json.dumps({"success": True, "message": "Grievance submitted successfully. HR will review and respond."})
        return json.dumps({"success": False, "error": data.get("message", "Failed to submit grievance")})

    return [
        # Profile & Team
        get_my_profile, get_my_team, get_special_events,
        # Leave
        get_leave_balance, get_my_leaves, apply_leave, cancel_leave,
        # Attendance
        get_attendance_summary, get_swipe_info, checkin_checkout,
        # Payslip
        get_payslips,
        # WFH
        get_wfh_list, apply_wfh,
        # Claims
        get_claim_types, get_my_claims, submit_claim,
        # Info
        get_notifications, get_public_holidays, get_my_shift, get_my_benefits,
        # Travel
        get_my_travel_requests,
        # Regularization
        get_regularizations,
        # Grievance
        get_my_grievances, raise_grievance,
    ]

# ---------------------------------------------------------------------------
# Agent runner
# ---------------------------------------------------------------------------

def _build_system_prompt(employee_name: str) -> str:
    today = datetime.now()
    weekdays = {}
    for i in range(1, 8):
        d = today + timedelta(days=i)
        weekdays[d.strftime("%A")] = d.strftime("%Y-%m-%d")

    return f"""You are an HR assistant for *{employee_name}*, helping them manage their HR tasks through Slack.

Today is {today.strftime("%A, %Y-%m-%d")}.
Upcoming dates: {", ".join(f"{k} = {v}" for k, v in weekdays.items())}

You have 24 tools covering: profile, leave, attendance, payslips, WFH, claims, notifications, holidays, shift, benefits, travel, regularization, grievances.

Key rules:
1. ALWAYS call get_leave_balance before apply_leave to get the leave_id.
2. ALWAYS call get_claim_types before submit_claim to get the claim_type_id.
3. Resolve all relative dates ("tomorrow", "next Monday") to YYYY-MM-DD using today's date.
4. Be concise and professional. Format ALL responses using Slack mrkdwn ONLY:
   - Bold: *text* (single asterisk — NEVER use **double**)
   - Bullet points: • (bullet character — NEVER use - or *)
   - Italic: _text_
   - Example leave balance format:
     *Your Leave Balance*
     • *Privilege Leave* — Available: 3 | Used: 0 | Total: 3
     • *Sick Leave* — Available: 3 | Used: 0 | Total: 3
5. If an action succeeds (leave applied, WFH submitted, etc.), confirm the key details clearly.
6. If an error occurs, explain it in plain language.
7. Never expose raw JSON to the user — always summarise it naturally.
8. If get_public_holidays returns no results for a year, do NOT silently fall back. Tell the user clearly: "No holidays have been added for [year] in the HRMS system yet." Then ask if they want to see a different year's holidays instead. Only fetch another year if the user explicitly asks for it.
9. You have conversation history above — use it to understand follow-up questions like "then the previous year", "what about last year", "cancel that", etc.
10. STRICT SCOPE: You are ONLY an HR assistant. If the user asks anything not related to HR (leaves, attendance, payslips, holidays, claims, profile, WFH, shifts, benefits, travel, grievances), politely decline and say you can only help with HR-related tasks."""


def _load_history(slack_user_id: str, limit: int = 10) -> list:
    """
    Load the last `limit` messages from MongoDB for this user
    and convert them to LangChain message objects for context.
    """
    if _conversations is None:
        return []
    try:
        doc = _conversations.find_one({"slack_user_id": slack_user_id})
        if not doc:
            return []
        recent = doc.get("messages", [])[-limit:]
        history = []
        for m in recent:
            if m["role"] == "user":
                history.append(HumanMessage(content=m["text"]))
            elif m["role"] == "bot":
                history.append(AIMessage(content=m["text"]))
        return history
    except Exception as e:
        log.error("MongoDB load_history failed: %s", e)
        return []


def run_hrms_agent(user_message: str, token: str, employee_name: str, slack_user_id: str = "") -> str:
    """
    Agentic loop:
      1. LLM decides which tool(s) to call
      2. Tool results fed back
      3. LLM may call more tools or produce final answer
      4. Repeats until no more tool calls (max 8 iterations)
    Includes the last 10 messages as conversation history for context.
    """
    log.info("AGENT start  employee=%s  message=%r", employee_name, user_message[:120])

    tools          = create_hrms_tools(token)
    tools_map      = {t.name: t for t in tools}
    llm            = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    # Build messages: system prompt + conversation history + current message
    history  = _load_history(slack_user_id) if slack_user_id else []
    messages = [SystemMessage(content=_build_system_prompt(employee_name))] \
               + history \
               + [HumanMessage(content=user_message)]

    for iteration in range(8):
        log.debug("AGENT iteration %d", iteration + 1)
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            log.info("AGENT done  iterations=%d  employee=%s", iteration + 1, employee_name)
            return response.content or "Done."

        for tc in response.tool_calls:
            log.info("TOOL call  name=%s  args=%s", tc["name"], json.dumps(tc["args"])[:200])
            fn = tools_map.get(tc["name"])
            try:
                result = fn.invoke(tc["args"]) if fn else f"Unknown tool: {tc['name']}"
                log.info("TOOL result  name=%s  preview=%s", tc["name"], str(result)[:150])
            except Exception as e:
                result = f"Tool error: {str(e)}"
                log.error("TOOL error  name=%s  error=%s", tc["name"], e, exc_info=True)
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    log.warning("AGENT max iterations reached  employee=%s", employee_name)
    return "I had trouble completing your request. Please try again."

# ---------------------------------------------------------------------------
# Slack helpers
# ---------------------------------------------------------------------------

def _get_user_display(user_id: str) -> str:
    try:
        info    = app.client.users_info(user=user_id)
        profile = info["user"]["profile"]
        return profile.get("real_name") or profile.get("display_name") or user_id
    except Exception:
        return user_id


HELP_TEXT = """\
*HR Bot — What I can do*

*Setup*
• `login <username> <password>` — connect your HRMS account
• `logout` — disconnect

*Just ask naturally, for example:*

*Profile & Team*
  _What's my profile?_ · _Who's in my team?_ · _Any birthdays this week?_

*Leave*
  _What's my leave balance?_ · _Show my leaves_ · _Apply sick leave from Monday to Wednesday_
  _Cancel my leave request_

*Attendance*
  _My attendance this month_ · _What time did I check in today?_ · _Check me in_ / _Check me out_

*Payslip*
  _Show my payslips_ · _What's my net salary this month?_

*Work From Home*
  _Apply WFH for tomorrow_ · _Show my WFH requests_

*Claims*
  _Show my claims_ · _Submit a travel claim for ₹500_

*Other*
  _My notifications_ · _Upcoming holidays_ · _My shift timings_ · _My benefits_
  _My travel requests_ · _My grievances_ · _Raise a grievance_

• `help` — show this message
"""

# ---------------------------------------------------------------------------
# Slack event handlers
# ---------------------------------------------------------------------------

@app.event("message")
def handle_dm(message, say):
    if message.get("subtype") or message.get("bot_id"):
        return
    if message.get("channel_type") != "im":
        return

    user_id = message.get("user", "")
    text    = message.get("text", "").strip()

    if not text:
        say(HELP_TEXT)
        return

    log.info("MSG  user=%s  text=%r", user_id, text[:120])

    # ── LOGIN ──────────────────────────────────────────────────────────────
    # Accepts: "login USER PASS"  or  "login USER password: PASS"  or  "login USER password:PASS"
    m = re.match(r"^login\s+(\S+)\s+(?:password:\s*)?(\S+)$", text, re.IGNORECASE)
    if m:
        username, password = m.group(1), m.group(2)
        say(":hourglass_flowing_sand: Authenticating...")
        ok, result = hrms_login(username, password)
        if ok:
            _set_session(user_id, result["token"], result["emp_id"], result["name"])
            log.info("SESSION set  user=%s  name=%s", user_id, result["name"])
            say(
                f":white_check_mark: Welcome *{result['name']}*! Connected to HRMS.\n"
                "Just ask me anything in plain English. Type `help` to see all examples."
            )
        else:
            say(f":x: Login failed: *{result['message']}*\nUsage: `login <username> <password>`")
        return

    # ── LOGOUT ─────────────────────────────────────────────────────────────
    if re.match(r"^logout$", text, re.IGNORECASE):
        if _get_session(user_id):
            _clear_session(user_id)
            log.info("SESSION cleared  user=%s", user_id)
            say(":wave: Logged out. Use `login <username> <password>` to reconnect.")
        else:
            say("You're not currently logged in.")
        return

    # ── HELP ───────────────────────────────────────────────────────────────
    if re.match(r"^help$", text, re.IGNORECASE):
        say(HELP_TEXT)
        return

    # ── Require login ──────────────────────────────────────────────────────
    session = _get_session(user_id)
    if not session:
        log.warning("UNAUTH message  user=%s  text=%r", user_id, text[:80])
        say(":lock: You're not logged in. Use: `login <username> <password>`")
        return

    say(":hourglass_flowing_sand: On it...")

    # Save user message to MongoDB
    save_message(user_id, "user", text,
                 employee_name=session.get("name", ""),
                 emp_id=session.get("emp_id", ""))

    try:
        reply = run_hrms_agent(
            user_message=text,
            token=session["token"],
            employee_name=session.get("name", "Employee"),
            slack_user_id=user_id,
        )
        if "Unauthorized" in reply or "401" in reply:
            log.warning("SESSION expired  user=%s", user_id)
            _clear_session(user_id)
            say(":lock: Session expired. Please `login` again.")
        else:
            # Save bot reply to MongoDB
            save_message(user_id, "bot", reply,
                         employee_name=session.get("name", ""),
                         emp_id=session.get("emp_id", ""))
            say(reply)
    except Exception as e:
        log.error("AGENT exception  user=%s  error=%s\n%s", user_id, e, traceback.format_exc())
        say(f":x: Something went wrong: {str(e)}")


@app.event("app_mention")
def handle_mention(event, say):
    say(f"<@{event['user']}> Please DM me directly — your HR info stays private!")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    log.info("=" * 60)
    log.info("HRMS AI Agent starting")
    log.info("API base : %s", HRMS_BASE_URL)
    log.info("Log file : %s", LOG_FILE)
    log.info("=" * 60)
    handler = SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"])
    handler.start()
