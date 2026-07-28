"""
Mock HR data and action implementations.
In a real system these would call an HRIS API or database.
"""
import random
from datetime import datetime

# Mock employee database
EMPLOYEES = {
    "E001": {
        "name": "Alice Johnson",
        "department": "Engineering",
        "role": "Senior Engineer",
        "email": "alice@company.com",
        "manager": "David Lee",
        "leave_balance": 15,
        "start_date": "2021-03-01",
    },
    "E002": {
        "name": "Bob Smith",
        "department": "Marketing",
        "role": "Marketing Manager",
        "email": "bob@company.com",
        "manager": "Sarah Chen",
        "leave_balance": 10,
        "start_date": "2020-06-15",
    },
    "E003": {
        "name": "Carol Davis",
        "department": "HR",
        "role": "HR Specialist",
        "email": "carol@company.com",
        "manager": "Michael Brown",
        "leave_balance": 12,
        "start_date": "2022-01-10",
    },
    "E004": {
        "name": "Abhizgn",
        "department": "Engineering",
        "role": "Engineer",
        "email": "abhizgn@photonxtech.com",
        "manager": "David Lee",
        "leave_balance": 20,
        "start_date": "2024-01-01",
    },
}


def process_leave_request(
    employee_name: str, leave_type: str, start_date: str, end_date: str
) -> dict:
    """
    Process a leave request using the employee's Slack name directly.
    No database lookup required — the name from Slack is used as-is.
    """
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end   = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError:
        return {"success": False, "error": "Invalid date format. Use YYYY-MM-DD."}

    if end < start:
        return {"success": False, "error": "End date cannot be before start date."}

    days_requested = (end - start).days + 1
    ref_id = f"LR-{random.randint(10000, 99999)}"

    return {
        "success": True,
        "reference_id": ref_id,
        "employee_name": employee_name,
        "leave_type": leave_type,
        "start_date": start_date,
        "end_date": end_date,
        "days_approved": days_requested,
        "status": "Approved and recorded in HRIS",
    }


def lookup_employee(query: str) -> dict:
    """
    Look up employee info. Returns the Slack name directly since
    identity comes from Slack, not the internal DB.
    Falls back to mock DB for demo data like department/role.
    """
    # Try mock DB for richer info (department, role, etc.)
    for eid, emp in EMPLOYEES.items():
        if query.lower() in emp["name"].lower() or query.upper() == eid:
            return {"found": True, "employee_id": eid, **emp}

    # If not in mock DB, return what we know from Slack name alone
    return {
        "found": True,
        "employee_name": query,
        "note": "Employee identified via Slack. Full HR profile not in demo database.",
    }
