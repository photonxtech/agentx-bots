#!/usr/bin/env python3
"""
Diagnostic script to test if the server can start and respond.
Run this instead of uvicorn to see detailed startup information.
"""
import sys
import os

print("="*60)
print("SERVER DIAGNOSTIC TEST")
print("="*60)
print(f"Python version: {sys.version}")
print(f"Working directory: {os.getcwd()}")
print()

print("Step 1: Testing imports...")
try:
    from backend import main
    print("✓ backend.main imported successfully")
except Exception as e:
    print(f"✗ FAILED to import backend.main: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("Step 2: Checking FastAPI app...")
try:
    app = main.app
    print(f"✓ FastAPI app created: {app.title}")
    print(f"  Routes registered: {len(app.routes)}")
    for route in list(app.routes)[:10]:  # Show first 10 routes
        if hasattr(route, 'path'):
            print(f"    - {route.path}")
except Exception as e:
    print(f"✗ FAILED to access FastAPI app: {e}")
    sys.exit(1)

print()
print("Step 3: Testing database connection...")
try:
    from backend.database import engine
    from sqlalchemy import text
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1"))
        print("✓ Database connection successful")
except Exception as e:
    print(f"✗ Database connection failed: {e}")
    print("  Make sure PostgreSQL is running and credentials in .env are correct")

print()
print("Step 4: Testing LLM configuration...")
try:
    from backend.services.deepeval_llm import build_generator_llm
    llm = build_generator_llm()
    print(f"✓ LLM configured: {llm.get_model_name()}")
except Exception as e:
    print(f"✗ LLM configuration failed: {e}")
    import traceback
    traceback.print_exc()

print()
print("="*60)
print("DIAGNOSTIC COMPLETE")
print("="*60)
print()
print("If all checks passed, start the server with:")
print("  python -m uvicorn backend.main:app --reload --port 8000")
print()
print("Then open: http://127.0.0.1:8000")
