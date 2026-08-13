"""Every module in this project does `import config`, `from rag import ...`
etc. assuming `backend/` (this file's parent) is on sys.path — true when the
app runs via `uvicorn api.main:app` from backend/, but not automatically true
for pytest run from elsewhere. Insert it once, before any test module imports.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
