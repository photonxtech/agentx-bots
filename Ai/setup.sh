#!/usr/bin/env bash
set -e
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "== Setting up backend =="
cd "$ROOT_DIR/backend"
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
playwright install chromium
[ -f .env ] || cp .env.example .env
deactivate

echo "== Setting up frontend =="
cd "$ROOT_DIR/frontend"
npm install
[ -f .env ] || cp .env.example .env

echo ""
echo "Setup complete."
echo "1. Edit Ai/backend/.env with your GROQ_API_KEY, JWT_SECRET, ADMIN_EMAIL, ADMIN_PASSWORD."
echo "2. Run ./backend/run.sh in one terminal."
echo "3. Run ./frontend/run.sh in another terminal."
echo "4. Open http://localhost:5173/admin to log in, and http://localhost:5173/ for the demo widget."
