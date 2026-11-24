@echo off
echo Starting IndicSent-M Backend and Frontend...
cd /d "%~dp0"

if not exist ".venv" (
    echo [NOTE] Python virtual environment (.venv) not found.
    echo Creating virtual environment and installing backend requirements...
    python -m venv .venv
    call .venv\Scripts\activate.bat
    pip install -r backend\requirements.txt
)

echo Launching Backend on http://localhost:8000 ...
start "IndicSent-M Backend" cmd /k "call .venv\Scripts\activate.bat && python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000"

echo Launching Frontend on http://localhost:5173 ...
start "IndicSent-M Frontend" cmd /k "cd frontend && npm run dev"

echo Both services launched in separate windows.
