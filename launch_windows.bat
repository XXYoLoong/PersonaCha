@echo off
setlocal

cd /d %~dp0

if not exist .venv (
  echo [PersonaCha] Creating virtual environment...
  python -m venv .venv
)

call .venv\Scripts\activate

echo [PersonaCha] Installing dependencies...
python -m pip install -U pip
pip install -r requirements.txt

echo [PersonaCha] Launching web UI...
python app.py

endlocal
