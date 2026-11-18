@echo off
cd /d "%~dp0.."
echo ============================================================================
echo Starting PathSnap Evaluator UI Server (Laptop 2)
echo Open http://127.0.0.1:8501 in your browser
echo ============================================================================
python -m uvicorn ui.app:ui_app --host 0.0.0.0 --port 8501
pause
