@echo off
cd /d "%~dp0.."
echo ============================================================================
echo Starting PathSnap API Server (Laptop 1: Model, HMM/Viterbi, Evaluation)
echo Listening on http://0.0.0.0:8000
echo ============================================================================
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
pause
