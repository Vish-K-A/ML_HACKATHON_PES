@echo off
REM Laptop 2: serve the evaluator UI on http://localhost:5500 and open it
cd /d "%~dp0ui"
start "" http://localhost:5500/index.html
python -m http.server 5500
