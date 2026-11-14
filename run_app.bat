@echo off
title Customer Churn Intelligence & Predictive Analytics Platform
echo ======================================================================
echo Starting Customer Churn Intelligence & Predictive Analytics Platform...
echo ======================================================================
cd /d "%~dp0"

if exist ".venv\Scripts\streamlit.exe" (
    echo Using dedicated virtual environment (.venv)...
    ".venv\Scripts\streamlit.exe" run app.py
) else (
    echo Launching with global Python environment...
    python -m streamlit run app.py
)

pause
