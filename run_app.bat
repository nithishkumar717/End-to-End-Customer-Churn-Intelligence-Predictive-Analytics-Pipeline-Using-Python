@echo off
title Customer Churn Intelligence & Predictive Analytics Platform
echo ======================================================================
echo Starting Customer Churn Intelligence & Predictive Analytics Platform...
echo ======================================================================
cd /d "%~dp0"
call .venv\Scripts\activate
echo Virtual environment activated.
echo Launching Streamlit web application on localhost:8501...
streamlit run app.py
pause
