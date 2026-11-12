@echo off
TITLE PressureIQ - Rajasthan Royals Analytics Dashboard
cd /d "%~dp0"
echo =====================================================================
echo Starting PressureIQ Streamlit Dashboard...
echo Local URL: http://localhost:8501
echo =====================================================================
.venv\Scripts\python.exe -m streamlit run app\app.py --browser.gatherUsageStats false
pause

