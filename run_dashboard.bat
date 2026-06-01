@echo off
REM Launch the Streamlit audit trail dashboard using the isolated .venv
REM Opens at http://localhost:8501

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] .venv not found. Run setup.bat first.
    exit /b 1
)

echo Starting dashboard at http://localhost:8501 ...
.venv\Scripts\streamlit run dashboard\app.py
