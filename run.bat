@echo off
REM Run the ROAS prediction pipeline using the isolated .venv
REM Usage: run.bat [--config path/to/config.yaml]

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] .venv not found. Run setup.bat first.
    exit /b 1
)

.venv\Scripts\python main.py %*
