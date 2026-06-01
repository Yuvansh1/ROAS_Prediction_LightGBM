@echo off
REM One-time setup: creates .venv and installs all dependencies
REM Run this once after cloning the repo.

echo Creating virtual environment...
python -m venv .venv

echo Installing dependencies...
.venv\Scripts\pip install -q -e ".[dev]"

echo.
echo Setup complete. Run the pipeline with:
echo   run.bat
echo.
echo Launch the dashboard with:
echo   run_dashboard.bat
echo.
echo Run tests with:
echo   .venv\Scripts\python -m pytest tests\ -v
