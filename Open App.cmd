@echo off
cd /d "%~dp0"
set "PYTHONPATH=%~dp0src"
if not exist "%~dp0.venv\Scripts\pythonw.exe" (
  echo Python environment missing. See README.md for setup.
  pause
  exit /b 1
)
start "" "%~dp0.venv\Scripts\pythonw.exe" -m motionlab.gui
