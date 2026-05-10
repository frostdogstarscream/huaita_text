@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_EXE=D:\Develop\Anaconda3\python.exe"

if exist "%PYTHON_EXE%" (
    "%PYTHON_EXE%" run_app.py
) else (
    python run_app.py
)

endlocal
