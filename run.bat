@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_EXE=D:\Develop\Anaconda3\python.exe"

if exist "%PYTHON_EXE%" (
    "%PYTHON_EXE%" gui_app.py
) else (
    python gui_app.py
)

if errorlevel 1 (
    echo.
    echo Huaita GUI failed to start. See the error above.
    pause
)

endlocal
