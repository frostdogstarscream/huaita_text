@echo off
setlocal
cd /d "%~dp0"

set "VENV_PYTHON=%~dp0.venv\Scripts\python.exe"
set "MODELSCOPE_PYTHON=%~dp0.venv-modelscope\Scripts\python.exe"
set "PYTHON_EXE=D:\Develop\Anaconda3\python.exe"

if exist "%VENV_PYTHON%" (
    "%VENV_PYTHON%" gui_app.py
) else if exist "%MODELSCOPE_PYTHON%" (
    "%MODELSCOPE_PYTHON%" gui_app.py
) else if exist "%PYTHON_EXE%" (
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
