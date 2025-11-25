@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul || (
    echo [!] Python launcher (py.exe) not found. Install Python 3.10 from https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [1/3] Using Python version:
py -3.10 -V || (
    echo [!] Python 3.10 not found. Install it with "winget install Python.Python.3.10"
    pause
    exit /b 1
)

echo [2/3] Upgrading pip...
py -3.10 -m pip install --upgrade pip || goto :error

echo [3/3] Installing project requirements (safe to rerun)...
py -3.10 -m pip install -r requirements.txt || goto :error

echo.
echo [✓] Dependencies installed. Starting the Flask app on http://127.0.0.1:8000 ...
py -3.10 app.py

endlocal
exit /b 0

:error
echo.
echo [!] Setup failed. See the error above, resolve it, then run run_app.bat again.
pause
exit /b 1

