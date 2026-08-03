@echo off
cd /d "%~dp0"
set PYTHON=C:\Python312\python.exe

echo Loading...
echo.

%PYTHON% main.py
if errorlevel 1 pause && exit /b

%PYTHON% scripts\dump_dashboard_data.py
if errorlevel 1 pause && exit /b

for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000 ^| findstr LISTENING') do taskkill /F /PID %%a >nul 2>&1
start http://127.0.0.1:8000/static/
%PYTHON% server.py
