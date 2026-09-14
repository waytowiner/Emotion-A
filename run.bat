@echo off
cd /d "%~dp0"
set PYTHON=C:\Python312\python.exe

rem Data update (fetch -> report -> dashboard, 20-40 min) moved to the dashboard
rem UI "Update Data" button; the weekday 18:00 auto-update job still runs it.

for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000 ^| findstr LISTENING') do taskkill /F /PID %%a >nul 2>&1
start http://127.0.0.1:8000/static/
%PYTHON% server.py
if errorlevel 1 pause
