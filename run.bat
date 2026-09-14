@echo off
cd /d "%~dp0"

if exist "C:\Python312\python.exe" (
    set PYTHON=C:\Python312\python.exe
) else (
    set PYTHON=python
)

rem 清理占用 8000 端口的残留进程
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000 ^| findstr LISTENING') do taskkill /F /PID %%a >nul 2>&1

rem 延迟 2 秒后打开浏览器，等待 server 完成数据加载与端口监听
start "" cmd /c "timeout /t 2 /nobreak >nul & start http://127.0.0.1:8000/"
%PYTHON% server.py
if errorlevel 1 pause
