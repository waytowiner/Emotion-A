@echo off
chcp 65001 >nul
title 板块情绪雷达
set PYTHON=C:\Python312\python.exe

echo.
echo 正在计算今日板块情绪...
echo.

echo [1/3] 生成今日报告...
%PYTHON% main.py
if errorlevel 1 goto ERROR

echo.
echo [2/3] 导出看板数据...
%PYTHON% scripts\dump_dashboard_data.py
if errorlevel 1 goto ERROR

echo.
echo [3/3] 启动看板...
REM 清除占用8000端口的旧进程
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000 ^| findstr LISTENING') do (
    taskkill /F /PID %%a >nul 2>&1
)
start http://127.0.0.1:8000/static/
%PYTHON% server.py
exit /b

:ERROR
echo.
echo 出错了！请把上面红色错误信息截图发给我。
pause
