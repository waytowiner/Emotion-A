@echo off
chcp 65001 >nul
title 板块情绪雷达
set PYTHON=C:\Python312\python.exe

echo.
echo 正在计算今日板块情绪...
echo.

echo [1/3] 生成今日报告...
%PYTHON% main.py
if errorlevel 1 echo 报告生成失败，请关闭本窗口后重试 && pause && exit /b

echo.
echo [2/3] 导出看板数据...
%PYTHON% scripts\dump_dashboard_data.py
if errorlevel 1 echo 看板导出失败 && pause && exit /b

echo.
echo [3/3] 启动看板...
start http://127.0.0.1:8000/static/
%PYTHON% server.py
