@echo off
chcp 65001 >nul
set PYTHON=C:\Python312\python.exe
title Emotion-A Board Sentiment Radar

:MENU
cls
echo ============================================================
echo            Emotion-A  Board Sentiment Radar
echo ============================================================
echo.
echo   [1] Update ALL (fetch stocks + report + dashboard)
echo   [2] Fetch stock history only (incremental, 20-40 min)
echo   [3] Generate daily sentiment report
echo   [4] Dump dashboard data
echo   [5] Start web dashboard server (http://127.0.0.1:8000)
echo   [6] Run backtest
echo   [0] Exit
echo.
set /p CHOICE=Enter choice:

if "%CHOICE%"=="1" goto ALL
if "%CHOICE%"=="2" goto FETCH
if "%CHOICE%"=="3" goto REPORT
if "%CHOICE%"=="4" goto DUMP
if "%CHOICE%"=="5" goto SERVER
if "%CHOICE%"=="6" goto BACKTEST
if "%CHOICE%"=="0" exit /b
goto MENU

:ALL
echo.
echo [1/3] Fetching stock history (incremental)...
%PYTHON% scripts\fetch_stock_history.py --update
echo.
echo [2/3] Generating daily report...
%PYTHON% main.py
echo.
echo [3/3] Dumping dashboard data...
%PYTHON% scripts\dump_dashboard_data.py
echo.
echo ALL DONE. Press any key to return to menu...
pause >nul
goto MENU

:FETCH
echo.
echo Fetching stock history (incremental, 20-40 min)...
%PYTHON% scripts\fetch_stock_history.py --update
echo.
echo Done. Press any key to return to menu...
pause >nul
goto MENU

:REPORT
echo.
echo Generating daily sentiment report...
%PYTHON% main.py
echo.
echo Done. Press any key to return to menu...
pause >nul
goto MENU

:DUMP
echo.
echo Dumping dashboard data...
%PYTHON% scripts\dump_dashboard_data.py
echo.
echo Done. Press any key to return to menu...
pause >nul
goto MENU

:SERVER
echo.
echo Starting dashboard server at http://127.0.0.1:8000
echo Browser opening automatically...
start http://127.0.0.1:8000/static/
%PYTHON% server.py
goto MENU

:BACKTEST
echo.
echo Running backtest...
%PYTHON% backtest.py
echo.
echo Done. Report saved to report\ directory.
pause >nul
goto MENU
