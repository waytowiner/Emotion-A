@echo off
chcp 65001 >nul
title 板块情绪雷达 - Emotion-A

:menu
cls
echo ========================================
echo         板块情绪雷达 Emotion-A
echo ========================================
echo.
echo   [1] 一键更新全部 (增量抓取 + 报告 + 看板)
echo   [2] 增量更新个股数据 (20-40 分钟)
echo   [3] 生成今日情绪报告
echo   [4] 导出看板数据
echo   [5] 启动 Web 看板服务器
echo   [6] 运行回测验证
echo   [0] 退出
echo.
set /p choice=请输入选项:

if "%choice%"=="1" goto update_all
if "%choice%"=="2" goto update_stocks
if "%choice%"=="3" goto gen_report
if "%choice%"=="4" goto dump_dashboard
if "%choice%"=="5" goto start_server
if "%choice%"=="6" goto run_backtest
if "%choice%"=="0" exit /b
goto menu

:update_all
echo.
echo [1/3] 增量更新个股数据...
python scripts\fetch_stock_history.py --update
if errorlevel 1 (
    echo [警告] 个股数据更新失败，继续后续步骤...
    timeout /t 3 >nul
)
echo.
echo [2/3] 生成情绪报告...
python main.py
if errorlevel 1 echo [失败] 报告生成出错
echo.
echo [3/3] 导出看板数据...
python scripts\dump_dashboard_data.py
if errorlevel 1 echo [失败] 看板导出出错
echo.
echo 全部完成！按任意键返回菜单...
pause >nul
goto menu

:update_stocks
echo.
echo 正在增量更新个股数据（预计 20-40 分钟）...
python scripts\fetch_stock_history.py --update
echo.
echo 完成！按任意键返回菜单...
pause >nul
goto menu

:gen_report
echo.
echo 正在生成情绪报告...
python main.py
echo.
echo 完成！按任意键返回菜单...
pause >nul
goto menu

:dump_dashboard
echo.
echo 正在导出看板数据...
python scripts\dump_dashboard_data.py
echo.
echo 完成！按任意键返回菜单...
pause >nul
goto menu

:start_server
echo.
echo 正在启动看板服务器 http://127.0.0.1:8000 ...
echo 浏览器打开: http://127.0.0.1:8000/static/
echo 按 Ctrl+C 停止服务器
echo.
start http://127.0.0.1:8000/static/
python server.py
goto menu

:run_backtest
echo.
echo 正在运行回测...
python backtest.py
echo.
echo 完成！报告已保存到 report/ 目录。按任意键返回菜单...
pause >nul
goto menu
