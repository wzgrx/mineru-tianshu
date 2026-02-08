@echo off
REM Tianshu - Docker Quick Setup for Windows 11 (WSL2 Edition)
REM Optimized for RTX 5090 & Ubuntu-24.04

setlocal enabledelayedexpansion

echo ===================================================
echo    Tianshu Deployment Script (WSL2 Bridge Mode)
echo ===================================================
echo.

REM 1. 切换到项目根目录
cd /d "%~dp0\.."

REM 2. 设置 WSL 发行版名称 (必须与您安装的一致)
set WSL_DISTRO=Ubuntu-24.04
set WSL_CMD=wsl -d %WSL_DISTRO%

REM ============================================================================
REM 环境检查
REM ============================================================================
:check_wsl
echo [INFO] Checking WSL environment (%WSL_DISTRO%)...

REM 尝试连接 WSL，不再依赖文本输出检测
wsl -d %WSL_DISTRO% true >nul 2>&1

if errorlevel 1 (
    echo [ERROR] WSL Distro '%WSL_DISTRO%' did not respond!
    echo [INFO] Please run: wsl --install -d %WSL_DISTRO%
    echo [INFO] Current installed distros:
    wsl -l -v
    pause
    exit /b 1
)
echo [OK] WSL Distro found and ready.

echo [INFO] Checking Docker inside WSL...
%WSL_CMD% docker --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker not found inside WSL!
    echo [INFO] Please enable 'WSL Integration' in Docker Desktop Settings.
    pause
    exit /b 1
)
echo [OK] Docker is ready inside WSL

echo [INFO] Checking GPU (RTX 5090)...
%WSL_CMD% nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo [WARNING] NVIDIA Driver not detected in WSL!
    echo [INFO] Ensure you have installed NVIDIA Drivers on Windows.
) else (
    echo [OK] GPU detected inside WSL
)
echo.

REM ============================================================================
REM 菜单
REM ============================================================================
:menu
echo.
echo ========================================
echo    Select Deployment Option (WSL2)
echo ========================================
echo.
echo   1. Full Deployment (Build + Start)
echo   2. Start Services (Production)
echo   3. Stop All Services
echo   4. View Logs (Backend)
echo   5. View Logs (Worker)
echo   6. Enter WSL Shell
echo   7. Create Model Directories (D Drive)
echo   0. Exit
echo.
set /p choice="Please enter option [0-7]: "

if "%choice%"=="1" goto full_setup
if "%choice%"=="2" goto start_prod
if "%choice%"=="3" goto stop
if "%choice%"=="4" goto logs_backend
if "%choice%"=="5" goto logs_worker
if "%choice%"=="6" goto shell
if "%choice%"=="7" goto init_dirs
if "%choice%"=="0" goto end
echo [ERROR] Invalid option
goto menu

REM ============================================================================
REM 1. 全量部署
REM ============================================================================
:full_setup
echo.
echo [INFO] Starting full deployment in WSL...

REM 检查 .env
if not exist .env (
    if exist .env.example (
        echo [INFO] Creating .env file from example...
        copy .env.example .env >nul
        echo [WARNING] Default .env created. Please edit it for RTX 5090 config!
        REM 这里暂停是为了让你看到警告，按键后继续
        pause
    )
)

REM 确保 D 盘模型目录存在 (传递参数 quiet 以避免跳回菜单)
call :create_dirs quiet

REM 在 WSL 中构建
echo [INFO] Building images in WSL (BuildKit enabled)...
REM 使用 Linux 语法设置环境变量并执行构建
%WSL_CMD% bash -c "export DOCKER_BUILDKIT=1 && docker compose build"

if errorlevel 1 (
    echo [ERROR] Build failed inside WSL.
    pause
    goto menu
)

REM 启动服务
echo [INFO] Starting services in WSL...
%WSL_CMD% docker compose up -d

echo.
echo [OK] Services started! 
echo      API: http://localhost:8000
echo      Worker: http://localhost:8001
timeout /t 5 >nul
goto menu

REM ============================================================================
REM 2. 仅启动
REM ============================================================================
:start_prod
echo [INFO] Starting services...
%WSL_CMD% docker compose up -d
goto menu

REM ============================================================================
REM 3. 停止服务
REM ============================================================================
:stop
echo [INFO] Stopping services...
%WSL_CMD% docker compose down
echo [OK] Stopped.
goto menu

REM ============================================================================
REM 4/5. 查看日志
REM ============================================================================
:logs_backend
echo [INFO] Tailing Backend logs (Ctrl+C to exit)...
%WSL_CMD% docker compose logs -f backend
goto menu

:logs_worker
echo [INFO] Tailing Worker logs (Ctrl+C to exit)...
%WSL_CMD% docker compose logs -f worker
goto menu

REM ============================================================================
REM 6. 进入 WSL 终端
REM ============================================================================
:shell
echo [INFO] Entering WSL shell...
wsl -d %WSL_DISTRO%
goto menu

REM ============================================================================
REM 7. 初始化目录 (辅助功能)
REM ============================================================================
:init_dirs
:create_dirs
echo [INFO] Creating local data directories...
if not exist data\uploads mkdir data\uploads
if not exist data\output mkdir data\output
if not exist data\db mkdir data\db
if not exist logs mkdir logs

echo [INFO] Checking D: drive model directories...
if not exist "D:\aiworkspace\models\mineru" mkdir "D:\aiworkspace\models\mineru"
if not exist "D:\aiworkspace\models\paddleocr" mkdir "D:\aiworkspace\models\paddleocr"
if not exist "D:\aiworkspace\models\paddlex" mkdir "D:\aiworkspace\models\paddlex"
if not exist "D:\aiworkspace\models\huggingface" mkdir "D:\aiworkspace\models\huggingface"
if not exist "D:\aiworkspace\models\modelscope" mkdir "D:\aiworkspace\models\modelscope"

echo [OK] Directories checked.

REM 关键修改：如果有参数（如 quiet），则直接返回；否则暂停并回菜单
if "%1"=="" (
    pause
    goto menu
)
exit /b

REM ============================================================================
REM 退出
REM ============================================================================
:end
exit /b 0
