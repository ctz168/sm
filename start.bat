@echo off
REM 分布式大模型推理系统 - 一键启动脚本 (Windows)

setlocal enabledelayedexpansion

:: 颜色
for /F %%a in ('echo prompt $E^| cmd') do set "ESC=%%a"
set "GREEN=!ESC![32m"
set "YELLOW=!ESC![33m"
set "BLUE=!ESC![34m"
set "CYAN=!ESC![36m"
set "NC=!ESC![0m"

:: 默认配置
set MODEL=Qwen/Qwen2.5-0.5B-Instruct
set PORT=7000
set MIN_MEMORY=2.0
set MIN_CPU=10.0

:: 查找安装目录
set SCRIPT_DIR=%~dp0
if exist "%SCRIPT_DIR%download\node_resource_aware.py" (
    set INSTALL_DIR=%SCRIPT_DIR%
) else if exist "%USERPROFILE%\.distributed-llm\sm\download\node_resource_aware.py" (
    set INSTALL_DIR=%USERPROFILE%\.distributed-llm\sm
) else (
    set INSTALL_DIR=%SCRIPT_DIR%
)

:: 激活虚拟环境
if exist "%USERPROFILE%\.distributed-llm\venv\Scripts\activate.bat" (
    call "%USERPROFILE%\.distributed-llm\venv\Scripts\activate.bat"
)

cd /d "%INSTALL_DIR%"

:: 打印Banner
echo.
echo !CYAN!╔══════════════════════════════════════════════════════════════════════╗!NC!
echo !CYAN!║                                                                      ║!NC!
echo !CYAN!║           分布式大模型推理系统 - 一键启动                            ║!NC!
echo !CYAN!║                                                                      ║!NC!
echo !CYAN!╚══════════════════════════════════════════════════════════════════════╝!NC!
echo.

:: 选择模式
echo !CYAN!请选择启动模式:!NC!
echo.
echo   1) 资源感知模式     - 自动检测资源，智能启停（推荐）
echo   2) Pipeline并行    - 多节点分片，内存最优
echo   3) 去中心化模式    - 无单点故障，高可用
echo   4) 中心化模式      - 有Web管理界面
echo.

set /p choice="请输入选择 [1-4]: "

if "%choice%"=="" set choice=1

if "%choice%"=="1" goto resource_aware
if "%choice%"=="2" goto pipeline
if "%choice%"=="3" goto decentralized
if "%choice%"=="4" goto centralized
goto resource_aware

:resource_aware
echo.
echo !BLUE!启动资源感知模式...!NC!
echo.
echo   模型: !YELLOW!%MODEL%!NC!
echo   端口: !YELLOW!%PORT%!NC!
echo.
python download\node_resource_aware.py --model %MODEL% --port %PORT% --min-memory %MIN_MEMORY% --min-cpu %MIN_CPU%
goto end

:pipeline
echo.
echo !BLUE!启动Pipeline并行模式...!NC!
set /p node_index="请输入节点索引 [0]: "
if "%node_index%"=="" set node_index=0
set /p total_nodes="请输入总节点数 [2]: "
if "%total_nodes%"=="" set total_nodes=2
set /a port=6000+%node_index%
echo.
echo   节点索引: !YELLOW!%node_index%!NC!
echo   总节点数: !YELLOW!%total_nodes%!NC!
echo   端口: !YELLOW!%port%!NC!
echo.
python download\node_pipeline_shard.py --model %MODEL% --index %node_index% --total %total_nodes% --port %port%
goto end

:decentralized
echo.
echo !BLUE!启动去中心化模式...!NC!
set /p port="请输入端口 [5000]: "
if "%port%"=="" set port=5000
echo.
echo   端口: !YELLOW!%port%!NC!
echo.
python download\node_decentralized.py --model %MODEL% --port %port%
goto end

:centralized
echo.
echo !BLUE!启动中心化模式...!NC!
echo !YELLOW!注意: 需要先启动Orchestrator服务!NC!
echo.
python download\node_service_optimized.py --model %MODEL% --server http://localhost:3003
goto end

:end
pause
