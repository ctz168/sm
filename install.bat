@echo off
REM 分布式大模型推理系统 - Windows一键安装脚本
REM
REM 支持: Windows 10/11
REM 用法: 以管理员身份运行

setlocal enabledelayedexpansion

:: 版本
set VERSION=1.0.0
set INSTALL_DIR=%USERPROFILE%\.distributed-llm
set REPO_URL=https://github.com/ctz168/sm.git

:: 颜色
for /F %%a in ('echo prompt $E^| cmd') do set "ESC=%%a"
set "RED=!ESC![31m"
set "GREEN=!ESC![32m"
set "YELLOW=!ESC![33m"
set "BLUE=!ESC![34m"
set "CYAN=!ESC![36m"
set "NC=!ESC![0m"

:: 打印Banner
echo.
echo !CYAN!╔══════════════════════════════════════════════════════════════════════╗!NC!
echo !CYAN!║                                                                      ║!NC!
echo !CYAN!║       分布式大模型推理系统 - 一键安装 (Windows)                      ║!NC!
echo !CYAN!║                                                                      ║!NC!
echo !CYAN!║       版本: %VERSION%                                                    ║!NC!
echo !CYAN!║                                                                      ║!NC!
echo !CYAN!╚══════════════════════════════════════════════════════════════════════╝!NC!
echo.

:: 检查管理员权限
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo !YELLOW!提示: 建议以管理员身份运行以获得完整功能!NC!
    echo.
)

:: 检查Python
echo !BLUE!▶ 检查Python...!NC!
python --version >nul 2>&1
if %errorLevel% neq 0 (
    echo !RED!❌ 未检测到Python，正在安装...!NC!
    
    :: 使用winget安装Python
    winget install Python.Python.3.12 --accept-source-agreements --accept-package-agreements
    
    :: 刷新环境变量
    call refreshenv >nul 2>&1 || echo 请重新打开命令行窗口
)
python --version
echo !GREEN!✅ Python已安装!NC!
echo.

:: 检查Git
echo !BLUE!▶ 检查Git...!NC!
git --version >nul 2>&1
if %errorLevel% neq 0 (
    echo !RED!❌ 未检测到Git，正在安装...!NC!
    winget install Git.Git --accept-source-agreements --accept-package-agreements
)
git --version
echo !GREEN!✅ Git已安装!NC!
echo.

:: 创建安装目录
echo !BLUE!▶ 创建安装目录...!NC!
if not exist "%INSTALL_DIR%" mkdir "%INSTALL_DIR%"
if not exist "%INSTALL_DIR%\config" mkdir "%INSTALL_DIR%\config"
echo !GREEN!✅ 安装目录: %INSTALL_DIR%!NC!
echo.

:: 克隆仓库
echo !BLUE!▶ 下载项目...!NC!
if exist "%INSTALL_DIR%\sm" (
    echo !YELLOW!项目目录已存在，更新...!NC!
    cd /d "%INSTALL_DIR%\sm"
    git pull
) else (
    git clone %REPO_URL% "%INSTALL_DIR%\sm"
)
echo !GREEN!✅ 项目下载完成!NC!
echo.

:: 创建虚拟环境
echo !BLUE!▶ 创建Python虚拟环境...!NC!
if not exist "%INSTALL_DIR%\venv" (
    python -m venv "%INSTALL_DIR%\venv"
)
echo !GREEN!✅ 虚拟环境创建完成!NC!
echo.

:: 安装Python依赖
echo !BLUE!▶ 安装Python依赖...!NC!
call "%INSTALL_DIR%\venv\Scripts\activate.bat"
python -m pip install --upgrade pip
pip install torch transformers psutil python-socketio accelerate
echo !GREEN!✅ Python依赖安装完成!NC!
echo.

:: 创建配置文件
echo !BLUE!▶ 创建配置文件...!NC!
(
echo # 分布式大模型推理系统配置
echo.
echo model:
echo   name: "Qwen/Qwen2.5-0.5B-Instruct"
echo   memory_gb: 2.0
echo.
echo resources:
echo   min_memory_gb: 2.0
echo   min_cpu_percent: 10.0
echo.
echo network:
echo   host: "0.0.0.0"
echo   port: 7000
echo.
echo mode: "resource_aware"
) > "%INSTALL_DIR%\config\config.yaml"
echo !GREEN!✅ 配置文件创建完成!NC!
echo.

:: 创建启动脚本
echo !BLUE!▶ 创建启动脚本...!NC!
(
echo @echo off
echo cd /d "%INSTALL_DIR%\sm"
echo call "%INSTALL_DIR%\venv\Scripts\activate.bat"
echo python download/node_resource_aware.py --model Qwen/Qwen2.5-0.5B-Instruct --port 7000
echo pause
) > "%INSTALL_DIR%\start.bat"
echo !GREEN!✅ 启动脚本创建完成!NC!
echo.

:: 创建停止脚本
echo !BLUE!▶ 创建停止脚本...!NC!
(
echo @echo off
echo taskkill /F /IM python.exe /FI "WINDOWTITLE eq distributed-llm*" 2>nul
echo taskkill /F /IM pythonw.exe /FI "WINDOWTITLE eq distributed-llm*" 2>nul
echo echo 服务已停止
echo pause
) > "%INSTALL_DIR%\stop.bat"
echo !GREEN!✅ 停止脚本创建完成!NC!
echo.

:: 创建状态脚本
echo !BLUE!▶ 创建状态脚本...!NC!
(
echo @echo off
echo echo 服务状态:
echo curl -s http://localhost:7000/status 2>nul
echo echo.
echo pause
) > "%INSTALL_DIR%\status.bat"
echo !GREEN!✅ 状态脚本创建完成!NC!
echo.

:: 创建桌面快捷方式
echo !BLUE!▶ 创建桌面快捷方式...!NC!
set SHORTCUT=%USERPROFILE%\Desktop\分布式大模型推理.lnk
powershell -Command "$ws = New-Object -ComObject WScript.Shell; $s = $ws.CreateShortcut('%SHORTCUT%'); $s.TargetPath = '%INSTALL_DIR%\start.bat'; $s.WorkingDirectory = '%INSTALL_DIR%'; $s.Description = '分布式大模型推理系统'; $s.Save()"
echo !GREEN!✅ 桌面快捷方式创建完成!NC!
echo.

:: 创建Windows服务注册脚本
echo !BLUE!▶ 创建Windows服务脚本...!NC!
(
echo @echo off
echo echo 注册Windows服务...
echo sc create "DistributedLLM" binPath= "\"%INSTALL_DIR%\venv\Scripts\python.exe\" \"%INSTALL_DIR%\sm\download\node_resource_aware.py\" --model Qwen/Qwen2.5-0.5B-Instruct --port 7000" start= auto
echo sc description "DistributedLLM" "分布式大模型推理服务"
echo echo 服务已注册，使用以下命令管理:
echo echo   启动: sc start DistributedLLM
echo echo   停止: sc stop DistributedLLM
echo echo   删除: sc delete DistributedLLM
echo pause
) > "%INSTALL_DIR%\install-service.bat"
echo !GREEN!✅ Windows服务脚本创建完成!NC!
echo.

:: 完成
echo.
echo !GREEN!╔══════════════════════════════════════════════════════════════════════╗!NC!
echo !GREEN!║                                                                      ║!NC!
echo !GREEN!║                    ✅ 安装完成！                                     ║!NC!
echo !GREEN!║                                                                      ║!NC!
echo !GREEN!╚══════════════════════════════════════════════════════════════════════╝!NC!
echo.
echo !CYAN!安装目录:!NC! %INSTALL_DIR%
echo.
echo !CYAN!快速使用:!NC!
echo.
echo   双击桌面快捷方式 "分布式大模型推理" 启动服务
echo.
echo   或运行:
echo   %INSTALL_DIR%\start.bat    - 启动服务
echo   %INSTALL_DIR%\stop.bat     - 停止服务
echo   %INSTALL_DIR%\status.bat   - 查看状态
echo.
echo !CYAN!配置文件:!NC! %INSTALL_DIR%\config\config.yaml
echo.
echo !CYAN!API地址:!NC! http://localhost:7000
echo.
echo !YELLOW!开机自启:!NC!
echo   以管理员身份运行: %INSTALL_DIR%\install-service.bat
echo.

pause
