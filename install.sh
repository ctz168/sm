#!/bin/bash
#
# 分布式大模型推理系统 - 一键安装脚本
#
# 支持: Linux, macOS
# 用法: curl -fsSL https://raw.githubusercontent.com/ctz168/sm/main/install.sh | bash
#

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# 版本
VERSION="1.0.0"
INSTALL_DIR="$HOME/.distributed-llm"
REPO_URL="https://github.com/ctz168/sm.git"

# 打印函数
print_banner() {
    echo -e "${CYAN}"
    echo "╔══════════════════════════════════════════════════════════════════════╗"
    echo "║                                                                      ║"
    echo "║       分布式大模型推理系统 - 一键安装                                ║"
    echo "║                                                                      ║"
    echo "║       版本: $VERSION                                                    ║"
    echo "║       支持: Linux, macOS                                             ║"
    echo "║                                                                      ║"
    echo "╚══════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_step() {
    echo -e "\n${BLUE}▶ $1${NC}\n"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# 检测操作系统
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
        PKG_MANAGER=$(command -v apt-get >/dev/null 2>&1 && echo "apt" || \
                     command -v yum >/dev/null 2>&1 && echo "yum" || \
                     command -v dnf >/dev/null 2>&1 && echo "dnf" || \
                     command -v pacman >/dev/null 2>&1 && echo "pacman" || echo "unknown")
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
        PKG_MANAGER="brew"
    else
        print_error "不支持的操作系统: $OSTYPE"
        exit 1
    fi
    print_success "检测到操作系统: $OS"
}

# 检查依赖
check_dependencies() {
    print_step "检查系统依赖..."
    
    local missing=()
    
    # 检查Python
    if ! command -v python3 &>/dev/null; then
        missing+=("python3")
    fi
    
    # 检查pip
    if ! command -v pip3 &>/dev/null && ! python3 -m pip --version &>/dev/null; then
        missing+=("pip3")
    fi
    
    # 检查git
    if ! command -v git &>/dev/null; then
        missing+=("git")
    fi
    
    if [ ${#missing[@]} -gt 0 ]; then
        print_warning "缺少依赖: ${missing[*]}"
        install_dependencies "${missing[@]}"
    else
        print_success "所有系统依赖已安装"
    fi
}

# 安装依赖
install_dependencies() {
    print_step "安装系统依赖..."
    
    local deps=("$@")
    
    case $PKG_MANAGER in
        apt)
            sudo apt-get update
            sudo apt-get install -y python3 python3-pip git curl
            ;;
        yum|dnf)
            sudo $PKG_MANAGER install -y python3 python3-pip git curl
            ;;
        pacman)
            sudo pacman -S --noconfirm python python-pip git curl
            ;;
        brew)
            brew install python3 git curl
            ;;
        *)
            print_error "未知的包管理器，请手动安装: ${deps[*]}"
            exit 1
            ;;
    esac
    
    print_success "系统依赖安装完成"
}

# 安装Python依赖
install_python_deps() {
    print_step "安装Python依赖..."
    
    # 创建虚拟环境
    if [ ! -d "$INSTALL_DIR/venv" ]; then
        python3 -m venv "$INSTALL_DIR/venv"
        print_success "创建虚拟环境"
    fi
    
    # 激活虚拟环境
    source "$INSTALL_DIR/venv/bin/activate"
    
    # 安装依赖
    pip install --upgrade pip
    pip install torch transformers psutil python-socketio accelerate
    
    print_success "Python依赖安装完成"
}

# 克隆仓库
clone_repo() {
    print_step "下载项目..."
    
    if [ -d "$INSTALL_DIR/sm" ]; then
        print_warning "项目目录已存在，更新..."
        cd "$INSTALL_DIR/sm"
        git pull
    else
        mkdir -p "$INSTALL_DIR"
        git clone "$REPO_URL" "$INSTALL_DIR/sm"
    fi
    
    print_success "项目下载完成"
}

# 创建配置文件
create_config() {
    print_step "创建配置文件..."
    
    mkdir -p "$INSTALL_DIR/config"
    
    # 创建默认配置
    cat > "$INSTALL_DIR/config/config.yaml" << 'EOF'
# 分布式大模型推理系统配置

# 模型配置
model:
  name: "Qwen/Qwen2.5-0.5B-Instruct"
  memory_gb: 2.0

# 资源配置
resources:
  min_memory_gb: 2.0
  min_cpu_percent: 10.0
  check_interval: 10

# 网络配置
network:
  host: "0.0.0.0"
  port: 7000

# 模式选择: resource_aware, pipeline, decentralized, centralized
mode: "resource_aware"

# Pipeline模式配置 (仅mode=pipeline时生效)
pipeline:
  node_index: 0
  total_nodes: 2
  next_host: ""
  next_port: 0

# 去中心化模式配置 (仅mode=decentralized时生效)
decentralized:
  discovery_port: 37000
  seeds: []
EOF

    print_success "配置文件创建完成: $INSTALL_DIR/config/config.yaml"
}

# 创建启动脚本
create_start_script() {
    print_step "创建启动脚本..."
    
    cat > "$INSTALL_DIR/start.sh" << 'SCRIPT'
#!/bin/bash
# 一键启动脚本

cd "$HOME/.distributed-llm/sm"
source "$HOME/.distributed-llm/venv/bin/activate"

# 读取配置
CONFIG="$HOME/.distributed-llm/config/config.yaml"
if [ -f "$CONFIG" ]; then
    MODE=$(grep "^mode:" "$CONFIG" | awk '{print $2}' | tr -d '"')
    MODEL=$(grep "name:" "$CONFIG" | head -1 | awk '{print $2}' | tr -d '"')
    PORT=$(grep "port:" "$CONFIG" | awk '{print $2}')
    MIN_MEM=$(grep "min_memory_gb:" "$CONFIG" | awk '{print $2}')
    MIN_CPU=$(grep "min_cpu_percent:" "$CONFIG" | awk '{print $2}')
else
    MODE="resource_aware"
    MODEL="Qwen/Qwen2.5-0.5B-Instruct"
    PORT=7000
    MIN_MEM=2.0
    MIN_CPU=10.0
fi

echo "启动模式: $MODE"
echo "模型: $MODEL"
echo "端口: $PORT"

case $MODE in
    resource_aware)
        python3 download/node_resource_aware.py \
            --model "$MODEL" \
            --port $PORT \
            --min-memory $MIN_MEM \
            --min-cpu $MIN_CPU
        ;;
    pipeline)
        NODE_INDEX=$(grep "node_index:" "$CONFIG" | awk '{print $2}')
        TOTAL_NODES=$(grep "total_nodes:" "$CONFIG" | awk '{print $2}')
        python3 download/node_pipeline_shard.py \
            --model "$MODEL" \
            --index $NODE_INDEX \
            --total $TOTAL_NODES \
            --port $PORT
        ;;
    decentralized)
        python3 download/node_decentralized.py \
            --model "$MODEL" \
            --port $PORT
        ;;
    centralized)
        python3 download/node_service_optimized.py \
            --model "$MODEL" \
            --server http://localhost:3003
        ;;
    *)
        echo "未知模式: $MODE"
        exit 1
        ;;
esac
SCRIPT

    chmod +x "$INSTALL_DIR/start.sh"
    print_success "启动脚本创建完成: $INSTALL_DIR/start.sh"
}

# 创建停止脚本
create_stop_script() {
    print_step "创建停止脚本..."
    
    cat > "$INSTALL_DIR/stop.sh" << 'SCRIPT'
#!/bin/bash
# 一键停止脚本

pkill -f "node_resource_aware" 2>/dev/null
pkill -f "node_pipeline_shard" 2>/dev/null
pkill -f "node_decentralized" 2>/dev/null
pkill -f "node_service" 2>/dev/null

echo "✅ 服务已停止"
SCRIPT

    chmod +x "$INSTALL_DIR/stop.sh"
    print_success "停止脚本创建完成: $INSTALL_DIR/stop.sh"
}

# 创建状态脚本
create_status_script() {
    print_step "创建状态脚本..."
    
    cat > "$INSTALL_DIR/status.sh" << 'SCRIPT'
#!/bin/bash
# 查看服务状态

echo "服务状态:"
echo ""

# 检查进程
if pgrep -f "node_resource_aware" > /dev/null; then
    echo "✅ 资源感知服务: 运行中"
    curl -s http://localhost:7000/status 2>/dev/null | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(f'   状态: {data.get(\"state\")}')
    print(f'   模型: {data.get(\"model_loaded\")}')
except: pass
" 2>/dev/null
elif pgrep -f "node_pipeline_shard" > /dev/null; then
    echo "✅ Pipeline服务: 运行中"
elif pgrep -f "node_decentralized" > /dev/null; then
    echo "✅ 去中心化服务: 运行中"
elif pgrep -f "node_service" > /dev/null; then
    echo "✅ 中心化服务: 运行中"
else
    echo "⭕ 服务未运行"
fi

echo ""
echo "系统资源:"
free -h 2>/dev/null || vm_stat 2>/dev/null | head -5
SCRIPT

    chmod +x "$INSTALL_DIR/status.sh"
    print_success "状态脚本创建完成: $INSTALL_DIR/status.sh"
}

# 安装系统服务
install_system_service() {
    print_step "安装系统服务..."
    
    if [ "$OS" == "linux" ]; then
        # 创建systemd服务
        cat > /tmp/distributed-llm.service << EOF
[Unit]
Description=Distributed LLM Inference Service
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$INSTALL_DIR/sm
ExecStart=$INSTALL_DIR/venv/bin/python3 download/node_resource_aware.py --model Qwen/Qwen2.5-0.5B-Instruct --port 7000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
        
        echo ""
        echo -e "${YELLOW}要安装为系统服务，请运行:${NC}"
        echo "  sudo cp /tmp/distributed-llm.service /etc/systemd/system/"
        echo "  sudo systemctl daemon-reload"
        echo "  sudo systemctl enable distributed-llm"
        echo "  sudo systemctl start distributed-llm"
        
    elif [ "$OS" == "macos" ]; then
        # 创建launchd服务
        cat > /tmp/com.distributed.llm.plist << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.distributed.llm</string>
    <key>ProgramArguments</key>
    <array>
        <string>$INSTALL_DIR/venv/bin/python3</string>
        <string>$INSTALL_DIR/sm/download/node_resource_aware.py</string>
        <string>--model</string>
        <string>Qwen/Qwen2.5-0.5B-Instruct</string>
        <string>--port</string>
        <string>7000</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>WorkingDirectory</key>
    <string>$INSTALL_DIR/sm</string>
</dict>
</plist>
EOF
        
        echo ""
        echo -e "${YELLOW}要安装为系统服务，请运行:${NC}"
        echo "  cp /tmp/com.distributed.llm.plist ~/Library/LaunchAgents/"
        echo "  launchctl load ~/Library/LaunchAgents/com.distributed.llm.plist"
    fi
    
    print_success "系统服务配置已生成"
}

# 设置环境变量
setup_env() {
    print_step "设置环境变量..."
    
    # 添加到shell配置
    SHELL_RC=""
    if [ -n "$ZSH_VERSION" ]; then
        SHELL_RC="$HOME/.zshrc"
    elif [ -n "$BASH_VERSION" ]; then
        SHELL_RC="$HOME/.bashrc"
    fi
    
    if [ -n "$SHELL_RC" ]; then
        if ! grep -q "DISTRIBUTED_LLM_HOME" "$SHELL_RC" 2>/dev/null; then
            echo "" >> "$SHELL_RC"
            echo "# Distributed LLM" >> "$SHELL_RC"
            echo "export DISTRIBUTED_LLM_HOME=\"$INSTALL_DIR\"" >> "$SHELL_RC"
            echo "alias dllm-start=\"$INSTALL_DIR/start.sh\"" >> "$SHELL_RC"
            echo "alias dllm-stop=\"$INSTALL_DIR/stop.sh\"" >> "$SHELL_RC"
            echo "alias dllm-status=\"$INSTALL_DIR/status.sh\"" >> "$SHELL_RC"
            print_success "环境变量已添加到 $SHELL_RC"
        fi
    fi
}

# 打印完成信息
print_complete() {
    echo ""
    echo -e "${GREEN}╔══════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                                                                      ║${NC}"
    echo -e "${GREEN}║                    ✅ 安装完成！                                     ║${NC}"
    echo -e "${GREEN}║                                                                      ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${CYAN}安装目录:${NC} $INSTALL_DIR"
    echo ""
    echo -e "${CYAN}快速使用:${NC}"
    echo ""
    echo "  # 启动服务"
    echo "  $INSTALL_DIR/start.sh"
    echo ""
    echo "  # 停止服务"
    echo "  $INSTALL_DIR/stop.sh"
    echo ""
    echo "  # 查看状态"
    echo "  $INSTALL_DIR/status.sh"
    echo ""
    echo -e "${CYAN}配置文件:${NC} $INSTALL_DIR/config/config.yaml"
    echo ""
    echo -e "${CYAN}API地址:${NC} http://localhost:7000"
    echo ""
    echo -e "${YELLOW}提示: 重新打开终端后可以使用以下快捷命令:${NC}"
    echo "  dllm-start   - 启动服务"
    echo "  dllm-stop    - 停止服务"
    echo "  dllm-status  - 查看状态"
    echo ""
}

# 主函数
main() {
    print_banner
    
    detect_os
    check_dependencies
    clone_repo
    install_python_deps
    create_config
    create_start_script
    create_stop_script
    create_status_script
    install_system_service
    setup_env
    
    print_complete
}

# 运行
main "$@"
