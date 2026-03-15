#!/bin/bash
#
# 分布式大模型推理系统 - 一键启动脚本
#
# 使用方法:
#   ./start.sh              # 启动所有服务
#   ./start.sh --dev        # 开发模式（前台运行）
#   ./start.sh --stop       # 停止所有服务
#   ./start.sh --status     # 查看服务状态
#

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 日志目录
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

# PID文件目录
PID_DIR="$PROJECT_ROOT/pids"
mkdir -p "$PID_DIR"

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查端口是否被占用
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0  # 端口被占用
    else
        return 1  # 端口空闲
    fi
}

# 启动Orchestrator服务
start_orchestrator() {
    print_info "启动 Orchestrator 服务..."
    
    if check_port 3003; then
        print_warning "端口 3003 已被占用，跳过 Orchestrator 启动"
        return 0
    fi
    
    cd "$PROJECT_ROOT/mini-services/orchestrator"
    
    # 检查依赖
    if [ ! -d "node_modules" ]; then
        print_info "安装 Orchestrator 依赖..."
        bun install
    fi
    
    # 启动服务
    bun run dev > "$LOG_DIR/orchestrator.log" 2>&1 &
    local pid=$!
    echo $pid > "$PID_DIR/orchestrator.pid"
    
    # 等待启动
    sleep 3
    if check_port 3003; then
        print_success "Orchestrator 服务已启动 (PID: $pid)"
        print_info "  WebSocket: ws://localhost:3003"
        print_info "  HTTP API:  http://localhost:3004"
    else
        print_error "Orchestrator 服务启动失败"
        return 1
    fi
}

# 启动Web界面
start_web() {
    print_info "启动 Web 管理界面..."
    
    if check_port 3000; then
        print_warning "端口 3000 已被占用，跳过 Web 界面启动"
        return 0
    fi
    
    cd "$PROJECT_ROOT"
    
    # 启动服务
    bun run dev > "$LOG_DIR/web.log" 2>&1 &
    local pid=$!
    echo $pid > "$PID_DIR/web.pid"
    
    # 等待启动
    sleep 5
    if check_port 3000; then
        print_success "Web 管理界面已启动 (PID: $pid)"
        print_info "  访问地址: http://localhost:3000"
    else
        print_error "Web 管理界面启动失败"
        return 1
    fi
}

# 启动模拟节点
start_simulator() {
    print_info "启动模拟节点..."
    
    cd "$PROJECT_ROOT/mini-services/simulator"
    
    # 检查依赖
    if [ ! -d "node_modules" ]; then
        print_info "安装模拟器依赖..."
        bun install
    fi
    
    # 启动服务
    bun run dev > "$LOG_DIR/simulator.log" 2>&1 &
    local pid=$!
    echo $pid > "$PID_DIR/simulator.pid"
    
    print_success "模拟节点已启动 (PID: $pid)"
    print_info "  4个模拟节点将陆续加入集群"
}

# 停止所有服务
stop_all() {
    print_info "停止所有服务..."
    
    # 停止模拟器
    if [ -f "$PID_DIR/simulator.pid" ]; then
        local pid=$(cat "$PID_DIR/simulator.pid")
        if kill -0 $pid 2>/dev/null; then
            kill $pid 2>/dev/null || true
            print_success "模拟节点已停止"
        fi
        rm -f "$PID_DIR/simulator.pid"
    fi
    
    # 停止Web
    if [ -f "$PID_DIR/web.pid" ]; then
        local pid=$(cat "$PID_DIR/web.pid")
        if kill -0 $pid 2>/dev/null; then
            kill $pid 2>/dev/null || true
            print_success "Web 界面已停止"
        fi
        rm -f "$PID_DIR/web.pid"
    fi
    
    # 停止Orchestrator
    if [ -f "$PID_DIR/orchestrator.pid" ]; then
        local pid=$(cat "$PID_DIR/orchestrator.pid")
        if kill -0 $pid 2>/dev/null; then
            kill $pid 2>/dev/null || true
            print_success "Orchestrator 已停止"
        fi
        rm -f "$PID_DIR/orchestrator.pid"
    fi
    
    # 清理残留进程
    for port in 3000 3003 3004; do
        local pids=$(lsof -t -i:$port 2>/dev/null || true)
        if [ -n "$pids" ]; then
            kill $pids 2>/dev/null || true
        fi
    done
    
    print_success "所有服务已停止"
}

# 查看服务状态
show_status() {
    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║              分布式大模型推理系统 - 服务状态                ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
    
    # Web界面状态
    if check_port 3000; then
        echo -e "Web 管理界面:  ${GREEN}● 运行中${NC}  http://localhost:3000"
    else
        echo -e "Web 管理界面:  ${RED}○ 已停止${NC}"
    fi
    
    # Orchestrator状态
    if check_port 3003 && check_port 3004; then
        echo -e "Orchestrator:  ${GREEN}● 运行中${NC}  ws://localhost:3003 / http://localhost:3004"
    else
        echo -e "Orchestrator:  ${RED}○ 已停止${NC}"
    fi
    
    # 模拟器状态
    if [ -f "$PID_DIR/simulator.pid" ]; then
        local pid=$(cat "$PID_DIR/simulator.pid")
        if kill -0 $pid 2>/dev/null; then
            echo -e "模拟节点:      ${GREEN}● 运行中${NC}  (PID: $pid)"
        else
            echo -e "模拟节点:      ${RED}○ 已停止${NC}"
        fi
    else
        echo -e "模拟节点:      ${YELLOW}○ 未启动${NC}"
    fi
    
    echo ""
    
    # 显示系统指标
    if check_port 3004; then
        echo "┌─────────────────────────────────────────────────────────────┐"
        echo "│ 系统指标                                                    │"
        echo "├─────────────────────────────────────────────────────────────┤"
        
        # 获取API数据
        local metrics=$(curl -s http://localhost:3004/api/metrics 2>/dev/null || echo "{}")
        
        local nodes=$(echo "$metrics" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('healthyNodes', 0))" 2>/dev/null || echo "0")
        local parallel=$(echo "$metrics" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"{d.get('currentParallelism', 0)}/{d.get('maxParallelism', 0)}\")" 2>/dev/null || echo "0/0")
        local load=$(echo "$metrics" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"{d.get('avgNodeLoad', 0):.0f}%\")" 2>/dev/null || echo "0%")
        
        echo "│  在线节点: $nodes          并行度: $parallel           平均负载: $load     │"
        echo "└─────────────────────────────────────────────────────────────┘"
    fi
    
    echo ""
}

# 开发模式（前台运行）
dev_mode() {
    print_info "开发模式启动..."
    echo ""
    
    # 启动Orchestrator
    print_info "启动 Orchestrator..."
    cd "$PROJECT_ROOT/mini-services/orchestrator"
    bun run dev &
    ORCH_PID=$!
    
    sleep 3
    
    # 启动Web
    print_info "启动 Web 界面..."
    cd "$PROJECT_ROOT"
    bun run dev &
    WEB_PID=$!
    
    echo ""
    print_success "服务已启动，按 Ctrl+C 停止"
    echo ""
    show_status
    
    # 捕获退出信号
    trap "kill $ORCH_PID $WEB_PID 2>/dev/null; exit 0" SIGINT SIGTERM
    
    # 等待
    wait
}

# 显示帮助
show_help() {
    echo ""
    echo "分布式大模型推理系统 - 启动脚本"
    echo ""
    echo "使用方法:"
    echo "  $0              启动所有服务（后台运行）"
    echo "  $0 --dev        开发模式（前台运行，显示日志）"
    echo "  $0 --stop       停止所有服务"
    echo "  $0 --status     查看服务状态"
    echo "  $0 --restart    重启所有服务"
    echo "  $0 --help       显示帮助信息"
    echo ""
    echo "服务端口:"
    echo "  Web 管理界面:   http://localhost:3000"
    echo "  Orchestrator:   ws://localhost:3003 / http://localhost:3004"
    echo ""
    echo "日志文件:"
    echo "  $LOG_DIR/orchestrator.log"
    echo "  $LOG_DIR/web.log"
    echo "  $LOG_DIR/simulator.log"
    echo ""
}

# 主函数
main() {
    case "${1:-}" in
        --dev|-d)
            dev_mode
            ;;
        --stop|-s)
            stop_all
            ;;
        --status|-st)
            show_status
            ;;
        --restart|-r)
            stop_all
            sleep 2
            start_orchestrator
            start_web
            show_status
            ;;
        --help|-h)
            show_help
            ;;
        --simulator)
            start_simulator
            ;;
        *)
            echo ""
            echo "╔════════════════════════════════════════════════════════════╗"
            echo "║          分布式大模型推理系统 - 启动中...                   ║"
            echo "╚════════════════════════════════════════════════════════════╝"
            echo ""
            
            start_orchestrator
            start_web
            
            echo ""
            print_success "所有服务启动完成！"
            echo ""
            
            show_status
            
            print_info "提示: 运行 '$0 --simulator' 启动模拟节点"
            print_info "提示: 运行 '$0 --stop' 停止所有服务"
            ;;
    esac
}

main "$@"
