#!/bin/bash
#
# 分布式大模型推理系统 - 一键启动脚本
#
# 用法: ./start.sh [模式]
#
# 模式:
#   1 或 resource_aware  - 资源感知模式（推荐）
#   2 或 pipeline        - Pipeline并行模式
#   3 或 decentralized   - 去中心化模式
#   4 或 centralized     - 中心化模式
#

set -e

# 颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# 默认配置
MODEL="Qwen/Qwen2.5-0.5B-Instruct"
PORT=7000
MIN_MEMORY=2.0
MIN_CPU=10.0

# 查找安装目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/download/node_resource_aware.py" ]; then
    INSTALL_DIR="$SCRIPT_DIR"
elif [ -f "$HOME/.distributed-llm/sm/download/node_resource_aware.py" ]; then
    INSTALL_DIR="$HOME/.distributed-llm/sm"
else
    INSTALL_DIR="$SCRIPT_DIR"
fi

# 激活虚拟环境
if [ -f "$HOME/.distributed-llm/venv/bin/activate" ]; then
    source "$HOME/.distributed-llm/venv/bin/activate"
elif [ -f "$INSTALL_DIR/../venv/bin/activate" ]; then
    source "$INSTALL_DIR/../venv/bin/activate"
fi

cd "$INSTALL_DIR"

# 打印Banner
print_banner() {
    echo -e "${CYAN}"
    echo "╔══════════════════════════════════════════════════════════════════════╗"
    echo "║                                                                      ║"
    echo "║           分布式大模型推理系统 - 一键启动                            ║"
    echo "║                                                                      ║"
    echo "╚══════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# 选择模式
select_mode() {
    echo -e "${CYAN}请选择启动模式:${NC}"
    echo ""
    echo "  1) 资源感知模式     - 自动检测资源，智能启停（推荐）"
    echo "  2) Pipeline并行    - 多节点分片，内存最优"
    echo "  3) 去中心化模式    - 无单点故障，高可用"
    echo "  4) 中心化模式      - 有Web管理界面"
    echo ""
    echo -n "请输入选择 [1-4]: "
    read -r choice
    echo ""
    
    case $choice in
        1) MODE="resource_aware" ;;
        2) MODE="pipeline" ;;
        3) MODE="decentralized" ;;
        4) MODE="centralized" ;;
        *) 
            echo -e "${YELLOW}使用默认模式: 资源感知${NC}"
            MODE="resource_aware"
            ;;
    esac
}

# 启动资源感知模式
start_resource_aware() {
    echo -e "${BLUE}启动资源感知模式...${NC}"
    echo ""
    echo -e "  模型: ${YELLOW}$MODEL${NC}"
    echo -e "  端口: ${YELLOW}$PORT${NC}"
    echo -e "  最小内存: ${YELLOW}${MIN_MEMORY}GB${NC}"
    echo ""
    
    python3 download/node_resource_aware.py \
        --model "$MODEL" \
        --port $PORT \
        --min-memory $MIN_MEMORY \
        --min-cpu $MIN_CPU
}

# 启动Pipeline模式
start_pipeline() {
    echo -e "${BLUE}启动Pipeline并行模式...${NC}"
    echo ""
    echo -n "请输入节点索引 [0]: "
    read -r node_index
    node_index=${node_index:-0}
    
    echo -n "请输入总节点数 [2]: "
    read -r total_nodes
    total_nodes=${total_nodes:-2}
    
    port=$((6000 + node_index))
    
    echo ""
    echo -e "  节点索引: ${YELLOW}$node_index${NC}"
    echo -e "  总节点数: ${YELLOW}$total_nodes${NC}"
    echo -e "  端口: ${YELLOW}$port${NC}"
    echo ""
    
    python3 download/node_pipeline_shard.py \
        --model "$MODEL" \
        --index $node_index \
        --total $total_nodes \
        --port $port
}

# 启动去中心化模式
start_decentralized() {
    echo -e "${BLUE}启动去中心化模式...${NC}"
    echo ""
    echo -n "请输入端口 [5000]: "
    read -r port
    port=${port:-5000}
    
    echo ""
    echo -e "  端口: ${YELLOW}$port${NC}"
    echo ""
    
    python3 download/node_decentralized.py \
        --model "$MODEL" \
        --port $port
}

# 启动中心化模式
start_centralized() {
    echo -e "${BLUE}启动中心化模式...${NC}"
    echo ""
    echo -e "${YELLOW}注意: 需要先启动Orchestrator服务${NC}"
    echo ""
    
    python3 download/node_service_optimized.py \
        --model "$MODEL" \
        --server http://localhost:3003
}

# 主函数
main() {
    print_banner
    
    # 检查参数
    if [ -n "$1" ]; then
        case $1 in
            1|resource_aware) MODE="resource_aware" ;;
            2|pipeline) MODE="pipeline" ;;
            3|decentralized) MODE="decentralized" ;;
            4|centralized) MODE="centralized" ;;
            *)
                echo -e "${RED}未知模式: $1${NC}"
                select_mode
                ;;
        esac
    else
        select_mode
    fi
    
    # 启动对应模式
    case $MODE in
        resource_aware) start_resource_aware ;;
        pipeline) start_pipeline ;;
        decentralized) start_decentralized ;;
        centralized) start_centralized ;;
    esac
}

# 运行
main "$@"
