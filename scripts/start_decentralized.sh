#!/bin/bash
#
# 分布式大模型推理系统 - 去中心化启动脚本
#
# 用法:
#   ./start_decentralized.sh [节点数]
#

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 配置
NUM_NODES=${1:-3}
BASE_PORT=5000
MODEL="Qwen/Qwen2.5-0.5B-Instruct"

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     分布式大模型推理系统 - 去中心化模式                      ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ 未找到Python3${NC}"
    exit 1
fi

# 检查依赖
echo -e "${YELLOW}检查依赖...${NC}"
pip3 install --user python-socketio psutil torch transformers 2>/dev/null || true

# 创建日志目录
mkdir -p logs
mkdir -p pids

# 停止旧进程
echo -e "${YELLOW}停止旧进程...${NC}"
pkill -f "node_decentralized.py" 2>/dev/null || true
sleep 2

# 启动节点
echo -e "${GREEN}启动 ${NUM_NODES} 个去中心化节点...${NC}"
echo ""

for i in $(seq 1 $NUM_NODES); do
    PORT=$((BASE_PORT + i - 1))
    NODE_NAME="DecentralNode-$i"
    
    # 第一个节点的种子节点为空，后续节点连接到第一个节点
    if [ $i -eq 1 ]; then
        SEEDS=""
    else
        SEEDS="localhost:${BASE_PORT}"
    fi
    
    echo -e "${BLUE}启动节点 ${i}:${NC}"
    echo -e "  名称: ${NODE_NAME}"
    echo -e "  端口: ${PORT}"
    echo -e "  种子: ${SEEDS:-"(无，首个节点)"}"
    
    nohup python3 download/node_decentralized.py \
        --host 0.0.0.0 \
        --port $PORT \
        --name "$NODE_NAME" \
        --model "$MODEL" \
        --seeds "$SEEDS" \
        --workers 2 \
        --min-nodes 1 \
        > "logs/node_${i}.log" 2>&1 &
    
    echo $! > "pids/node_${i}.pid"
    
    echo -e "${GREEN}  ✅ 已启动 (PID: $(cat pids/node_${i}.pid))${NC}"
    echo ""
    
    # 等待节点启动
    sleep 2
done

# 等待集群稳定
echo -e "${YELLOW}等待集群稳定...${NC}"
sleep 5

# 显示状态
echo ""
echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     集群状态                                                 ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

for i in $(seq 1 $NUM_NODES); do
    PORT=$((BASE_PORT + i - 1))
    echo -e "${BLUE}节点 ${i}:${NC}"
    tail -5 "logs/node_${i}.log" 2>/dev/null | grep -E "\[Raft\]|\[模型\]|\[P2P\]" | tail -3
    echo ""
done

echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║     启动完成！                                               ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "节点列表:"
for i in $(seq 1 $NUM_NODES); do
    PORT=$((BASE_PORT + i - 1))
    echo -e "  节点 ${i}: localhost:${PORT}"
done
echo ""
echo -e "日志文件: logs/node_*.log"
echo -e "停止集群: pkill -f node_decentralized.py"
echo ""
echo -e "${YELLOW}提示: 只要有一个节点在线，服务就能继续运行${NC}"
