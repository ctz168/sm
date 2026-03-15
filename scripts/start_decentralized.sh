#!/bin/bash
#
# 分布式大模型推理系统 - 去中心化版本启动脚本
#
# 特性:
# - 无单点故障
# - 自动节点发现
# - 自动领导者选举
# - 动态算力管理
#

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 配置
DISCOVERY_PORT=${DISCOVERY_PORT:-37000}
API_PORT=${API_PORT:-37002}
MODEL_NAME=${MODEL_NAME:-"Qwen/Qwen2.5-0.5B-Instruct"}
NUM_NODES=${NUM_NODES:-3}

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     分布式大模型推理系统 - 去中心化版本                               ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# 检查Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python3 未安装${NC}"
    exit 1
fi

# 检查依赖
echo -e "${YELLOW}检查依赖...${NC}"
pip3 install --user psutil 2>/dev/null || true

# 创建日志目录
mkdir -p logs pids

# 停止旧进程
echo -e "${YELLOW}停止旧进程...${NC}"
pkill -f "node_decentralized.py" 2>/dev/null || true
sleep 2

# 启动节点
echo -e "${GREEN}启动去中心化节点...${NC}"
echo ""
echo -e "  模型: ${YELLOW}${MODEL_NAME}${NC}"
echo -e "  发现端口: ${YELLOW}${DISCOVERY_PORT}${NC}"
echo -e "  API端口: ${YELLOW}${API_PORT}${NC}"
echo ""

# 启动多个节点
for i in $(seq 1 $NUM_NODES); do
    PORT=$((API_PORT + i - 1))
    
    echo -e "${GREEN}启动节点 $i (端口: $PORT)...${NC}"
    
    python3 download/node_decentralized.py \
        --port $PORT \
        --discovery-port $DISCOVERY_PORT \
        --model "$MODEL_NAME" \
        > "logs/node_$i.log" 2>&1 &
    
    echo $! > "pids/node_$i.pid"
    
    sleep 1
done

echo ""
echo -e "${GREEN}✅ 启动完成！${NC}"
echo ""
echo -e "节点状态:"
for i in $(seq 1 $NUM_NODES); do
    PORT=$((API_PORT + i - 1))
    echo -e "  节点 $i: http://localhost:$PORT"
done

echo ""
echo -e "${BLUE}测试命令:${NC}"
echo -e "  # 查看节点状态"
echo -e "  curl -s http://localhost:${API_PORT} -d '{\"type\":\"status\"}' | python3 -m json.tool"
echo ""
echo -e "  # 发送推理请求"
echo -e "  curl -s http://localhost:${API_PORT} -d '{\"type\":\"inference\",\"prompt\":\"你好\"}' | python3 -m json.tool"
echo ""
echo -e "${BLUE}停止服务:${NC}"
echo -e "  pkill -f node_decentralized.py"
echo ""
echo -e "${GREEN}服务运行中... 按 Ctrl+C 停止${NC}"

# 等待
wait
