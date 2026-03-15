# 分布式大模型推理系统

<div align="center">

**动态算力调度 · 自动故障恢复 · 网络感知负载均衡**

一个轻量级、生产级的分布式大模型推理系统，支持多节点协同推理。

</div>

---

## 功能特性

- ✅ **动态节点管理** - 节点自动注册、心跳检测、故障恢复
- ✅ **智能负载均衡** - 6维度综合评分算法选择最优节点
- ✅ **网络感知调度** - 根据延迟、带宽优化任务分配
- ✅ **多槽位并行** - 每个节点支持多个并发推理任务
- ✅ **实时监控** - Web界面实时查看节点状态和任务进度
- ✅ **健康检查** - 支持/health、/ready、/metrics端点

---

## 快速开始

### 方式一：一键启动

```bash
# 克隆项目
git clone https://github.com/ctz168/sm.git
cd sm

# 启动所有服务
bash scripts/start.sh
```

### 方式二：手动启动

```bash
# 1. 启动中央调度服务
cd mini-services/orchestrator
bun install
bun run dev

# 2. 启动Web管理界面（新终端）
cd ../..
bun run dev

# 3. 启动推理节点（新终端）
pip install torch transformers python-socketio psutil
python download/node_service_optimized.py \
  --server http://localhost:3003 \
  --model Qwen/Qwen2.5-0.5B-Instruct
```

### 访问地址

| 服务 | 地址 | 说明 |
|------|------|------|
| Web管理界面 | http://localhost:3000 | 节点监控和任务管理 |
| Orchestrator API | http://localhost:3004/api | REST API |
| 健康检查 | http://localhost:3004/health | 系统状态 |
| Prometheus指标 | http://localhost:3004/metrics | 监控指标 |

---

## 项目结构

```
sm/
├── download/                      # 节点服务
│   ├── node_service_optimized.py  # 优化版节点服务 ⭐
│   ├── node_service_production.py # 生产版节点服务
│   └── model_sharding_analysis.py # 模型分片分析工具
│
├── mini-services/
│   └── orchestrator/              # 中央调度服务
│       ├── index.ts               # 调度器核心代码
│       └── package.json
│
├── src/app/                       # Web管理界面
│   └── page.tsx                   # 主页面
│
├── config/
│   └── default.json               # 系统配置
│
├── scripts/
│   └── start.sh                   # 一键启动脚本
│
└── README.md                      # 本文件
```

---

## 使用指南

### 启动推理节点

```bash
# 基本用法
python download/node_service_optimized.py \
  --server http://localhost:3003 \
  --model Qwen/Qwen2.5-0.5B-Instruct

# 完整参数
python download/node_service_optimized.py \
  --server http://localhost:3003 \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --name MyNode \
  --workers 4 \
  --memory-limit 8 \
  --quantize
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--server` | 中央调度服务地址 | 必填 |
| `--model` | 模型名称或路径 | Qwen/Qwen2.5-0.5B-Instruct |
| `--name` | 节点名称 | 自动生成 |
| `--workers` | 并行工作线程数 | 2 |
| `--memory-limit` | 内存限制(GB) | 系统内存 |
| `--quantize` | 启用量化 | 否 |

### 发送推理请求

```bash
# HTTP API
curl -X POST http://localhost:3004/api/inference \
  -H "Content-Type: application/json" \
  -d '{"prompt": "你好，请介绍一下你自己。"}'

# 响应
{"taskId": "xxx", "status": "queued"}
```

---

## 模型选择建议

| 模型 | 内存需求 | 推荐配置 | 下载时间(100Mbps) |
|------|---------|---------|------------------|
| Qwen2.5-0.5B | ~1GB | 4GB内存 | 1.4分钟 |
| Qwen2.5-1.5B | ~3GB | 8GB内存 | 3.8分钟 |
| Qwen2.5-3B | ~6GB | 16GB内存 | 8分钟 |
| Qwen2.5-7B | ~14GB | 32GB内存 | 16分钟 |

---

## 性能指标

### 单节点性能 (Qwen2.5-0.5B, CPU)

| 指标 | 数值 |
|------|------|
| 模型加载 | 4秒 |
| 推理延迟 | 2.6秒 (50 tokens) |
| 吞吐量 | 18.9 tokens/s |
| 内存占用 | 1.84GB |

### 多节点扩展

| 节点数 | 吞吐量 (理论) |
|--------|--------------|
| 1 | 18.9 t/s |
| 2 | 37.8 t/s |
| 4 | 75.6 t/s |
| N | 18.9×N t/s |

---

## API 文档

### REST API

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/status` | GET | 获取系统状态 |
| `/api/inference` | POST | 创建推理任务 |
| `/api/task/:id` | GET | 查询任务状态 |
| `/health` | GET | 健康检查 |
| `/ready` | GET | 就绪检查 |
| `/metrics` | GET | Prometheus指标 |

### WebSocket 事件

| 事件 | 方向 | 说明 |
|------|------|------|
| `node:register` | 节点→服务器 | 注册节点 |
| `node:heartbeat` | 节点→服务器 | 发送心跳 |
| `task:inference` | 服务器→节点 | 分配推理任务 |
| `inference:result` | 节点→服务器 | 返回推理结果 |

---

## 配置说明

编辑 `config/default.json`：

```json
{
  "scheduling": {
    "heartbeatTimeout": 60000,
    "maxRetries": 3,
    "taskTimeout": 30000
  },
  "parallel": {
    "maxBatchSize": 8,
    "maxParallelism": 16
  }
}
```

---

## 故障排除

### 模型下载慢

```bash
# 使用镜像
export HF_ENDPOINT=https://hf-mirror.com
```

### 内存不足

```bash
# 使用更小的模型
python download/node_service_optimized.py \
  --model Qwen/Qwen2.5-0.5B-Instruct

# 或启用量化
python download/node_service_optimized.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --quantize
```

### GPU不被识别

```bash
# 检查CUDA
python -c "import torch; print(torch.cuda.is_available())"

# 安装正确的PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

---

## 技术栈

- **后端**: TypeScript + Socket.IO
- **前端**: Next.js + shadcn/ui
- **推理**: PyTorch + Transformers
- **通信**: WebSocket + REST API

---

## 许可证

MIT License

---

## 致谢

- [Qwen](https://github.com/QwenLM/Qwen) - 模型支持
- [Socket.IO](https://socket.io/) - 实时通信
- [Next.js](https://nextjs.org/) - Web框架
- [shadcn/ui](https://ui.shadcn.com/) - UI组件
