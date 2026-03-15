# 分布式大模型推理系统

<div align="center">

![Version](https://img.shields.io/badge/version-2.0.0-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-production%20ready-success)

**动态算力调度 · 自动故障恢复 · 网络感知负载均衡**

[功能特性](#功能特性) • [快速开始](#快速开始) • [架构设计](#架构设计) • [使用指南](#使用指南)

</div>

---

## 📋 功能特性

### 核心功能

| 功能 | 状态 | 说明 |
|------|------|------|
| 动态节点加入 | ✅ | 新节点自动注册并获取任务分配 |
| 自动故障恢复 | ✅ | 节点离线时自动重新路由任务 |
| 网络感知调度 | ✅ | 根据延迟、带宽选择最优节点 |
| 智能负载均衡 | ✅ | 综合评分算法选择最佳节点 |
| 多槽位并行 | ✅ | 每个节点支持多个并发任务 |
| 实时监控 | ✅ | 节点状态、性能指标可视化 |

### 并行策略

| 策略 | 状态 | 说明 |
|------|------|------|
| 数据并行 | ✅ | 多节点处理不同请求（当前实现） |
| 请求级并行 | ✅ | 多个请求并行处理 |
| Pipeline并行 | ⚠️ | 需要使用Petals框架 |

---

## 🚀 快速开始

### 方式一：一键启动（推荐）

```bash
# 启动所有服务
./scripts/start.sh

# 查看服务状态
./scripts/start.sh --status

# 停止所有服务
./scripts/start.sh --stop
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

# 3. 启动节点服务（新终端）
pip install torch transformers python-socketio psutil
python download/node_service_production.py \
  --server http://localhost:3003 \
  --model Qwen/Qwen2.5-0.5B-Instruct
```

### 访问地址

- **Web管理界面**: http://localhost:3000
- **Orchestrator API**: http://localhost:3004/api
- **WebSocket**: ws://localhost:3003

---

## 🏗️ 架构设计

```
┌─────────────────────────────────────────────────────────────────┐
│                    Web管理界面 (Next.js)                         │
│                      http://localhost:3000                       │
└─────────────────────────┬───────────────────────────────────────┘
                          │ WebSocket
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                  中央调度服务 (Orchestrator)                      │
│              WebSocket: 3003 / HTTP API: 3004                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ 节点注册中心 │  │ 任务调度器   │  │ 负载均衡器   │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────┬───────────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│   节点 A      │ │   节点 B      │ │   节点 C      │
│ (完整模型)   │ │ (完整模型)   │ │ (完整模型)   │
│ Windows 4GB   │ │ macOS 8GB     │ │ Linux 16GB    │
└───────────────┘ └───────────────┘ └───────────────┘
```

---

## 📖 使用指南

### 生产模式（真实推理）

```bash
# 安装依赖
pip install torch transformers python-socketio psutil

# 启动节点（小模型，适合4GB内存）
python download/node_service_production.py \
  --server http://localhost:3003 \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --workers 2

# 启动节点（中等模型，适合8GB内存）
python download/node_service_production.py \
  --server http://localhost:3003 \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --workers 2
```

### 模型选择建议

| 模型 | 内存需求 | 推荐配置 |
|------|---------|---------|
| Qwen2.5-0.5B | ~1GB | 4GB内存 |
| Qwen2.5-1.5B | ~3GB | 8GB内存 |
| Qwen2.5-3B | ~6GB | 16GB内存 |
| Qwen2.5-7B | ~14GB | 32GB内存 |

---

## 📁 项目结构

```
my-project/
├── src/app/                    # Web管理界面
│   └── page.tsx                # 主页面
├── mini-services/
│   ├── orchestrator/           # 中央调度服务
│   │   └── index.ts
│   └── simulator/              # 模拟节点服务
│       └── index.ts
├── download/
│   ├── node_service_production.py  # 生产级节点服务 ⭐
│   ├── node_service_network.py     # 网络感知节点服务
│   ├── node_service_real.py        # 真实推理节点
│   └── node_service.py             # 模拟节点脚本
├── config/
│   └── default.json            # 系统配置
├── scripts/
│   └── start.sh                # 启动脚本
├── venv/                       # Python虚拟环境
├── IMPLEMENTATION.md           # 实现说明
├── IMPLEMENTATION_DETAILS.md   # 详细实现说明
└── README.md                   # 本文件
```

---

## ⚙️ 配置说明

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

## 🔧 API 文档

### HTTP API

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/status` | GET | 获取系统状态 |
| `/api/metrics` | GET | 获取系统指标 |
| `/api/inference` | POST | 创建推理任务 |
| `/api/task/:id` | GET | 查询任务状态 |

### WebSocket 事件

| 事件 | 方向 | 说明 |
|------|------|------|
| `node:register` | 客户端→服务端 | 注册节点 |
| `node:heartbeat` | 客户端→服务端 | 发送心跳 |
| `task:inference` | 服务端→客户端 | 分配推理任务 |
| `inference:result` | 客户端→服务端 | 返回推理结果 |

---

## 🐛 常见问题

### Q: 模型下载慢怎么办？

```bash
# 使用镜像
export HF_ENDPOINT=https://hf-mirror.com
```

### Q: 内存不足怎么办？

```bash
# 使用更小的模型或量化
python node_service_production.py \
  --model Qwen/Qwen2.5-0.5B-Instruct
```

### Q: GPU不被识别？

```bash
# 检查CUDA
python -c "import torch; print(torch.cuda.is_available())"

# 安装正确的PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

---

## 📊 负载均衡算法

综合评分 = 负载(15%) + 吞吐量(20%) + 网络(25%) + 槽位(15%) + 成功率(15%) + 历史延迟(10%)

```typescript
function calculateNodeScore(node: NodeInfo): number {
  return (
    loadScore * 0.15 +
    throughputScore * 0.20 +
    networkScore * 0.25 +
    slotScore * 0.15 +
    successRate * 0.15 +
    latencyHistoryScore * 0.10
  );
}
```

---

## 📄 许可证

MIT License

---

## 🙏 致谢

- [Qwen](https://github.com/QwenLM/Qwen) - 模型支持
- [Socket.IO](https://socket.io/) - 实时通信
- [Next.js](https://nextjs.org/) - Web框架
- [shadcn/ui](https://ui.shadcn.com/) - UI组件

---

<div align="center">

**⭐ 如果这个项目对你有帮助，请给一个 Star！**

</div>
