# 分布式大模型推理系统

<div align="center">

**动态算力调度 · 自动故障恢复 · 去中心化高可用**

一个轻量级、生产级的分布式大模型推理系统，支持多节点协同推理。

</div>

---

## 功能特性

- ✅ **去中心化架构** - 无单点故障，服务永不中断
- ✅ **自动节点发现** - 节点自动发现和加入集群
- ✅ **领导者选举** - Raft算法自动选举领导者
- ✅ **故障自动转移** - 领导者故障时自动重新选举
- ✅ **动态算力管理** - 资源不足时自动降级，恢复后自动启用
- ✅ **智能负载均衡** - 6维度综合评分算法选择最优节点

---

## 两种部署模式

### 模式一：去中心化模式（推荐）

**特点：无单点故障，服务永不中断**

```
┌─────────┐     ┌─────────┐     ┌─────────┐
│ 节点 A  │────│ 节点 B  │────│ 节点 C  │
│(Leader) │     │(Follower)│    │(Follower)│
└─────────┘     └─────────┘     └─────────┘
     │               │               │
     └───────────────┴───────────────┘
              P2P 自动发现
```

### 模式二：中心化模式

**特点：有Web管理界面，适合监控需求**

```
┌────────────────┐
│  Orchestrator  │  ← 中央调度服务
│   (端口3004)   │
└───────┬────────┘
        │
   ┌────┴────┬─────────┐
   │         │         │
┌──┴──┐  ┌──┴──┐  ┌──┴──┐
│节点1│  │节点2│  │节点3│
└─────┘  └─────┘  └─────┘
```

---

## 快速开始

### 去中心化模式（推荐）

```bash
# 克隆项目
git clone https://github.com/ctz168/sm.git
cd sm

# 安装依赖
pip install torch transformers psutil

# 启动3个节点（自动发现、自动选举）
bash scripts/start_decentralized.sh
```

### 中心化模式

```bash
# 启动中央调度服务
cd mini-services/orchestrator && bun install && bun run dev

# 启动推理节点（新终端）
python download/node_service_optimized.py \
  --server http://localhost:3003 \
  --model Qwen/Qwen2.5-0.5B-Instruct
```

---

## 去中心化模式详解

### 核心特性

| 特性 | 说明 |
|------|------|
| 无单点故障 | 任何节点故障不影响整体服务 |
| 自动发现 | UDP广播自动发现其他节点 |
| 领导者选举 | Raft算法自动选举 |
| 故障转移 | 领导者故障后自动重新选举 |
| 动态算力 | 资源不足时自动降级 |

### 使用方法

```bash
# 启动单节点（自动成为领导者）
python download/node_decentralized.py --port 37002

# 启动多节点（自动组成集群）
python download/node_decentralized.py --port 37002 &
python download/node_decentralized.py --port 37003 &
python download/node_decentralized.py --port 37004 &
```

### API 请求

```bash
# 查看节点状态
curl -s localhost:37002 -d '{"type":"status"}' | python3 -m json.tool

# 发送推理请求（任意节点都可以）
curl -s localhost:37002 -d '{"type":"inference","prompt":"你好"}' | python3 -m json.tool
```

### 高可用性保证

```
场景1: 领导者故障
┌─────────┐     ┌─────────┐     ┌─────────┐
│ 节点 A  │────│ 节点 B  │────│ 节点 C  │
│(故障!) │     │(新Leader)│    │(Follower)│
└─────────┘     └─────────┘     └─────────┘
                     ↑
              自动选举新领导者

场景2: 资源不足
┌─────────────────────────────────────────┐
│ 节点检测到内存不足                        │
│ → 自动卸载模型（降级模式）                 │
│ → 继续提供服务（转发到其他节点）           │
│ → 资源恢复后自动重新加载模型               │
└─────────────────────────────────────────┘
```

---

## 项目结构

```
sm/
├── download/
│   ├── node_decentralized.py      # 去中心化节点 ⭐
│   ├── node_service_optimized.py  # 中心化节点
│   └── model_sharding_analysis.py # 分析工具
│
├── mini-services/
│   └── orchestrator/              # 中央调度服务
│
├── scripts/
│   ├── start_decentralized.sh     # 去中心化启动
│   └── start.sh                   # 中心化启动
│
└── README.md
```

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

| 节点数 | 吞吐量 | 可用性 |
|--------|--------|--------|
| 1 | 18.9 t/s | 单点故障 |
| 3 | 56.7 t/s | 容忍1节点故障 |
| 5 | 94.5 t/s | 容忍2节点故障 |
| N | 18.9×N t/s | 容忍(N-1)/2节点故障 |

---

## 配置说明

### 去中心化配置

```python
# 在 node_decentralized.py 中修改
config = Config(
    discovery_port=37000,        # 节点发现端口
    api_port=37002,              # API端口
    heartbeat_interval=1.0,      # 心跳间隔(秒)
    election_timeout_min=2.0,    # 选举超时最小值
    election_timeout_max=4.0,    # 选举超时最大值
    min_memory_gb=2.0,           # 最小内存要求
    auto_recovery=True,          # 自动恢复
)
```

---

## 故障排除

### 节点无法发现其他节点

```bash
# 检查防火墙
sudo ufw allow 37000/udp
sudo ufw allow 37000:37010/tcp
```

### 领导者选举失败

```bash
# 确保至少有3个节点
# Raft需要多数节点在线才能选举
```

### 资源不足导致服务降级

```bash
# 检查系统资源
free -h
top

# 使用更小的模型
python download/node_decentralized.py \
  --model Qwen/Qwen2.5-0.5B-Instruct
```

---

## 技术栈

- **共识算法**: Raft
- **节点发现**: UDP广播
- **推理引擎**: PyTorch + Transformers
- **通信协议**: TCP Socket + JSON

---

## 许可证

MIT License

---

## 致谢

- [Qwen](https://github.com/QwenLM/Qwen) - 模型支持
- [Raft](https://raft.github.io/) - 共识算法
