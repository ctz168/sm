# 分布式大模型推理系统

<div align="center">

**动态算力调度 · 自动故障恢复 · Pipeline并行分片**

一个轻量级、生产级的分布式大模型推理系统，支持多节点协同推理。

</div>

---

## 功能特性

- ✅ **Pipeline并行分片** - 每个节点只加载部分层，内存节省33%
- ✅ **去中心化架构** - 无单点故障，服务永不中断
- ✅ **自动节点发现** - 节点自动发现和加入集群
- ✅ **领导者选举** - Raft算法自动选举领导者
- ✅ **故障自动转移** - 领导者故障时自动重新选举
- ✅ **动态算力管理** - 资源不足时自动降级，恢复后自动启用

---

## 三种部署模式

### 模式一：Pipeline并行分片（推荐，内存最优）

**特点：每个节点只加载部分层，内存效率最高**

```
┌─────────────┐     ┌─────────────┐
│   节点 1    │────│   节点 2    │
│ 层 0-11    │     │ 层 12-23   │
│ 1.2GB      │     │ 1.2GB      │
└─────────────┘     └─────────────┘
      ↓                   ↓
   嵌入层              输出层

总内存: 2.4GB (vs 完整模型3.6GB)
```

### 模式二：去中心化模式

**特点：无单点故障，服务永不中断**

```
┌─────────┐     ┌─────────┐     ┌─────────┐
│ 节点 A  │────│ 节点 B  │────│ 节点 C  │
│(Leader) │     │(Follower)│    │(Follower)│
└─────────┘     └─────────┘     └─────────┘
```

### 模式三：中心化模式

**特点：有Web管理界面，适合监控需求**

```
┌────────────────┐
│  Orchestrator  │
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

### Pipeline并行分片（推荐）

```bash
# 克隆项目
git clone https://github.com/ctz168/sm.git
cd sm

# 安装依赖
pip install torch transformers psutil

# 启动2个节点（每个节点加载12层）
# 节点1: 层0-11
python download/node_pipeline_shard.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --index 0 --total 2 \
    --port 6000 \
    --next-host localhost --next-port 6001 &

# 节点2: 层12-23
python download/node_pipeline_shard.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --index 1 --total 2 \
    --port 6001 &
```

### 去中心化模式

```bash
# 启动3个节点（自动发现、自动选举）
bash scripts/start_decentralized.sh
```

---

## Pipeline并行优势

### 内存对比

| 模型 | 完整模型 | Pipeline 2节点 | Pipeline 4节点 |
|------|---------|---------------|---------------|
| Qwen2.5-0.5B | 1.8GB/节点 | 1.2GB/节点 | 0.6GB/节点 |
| Qwen2.5-7B | 14GB/节点 | 7GB/节点 | 3.5GB/节点 |
| Qwen2.5-32B | 64GB/节点 | 32GB/节点 | 16GB/节点 |

### 大模型支持

```
Qwen2.5-7B (14GB):
  ❌ 完整模型: 单节点8GB内存无法运行
  ✅ Pipeline 4节点: 每节点3.5GB，可以运行!

Qwen2.5-32B (64GB):
  ❌ 完整模型: 需要高端服务器
  ✅ Pipeline 8节点: 每节点8GB，普通PC即可!
```

---

## 项目结构

```
sm/
├── download/
│   ├── node_pipeline_shard.py     # Pipeline并行分片 ⭐
│   ├── node_decentralized.py      # 去中心化节点
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

### Pipeline并行 (Qwen2.5-0.5B, 2节点)

| 指标 | 数值 |
|------|------|
| 每节点内存 | 1.2GB (vs 1.8GB) |
| 总内存 | 2.4GB (vs 3.6GB) |
| 内存节省 | 33% |
| 推理延迟 | ~3秒 (含节点间通信) |

### 单节点性能 (Qwen2.5-0.5B, CPU)

| 指标 | 数值 |
|------|------|
| 模型加载 | 4秒 |
| 推理延迟 | 2.6秒 (50 tokens) |
| 吞吐量 | 18.9 tokens/s |

---

## 技术栈

- **Pipeline并行**: 按层分片
- **共识算法**: Raft
- **推理引擎**: PyTorch + Transformers
- **通信协议**: TCP Socket + JSON

---

## 许可证

MIT License

---

## 致谢

- [Qwen](https://github.com/QwenLM/Qwen) - 模型支持
- [Raft](https://raft.github.io/) - 共识算法
