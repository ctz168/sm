# 分布式大模型推理系统 - 实现说明

## 📋 系统架构

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
│  │ 节点注册中心 │  │ 任务调度器   │  │ 分片分配器   │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────┬───────────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│   节点 A      │ │   节点 B      │ │   节点 C      │
│ (模型分片1)   │ │ (模型分片2)   │ │ (模型分片3)   │
│ Windows 4GB   │ │ macOS 8GB     │ │ Linux 16GB    │
└───────────────┘ └───────────────┘ └───────────────┘
```

## 🔄 实现状态对比

### 当前版本功能状态

| 功能 | 模拟版本 | 真实版本 | 说明 |
|------|---------|---------|------|
| 节点注册 | ✅ | ✅ | 完全实现 |
| 心跳检测 | ✅ | ✅ | 完全实现 |
| 分片分配 | ✅ | ✅ | 完全实现 |
| 故障恢复 | ✅ | ✅ | 完全实现 |
| 负载均衡 | ✅ | ✅ | 完全实现 |
| **模型加载** | ⚠️ 模拟 | ✅ 真实 | 见下方说明 |
| **推理执行** | ⚠️ 模拟 | ✅ 真实 | 见下方说明 |
| **Pipeline并行** | ⚠️ 未实现 | ⚠️ 部分实现 | 需要更多工作 |

### 模拟版本 (node_service.py)

**用途**: 系统测试、演示、功能验证

**特点**:
- 不需要GPU或大量内存
- 快速启动，用于测试调度逻辑
- 返回模拟的推理结果

**限制**:
- ❌ 无法运行真实模型
- ❌ 无法处理真实数据
- ❌ 性能数据是模拟的

### 真实版本 (node_service_real.py)

**用途**: 生产环境、实际推理

**特点**:
- ✅ 真正加载模型权重
- ✅ 真正执行推理
- ✅ 支持GPU加速
- ✅ 真实的性能指标

**要求**:
- Python 3.8+
- PyTorch
- Transformers
- 足够的内存/显存

## 🚀 快速开始

### 1. 启动中央调度服务

```bash
cd mini-services/orchestrator
bun install
bun run dev
```

### 2. 启动Web管理界面

```bash
cd /home/z/my-project
bun run dev
```

### 3. 启动节点服务

**模拟模式（测试用）**:
```bash
cd mini-services/simulator
bun run dev
```

**真实模式（生产用）**:
```bash
# 安装依赖
pip install socketio-client psutil torch transformers accelerate

# 启动节点（小模型，适合4GB内存）
python download/node_service_real.py --server http://localhost:3003 --model Qwen/Qwen2.5-0.5B-Instruct

# 启动节点（中等模型，适合8GB内存）
python download/node_service_real.py --server http://localhost:3003 --model Qwen/Qwen2.5-1.5B-Instruct

# 启动节点（大模型，适合16GB+内存）
python download/node_service_real.py --server http://localhost:3003 --model Qwen/Qwen2.5-3B-Instruct
```

## 📊 模型选择建议

| 模型 | 内存需求 | 推荐配置 | 说明 |
|------|---------|---------|------|
| Qwen2.5-0.5B-Instruct | ~1GB | 4GB内存 | 最小模型，适合测试 |
| Qwen2.5-1.5B-Instruct | ~3GB | 8GB内存 | 小型模型，基本可用 |
| Qwen2.5-3B-Instruct | ~6GB | 16GB内存 | 中型模型，效果较好 |
| Qwen2.5-7B-Instruct | ~14GB | 32GB内存 | 大型模型，需要量化 |
| Qwen2.5-14B-Instruct | ~28GB | 64GB内存 | 超大模型，需要分布式 |

## ⚠️ 当前限制

### 1. Pipeline并行

**当前状态**: 部分实现

**问题**:
- 层间数据传递需要更多工作
- 隐藏状态序列化/反序列化开销大
- 需要精确的层边界处理

**解决方案**:
- 使用 Petals 框架（专门为分布式推理设计）
- 或使用 DeepSpeed/Megatron-LM

### 2. 模型分片

**当前状态**: 每个节点加载完整模型

**问题**:
- 内存利用率不高
- 无法运行超大模型

**解决方案**:
```python
# 使用 Petals 进行真正的模型分片
from petals import DistributedModel

model = DistributedModel(
    "bigscience/bloom-petals",
    initial_peers=["peer1", "peer2"]
)
```

### 3. KV Cache共享

**当前状态**: 未实现

**问题**:
- 长对话时内存占用大
- 无法跨节点共享缓存

## 🔧 推荐的生产配置

### 方案一：使用 Petals（推荐）

Petals 是专门为分布式大模型推理设计的框架：

```bash
pip install petals

# 启动节点
python -m petals.cli.run_server --model bigscience/bloom-petals
```

### 方案二：使用 vLLM

vLLM 提供高效的推理服务：

```bash
pip install vllm

# 启动服务
python -m vllm.entrypoints.api_server --model Qwen/Qwen2.5-7B-Instruct
```

### 方案三：使用本系统（简化版）

适合小规模部署：

```bash
# 每个节点运行完整的小模型
python node_service_real.py --server http://orchestrator:3003 --model Qwen/Qwen2.5-1.5B-Instruct
```

## 📈 性能优化建议

1. **使用GPU**: 确保安装CUDA版本的PyTorch
2. **量化**: 使用8-bit或4-bit量化减少内存
3. **批处理**: 合并多个请求提高吞吐量
4. **Flash Attention**: 启用Flash Attention加速

```python
# 启用量化
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,  # 或 load_in_4bit=True
    device_map="auto"
)
```

## 🐛 常见问题

### Q: 模型下载慢怎么办？
```bash
# 使用镜像
export HF_ENDPOINT=https://hf-mirror.com
```

### Q: 内存不足怎么办？
```bash
# 使用更小的模型或量化
python node_service_real.py --model Qwen/Qwen2.5-0.5B-Instruct
```

### Q: GPU不被识别？
```bash
# 检查CUDA
python -c "import torch; print(torch.cuda.is_available())"

# 安装正确的PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## 📝 总结

| 场景 | 推荐方案 |
|------|---------|
| 测试/演示 | 使用模拟版本 |
| 小规模生产 | 使用真实版本 + 小模型 |
| 大规模生产 | 使用 Petals/vLLM |
| 超大模型 | 使用专业分布式框架 |
