# 分布式大模型推理系统 - 实现说明

## 🔍 代码审查发现的问题及解决方案

### 问题1: Pipeline并行未真正实现

**问题描述**:
- 分片定义了层范围（layerStart, layerEnd），但没有层间数据传递机制
- 每个节点独立运行完整模型，无法实现真正的Pipeline并行

**解决方案**:
- 采用**数据并行**方案：每个节点加载完整模型，处理不同请求
- 这是业界主流方案（vLLM、TGI等）
- 真正的Pipeline并行需要更复杂的实现（如Petals框架）

**当前实现**:
```python
# 每个节点加载完整模型
self.model = AutoModelForCausalLM.from_pretrained(model_name)

# 独立处理请求
response = self.model.generate(prompt)
```

### 问题2: 结果合并逻辑简单

**问题描述**:
- 没有合并多个并行任务结果的逻辑
- 并行任务结果无法正确合并

**解决方案**:
- 对于数据并行，每个任务独立完成，无需合并
- 对于批处理，结果按任务ID分别返回

**当前实现**:
```typescript
// 每个任务独立返回结果
task.status = 'completed';
task.result = data.result;
```

### 问题3: 分片迁移是模拟的

**问题描述**:
- 没有真正的模型权重传输
- 节点故障时无法真正恢复模型

**解决方案**:
- 采用**重新加载**方案：新节点从HuggingFace下载模型
- 或者使用共享存储（NFS、S3）存储模型

**当前实现**:
```python
# 新节点加载模型
engine = InferenceEngine(model_name)
engine.load()  # 从HuggingFace下载
```

### 问题4: 负载均衡可以优化

**问题描述**:
- 没有考虑历史性能数据
- 可能选择次优节点

**解决方案**:
- 已实现综合评分算法，考虑：
  - CPU负载 (15%)
  - 吞吐量 (20%)
  - 网络延迟 (25%)
  - 槽位可用性 (15%)
  - 历史成功率 (15%)
  - 历史延迟 (10%)

**当前实现**:
```typescript
function calculateNodeScore(node: NodeInfo): number {
  const compositeScore = 
    loadScore * 0.15 +
    throughputScore * 0.20 +
    networkScore * 0.25 +
    slotScore * 0.15 +
    successRate * 0.15 +
    latencyHistoryScore * 0.10;
  return compositeScore;
}
```

### 问题5: 没有模型文件管理

**问题描述**:
- 没有下载/存储模型分片的逻辑
- 节点无法获取模型文件

**解决方案**:
- 使用HuggingFace自动下载
- 支持本地模型路径
- 可配置模型缓存目录

**当前实现**:
```python
# 自动从HuggingFace下载
model = AutoModelForCausalLM.from_pretrained(
    model_name,  # 可以是HF模型名或本地路径
    cache_dir="/path/to/cache"
)
```

---

## 🏗️ 当前架构

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
│ 数据并行     │ │ 数据并行     │ │ 数据并行     │
└───────────────┘ └───────────────┘ └───────────────┘
```

---

## 📊 并行策略说明

### 数据并行（当前实现）

每个节点加载完整模型，独立处理请求：

```
请求1 ──→ 节点A (完整模型) ──→ 结果1
请求2 ──→ 节点B (完整模型) ──→ 结果2
请求3 ──→ 节点C (完整模型) ──→ 结果3
```

**优点**:
- 实现简单
- 故障恢复容易
- 无需节点间通信

**缺点**:
- 每个节点需要足够内存
- 无法运行超大模型

### Pipeline并行（未实现）

模型按层切分，数据流水线处理：

```
请求 ──→ 节点A (层1-10) ──→ 节点B (层11-20) ──→ 节点C (层21-30) ──→ 结果
```

**优点**:
- 可以运行超大模型
- 内存利用率高

**缺点**:
- 实现复杂
- 需要节点间高速通信
- 故障恢复困难

**推荐方案**: 使用Petals框架

---

## 🚀 使用指南

### 1. 启动中央调度服务

```bash
cd mini-services/orchestrator
bun install
bun run dev
```

### 2. 启动Web管理界面

```bash
bun run dev
```

### 3. 启动节点服务

**生产级节点（推荐）**:
```bash
# 安装依赖
pip install torch transformers python-socketio psutil

# 启动节点
python download/node_service_production.py \
  --server http://localhost:3003 \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --workers 2
```

**网络感知节点**:
```bash
source venv/bin/activate
python download/node_service_network.py \
  --server http://localhost:3003 \
  --model Qwen/Qwen2.5-0.5B-Instruct
```

### 4. 发送推理请求

```bash
curl -X POST http://localhost:3004/api/inference \
  -H "Content-Type: application/json" \
  -d '{"prompt": "你好，请介绍一下你自己。"}'
```

---

## 📈 性能优化建议

### 1. 使用GPU

```python
# 检测GPU
if torch.cuda.is_available():
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
```

### 2. 使用量化

```python
# 8-bit量化
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,
    device_map="auto"
)

# 4-bit量化
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    device_map="auto"
)
```

### 3. 调整并行槽位

```bash
# 根据CPU核心数调整
python node_service_production.py \
  --workers 4  # 建议: CPU核心数 / 2
```

---

## 📁 文件说明

| 文件 | 说明 | 状态 |
|------|------|------|
| `node_service_production.py` | 生产级节点服务 | ✅ 推荐 |
| `node_service_network.py` | 网络感知节点服务 | ✅ 可用 |
| `node_service_real.py` | 真实推理节点 | ⚠️ 旧版本 |
| `node_service.py` | 模拟节点服务 | ⚠️ 仅测试用 |

---

## 🔧 故障排除

### 问题: 模型下载慢

```bash
# 使用镜像
export HF_ENDPOINT=https://hf-mirror.com
```

### 问题: 内存不足

```bash
# 使用更小的模型或量化
python node_service_production.py \
  --model Qwen/Qwen2.5-0.5B-Instruct  # 最小模型
```

### 问题: GPU不被识别

```bash
# 检查CUDA
python -c "import torch; print(torch.cuda.is_available())"

# 安装正确的PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

---

## 📝 总结

当前系统实现了：
- ✅ 真实模型加载和推理
- ✅ 数据并行（多节点处理不同请求）
- ✅ 网络感知负载均衡
- ✅ 故障检测和恢复
- ✅ 多槽位并行处理

未实现（需要更复杂方案）：
- ⚠️ Pipeline并行（推荐使用Petals）
- ⚠️ 模型分片传输（推荐使用共享存储）
- ⚠️ 跨节点KV Cache共享

对于大多数场景，当前的数据并行方案已经足够。
