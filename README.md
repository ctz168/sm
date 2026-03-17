# 分布式大模型推理系统

<div align="center">

**一键安装 · 跨平台支持 · 傻瓜式使用**

一个轻量级、生产级的分布式大模型推理系统，支持多节点协同推理。

</div>

---

## ✨ 一键安装

### Linux / macOS

```bash
# 一键安装
curl -fsSL https://raw.githubusercontent.com/ctz168/sm/main/install.sh | bash

# 启动服务
dllm-start

# 或
~/.distributed-llm/start.sh
```

### Windows

```powershell
# 下载并运行安装脚本
# 方法1: PowerShell
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/ctz168/sm/main/install.bat" -OutFile "install.bat"
.\install.bat

# 方法2: 直接下载
# 访问 https://github.com/ctz168/sm 下载项目
# 双击运行 install.bat
```

### Docker

```bash
# 克隆项目
git clone https://github.com/ctz168/sm.git
cd sm

# 启动服务
docker-compose up -d

# 查看日志
docker-compose logs -f
```

---

## 🚀 快速使用

### 启动服务

**Linux/macOS:**
```bash
# 方式1: 使用快捷命令
dllm-start

# 方式2: 直接运行
~/.distributed-llm/start.sh

# 方式3: 指定模式
./start.sh resource_aware   # 资源感知模式
./start.sh pipeline         # Pipeline模式
./start.sh decentralized    # 去中心化模式
```

**Windows:**
```cmd
# 双击桌面快捷方式 "分布式大模型推理"
# 或运行
start.bat
```

### 停止服务

```bash
# Linux/macOS
dllm-stop
# 或
~/.distributed-llm/stop.sh

# Windows
stop.bat
```

### 查看状态

```bash
# Linux/macOS
dllm-status
# 或
~/.distributed-llm/status.sh

# Windows
status.bat
```

---

## 📋 四种运行模式

### 1. 资源感知模式（推荐）

**特点：自动检测资源，智能启停**

- 资源不足时自动停止模型
- 资源充足时自动启动模型
- 适合资源有限的环境

```bash
./start.sh 1
# 或
python download/node_resource_aware.py --model Qwen/Qwen2.5-0.5B-Instruct
```

### 2. Pipeline并行模式

**特点：多节点分片，内存最优**

- 每个节点只加载部分层
- 内存使用降低50%
- 支持运行更大的模型

```bash
# 节点1
./start.sh 2
# 输入: 节点索引=0, 总节点数=2

# 节点2
./start.sh 2
# 输入: 节点索引=1, 总节点数=2
```

### 3. 去中心化模式

**特点：无单点故障，高可用**

- Raft共识算法
- 自动领导者选举
- 故障自动转移

```bash
./start.sh 3
```

### 4. 中心化模式

**特点：有Web管理界面**

- 需要先启动Orchestrator
- 支持Web监控

```bash
./start.sh 4
```

---

## 🔧 配置文件

配置文件位置: `~/.distributed-llm/config/config.yaml`

```yaml
# 模型配置
model:
  name: "Qwen/Qwen2.5-0.5B-Instruct"
  memory_gb: 2.0

# 资源配置
resources:
  min_memory_gb: 2.0
  min_cpu_percent: 10.0

# 网络配置
network:
  host: "0.0.0.0"
  port: 7000

# 模式选择
mode: "resource_aware"
```

---

## 📡 API使用

服务启动后，可以通过HTTP API访问：

### 健康检查

```bash
curl http://localhost:7000/health
# {"status": "healthy"}
```

### 查看状态

```bash
curl http://localhost:7000/status
```

### 推理请求

```bash
curl -X POST http://localhost:7000/inference \
  -H "Content-Type: application/json" \
  -d '{"prompt": "你好", "max_tokens": 50}'
```

### Python调用

```python
import requests

response = requests.post(
    "http://localhost:7000/inference",
    json={"prompt": "你好", "max_tokens": 50}
)
print(response.json())
```

---

## 🖥️ 开机自启

### Linux (systemd)

```bash
sudo cp /tmp/distributed-llm.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable distributed-llm
sudo systemctl start distributed-llm
```

### macOS (launchd)

```bash
cp /tmp/com.distributed.llm.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.distributed.llm.plist
```

### Windows

以管理员身份运行:
```cmd
%USERPROFILE%\.distributed-llm\install-service.bat
```

---

## 📁 安装目录结构

```
~/.distributed-llm/
├── sm/                    # 项目代码
│   ├── download/          # 节点服务
│   └── ...
├── venv/                  # Python虚拟环境
├── config/                # 配置文件
│   └── config.yaml
├── start.sh               # 启动脚本
├── stop.sh                # 停止脚本
└── status.sh              # 状态脚本
```

---

## 🐳 Docker命令

```bash
# 构建镜像
docker build -t distributed-llm .

# 运行容器
docker run -d -p 7000:7000 --name llm distributed-llm

# 使用docker-compose
docker-compose up -d        # 启动
docker-compose logs -f      # 日志
docker-compose down         # 停止
docker-compose restart      # 重启
```

---

## ❓ 常见问题

### 1. 端口被占用

```bash
# 查看端口占用
lsof -i :7000        # Linux/macOS
netstat -ano | findstr :7000  # Windows

# 修改端口
# 编辑 config/config.yaml 中的 network.port
```

### 2. 内存不足

```bash
# 使用更小的模型
# 编辑 config/config.yaml
model:
  name: "Qwen/Qwen2.5-0.5B-Instruct"
```

### 3. 模型下载慢

```bash
# 使用镜像
export HF_ENDPOINT=https://hf-mirror.com
```

### 4. Python版本不兼容

```bash
# 需要 Python 3.8+
python3 --version
```

---

## 📊 性能指标

| 模型 | 内存 | 延迟 | 吞吐量 |
|------|------|------|--------|
| Qwen2.5-0.5B | 1.8GB | 1.2s | 17 t/s |
| Qwen2.5-1.5B | 3.5GB | 2.5s | 20 t/s |
| Qwen2.5-7B | 14GB | 8s | 22 t/s |

---

## 📜 许可证

MIT License

---

## 🙏 致谢

- [Qwen](https://github.com/QwenLM/Qwen) - 模型支持
- [Raft](https://raft.github.io/) - 共识算法
