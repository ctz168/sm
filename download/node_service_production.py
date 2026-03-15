#!/usr/bin/env python3
"""
分布式大模型推理系统 - 生产级节点服务
=====================================

这是真正可运行的版本，支持：
1. 真实模型加载和推理
2. 数据并行（每节点完整模型）
3. 多槽位并行处理
4. 网络感知调度
5. 故障恢复

依赖安装:
    pip install torch transformers accelerate python-socketio psutil

使用方法:
    python node_service_production.py --server <服务器地址> --model <模型名称>
"""

import os
import sys
import json
import time
import uuid
import argparse
import platform
import threading
import queue
import socket
import signal
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from collections import deque

try:
    import socketio
    import psutil
except ImportError as e:
    print(f"❌ 缺少依赖: {e}")
    print("请运行: pip install python-socketio psutil")
    sys.exit(1)

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("❌ 未安装torch/transformers")
    print("请运行: pip install torch transformers accelerate")
    sys.exit(1)


@dataclass
class PerformanceStats:
    """性能统计"""
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_tokens: int = 0
    total_latency: float = 0.0
    latency_history: deque = field(default_factory=lambda: deque(maxlen=100))
    throughput_history: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def record_task(self, latency: float, tokens: int, success: bool):
        """记录任务执行"""
        if success:
            self.tasks_completed += 1
            self.total_tokens += tokens
            self.total_latency += latency
            self.latency_history.append(latency)
            if latency > 0:
                self.throughput_history.append(tokens / latency)
        else:
            self.tasks_failed += 1
    
    def get_avg_latency(self) -> float:
        if not self.latency_history:
            return 0.0
        return sum(self.latency_history) / len(self.latency_history)
    
    def get_avg_throughput(self) -> float:
        if not self.throughput_history:
            return 0.0
        return sum(self.throughput_history) / len(self.throughput_history)
    
    def get_load_score(self) -> float:
        """计算负载分数 (0-100)"""
        cpu = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory().percent
        return min(100, (cpu * 0.5 + memory * 0.5))


@dataclass
class NetworkMetrics:
    """网络指标"""
    latency_ms: float = 0.0
    jitter_ms: float = 0.0
    bandwidth_mbps: float = 0.0
    last_ping: float = 0.0
    ping_history: deque = field(default_factory=lambda: deque(maxlen=10))
    
    def record_ping(self, latency: float):
        """记录ping延迟"""
        self.ping_history.append(latency)
        self.latency_ms = sum(self.ping_history) / len(self.ping_history)
        
        if len(self.ping_history) > 1:
            diffs = [abs(self.ping_history[i] - self.ping_history[i-1]) 
                     for i in range(1, len(self.ping_history))]
            self.jitter_ms = sum(diffs) / len(diffs)
        
        self.last_ping = time.time()
    
    def get_network_score(self) -> float:
        """计算网络分数 (0-100)"""
        if self.latency_ms == 0:
            return 50.0
        latency_score = max(0, 100 - self.latency_ms / 2)
        jitter_score = max(0, 100 - self.jitter_ms * 5)
        return (latency_score + jitter_score) / 2


class InferenceEngine:
    """推理引擎 - 真实模型推理"""
    
    def __init__(self, model_name: str, device: str = "auto"):
        self.model_name = model_name
        self.device = self._detect_device() if device == "auto" else device
        self.model = None
        self.tokenizer = None
        self.generation_config = None
        self.loaded = False
        self.load_time = 0.0
        self.model_size_mb = 0.0
    
    def _detect_device(self) -> str:
        """检测最佳设备"""
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(f"🎮 检测到GPU: {device_name}")
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("🍎 检测到Apple Silicon GPU")
            return "mps"
        else:
            print("💻 使用CPU推理")
            return "cpu"
    
    def load(self) -> bool:
        """加载模型"""
        start_time = time.time()
        
        try:
            print(f"📥 加载模型: {self.model_name}")
            print(f"   设备: {self.device}")
            
            # 加载tokenizer
            print("   [1/3] 加载Tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                use_fast=True
            )
            
            # 确保有pad_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 加载模型
            print("   [2/3] 加载模型权重...")
            
            # 根据设备和模型大小选择加载方式
            if self.device == "cuda":
                # GPU: 使用float16
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
            elif self.device == "mps":
                # Apple Silicon: 使用float16
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    trust_remote_code=True
                ).to(self.device)
            else:
                # CPU: 使用float32
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32,
                    trust_remote_code=True
                )
            
            # 设置为评估模式
            print("   [3/3] 初始化推理引擎...")
            self.model.eval()
            
            # 创建生成配置
            self.generation_config = GenerationConfig(
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            
            # 计算模型大小
            param_count = sum(p.numel() for p in self.model.parameters())
            self.model_size_mb = (param_count * 4) / (1024 * 1024)  # 假设float32
            
            self.load_time = time.time() - start_time
            self.loaded = True
            
            print(f"✅ 模型加载完成!")
            print(f"   参数量: {param_count / 1e9:.2f}B")
            print(f"   加载时间: {self.load_time:.1f}s")
            print(f"   设备: {self.device}")
            
            return True
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def unload(self):
        """卸载模型释放内存"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.loaded = False
        print("📤 模型已卸载")
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> Tuple[str, int, float, Dict]:
        """
        执行推理生成
        
        返回: (生成的文本, token数量, 延迟, 详细信息)
        """
        if not self.loaded:
            raise RuntimeError("模型未加载")
        
        start_time = time.time()
        
        # 编码输入
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
            padding=True
        )
        
        # 移动到设备
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        input_token_count = input_ids.shape[1]
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else 1.0,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
        
        # 解码输出
        output_token_count = outputs.shape[1]
        new_tokens = output_token_count - input_token_count
        
        # 只解码新生成的部分
        response = self.tokenizer.decode(
            outputs[0][input_token_count:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        latency = time.time() - start_time
        throughput = new_tokens / latency if latency > 0 else 0
        
        details = {
            "input_tokens": input_token_count,
            "output_tokens": output_token_count,
            "new_tokens": new_tokens,
            "latency": latency,
            "throughput": throughput,
            "device": self.device
        }
        
        return response, new_tokens, latency, details


class TaskProcessor:
    """任务处理器 - 支持并行处理"""
    
    def __init__(self, engine: InferenceEngine, max_workers: int = 2):
        self.engine = engine
        self.max_workers = max_workers
        self.active_tasks = 0
        self.lock = threading.Lock()
        self.task_queue = queue.Queue()
        self.results: Dict[str, Dict] = {}
    
    def can_accept_task(self) -> bool:
        """检查是否可以接受新任务"""
        with self.lock:
            return self.active_tasks < self.max_workers
    
    def get_available_slots(self) -> int:
        """获取可用槽位数"""
        with self.lock:
            return self.max_workers - self.active_tasks
    
    def submit_task(self, task_id: str, prompt: str, params: Dict, callback) -> bool:
        """
        提交任务
        
        callback: 回调函数，接收 (task_id, result, tokens, latency, details)
        """
        if not self.can_accept_task():
            return False
        
        with self.lock:
            self.active_tasks += 1
        
        def worker():
            try:
                result, tokens, latency, details = self.engine.generate(
                    prompt,
                    max_new_tokens=params.get("max_tokens", 256),
                    temperature=params.get("temperature", 0.7),
                    top_p=params.get("top_p", 0.9)
                )
                callback(task_id, result, tokens, latency, details, None)
            except Exception as e:
                callback(task_id, None, 0, 0, {}, str(e))
            finally:
                with self.lock:
                    self.active_tasks -= 1
        
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        return True


class ProductionNodeService:
    """生产级节点服务"""
    
    def __init__(
        self,
        server_url: str,
        model_name: str,
        name: Optional[str] = None,
        max_workers: int = 2
    ):
        self.server_url = server_url
        self.model_name = model_name
        self.node_id = self._load_or_create_node_id()
        self.node_name = name or f"Node-{self.node_id[:8]}"
        
        # 系统信息
        self.system_info = self._get_system_info()
        
        # 推理引擎
        self.engine: Optional[InferenceEngine] = None
        self.task_processor: Optional[TaskProcessor] = None
        self.max_workers = max_workers
        
        # 性能统计
        self.stats = PerformanceStats()
        self.network = NetworkMetrics()
        
        # 重连计数
        self._reconnect_attempt = 0
        self._max_reconnect_attempts = 10
        
        # Socket.IO - 配置指数退避重连
        self.sio = socketio.Client(
            reconnection=True,
            reconnection_attempts=self._max_reconnect_attempts,
            reconnection_delay=1,        # 初始延迟1秒
            reconnection_delay_max=60,   # 最大延迟60秒
            reconnection_delay_multiplier=2,  # 指数退避
            logger=False,                # 禁用内置日志
            engineio_logger=False        # 禁用Engine.IO日志
        )
        
        # 状态
        self.running = False
        self.connected = False
        self.model_ready = False
        
        self._setup_events()
    
    def _load_or_create_node_id(self) -> str:
        """加载或创建节点ID"""
        id_file = os.path.expanduser("~/.distributed_llm_node_id")
        if os.path.exists(id_file):
            with open(id_file, 'r') as f:
                return f.read().strip()
        else:
            node_id = str(uuid.uuid4())
            os.makedirs(os.path.dirname(id_file), exist_ok=True)
            with open(id_file, 'w') as f:
                f.write(node_id)
            return node_id
    
    def _get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        memory = psutil.virtual_memory()
        
        gpu_info = ""
        if torch.cuda.is_available():
            gpu_info = f", GPU: {torch.cuda.get_device_name(0)}"
        
        return {
            "nodeId": self.node_id,
            "name": self.node_name,
            "os": f"{platform.system()} {platform.release()}{gpu_info}",
            "cpuCores": psutil.cpu_count(logical=True),
            "totalMemory": int(memory.total / (1024 * 1024)),  # MB
            "availableMemory": int(memory.available / (1024 * 1024)),  # MB
            "modelName": self.model_name,
            "pythonVersion": platform.python_version(),
            "torchVersion": torch.__version__,
        }
    
    def _setup_events(self):
        """设置Socket.IO事件"""
        
        @self.sio.event
        def connect():
            print(f"✅ 已连接到服务器: {self.server_url}")
            self.connected = True
            self._reconnect_attempt = 0  # 重置重连计数
            self._register()
        
        @self.sio.event
        def disconnect():
            print("❌ 与服务器断开连接")
            self.connected = False
            # 自动重连由Socket.IO客户端处理
        
        @self.sio.event
        def connect_error(data):
            self._reconnect_attempt += 1
            # 指数退避延迟
            delay = min(2 ** self._reconnect_attempt, 60)  # 最大60秒
            print(f"❌ 连接错误 (尝试 {self._reconnect_attempt}): {data}")
            print(f"   将在 {delay} 秒后重试...")
        
        @self.sio.on('node:registered')
        def on_registered(data):
            print(f"✅ 节点已注册: {data.get('nodeId', 'unknown')}")
            # 启动心跳
            threading.Thread(target=self._heartbeat_loop, daemon=True).start()
        
        @self.sio.on('task:inference')
        def on_task(data):
            """接收推理任务"""
            if not self.model_ready:
                self.sio.emit('inference:result', {
                    'taskId': data.get('taskId'),
                    'status': 'failed',
                    'error': '模型未就绪'
                })
                return
            
            task_id = data.get('taskId')
            prompt = data.get('prompt', '')
            
            # 输入验证
            if not prompt or len(prompt) > 10000:
                self.sio.emit('inference:result', {
                    'taskId': task_id,
                    'status': 'failed',
                    'error': '无效的输入'
                })
                return
            
            params = {
                'max_tokens': data.get('maxTokens', 256),
                'temperature': data.get('temperature', 0.7),
                'top_p': data.get('topP', 0.9)
            }
            
            print(f"🎯 收到任务: {task_id[:8]}...")
            
            # 提交任务
            if not self.task_processor.submit_task(
                task_id, prompt, params, self._on_task_complete
            ):
                self.sio.emit('inference:result', {
                    'taskId': task_id,
                    'status': 'queued',
                    'message': '节点繁忙，任务排队中'
                })
        
        @self.sio.on('network:probe')
        def on_probe(data):
            """网络探测"""
            self.network.record_ping(time.time() - data.get('timestamp', time.time()))
            self.sio.emit('network:probe-response', {
                'timestamp': time.time(),
                'nodeId': self.node_id
            })
        
        @self.sio.on('model:status')
        def on_model_status(data):
            """模型状态查询"""
            self.sio.emit('model:status-response', {
                'nodeId': self.node_id,
                'modelName': self.model_name,
                'loaded': self.model_ready,
                'device': self.engine.device if self.engine else None,
                'loadTime': self.engine.load_time if self.engine else 0
            })
    
    def _register(self):
        """注册节点"""
        self.sio.emit('node:register', {
            **self.system_info,
            'parallelSlots': self.max_workers,
            'network': {
                'latency_ms': self.network.latency_ms,
                'jitter_ms': self.network.jitter_ms,
                'network_score': self.network.get_network_score()
            }
        })
    
    def _on_task_complete(
        self,
        task_id: str,
        result: Optional[str],
        tokens: int,
        latency: float,
        details: Dict,
        error: Optional[str]
    ):
        """任务完成回调"""
        if error:
            self.stats.record_task(0, 0, False)
            self.sio.emit('inference:result', {
                'taskId': task_id,
                'status': 'failed',
                'error': error
            })
            print(f"❌ 任务失败: {task_id[:8]}... - {error}")
        else:
            self.stats.record_task(latency, tokens, True)
            self.sio.emit('inference:result', {
                'taskId': task_id,
                'status': 'completed',
                'result': result,
                'tokens': tokens,
                'latency': latency,
                'throughput': details.get('throughput', 0),
                'device': details.get('device', 'unknown')
            })
            print(f"✅ 任务完成: {task_id[:8]}... ({latency:.2f}s, {tokens} tokens, {details.get('throughput', 0):.1f} t/s)")
    
    def _heartbeat_loop(self):
        """心跳循环"""
        while self.running and self.connected:
            try:
                memory = psutil.virtual_memory()
                
                self.sio.emit('node:heartbeat', {
                    'availableMemory': int(memory.available / (1024 * 1024)),
                    'status': 'busy' if self.task_processor and self.task_processor.active_tasks > 0 else 'online',
                    'loadScore': self.stats.get_load_score(),
                    'activeSlots': self.task_processor.active_tasks if self.task_processor else 0,
                    'tasksCompleted': self.stats.tasks_completed,
                    'avgLatency': self.stats.get_avg_latency(),
                    'throughput': self.stats.get_avg_throughput(),
                    'network': {
                        'latency_ms': self.network.latency_ms,
                        'jitter_ms': self.network.jitter_ms,
                        'network_score': self.network.get_network_score()
                    },
                    'modelReady': self.model_ready
                })
            except Exception as e:
                print(f"[心跳错误] {e}")
            
            time.sleep(10)
    
    def start(self):
        """启动服务"""
        print(f"\n{'='*60}")
        print(f"  分布式大模型推理系统 - 生产级节点")
        print(f"{'='*60}")
        print(f"  节点ID: {self.node_id}")
        print(f"  节点名称: {self.node_name}")
        print(f"  模型: {self.model_name}")
        print(f"  并行槽位: {self.max_workers}")
        print(f"  服务器: {self.server_url}")
        print(f"{'='*60}\n")
        
        self.running = True
        
        # 加载模型
        print("📦 加载模型...")
        self.engine = InferenceEngine(self.model_name)
        if not self.engine.load():
            print("❌ 模型加载失败，退出")
            return
        
        # 创建任务处理器
        self.task_processor = TaskProcessor(self.engine, self.max_workers)
        self.model_ready = True
        
        print("\n🚀 节点就绪，等待任务...\n")
        
        # 连接服务器
        while self.running:
            try:
                print(f"🔗 连接到 {self.server_url}...")
                self.sio.connect(
                    self.server_url,
                    transports=["polling", "websocket"],
                    wait_timeout=10
                )
                self.sio.wait()
            except Exception as e:
                print(f"❌ 连接错误: {e}")
                if self.running:
                    print("⏳ 5秒后重试...")
                    time.sleep(5)
    
    def stop(self):
        """停止服务"""
        print("\n🛑 正在停止服务...")
        self.running = False
        self.model_ready = False
        
        if self.engine:
            self.engine.unload()
        
        if self.sio.connected:
            self.sio.disconnect()
        
        print("✅ 服务已停止")


def main():
    parser = argparse.ArgumentParser(
        description="分布式大模型推理系统 - 生产级节点服务",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 使用Qwen小模型
    python node_service_production.py --server http://localhost:3003 --model Qwen/Qwen2.5-0.5B-Instruct
    
    # 指定并行槽位数
    python node_service_production.py --server http://localhost:3003 --model Qwen/Qwen2.5-1.5B-Instruct --workers 4

推荐模型:
    - Qwen/Qwen2.5-0.5B-Instruct (约1GB, 适合测试)
    - Qwen/Qwen2.5-1.5B-Instruct (约3GB, 适合4-8GB内存)
    - Qwen/Qwen2.5-3B-Instruct (约6GB, 适合8-16GB内存)
        """
    )
    
    parser.add_argument(
        "--server", "-s",
        required=True,
        help="中央调度服务器地址"
    )
    
    parser.add_argument(
        "--model", "-m",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="模型名称或路径"
    )
    
    parser.add_argument(
        "--name", "-n",
        default=None,
        help="节点名称"
    )
    
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=2,
        help="并行工作线程数"
    )
    
    args = parser.parse_args()
    
    service = ProductionNodeService(
        server_url=args.server,
        model_name=args.model,
        name=args.name,
        max_workers=args.workers
    )
    
    # 信号处理
    def signal_handler(sig, frame):
        service.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        service.start()
    except KeyboardInterrupt:
        service.stop()


if __name__ == "__main__":
    main()
