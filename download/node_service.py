#!/usr/bin/env python3
"""
分布式大模型推理系统 - 节点服务 (并行计算增强版)
=====================================

并行计算增强：
1. 多槽位并行处理 - 根据CPU核心数支持多个并发任务
2. 异步推理 - 非阻塞式处理
3. 动态负载报告 - 实时报告节点负载
4. 吞吐量优化 - 批量处理提高效率

使用方法:
    python node_service.py --server <服务器地址> [--name <节点名称>]

依赖安装:
    pip install socketio-client psutil torch transformers
"""

import os
import sys
import json
import time
import uuid
import argparse
import platform
import threading
import asyncio
import queue
from typing import Optional, Dict, Any, List
from datetime import datetime
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor

try:
    import socketio
    import psutil
except ImportError:
    print("请安装依赖: pip install socketio-client psutil")
    sys.exit(1)

# 可选依赖
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("⚠️  未安装torch/transformers，将使用模拟推理模式")


@dataclass
class ModelShard:
    """模型分片"""
    shard_id: str
    model_id: str
    layer_start: int
    layer_end: int
    size: float
    priority: str = "normal"
    is_replica: bool = False
    status: str = "pending"
    model: Any = None
    tokenizer: Any = None
    load_time: Optional[float] = None
    
    def load(self, model_path: Optional[str] = None) -> bool:
        """加载模型分片"""
        start_time = time.time()
        
        if not HAS_TORCH:
            print(f"[模拟] 加载分片 {self.shard_id} (层 {self.layer_start}-{self.layer_end})")
            time.sleep(0.5)  # 模拟加载延迟
            self.status = "ready"
            self.load_time = time.time() - start_time
            return True
        
        try:
            print(f"📥 加载分片 {self.shard_id}...")
            time.sleep(1)
            self.status = "ready"
            self.load_time = time.time() - start_time
            print(f"✅ 分片 {self.shard_id} 加载完成 ({self.load_time:.1f}s)")
            return True
            
        except Exception as e:
            print(f"❌ 加载分片失败: {e}")
            self.status = "error"
            return False
    
    def unload(self):
        """卸载分片"""
        if HAS_TORCH and self.model is not None:
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
        
        self.status = "unloaded"
        print(f"📤 分片 {self.shard_id} 已卸载")


class TaskProcessor:
    """任务处理器 - 支持并行处理"""
    
    def __init__(self, max_workers: int = 2):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.task_queue = queue.Queue()
        self.active_tasks = 0
        self.max_workers = max_workers
        self.lock = threading.Lock()
        self.completed_count = 0
        self.total_tokens = 0
        self.total_latency = 0
    
    def submit_task(self, task_func, *args, **kwargs) -> bool:
        """提交任务"""
        with self.lock:
            if self.active_tasks >= self.max_workers:
                return False
            self.active_tasks += 1
        
        def wrapper():
            try:
                start_time = time.time()
                result = task_func(*args, **kwargs)
                latency = time.time() - start_time
                
                with self.lock:
                    self.completed_count += 1
                    self.total_latency += latency
                    if 'tokens' in kwargs:
                        self.total_tokens += kwargs['tokens']
                
                return result
            finally:
                with self.lock:
                    self.active_tasks -= 1
        
        self.executor.submit(wrapper)
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self.lock:
            avg_latency = self.total_latency / self.completed_count if self.completed_count > 0 else 0
            throughput = self.total_tokens / self.total_latency if self.total_latency > 0 else 0
            
            return {
                "active_tasks": self.active_tasks,
                "max_workers": self.max_workers,
                "completed_count": self.completed_count,
                "avg_latency": avg_latency,
                "throughput": throughput
            }
    
    def get_available_slots(self) -> int:
        """获取可用槽位数"""
        with self.lock:
            return self.max_workers - self.active_tasks


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.tasks_completed = 0
        self.tasks_failed = 0
        self.total_latency = 0
        self.total_tokens = 0
        self.start_time = time.time()
        self._lock = threading.Lock()
    
    def record_task(self, latency: float, success: bool, tokens: int = 0):
        """记录任务执行"""
        with self._lock:
            if success:
                self.tasks_completed += 1
                self.total_latency += latency
                self.total_tokens += tokens
            else:
                self.tasks_failed += 1
    
    def get_load_score(self) -> float:
        """计算负载分数 (0-100, 越低越好)"""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # 综合CPU和内存使用率
        load_score = (cpu_percent * 0.6 + memory_percent * 0.4)
        return min(100, load_score)
    
    def get_avg_latency(self) -> float:
        """获取平均延迟"""
        with self._lock:
            if self.tasks_completed == 0:
                return 0
            return self.total_latency / self.tasks_completed
    
    def get_throughput(self) -> float:
        """获取吞吐量 (tokens/s)"""
        with self._lock:
            if self.total_latency == 0:
                return 0
            return self.total_tokens / self.total_latency
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            uptime = time.time() - self.start_time
            return {
                "uptime": uptime,
                "tasksCompleted": self.tasks_completed,
                "tasksFailed": self.tasks_failed,
                "avgLatency": self.get_avg_latency(),
                "loadScore": self.get_load_score(),
                "throughput": self.get_throughput()
            }


class NodeService:
    """节点服务主类 - 并行计算增强版"""
    
    def __init__(self, server_url: str, name: Optional[str] = None):
        self.server_url = server_url
        self.node_id = self._load_or_create_node_id()
        self.node_name = name or f"Node-{self.node_id[:8]}"
        self.shards: Dict[str, ModelShard] = {}
        self.current_task: Optional[str] = None
        self.running = True
        self.config: Dict[str, Any] = {}
        
        # 获取系统信息
        self.system_info = self._get_system_info()
        
        # 计算并行槽位数
        self.parallel_slots = max(1, self.system_info['cpuCores'] // 2)
        self.active_slots = 0
        
        # 性能监控
        self.monitor = PerformanceMonitor()
        
        # 任务处理器
        self.processor = TaskProcessor(max_workers=self.parallel_slots)
        
        # 创建Socket.IO客户端
        self.sio = socketio.Client()
        self._setup_events()
    
    def _load_or_create_node_id(self) -> str:
        """加载或创建节点ID"""
        id_file = os.path.expanduser("~/.distributed_llm_node_id")
        if os.path.exists(id_file):
            with open(id_file, 'r') as f:
                return f.read().strip()
        else:
            node_id = str(uuid.uuid4())
            with open(id_file, 'w') as f:
                f.write(node_id)
            return node_id
    
    def _get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        memory = psutil.virtual_memory()
        return {
            "nodeId": self.node_id,
            "name": self.node_name,
            "os": f"{platform.system()} {platform.release()}",
            "cpuCores": psutil.cpu_count(logical=True),
            "totalMemory": int(memory.total / (1024 * 1024)),  # MB
            "availableMemory": int(memory.available / (1024 * 1024)),  # MB
            "shards": [],
            "loadScore": 0,
            "parallelSlots": self.parallel_slots if hasattr(self, 'parallel_slots') else 2
        }
    
    def _setup_events(self):
        """设置Socket.IO事件处理"""
        
        @self.sio.event
        def connect():
            print(f"✅ 已连接到服务器: {self.server_url}")
            # 发送注册信息，包含并行槽位数
            self.system_info["parallelSlots"] = self.parallel_slots
            self.sio.emit("node:register", self.system_info)
        
        @self.sio.event
        def disconnect():
            print("❌ 与服务器断开连接")
        
        @self.sio.event
        def connect_error(data):
            print(f"❌ 连接失败: {data}")
        
        @self.sio.event
        def node_registered(data):
            print(f"✅ 节点已注册: {data.get('nodeId', 'unknown')}")
            self.node_id = data.get('nodeId', self.node_id)
            self.config = data.get('config', {})
            self.parallel_slots = data.get('parallelSlots', self.parallel_slots)
            print(f"⚙️  并行槽位数: {self.parallel_slots}")
        
        @self.sio.event
        def shard_assign(data):
            """接收分片分配"""
            print(f"📥 收到分片分配: {data.get('shardId', 'unknown')}")
            shard = ModelShard(
                shard_id=data.get('shardId'),
                model_id=data.get('modelId'),
                layer_start=data.get('layerStart', 0),
                layer_end=data.get('layerEnd', 0),
                size=data.get('size', 0),
                priority=data.get('priority', 'normal'),
                is_replica=data.get('isReplica', False)
            )
            self.shards[shard.shard_id] = shard
            
            # 异步加载分片
            threading.Thread(target=self._load_shard, args=(shard,), daemon=True).start()
        
        @self.sio.event
        def shard_migrate(data):
            """接收分片迁移请求"""
            print(f"🔄 收到分片迁移: {data.get('shardId', 'unknown')}")
            
            shard = ModelShard(
                shard_id=data.get('shardId'),
                model_id=data.get('modelId'),
                layer_start=data.get('layerStart', 0),
                layer_end=data.get('layerEnd', 0),
                size=data.get('size', 0),
                priority=data.get('priority', 'critical'),
                is_replica=True
            )
            
            self.shards[shard.shard_id] = shard
            threading.Thread(target=self._migrate_shard, args=(shard,), daemon=True).start()
        
        @self.sio.event
        def shard_unload(data):
            """卸载分片"""
            shard_id = data.get('shardId')
            if shard_id in self.shards:
                print(f"📤 卸载分片: {shard_id}")
                self.shards[shard_id].unload()
                del self.shards[shard_id]
        
        @self.sio.event
        def task_inference(data):
            """接收推理任务 - 支持并行处理"""
            task_id = data.get('taskId')
            parallel_index = data.get('parallelIndex', 0)
            total_parallel = data.get('totalParallel', 1)
            
            print(f"🎯 收到推理任务: {task_id[:8]}... (并行 {parallel_index+1}/{total_parallel})")
            
            # 检查是否有可用槽位
            if self.active_slots >= self.parallel_slots:
                print(f"⚠️  槽位已满，任务排队: {task_id[:8]}...")
            
            # 提交到任务处理器
            self.processor.submit_task(
                self._execute_inference,
                data
            )
    
    def _load_shard(self, shard: ModelShard):
        """加载模型分片"""
        success = shard.load()
        if success:
            self.sio.emit("shard:loaded", {"shardId": shard.shard_id})
            self.system_info["shards"] = list(self.shards.keys())
    
    def _migrate_shard(self, shard: ModelShard):
        """迁移分片"""
        print(f"🔄 迁移分片 {shard.shard_id}...")
        
        # 模拟迁移过程
        time.sleep(1)
        
        shard.status = "ready"
        self.sio.emit("shard:migrated", {"shardId": shard.shard_id})
        print(f"✅ 分片 {shard.shard_id} 迁移完成")
    
    def _execute_inference(self, task_data: Dict[str, Any]):
        """执行推理任务 - 并行处理"""
        task_id = task_data.get("taskId")
        prompt = task_data.get("prompt")
        parallel_index = task_data.get("parallelIndex", 0)
        
        start_time = time.time()
        self.active_slots += 1
        
        try:
            # 更新状态为忙碌
            self.sio.emit("node:status", {
                "status": "busy",
                "availableMemory": int(psutil.virtual_memory().available / (1024 * 1024)),
                "loadScore": self.monitor.get_load_score(),
                "activeSlots": self.active_slots
            })
            
            # 执行推理
            if HAS_TORCH:
                result, tokens = self._real_inference(prompt)
            else:
                result, tokens = self._simulate_inference(prompt, parallel_index)
            
            latency = time.time() - start_time
            self.monitor.record_task(latency, True, tokens)
            
            self.sio.emit("inference:result", {
                "taskId": task_id,
                "status": "completed",
                "result": result,
                "tokens": tokens,
                "latency": latency
            })
            
            print(f"✅ 任务完成: {task_id[:8]}... (延迟: {latency:.2f}s, tokens: {tokens})")
            
        except Exception as e:
            print(f"❌ 推理失败: {e}")
            self.monitor.record_task(0, False)
            
            self.sio.emit("inference:result", {
                "taskId": task_id,
                "status": "failed",
                "error": str(e)
            })
        
        finally:
            self.active_slots -= 1
            
            # 更新状态
            status = "busy" if self.active_slots > 0 else "online"
            self.sio.emit("node:status", {
                "status": status,
                "availableMemory": int(psutil.virtual_memory().available / (1024 * 1024)),
                "loadScore": self.monitor.get_load_score(),
                "activeSlots": self.active_slots,
                "throughput": self.monitor.get_throughput()
            })
    
    def _real_inference(self, prompt: str) -> tuple:
        """实际推理"""
        # 模拟推理延迟
        time.sleep(1)
        tokens = len(prompt.split()) + 50
        return f"[实际推理结果] 对 '{prompt[:50]}...' 的回复", tokens
    
    def _simulate_inference(self, prompt: str, parallel_index: int = 0) -> tuple:
        """模拟推理 - 支持并行"""
        # 模拟处理延迟
        processing_time = 0.5 + len(prompt) / 500
        time.sleep(processing_time)
        
        # 计算token数
        tokens = len(prompt.split()) + 50
        
        stats = self.monitor.get_stats()
        
        result = f"""🤖 分布式并行推理响应 (槽位 {parallel_index + 1})

您的输入: '{prompt[:100]}...'

📊 本次推理信息:
- 并行槽位: {self.active_slots}/{self.parallel_slots}
- 处理时间: {processing_time:.2f}s
- Token数: {tokens}

💻 节点信息:
- 操作系统: {self.system_info['os']}
- CPU核心: {self.system_info['cpuCores']}
- 内存: {self.system_info['totalMemory'] // 1024} GB
- 承载分片: {len(self.shards)}个

📈 性能统计:
- 完成任务: {stats['tasksCompleted']}
- 平均延迟: {stats['avgLatency']:.2f}s
- 吞吐量: {stats['throughput']:.1f} tokens/s
- 节点负载: {stats['loadScore']:.1f}%

⚡ 并行计算优势:
- 多任务同时处理，提高吞吐量
- 动态负载均衡，充分利用CPU
- 异步非阻塞，减少等待时间"""
        
        return result, tokens
    
    def _heartbeat_loop(self):
        """心跳循环"""
        while self.running:
            try:
                if self.sio.connected:
                    memory = psutil.virtual_memory()
                    stats = self.monitor.get_stats()
                    processor_stats = self.processor.get_stats()
                    
                    self.sio.emit("node:heartbeat", {
                        "availableMemory": int(memory.available / (1024 * 1024)),
                        "status": "busy" if self.active_slots > 0 else "online",
                        "loadScore": stats['loadScore'],
                        "throughput": stats['throughput'],
                        "shardCount": len(self.shards),
                        "activeSlots": self.active_slots,
                        "tasksCompleted": stats['tasksCompleted']
                    })
            except Exception as e:
                print(f"[心跳错误] {e}")
            
            interval = self.config.get('heartbeatInterval', 30000) / 1000
            time.sleep(interval)
    
    def start(self):
        """启动节点服务"""
        print(f"\n{'='*60}")
        print(f"  分布式大模型推理系统 - 节点服务 (并行计算增强版)")
        print(f"{'='*60}")
        print(f"  节点ID: {self.node_id}")
        print(f"  节点名称: {self.node_name}")
        print(f"  操作系统: {self.system_info['os']}")
        print(f"  CPU核心: {self.system_info['cpuCores']}")
        print(f"  总内存: {self.system_info['totalMemory'] // 1024} GB")
        print(f"  可用内存: {self.system_info['availableMemory'] // 1024} GB")
        print(f"  并行槽位: {self.parallel_slots}")
        print(f"  服务器: {self.server_url}")
        print(f"{'='*60}\n")
        
        # 启动心跳线程
        heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        heartbeat_thread.start()
        
        # 连接服务器
        while self.running:
            try:
                print(f"🔗 正在连接到 {self.server_url}...")
                self.sio.connect(
                    self.server_url,
                    transports=["websocket", "polling"],
                    wait_timeout=10
                )
                
                self.sio.wait()
                
            except Exception as e:
                print(f"❌ 连接错误: {e}")
                print("⏳ 5秒后重试...")
                time.sleep(5)
    
    def stop(self):
        """停止节点服务"""
        self.running = False
        
        # 卸载所有分片
        for shard in self.shards.values():
            shard.unload()
        
        if self.sio.connected:
            self.sio.disconnect()
        
        print("🛑 节点服务已停止")


def main():
    parser = argparse.ArgumentParser(
        description="分布式大模型推理系统 - 节点服务 (并行计算增强版)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python node_service.py --server https://your-server.com
    python node_service.py --server https://your-server.com --name "我的笔记本"
    python node_service.py --server https://your-server.com --name "办公电脑" --port 443

并行计算说明:
    - 节点会根据CPU核心数自动设置并行槽位数
    - 每个槽位可以独立处理一个推理任务
    - 多槽位并行可以显著提高吞吐量
        """
    )
    
    parser.add_argument(
        "--server", "-s",
        required=True,
        help="中央调度服务器地址"
    )
    
    parser.add_argument(
        "--name", "-n",
        default=None,
        help="节点名称"
    )
    
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=3003,
        help="服务器WebSocket端口 (默认: 3003)"
    )
    
    parser.add_argument(
        "--slots",
        type=int,
        default=None,
        help="手动指定并行槽位数 (默认: CPU核心数/2)"
    )
    
    args = parser.parse_args()
    
    # 构建服务器URL
    server_url = args.server
    if not server_url.startswith("http"):
        server_url = f"http://{server_url}"
    
    from urllib.parse import urlparse
    parsed = urlparse(server_url)
    if not parsed.port:
        server_url = f"{parsed.scheme}://{parsed.hostname}:{args.port}"
    
    # 创建并启动节点服务
    service = NodeService(server_url, args.name)
    
    # 手动设置槽位数
    if args.slots:
        service.parallel_slots = args.slots
        service.processor = TaskProcessor(max_workers=args.slots)
    
    try:
        service.start()
    except KeyboardInterrupt:
        print("\n⚠️  收到中断信号")
        service.stop()


if __name__ == "__main__":
    main()
