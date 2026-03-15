#!/usr/bin/env python3
"""
分布式大模型推理系统 - 节点服务 (网络感知增强版)
=====================================

新增功能：
1. 网络延迟检测 - 实时测量到服务器的延迟
2. 带宽感知 - 估算可用带宽
3. 智能调度 - 根据网络状况选择最优节点

使用方法:
    source venv/bin/activate
    python node_service_network.py --server http://localhost:3003 --model Qwen/Qwen2.5-0.5B-Instruct
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
import struct
import select
import hashlib
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

try:
    import socketio
    import psutil
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError as e:
    print(f"❌ 缺少依赖: {e}")
    print("请运行: source venv/bin/activate")
    sys.exit(1)


@dataclass
class NetworkMetrics:
    """网络指标"""
    latency_ms: float = 0.0          # 延迟（毫秒）
    jitter_ms: float = 0.0           # 抖动（毫秒）
    bandwidth_mbps: float = 0.0      # 带宽（Mbps）
    packet_loss: float = 0.0         # 丢包率（%）
    last_update: float = 0.0         # 上次更新时间
    latency_history: List[float] = field(default_factory=list)  # 延迟历史
    
    def update_latency(self, latency: float):
        """更新延迟"""
        self.latency_history.append(latency)
        if len(self.latency_history) > 10:
            self.latency_history.pop(0)
        
        self.latency_ms = sum(self.latency_history) / len(self.latency_history)
        
        # 计算抖动
        if len(self.latency_history) > 1:
            diffs = [abs(self.latency_history[i] - self.latency_history[i-1]) 
                     for i in range(1, len(self.latency_history))]
            self.jitter_ms = sum(diffs) / len(diffs)
        
        self.last_update = time.time()
    
    def get_network_score(self) -> float:
        """计算网络分数（0-100，越高越好）"""
        # 延迟分数（延迟越低越好）
        latency_score = max(0, 100 - self.latency_ms / 2)
        
        # 抖动分数（抖动越小越好）
        jitter_score = max(0, 100 - self.jitter_ms * 5)
        
        # 带宽分数（带宽越高越好）
        bandwidth_score = min(100, self.bandwidth_mbps / 10)
        
        # 综合分数
        score = latency_score * 0.5 + jitter_score * 0.3 + bandwidth_score * 0.2
        
        return min(100, max(0, score))
    
    def to_dict(self) -> Dict:
        return {
            "latency_ms": round(self.latency_ms, 2),
            "jitter_ms": round(self.jitter_ms, 2),
            "bandwidth_mbps": round(self.bandwidth_mbps, 2),
            "packet_loss": round(self.packet_loss, 2),
            "network_score": round(self.get_network_score(), 2)
        }


class NetworkMonitor:
    """网络监控器 - 测量延迟和带宽"""
    
    def __init__(self, server_url: str):
        self.server_url = server_url
        self.metrics = NetworkMetrics()
        self.running = True
        self.ping_thread = None
        
        # 解析服务器地址
        from urllib.parse import urlparse
        parsed = urlparse(server_url)
        self.server_host = parsed.hostname or "localhost"
        self.server_port = parsed.port or 3003
    
    def start(self):
        """启动网络监控"""
        self.ping_thread = threading.Thread(target=self._ping_loop, daemon=True)
        self.ping_thread.start()
    
    def stop(self):
        """停止网络监控"""
        self.running = False
    
    def _ping_loop(self):
        """持续测量网络延迟"""
        while self.running:
            try:
                # TCP连接测量延迟
                start_time = time.time()
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5)
                
                try:
                    sock.connect((self.server_host, self.server_port))
                    latency = (time.time() - start_time) * 1000  # 转换为毫秒
                    self.metrics.update_latency(latency)
                    
                    # 测量带宽（发送小数据包）
                    if latency < 100:  # 只在低延迟时测量带宽
                        self._measure_bandwidth(sock)
                    
                except socket.timeout:
                    self.metrics.packet_loss += 1
                except ConnectionRefusedError:
                    self.metrics.packet_loss += 1
                finally:
                    sock.close()
                    
            except Exception as e:
                pass
            
            time.sleep(5)  # 每5秒测量一次
    
    def _measure_bandwidth(self, sock):
        """估算带宽"""
        try:
            # 发送1KB数据测量带宽
            data = b'x' * 1024
            start = time.time()
            sock.sendall(data)
            elapsed = time.time() - start
            
            if elapsed > 0:
                # 简单估算（实际带宽需要更复杂的测量）
                bandwidth = (len(data) * 8) / elapsed / 1e6  # Mbps
                self.metrics.bandwidth_mbps = bandwidth
                
        except Exception:
            pass
    
    def get_metrics(self) -> Dict:
        """获取网络指标"""
        return self.metrics.to_dict()


class ModelShard:
    """模型分片"""
    
    def __init__(self, shard_id: str, model_id: str, layer_start: int, layer_end: int, size: float):
        self.shard_id = shard_id
        self.model_id = model_id
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.size = size
        self.status = "pending"
        self.model = None
        self.tokenizer = None
        self.load_time = None
    
    def load(self, model_name: str) -> bool:
        """加载模型"""
        start_time = time.time()
        
        try:
            print(f"📥 加载模型: {model_name}")
            
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            
            # 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=True
            )
            self.model.eval()
            
            self.status = "ready"
            self.load_time = time.time() - start_time
            
            print(f"✅ 模型加载完成 ({self.load_time:.1f}s)")
            return True
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            self.status = "error"
            return False
    
    def unload(self):
        """卸载模型"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self.status = "unloaded"


class InferenceEngine:
    """推理引擎"""
    
    def __init__(self, shard: ModelShard):
        self.shard = shard
    
    def generate(self, prompt: str, max_tokens: int = 100, temperature: float = 0.7) -> Tuple[str, int, float]:
        """执行推理"""
        start_time = time.time()
        
        if self.shard.model is None:
            raise ValueError("模型未加载")
        
        # 编码
        inputs = self.shard.tokenizer(prompt, return_tensors="pt")
        
        # 生成
        with torch.no_grad():
            outputs = self.shard.model.generate(
                inputs.input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.shard.tokenizer.eos_token_id
            )
        
        # 解码
        input_len = inputs.input_ids.shape[1]
        response = self.shard.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        
        latency = time.time() - start_time
        tokens = outputs.shape[1] - input_len
        
        return response, tokens, latency


class NetworkAwareNodeService:
    """网络感知节点服务"""
    
    def __init__(self, server_url: str, name: Optional[str] = None, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        self.server_url = server_url
        self.node_id = self._load_or_create_node_id()
        self.node_name = name or f"Node-{self.node_id[:8]}"
        self.model_name = model_name
        
        # 系统信息
        self.system_info = self._get_system_info()
        self.parallel_slots = max(1, self.system_info['cpuCores'] // 2)
        self.active_slots = 0
        
        # 模型和推理
        self.shard: Optional[ModelShard] = None
        self.engine: Optional[InferenceEngine] = None
        
        # 网络监控
        self.network_monitor = NetworkMonitor(server_url)
        
        # 性能统计
        self.tasks_completed = 0
        self.total_latency = 0
        self.total_tokens = 0
        
        # Socket.IO
        self.sio = socketio.Client()
        self.running = True
        self._setup_events()
    
    def _load_or_create_node_id(self) -> str:
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
        memory = psutil.virtual_memory()
        return {
            "nodeId": self.node_id,
            "name": self.node_name,
            "os": f"{platform.system()} {platform.release()}",
            "cpuCores": psutil.cpu_count(logical=True),
            "totalMemory": int(memory.total / (1024 * 1024)),
            "availableMemory": int(memory.available / (1024 * 1024)),
            "modelName": self.model_name
        }
    
    def _setup_events(self):
        """设置Socket.IO事件"""
        
        @self.sio.event
        def connect():
            print(f"✅ 已连接到服务器: {self.server_url}")
            self._register()
        
        @self.sio.event
        def disconnect():
            print("❌ 与服务器断开连接")
        
        @self.sio.event
        def node_registered(data):
            print(f"✅ 节点已注册: {data.get('nodeId', 'unknown')}")
            # 加载模型
            threading.Thread(target=self._load_model, daemon=True).start()
        
        @self.sio.event
        def shard_assign(data):
            print(f"📥 收到分片分配: {data.get('shardId', 'unknown')}")
        
        @self.sio.event
        def task_inference(data):
            """接收推理任务"""
            task_id = data.get('taskId')
            prompt = data.get('prompt')
            print(f"🎯 收到推理任务: {task_id[:8]}...")
            
            threading.Thread(
                target=self._execute_inference,
                args=(task_id, prompt),
                daemon=True
            ).start()
        
        @self.sio.event
        def network_probe(data):
            """网络探测请求"""
            # 立即响应以测量延迟
            self.sio.emit('network_probe_response', {
                'timestamp': time.time(),
                'nodeId': self.node_id
            })
    
    def _register(self):
        """注册节点"""
        network_metrics = self.network_monitor.get_metrics()
        
        self.sio.emit('node:register', {
            **self.system_info,
            'parallelSlots': self.parallel_slots,
            'network': network_metrics
        })
    
    def _load_model(self):
        """加载模型"""
        self.shard = ModelShard(
            shard_id=f"{self.node_id}-shard-0",
            model_id=self.model_name,
            layer_start=0,
            layer_end=0,
            size=500
        )
        
        if self.shard.load(self.model_name):
            self.engine = InferenceEngine(self.shard)
            self.sio.emit('shard:loaded', {
                'shardId': self.shard.shard_id,
                'modelName': self.model_name,
                'loadTime': self.shard.load_time
            })
    
    def _execute_inference(self, task_id: str, prompt: str):
        """执行推理"""
        self.active_slots += 1
        
        try:
            if self.engine is None:
                raise ValueError("推理引擎未就绪")
            
            # 执行推理
            response, tokens, latency = self.engine.generate(prompt)
            
            # 更新统计
            self.tasks_completed += 1
            self.total_latency += latency
            self.total_tokens += tokens
            
            # 发送结果
            self.sio.emit('inference:result', {
                'taskId': task_id,
                'status': 'completed',
                'result': response,
                'tokens': tokens,
                'latency': latency,
                'throughput': tokens / latency if latency > 0 else 0
            })
            
            print(f"✅ 任务完成: {task_id[:8]}... ({latency:.2f}s, {tokens} tokens)")
            
        except Exception as e:
            print(f"❌ 推理失败: {e}")
            self.sio.emit('inference:result', {
                'taskId': task_id,
                'status': 'failed',
                'error': str(e)
            })
        
        finally:
            self.active_slots -= 1
            self._send_heartbeat()
    
    def _send_heartbeat(self):
        """发送心跳"""
        memory = psutil.virtual_memory()
        network_metrics = self.network_monitor.get_metrics()
        
        self.sio.emit('node:heartbeat', {
            'availableMemory': int(memory.available / (1024 * 1024)),
            'status': 'busy' if self.active_slots > 0 else 'online',
            'loadScore': psutil.cpu_percent(),
            'activeSlots': self.active_slots,
            'tasksCompleted': self.tasks_completed,
            'avgLatency': self.total_latency / self.tasks_completed if self.tasks_completed > 0 else 0,
            'throughput': self.total_tokens / self.total_latency if self.total_latency > 0 else 0,
            'network': network_metrics
        })
    
    def _heartbeat_loop(self):
        """心跳循环"""
        while self.running:
            try:
                if self.sio.connected:
                    self._send_heartbeat()
            except Exception as e:
                print(f"[心跳错误] {e}")
            time.sleep(10)
    
    def start(self):
        """启动服务"""
        print(f"\n{'='*60}")
        print(f"  分布式大模型推理系统 - 网络感知节点")
        print(f"{'='*60}")
        print(f"  节点ID: {self.node_id}")
        print(f"  节点名称: {self.node_name}")
        print(f"  模型: {self.model_name}")
        print(f"  并行槽位: {self.parallel_slots}")
        print(f"  服务器: {self.server_url}")
        print(f"{'='*60}\n")
        
        # 启动网络监控
        self.network_monitor.start()
        
        # 启动心跳线程
        threading.Thread(target=self._heartbeat_loop, daemon=True).start()
        
        # 连接服务器
        while self.running:
            try:
                print(f"🔗 连接到 {self.server_url}...")
                self.sio.connect(self.server_url, transports=["websocket", "polling"])
                self.sio.wait()
            except Exception as e:
                print(f"❌ 连接错误: {e}")
                print("⏳ 5秒后重试...")
                time.sleep(5)
    
    def stop(self):
        """停止服务"""
        self.running = False
        self.network_monitor.stop()
        if self.shard:
            self.shard.unload()
        if self.sio.connected:
            self.sio.disconnect()
        print("🛑 节点服务已停止")


def main():
    parser = argparse.ArgumentParser(description="网络感知节点服务")
    parser.add_argument('--server', '-s', required=True, help='服务器地址')
    parser.add_argument('--name', '-n', default=None, help='节点名称')
    parser.add_argument('--model', '-m', default='Qwen/Qwen2.5-0.5B-Instruct', help='模型名称')
    
    args = parser.parse_args()
    
    service = NetworkAwareNodeService(
        server_url=args.server,
        name=args.name,
        model_name=args.model
    )
    
    try:
        service.start()
    except KeyboardInterrupt:
        print("\n⚠️ 收到中断信号")
        service.stop()


if __name__ == "__main__":
    main()
