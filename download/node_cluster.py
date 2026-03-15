#!/usr/bin/env python3
"""
分布式大模型推理系统 - 集群协调版本
====================================

核心功能:
1. 节点自动发现和注册
2. 集群资源聚合评估
3. 分布式推理协调
4. 统一API入口
5. 请求转发机制

工作流程:
1. 节点启动，检测本地资源
2. 如果资源不足，进入待机模式
3. 发现其他节点，聚合集群资源
4. 集群资源充足时，协调启动模型
5. 任何节点都可以接收API请求
6. 没有模型的节点转发请求到有模型的节点

使用方法:
    python download/node_cluster.py --port 8000 --seeds "localhost:8001,localhost:8002"
"""

import os
import sys
import time
import json
import uuid
import socket
import threading
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
import struct

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


# ==================== 枚举 ====================

class NodeRole(Enum):
    """节点角色"""
    STANDBY = "standby"      # 待机（资源不足）
    WORKER = "worker"        # 工作节点（有模型）
    COORDINATOR = "coordinator"  # 协调者
    UNKNOWN = "unknown"


class ClusterState(Enum):
    """集群状态"""
    INITIALIZING = "initializing"
    INSUFFICIENT = "insufficient"  # 资源不足
    READY = "ready"               # 就绪
    RUNNING = "running"           # 运行中


# ==================== 配置 ====================

@dataclass
class ClusterConfig:
    """集群配置"""
    # 节点配置
    node_id: str = ""
    node_name: str = ""
    host: str = "0.0.0.0"
    port: int = 8000
    
    # 模型配置
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    model_memory_gb: float = 2.0
    
    # 资源阈值
    min_memory_gb: float = 2.0
    min_cpu_percent: float = 10.0
    
    # 集群配置
    seeds: List[str] = field(default_factory=list)
    discovery_port: int = 9000
    heartbeat_interval: float = 5.0
    
    # 协调配置
    cluster_min_memory_gb: float = 4.0  # 集群最小内存
    cluster_min_nodes: int = 1          # 最小节点数
    
    def __post_init__(self):
        if not self.node_id:
            self.node_id = str(uuid.uuid4())[:8]
        if not self.node_name:
            self.node_name = f"Node-{self.node_id}"


# ==================== 节点信息 ====================

@dataclass
class NodeInfo:
    """节点信息"""
    node_id: str
    node_name: str
    host: str
    port: int
    role: NodeRole = NodeRole.STANDBY
    
    # 资源信息
    memory_total_gb: float = 0.0
    memory_available_gb: float = 0.0
    cpu_percent: float = 0.0
    cpu_cores: int = 0
    
    # 模型信息
    model_loaded: bool = False
    model_name: str = ""
    
    # 状态
    last_heartbeat: float = 0.0
    is_alive: bool = True
    
    def to_dict(self) -> Dict:
        return {
            "node_id": self.node_id,
            "node_name": self.node_name,
            "host": self.host,
            "port": self.port,
            "role": self.role.value,
            "memory_total_gb": self.memory_total_gb,
            "memory_available_gb": self.memory_available_gb,
            "cpu_percent": self.cpu_percent,
            "cpu_cores": self.cpu_cores,
            "model_loaded": self.model_loaded,
            "model_name": self.model_name,
            "last_heartbeat": self.last_heartbeat,
            "is_alive": self.is_alive,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'NodeInfo':
        return cls(
            node_id=data.get("node_id", ""),
            node_name=data.get("node_name", ""),
            host=data.get("host", ""),
            port=data.get("port", 0),
            role=NodeRole(data.get("role", "standby")),
            memory_total_gb=data.get("memory_total_gb", 0.0),
            memory_available_gb=data.get("memory_available_gb", 0.0),
            cpu_percent=data.get("cpu_percent", 0.0),
            cpu_cores=data.get("cpu_cores", 0),
            model_loaded=data.get("model_loaded", False),
            model_name=data.get("model_name", ""),
            last_heartbeat=data.get("last_heartbeat", 0.0),
            is_alive=data.get("is_alive", True),
        )


# ==================== 资源检测 ====================

class ResourceMonitor:
    """资源监控"""
    
    @staticmethod
    def get_local_resources() -> Dict:
        """获取本地资源"""
        if not HAS_PSUTIL:
            return {
                "memory_total_gb": 8.0,
                "memory_available_gb": 4.0,
                "cpu_percent": 50.0,
                "cpu_cores": 4,
            }
        
        mem = psutil.virtual_memory()
        return {
            "memory_total_gb": mem.total / (1024**3),
            "memory_available_gb": mem.available / (1024**3),
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "cpu_cores": psutil.cpu_count() or 4,
        }
    
    @staticmethod
    def can_run_model(config: ClusterConfig) -> Tuple[bool, str]:
        """检查本地资源是否足够运行模型"""
        resources = ResourceMonitor.get_local_resources()
        
        if resources["memory_available_gb"] < config.min_memory_gb:
            return False, f"内存不足: {resources['memory_available_gb']:.1f}GB < {config.min_memory_gb}GB"
        
        if resources["cpu_percent"] > (100 - config.min_cpu_percent):
            return False, f"CPU空闲不足: {100 - resources['cpu_percent']:.1f}% < {config.min_cpu_percent}%"
        
        return True, "资源充足"
    
    @staticmethod
    def get_resource_score() -> float:
        """获取资源评分 (0-100)"""
        resources = ResourceMonitor.get_local_resources()
        
        # 内存评分 (权重60%)
        mem_score = min(60, resources["memory_available_gb"] * 10)
        
        # CPU评分 (权重40%)
        cpu_idle = 100 - resources["cpu_percent"]
        cpu_score = min(40, cpu_idle * 0.4)
        
        return mem_score + cpu_score


# ==================== 集群管理 ====================

class ClusterManager:
    """集群管理器"""
    
    def __init__(self, config: ClusterConfig):
        self.config = config
        self.node_id = config.node_id
        
        # 节点列表
        self.nodes: Dict[str, NodeInfo] = {}
        self.lock = threading.Lock()
        
        # 本节点信息
        self.local_node = self._create_local_node()
        self.nodes[self.node_id] = self.local_node
        
        # 集群状态
        self.cluster_state = ClusterState.INITIALIZING
        self.coordinator_id: Optional[str] = None
        
        # 发现socket
        self.discovery_socket = None
    
    def _create_local_node(self) -> NodeInfo:
        """创建本节点信息"""
        resources = ResourceMonitor.get_local_resources()
        return NodeInfo(
            node_id=self.node_id,
            node_name=self.config.node_name,
            host=self.config.host,
            port=self.config.port,
            role=NodeRole.STANDBY,
            memory_total_gb=resources["memory_total_gb"],
            memory_available_gb=resources["memory_available_gb"],
            cpu_percent=resources["cpu_percent"],
            cpu_cores=resources["cpu_cores"],
        )
    
    def update_local_node(self, model_loaded: bool = False):
        """更新本节点信息"""
        resources = ResourceMonitor.get_local_resources()
        
        with self.lock:
            self.local_node.memory_available_gb = resources["memory_available_gb"]
            self.local_node.cpu_percent = resources["cpu_percent"]
            self.local_node.model_loaded = model_loaded
            self.local_node.last_heartbeat = time.time()
            
            self.nodes[self.node_id] = self.local_node
    
    def add_node(self, node: NodeInfo):
        """添加节点"""
        with self.lock:
            node.last_heartbeat = time.time()
            node.is_alive = True
            self.nodes[node.node_id] = node
    
    def remove_node(self, node_id: str):
        """移除节点"""
        with self.lock:
            if node_id in self.nodes:
                del self.nodes[node_id]
    
    def get_alive_nodes(self) -> List[NodeInfo]:
        """获取存活节点"""
        now = time.time()
        with self.lock:
            return [
                node for node in self.nodes.values()
                if node.is_alive and (now - node.last_heartbeat) < 30
            ]
    
    def get_worker_nodes(self) -> List[NodeInfo]:
        """获取工作节点（有模型的）"""
        return [n for n in self.get_alive_nodes() if n.model_loaded]
    
    def get_cluster_resources(self) -> Dict:
        """获取集群总资源"""
        nodes = self.get_alive_nodes()
        
        total_memory = sum(n.memory_total_gb for n in nodes)
        available_memory = sum(n.memory_available_gb for n in nodes)
        total_cores = sum(n.cpu_cores for n in nodes)
        
        return {
            "node_count": len(nodes),
            "total_memory_gb": total_memory,
            "available_memory_gb": available_memory,
            "total_cores": total_cores,
            "worker_count": len(self.get_worker_nodes()),
        }
    
    def evaluate_cluster(self) -> Tuple[bool, str]:
        """评估集群资源是否足够"""
        resources = self.get_cluster_resources()
        
        if resources["node_count"] < self.config.cluster_min_nodes:
            return False, f"节点数不足: {resources['node_count']} < {self.config.cluster_min_nodes}"
        
        if resources["available_memory_gb"] < self.config.cluster_min_memory_gb:
            return False, f"集群内存不足: {resources['available_memory_gb']:.1f}GB < {self.config.cluster_min_memory_gb}GB"
        
        return True, "集群资源充足"
    
    def select_best_node_for_model(self) -> Optional[str]:
        """选择最佳节点运行模型"""
        nodes = self.get_alive_nodes()
        if not nodes:
            return None
        
        # 选择资源评分最高的节点
        best_node = max(nodes, key=lambda n: (
            n.memory_available_gb,
            -n.cpu_percent,
            -len([x for x in nodes if x.model_loaded])
        ))
        
        return best_node.node_id
    
    def select_worker_for_request(self) -> Optional[NodeInfo]:
        """选择工作节点处理请求"""
        workers = self.get_worker_nodes()
        if not workers:
            return None
        
        # 选择负载最低的工作节点
        return min(workers, key=lambda n: n.cpu_percent)
    
    def start_discovery(self):
        """启动节点发现"""
        # UDP广播发现
        self.discovery_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.discovery_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.discovery_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.discovery_socket.bind(('0.0.0.0', self.config.discovery_port))
        self.discovery_socket.settimeout(1.0)
        
        # 启动发现线程
        threading.Thread(target=self._discovery_loop, daemon=True).start()
        threading.Thread(target=self._broadcast_loop, daemon=True).start()
    
    def _broadcast_loop(self):
        """广播本节点信息"""
        while True:
            try:
                message = json.dumps({
                    "type": "discovery",
                    "node": self.local_node.to_dict(),
                }).encode()
                
                self.discovery_socket.sendto(
                    message,
                    ('<broadcast>', self.config.discovery_port)
                )
            except:
                pass
            
            time.sleep(self.config.heartbeat_interval)
    
    def _discovery_loop(self):
        """接收其他节点信息"""
        while True:
            try:
                data, addr = self.discovery_socket.recvfrom(4096)
                message = json.loads(data.decode())
                
                if message.get("type") == "discovery":
                    node_data = message.get("node", {})
                    node = NodeInfo.from_dict(node_data)
                    
                    if node.node_id != self.node_id:
                        self.add_node(node)
            except socket.timeout:
                continue
            except:
                pass
    
    def connect_to_seeds(self):
        """连接到种子节点"""
        if not HAS_REQUESTS:
            return
        
        for seed in self.config.seeds:
            try:
                host, port = seed.split(":")
                url = f"http://{host}:{port}/cluster/register"
                
                response = requests.post(url, json=self.local_node.to_dict(), timeout=5)
                if response.status_code == 200:
                    print(f"[集群] 已连接到种子节点: {seed}")
            except Exception as e:
                print(f"[集群] 连接种子节点失败 {seed}: {e}")
    
    def to_dict(self) -> Dict:
        """导出集群信息"""
        return {
            "cluster_state": self.cluster_state.value,
            "coordinator_id": self.coordinator_id,
            "nodes": {nid: node.to_dict() for nid, node in self.nodes.items()},
            "resources": self.get_cluster_resources(),
        }


# ==================== 模型管理 ====================

class ModelManager:
    """模型管理器"""
    
    def __init__(self, config: ClusterConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.loaded = False
    
    def load(self) -> bool:
        """加载模型"""
        if not HAS_TORCH:
            print("[模型] PyTorch未安装")
            return False
        
        if self.loaded:
            return True
        
        try:
            print(f"[模型] 加载模型: {self.config.model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            self.model.eval()
            
            self.loaded = True
            print("[模型] 加载完成")
            return True
            
        except Exception as e:
            print(f"[模型] 加载失败: {e}")
            return False
    
    def unload(self):
        """卸载模型"""
        if self.model:
            del self.model
            self.model = None
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        
        self.loaded = False
        
        import gc
        gc.collect()
        
        print("[模型] 已卸载")
    
    def inference(self, prompt: str, max_tokens: int = 50) -> Dict:
        """推理"""
        if not self.loaded:
            return {"success": False, "error": "模型未加载"}
        
        try:
            import time
            start = time.time()
            
            inputs = self.tokenizer(prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=max_tokens,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            latency = time.time() - start
            new_tokens = outputs.shape[1] - inputs.input_ids.shape[1]
            response = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
            
            return {
                "success": True,
                "response": response,
                "tokens": new_tokens,
                "latency": latency,
                "throughput": new_tokens / latency if latency > 0 else 0,
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}


# ==================== 集群节点服务 ====================

class ClusterNode:
    """集群节点"""
    
    def __init__(self, config: ClusterConfig):
        self.config = config
        self.node_id = config.node_id
        
        # 组件
        self.cluster = ClusterManager(config)
        self.model = ModelManager(config)
        
        # 状态
        self.running = False
        self.role = NodeRole.STANDBY
    
    def start(self):
        """启动节点"""
        print(f"\n{'='*60}")
        print(f"  分布式大模型推理系统 - 集群节点")
        print(f"{'='*60}")
        print(f"  节点ID: {self.node_id}")
        print(f"  节点名称: {self.config.node_name}")
        print(f"  端口: {self.config.port}")
        print(f"{'='*60}\n")
        
        # 检查本地资源
        can_run, reason = ResourceMonitor.can_run_model(self.config)
        print(f"[资源] 本地资源: {reason}")
        
        # 启动集群发现
        self.cluster.start_discovery()
        print("[集群] 节点发现已启动")
        
        # 连接种子节点
        self.cluster.connect_to_seeds()
        
        # 启动协调循环
        threading.Thread(target=self._coordination_loop, daemon=True).start()
        
        # 启动HTTP服务
        self._start_http_server()
        
        self.running = True
        print(f"[服务] HTTP服务已启动: http://localhost:{self.config.port}\n")
    
    def _coordination_loop(self):
        """协调循环"""
        while True:
            try:
                # 更新本节点信息
                self.cluster.update_local_node(self.model.loaded)
                
                # 评估集群资源
                can_run, reason = self.cluster.evaluate_cluster()
                
                if can_run and not self.model.loaded:
                    # 集群资源充足，检查是否应该在本节点加载模型
                    local_can, _ = ResourceMonitor.can_run_model(self.config)
                    
                    if local_can:
                        # 本节点可以加载模型
                        workers = self.cluster.get_worker_nodes()
                        
                        if len(workers) < 2:  # 保持至少2个工作节点
                            print(f"[协调] {reason}，本节点加载模型")
                            if self.model.load():
                                self.role = NodeRole.WORKER
                                self.cluster.local_node.role = NodeRole.WORKER
                
                elif not can_run and self.model.loaded:
                    # 集群资源不足，检查是否需要卸载
                    workers = self.cluster.get_worker_nodes()
                    if len(workers) > 1:  # 还有其他工作节点
                        print("[协调] 集群资源不足，本节点卸载模型")
                        self.model.unload()
                        self.role = NodeRole.STANDBY
                
            except Exception as e:
                print(f"[协调] 错误: {e}")
            
            time.sleep(10)
    
    def _start_http_server(self):
        """启动HTTP服务"""
        from http.server import HTTPServer, BaseHTTPRequestHandler
        
        node = self
        
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/health':
                    self._send_json({"status": "healthy"})
                elif self.path == '/status':
                    self._send_json(node.get_status())
                elif self.path == '/cluster':
                    self._send_json(node.cluster.to_dict())
                elif self.path == '/resources':
                    self._send_json(ResourceMonitor.get_local_resources())
                else:
                    self.send_error(404)
            
            def do_POST(self):
                if self.path == '/inference':
                    self._handle_inference()
                elif self.path == '/cluster/register':
                    self._handle_register()
                else:
                    self.send_error(404)
            
            def _handle_inference(self):
                content_length = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(content_length)
                
                try:
                    data = json.loads(body)
                    prompt = data.get('prompt', '')
                    max_tokens = data.get('max_tokens', 50)
                    
                    # 如果本节点有模型，直接推理
                    if node.model.loaded:
                        result = node.model.inference(prompt, max_tokens)
                        self._send_json(result)
                    else:
                        # 转发到工作节点
                        result = node._forward_request(prompt, max_tokens)
                        self._send_json(result)
                        
                except Exception as e:
                    self._send_json({"success": False, "error": str(e)}, 500)
            
            def _handle_register(self):
                content_length = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(content_length)
                
                try:
                    data = json.loads(body)
                    node_info = NodeInfo.from_dict(data)
                    node.cluster.add_node(node_info)
                    self._send_json({"success": True})
                except Exception as e:
                    self._send_json({"success": False, "error": str(e)}, 500)
            
            def _send_json(self, data, code=200):
                self.send_response(code)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(data, ensure_ascii=False).encode())
            
            def log_message(self, format, *args):
                pass
        
        server = HTTPServer((self.config.host, self.config.port), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
    
    def _forward_request(self, prompt: str, max_tokens: int) -> Dict:
        """转发请求到工作节点"""
        if not HAS_REQUESTS:
            return {"success": False, "error": "无工作节点可用"}
        
        worker = self.cluster.select_worker_for_request()
        if not worker:
            return {"success": False, "error": "无工作节点可用"}
        
        try:
            url = f"http://{worker.host}:{worker.port}/inference"
            response = requests.post(
                url,
                json={"prompt": prompt, "max_tokens": max_tokens},
                timeout=60
            )
            return response.json()
        except Exception as e:
            return {"success": False, "error": f"转发失败: {e}"}
    
    def get_status(self) -> Dict:
        """获取状态"""
        resources = ResourceMonitor.get_local_resources()
        cluster_resources = self.cluster.get_cluster_resources()
        
        return {
            "node_id": self.node_id,
            "node_name": self.config.node_name,
            "role": self.role.value,
            "model_loaded": self.model.loaded,
            "local_resources": resources,
            "cluster": {
                "node_count": cluster_resources["node_count"],
                "worker_count": cluster_resources["worker_count"],
                "available_memory_gb": cluster_resources["available_memory_gb"],
            },
            "can_run_locally": ResourceMonitor.can_run_model(self.config)[0],
            "cluster_ready": self.cluster.evaluate_cluster(self.config)[0],
        }


# ==================== 主函数 ====================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="集群协调节点")
    parser.add_argument("--port", "-p", type=int, default=8000)
    parser.add_argument("--name", "-n", default="")
    parser.add_argument("--model", "-m", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--seeds", "-s", default="", help="种子节点，逗号分隔")
    parser.add_argument("--min-memory", type=float, default=2.0)
    parser.add_argument("--min-cpu", type=float, default=10.0)
    
    args = parser.parse_args()
    
    seeds = []
    if args.seeds:
        seeds = [s.strip() for s in args.seeds.split(",")]
    
    config = ClusterConfig(
        port=args.port,
        node_name=args.name or f"Node-{args.port}",
        model_name=args.model,
        seeds=seeds,
        min_memory_gb=args.min_memory,
        min_cpu_percent=args.min_cpu,
    )
    
    node = ClusterNode(config)
    node.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n停止...")


if __name__ == "__main__":
    main()
