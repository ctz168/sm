#!/usr/bin/env python3
"""
分布式大模型推理系统 - 去中心化版本
====================================

核心特性:
1. 去中心化 - 无单点故障
2. 自动发现 - 节点自动发现和加入
3. 领导者选举 - Raft算法自动选举
4. 故障转移 - 自动故障检测和恢复
5. 动态算力 - 根据资源自动启停推理服务
6. 高可用 - 只要有一个节点在线，服务就能运行

架构:
- 每个节点既是调度器也是推理节点
- 使用Raft算法选举领导者
- 领导者负责任务调度和状态同步
- 所有节点都可以处理推理请求
"""

import os
import sys
import time
import json
import uuid
import socket
import threading
import random
import hashlib
import signal
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
import struct
import pickle

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("警告: psutil未安装，部分功能不可用")

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("警告: torch/transformers未安装，推理功能不可用")


# ==================== 常量定义 ====================

class NodeState(Enum):
    """节点状态"""
    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    LEADER = "leader"


class ServiceState(Enum):
    """服务状态"""
    STARTING = "starting"
    RUNNING = "running"
    DEGRADED = "degraded"      # 降级模式（暂停推理）
    STOPPED = "stopped"
    ERROR = "error"


# ==================== 配置 ====================

@dataclass
class Config:
    """系统配置"""
    # 网络配置
    discovery_port: int = 37000        # 节点发现端口
    communication_port: int = 37001    # 节点通信端口
    api_port: int = 37002              # API端口
    
    # Raft配置
    heartbeat_interval: float = 1.0    # 心跳间隔(秒)
    election_timeout_min: float = 2.0  # 选举超时最小值
    election_timeout_max: float = 4.0  # 选举超时最大值
    
    # 资源配置
    min_memory_gb: float = 2.0         # 最小内存要求
    min_cpu_percent: float = 10.0      # 最小CPU空闲
    
    # 推理配置
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    max_workers: int = 2
    
    # 高可用配置
    min_nodes_for_inference: int = 1   # 最少节点数才能开启推理
    auto_recovery: bool = True         # 自动恢复


# ==================== 节点信息 ====================

@dataclass
class NodeInfo:
    """节点信息"""
    node_id: str
    address: str
    port: int
    state: NodeState = NodeState.FOLLOWER
    term: int = 0
    last_heartbeat: float = 0.0
    
    # 资源信息
    cpu_cores: int = 0
    memory_gb: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    
    # 服务信息
    service_state: ServiceState = ServiceState.STOPPED
    model_loaded: bool = False
    can_inference: bool = False
    
    # 统计信息
    tasks_completed: int = 0
    uptime: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "node_id": self.node_id,
            "address": self.address,
            "port": self.port,
            "state": self.state.value,
            "term": self.term,
            "cpu_cores": self.cpu_cores,
            "memory_gb": self.memory_gb,
            "cpu_usage": self.cpu_usage,
            "memory_usage": self.memory_usage,
            "service_state": self.service_state.value,
            "model_loaded": self.model_loaded,
            "can_inference": self.can_inference,
            "tasks_completed": self.tasks_completed,
            "uptime": self.uptime,
        }


# ==================== 资源监控器 ====================

class ResourceMonitor:
    """资源监控器"""
    
    def __init__(self):
        self.min_memory_gb = 2.0
        self.min_cpu_percent = 10.0
    
    def get_system_info(self) -> Dict:
        """获取系统信息"""
        info = {
            "cpu_cores": os.cpu_count() or 4,
            "memory_gb": 8.0,
            "cpu_usage": 0.0,
            "memory_usage": 0.0,
            "available_memory_gb": 4.0,
        }
        
        if HAS_PSUTIL:
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            info["memory_gb"] = memory.total / (1024**3)
            info["cpu_usage"] = cpu_percent
            info["memory_usage"] = memory.percent
            info["available_memory_gb"] = memory.available / (1024**3)
        
        return info
    
    def can_run_inference(self, model_size_gb: float = 2.0) -> Tuple[bool, str]:
        """检查是否可以运行推理"""
        info = self.get_system_info()
        
        # 检查内存
        if info["available_memory_gb"] < model_size_gb * 1.5:
            return False, f"内存不足: 需要{model_size_gb * 1.5:.1f}GB, 可用{info['available_memory_gb']:.1f}GB"
        
        # 检查CPU
        if info["cpu_usage"] > 90:
            return False, f"CPU负载过高: {info['cpu_usage']:.1f}%"
        
        return True, "资源充足"
    
    def get_load_score(self) -> float:
        """获取负载评分 (0-100, 越低越好)"""
        info = self.get_system_info()
        
        # CPU负载评分
        cpu_score = info["cpu_usage"]
        
        # 内存负载评分
        memory_score = info["memory_usage"]
        
        # 综合评分
        return cpu_score * 0.5 + memory_score * 0.5


# ==================== 节点发现服务 ====================

class NodeDiscovery:
    """节点发现服务 - 使用UDP广播"""
    
    def __init__(self, config: Config, node_id: str, port: int):
        self.config = config
        self.node_id = node_id
        self.port = port
        self.known_nodes: Dict[str, NodeInfo] = {}
        self.running = False
        
        self.broadcast_socket = None
        self.listen_socket = None
    
    def start(self):
        """启动发现服务"""
        self.running = True
        
        # 创建广播socket
        self.broadcast_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.broadcast_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.broadcast_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # 创建监听socket
        self.listen_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.listen_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listen_socket.bind(('0.0.0.0', self.config.discovery_port))
        self.listen_socket.settimeout(1.0)
        
        # 启动监听线程
        threading.Thread(target=self._listen_loop, daemon=True).start()
        
        # 启动广播线程
        threading.Thread(target=self._broadcast_loop, daemon=True).start()
        
        print(f"✅ 节点发现服务已启动 (端口: {self.config.discovery_port})")
    
    def stop(self):
        """停止发现服务"""
        self.running = False
        if self.broadcast_socket:
            self.broadcast_socket.close()
        if self.listen_socket:
            self.listen_socket.close()
    
    def _broadcast_loop(self):
        """广播节点信息"""
        while self.running:
            try:
                message = json.dumps({
                    "type": "discovery",
                    "node_id": self.node_id,
                    "port": self.port,
                    "timestamp": time.time(),
                }).encode()
                
                self.broadcast_socket.sendto(
                    message,
                    ('<broadcast>', self.config.discovery_port)
                )
                
            except Exception as e:
                if self.running:
                    print(f"[发现] 广播错误: {e}")
            
            time.sleep(5)  # 每5秒广播一次
    
    def _listen_loop(self):
        """监听其他节点的广播"""
        while self.running:
            try:
                data, addr = self.listen_socket.recvfrom(4096)
                message = json.loads(data.decode())
                
                if message["type"] == "discovery":
                    node_id = message["node_id"]
                    if node_id != self.node_id:
                        # 更新已知节点
                        if node_id not in self.known_nodes:
                            print(f"[发现] 发现新节点: {node_id} ({addr[0]})")
                        
                        self.known_nodes[node_id] = NodeInfo(
                            node_id=node_id,
                            address=addr[0],
                            port=message["port"],
                            last_heartbeat=time.time(),
                        )
                
            except socket.timeout:
                pass
            except Exception as e:
                if self.running:
                    pass  # 忽略错误
    
    def get_nodes(self) -> List[NodeInfo]:
        """获取所有已知节点"""
        # 清理超时节点
        current_time = time.time()
        timeout = 30  # 30秒超时
        
        expired = [
            node_id for node_id, node in self.known_nodes.items()
            if current_time - node.last_heartbeat > timeout
        ]
        
        for node_id in expired:
            print(f"[发现] 节点超时: {node_id}")
            del self.known_nodes[node_id]
        
        return list(self.known_nodes.values())


# ==================== Raft共识实现 ====================

class RaftNode:
    """Raft共识节点"""
    
    def __init__(self, node_id: str, config: Config):
        self.node_id = node_id
        self.config = config
        
        # 持久状态
        self.current_term = 0
        self.voted_for: Optional[str] = None
        
        # 易失状态
        self.state = NodeState.FOLLOWER
        self.leader_id: Optional[str] = None
        
        # 选举超时
        self.election_timeout = self._random_election_timeout()
        self.last_heartbeat = time.time()
        
        # 领导者状态
        self.next_index: Dict[str, int] = {}
        self.match_index: Dict[str, int] = {}
        
        # 日志
        self.log: List[Dict] = []
        self.commit_index = 0
        self.last_applied = 0
        
        # 投票
        self.votes_received: Set[str] = set()
        
        # 锁
        self.lock = threading.Lock()
    
    def _random_election_timeout(self) -> float:
        """随机选举超时"""
        return random.uniform(
            self.config.election_timeout_min,
            self.config.election_timeout_max
        )
    
    def tick(self) -> Optional[str]:
        """
        时钟滴答，返回需要执行的动作
        返回: "election" 表示需要开始选举
        """
        with self.lock:
            current_time = time.time()
            
            if self.state == NodeState.LEADER:
                # 领导者发送心跳
                return "heartbeat"
            
            elif self.state in [NodeState.FOLLOWER, NodeState.CANDIDATE]:
                # 检查选举超时
                if current_time - self.last_heartbeat > self.election_timeout:
                    return "election"
            
            return None
    
    def become_candidate(self):
        """成为候选人"""
        with self.lock:
            self.state = NodeState.CANDIDATE
            self.current_term += 1
            self.voted_for = self.node_id
            self.votes_received = {self.node_id}
            self.leader_id = None
            self.election_timeout = self._random_election_timeout()
            
            print(f"[Raft] 成为候选人 (任期: {self.current_term})")
    
    def become_leader(self):
        """成为领导者"""
        with self.lock:
            self.state = NodeState.LEADER
            self.leader_id = self.node_id
            
            print(f"[Raft] 成为领导者 (任期: {self.current_term})")
    
    def become_follower(self, term: int, leader_id: str):
        """成为跟随者"""
        with self.lock:
            self.state = NodeState.FOLLOWER
            self.current_term = term
            self.leader_id = leader_id
            self.voted_for = None
            self.last_heartbeat = time.time()
            self.election_timeout = self._random_election_timeout()
    
    def receive_heartbeat(self, term: int, leader_id: str) -> bool:
        """接收心跳"""
        with self.lock:
            if term >= self.current_term:
                self.current_term = term
                self.state = NodeState.FOLLOWER
                self.leader_id = leader_id
                self.last_heartbeat = time.time()
                self.election_timeout = self._random_election_timeout()
                return True
            return False
    
    def request_vote(self, term: int, candidate_id: str) -> Tuple[bool, int]:
        """处理投票请求"""
        with self.lock:
            if term > self.current_term:
                self.current_term = term
                self.state = NodeState.FOLLOWER
                self.voted_for = None
            
            vote_granted = False
            if term >= self.current_term:
                if self.voted_for is None or self.voted_for == candidate_id:
                    self.voted_for = candidate_id
                    vote_granted = True
                    self.last_heartbeat = time.time()
            
            return vote_granted, self.current_term
    
    def receive_vote(self, term: int, voter_id: str, vote_granted: bool) -> bool:
        """接收投票"""
        with self.lock:
            if self.state != NodeState.CANDIDATE:
                return False
            
            if term != self.current_term:
                return False
            
            if vote_granted:
                self.votes_received.add(voter_id)
            
            return False
    
    def is_leader(self) -> bool:
        return self.state == NodeState.LEADER
    
    def get_leader(self) -> Optional[str]:
        return self.leader_id


# ==================== 推理引擎 ====================

class InferenceEngine:
    """推理引擎"""
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.loaded = False
        self.model_size_gb = 2.0
        self.resource_monitor = ResourceMonitor()
        
        # 统计
        self.stats = {
            "total_requests": 0,
            "total_tokens": 0,
            "total_latency": 0.0,
        }
    
    def load(self) -> bool:
        """加载模型"""
        if not HAS_TORCH:
            print("❌ PyTorch未安装，无法加载模型")
            return False
        
        # 检查资源
        can_run, reason = self.resource_monitor.can_run_inference(self.model_size_gb)
        if not can_run:
            print(f"❌ 资源不足: {reason}")
            return False
        
        try:
            print(f"📥 加载模型: {self.config.model_name}")
            
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
            
            # 计算模型大小
            param_count = sum(p.numel() for p in self.model.parameters())
            self.model_size_gb = param_count * 4 / (1024**3)
            
            self.loaded = True
            print(f"✅ 模型加载完成 ({self.model_size_gb:.2f}GB)")
            return True
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
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
        
        # 清理内存
        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        import gc
        gc.collect()
        
        print("📤 模型已卸载")
    
    def generate(self, prompt: str, max_tokens: int = 100) -> Tuple[str, int, float]:
        """生成文本"""
        if not self.loaded:
            raise RuntimeError("模型未加载")
        
        start_time = time.time()
        
        # 编码
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=max_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # 解码
        new_tokens = outputs.shape[1] - inputs.input_ids.shape[1]
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        latency = time.time() - start_time
        
        # 更新统计
        self.stats["total_requests"] += 1
        self.stats["total_tokens"] += new_tokens
        self.stats["total_latency"] += latency
        
        return response, new_tokens, latency
    
    def can_inference(self) -> Tuple[bool, str]:
        """检查是否可以推理"""
        if not self.loaded:
            return False, "模型未加载"
        
        return self.resource_monitor.can_run_inference(self.model_size_gb)


# ==================== 去中心化节点 ====================

class DecentralizedNode:
    """去中心化节点"""
    
    def __init__(self, config: Config):
        self.config = config
        self.node_id = self._generate_node_id()
        self.start_time = time.time()
        
        # 组件
        self.raft = RaftNode(self.node_id, config)
        self.discovery = NodeDiscovery(config, self.node_id, config.api_port)
        self.engine = InferenceEngine(config)
        self.resource_monitor = ResourceMonitor()
        
        # 状态
        self.running = False
        self.service_state = ServiceState.STARTING
        
        # 任务队列
        self.task_queue: List[Dict] = []
        self.task_results: Dict[str, Dict] = {}
        
        # 网络
        self.api_socket: Optional[socket.socket] = None
    
    def _generate_node_id(self) -> str:
        """生成节点ID"""
        # 基于MAC地址和端口生成唯一ID
        try:
            mac = ':'.join(['{:02x}'.format((uuid.getnode() >> elements) & 0xff) 
                          for elements in range(0,2*6,2)][::-1])
        except:
            mac = str(uuid.uuid4())[:8]
        
        return f"node-{mac}-{random.randint(1000, 9999)}"
    
    def start(self):
        """启动节点"""
        print(f"\n{'='*60}")
        print(f"  分布式大模型推理系统 - 去中心化节点")
        print(f"{'='*60}")
        print(f"  节点ID: {self.node_id}")
        print(f"  API端口: {self.config.api_port}")
        print(f"{'='*60}\n")
        
        self.running = True
        self.service_state = ServiceState.STARTING
        
        # 启动发现服务
        self.discovery.start()
        
        # 启动API服务
        self._start_api_server()
        
        # 启动主循环
        threading.Thread(target=self._main_loop, daemon=True).start()
        
        # 启动资源监控
        threading.Thread(target=self._resource_loop, daemon=True).start()
        
        print("✅ 节点启动完成\n")
    
    def stop(self):
        """停止节点"""
        print("\n🛑 正在停止节点...")
        self.running = False
        self.service_state = ServiceState.STOPPED
        
        # 卸载模型
        if self.engine.loaded:
            self.engine.unload()
        
        # 停止发现服务
        self.discovery.stop()
        
        # 关闭API
        if self.api_socket:
            self.api_socket.close()
        
        print("✅ 节点已停止")
    
    def _start_api_server(self):
        """启动API服务器"""
        self.api_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.api_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.api_socket.bind(('0.0.0.0', self.config.api_port))
        self.api_socket.listen(10)
        self.api_socket.settimeout(1.0)
        
        threading.Thread(target=self._api_loop, daemon=True).start()
        
        print(f"✅ API服务已启动 (端口: {self.config.api_port})")
    
    def _api_loop(self):
        """API请求处理循环"""
        while self.running:
            try:
                conn, addr = self.api_socket.accept()
                threading.Thread(target=self._handle_api_request, args=(conn, addr), daemon=True).start()
            except socket.timeout:
                pass
            except Exception as e:
                if self.running:
                    pass
    
    def _handle_api_request(self, conn: socket.socket, addr: Tuple[str, int]):
        """处理API请求"""
        try:
            data = conn.recv(4096)
            if not data:
                return
            
            request = json.loads(data.decode())
            response = self._process_request(request)
            
            conn.sendall(json.dumps(response).encode())
            
        except Exception as e:
            try:
                conn.sendall(json.dumps({"error": str(e)}).encode())
            except:
                pass
        finally:
            conn.close()
    
    def _process_request(self, request: Dict) -> Dict:
        """处理请求"""
        request_type = request.get("type", "unknown")
        
        if request_type == "status":
            return self._get_status()
        
        elif request_type == "inference":
            return self._handle_inference(request)
        
        elif request_type == "heartbeat":
            # Raft心跳
            term = request.get("term", 0)
            leader_id = request.get("leader_id", "")
            self.raft.receive_heartbeat(term, leader_id)
            return {"success": True}
        
        elif request_type == "vote_request":
            # 投票请求
            term = request.get("term", 0)
            candidate_id = request.get("candidate_id", "")
            granted, current_term = self.raft.request_vote(term, candidate_id)
            return {"vote_granted": granted, "term": current_term}
        
        else:
            return {"error": "Unknown request type"}
    
    def _get_status(self) -> Dict:
        """获取状态"""
        info = self.resource_monitor.get_system_info()
        
        return {
            "node_id": self.node_id,
            "state": self.raft.state.value,
            "term": self.raft.current_term,
            "leader_id": self.raft.get_leader(),
            "service_state": self.service_state.value,
            "model_loaded": self.engine.loaded,
            "uptime": time.time() - self.start_time,
            "known_nodes": len(self.discovery.known_nodes),
            **info,
            "stats": self.engine.stats,
        }
    
    def _handle_inference(self, request: Dict) -> Dict:
        """处理推理请求"""
        # 检查是否可以推理
        if not self.engine.loaded:
            # 尝试加载模型
            if not self._try_load_model():
                return {"error": "无法加载模型，资源不足"}
        
        # 检查资源
        can_run, reason = self.engine.can_inference()
        if not can_run:
            return {"error": f"资源不足: {reason}"}
        
        # 执行推理
        prompt = request.get("prompt", "")
        max_tokens = request.get("max_tokens", 100)
        
        try:
            response, tokens, latency = self.engine.generate(prompt, max_tokens)
            return {
                "success": True,
                "response": response,
                "tokens": tokens,
                "latency": latency,
                "node_id": self.node_id,
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _main_loop(self):
        """主循环"""
        while self.running:
            # Raft时钟滴答
            action = self.raft.tick()
            
            if action == "election":
                # 开始选举
                self._start_election()
            
            elif action == "heartbeat":
                # 发送心跳
                self._send_heartbeats()
            
            time.sleep(0.1)
    
    def _start_election(self):
        """开始选举"""
        self.raft.become_candidate()
        
        # 向所有已知节点请求投票
        nodes = self.discovery.get_nodes()
        votes_needed = (len(nodes) + 1) // 2 + 1  # 多数票
        
        for node in nodes:
            threading.Thread(
                target=self._request_vote_from_node,
                args=(node,),
                daemon=True
            ).start()
        
        # 等待投票结果
        time.sleep(self.config.election_timeout_min / 2)
        
        # 检查是否获得多数票
        if len(self.raft.votes_received) >= votes_needed:
            self.raft.become_leader()
            self._on_become_leader()
    
    def _request_vote_from_node(self, node: NodeInfo):
        """向节点请求投票"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2.0)
            sock.connect((node.address, node.port))
            
            request = {
                "type": "vote_request",
                "term": self.raft.current_term,
                "candidate_id": self.node_id,
            }
            
            sock.sendall(json.dumps(request).encode())
            response = json.loads(sock.recv(4096).decode())
            
            if response.get("vote_granted"):
                self.raft.receive_vote(
                    response.get("term", 0),
                    node.node_id,
                    True
                )
            
            sock.close()
            
        except Exception as e:
            pass
    
    def _send_heartbeats(self):
        """发送心跳"""
        nodes = self.discovery.get_nodes()
        
        for node in nodes:
            threading.Thread(
                target=self._send_heartbeat_to_node,
                args=(node,),
                daemon=True
            ).start()
    
    def _send_heartbeat_to_node(self, node: NodeInfo):
        """向节点发送心跳"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1.0)
            sock.connect((node.address, node.port))
            
            request = {
                "type": "heartbeat",
                "term": self.raft.current_term,
                "leader_id": self.node_id,
            }
            
            sock.sendall(json.dumps(request).encode())
            sock.close()
            
        except Exception as e:
            pass
    
    def _on_become_leader(self):
        """成为领导者时的回调"""
        print(f"[领导者] 我已成为领导者")
        
        # 尝试加载模型
        self._try_load_model()
    
    def _resource_loop(self):
        """资源监控循环"""
        while self.running:
            # 检查资源
            can_run, reason = self.resource_monitor.can_run_inference(2.0)
            
            if can_run and not self.engine.loaded:
                # 资源充足，尝试加载模型
                if self.config.auto_recovery:
                    self._try_load_model()
            
            elif not can_run and self.engine.loaded:
                # 资源不足，卸载模型
                print(f"[资源] 资源不足，卸载模型: {reason}")
                self.engine.unload()
                self.service_state = ServiceState.DEGRADED
            
            # 更新服务状态
            if self.engine.loaded:
                self.service_state = ServiceState.RUNNING
            else:
                self.service_state = ServiceState.DEGRADED
            
            time.sleep(10)
    
    def _try_load_model(self) -> bool:
        """尝试加载模型"""
        if self.engine.loaded:
            return True
        
        can_run, reason = self.resource_monitor.can_run_inference(2.0)
        if not can_run:
            print(f"[资源] 无法加载模型: {reason}")
            return False
        
        return self.engine.load()
    
    def run_forever(self):
        """运行直到停止"""
        self.start()
        
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()


# ==================== 主函数 ====================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="分布式大模型推理 - 去中心化节点")
    parser.add_argument("--port", "-p", type=int, default=37002, help="API端口")
    parser.add_argument("--model", "-m", default="Qwen/Qwen2.5-0.5B-Instruct", help="模型名称")
    parser.add_argument("--discovery-port", type=int, default=37000, help="发现端口")
    
    args = parser.parse_args()
    
    config = Config(
        discovery_port=args.discovery_port,
        api_port=args.port,
        model_name=args.model,
    )
    
    node = DecentralizedNode(config)
    
    def signal_handler(sig, frame):
        node.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    node.run_forever()


if __name__ == "__main__":
    main()
