#!/usr/bin/env python3
"""
分布式大模型推理系统 - 去中心化版本
====================================

特性:
- 无单点故障
- 自动主节点选举
- 节点自动发现
- 故障自动恢复
- 只要有一个节点在线，服务就能运行

架构:
- P2P网络拓扑
- Raft共识协议
- 分布式状态存储
"""

import os

# HuggingFace 镜像配置（国内网络优化）
os.environ['HF_ENDPOINT'] = os.environ.get('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ['HF_HOME'] = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
import sys
import time
import json
import uuid
import socket
import threading
import hashlib
import random
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
import struct
import pickle

try:
    import socketio
    import psutil
except ImportError:
    print("请安装: pip install python-socketio psutil")
    sys.exit(1)

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ==================== 常量定义 ====================

class NodeState(Enum):
    """节点状态"""
    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    LEADER = "leader"


class MessageType(Enum):
    """消息类型"""
    # 节点发现
    DISCOVER = "discover"
    DISCOVER_RESPONSE = "discover_response"
    
    # 心跳
    HEARTBEAT = "heartbeat"
    HEARTBEAT_RESPONSE = "heartbeat_response"
    
    # 选举
    REQUEST_VOTE = "request_vote"
    VOTE_RESPONSE = "vote_response"
    
    # 状态同步
    STATE_SYNC = "state_sync"
    STATE_SYNC_ACK = "state_sync_ack"
    
    # 任务
    TASK_ASSIGN = "task_assign"
    TASK_RESULT = "task_result"
    
    # 节点加入/离开
    NODE_JOIN = "node_join"
    NODE_LEAVE = "node_leave"


# ==================== 配置 ====================

@dataclass
class DecentralizedConfig:
    """去中心化配置"""
    # 节点配置
    node_id: str = ""
    node_name: str = ""
    host: str = "0.0.0.0"
    port: int = 5000
    
    # 模型配置
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    max_workers: int = 2
    
    # 集群配置
    seed_nodes: List[str] = field(default_factory=list)  # 种子节点列表
    heartbeat_interval: float = 2.0  # 心跳间隔(秒)
    election_timeout: float = 5.0  # 选举超时(秒)
    leader_timeout: float = 10.0  # 主节点超时(秒)
    
    # 推理配置
    min_nodes_for_inference: int = 1  # 最少节点数才开启推理
    memory_limit_gb: float = 0.0
    
    def __post_init__(self):
        if not self.node_id:
            self.node_id = str(uuid.uuid4())
        if not self.node_name:
            self.node_name = f"Node-{self.node_id[:8]}"
        if self.memory_limit_gb == 0:
            self.memory_limit_gb = psutil.virtual_memory().total / (1024**3)


# ==================== 分布式状态存储 ====================

@dataclass
class NodeInfo:
    """节点信息"""
    node_id: str
    node_name: str
    host: str
    port: int
    state: NodeState
    is_leader: bool = False
    last_heartbeat: float = 0.0
    model_loaded: bool = False
    model_name: str = ""
    available_memory: int = 0
    cpu_cores: int = 0
    active_tasks: int = 0
    max_workers: int = 2
    term: int = 0  # 选举任期


@dataclass
class TaskInfo:
    """任务信息"""
    task_id: str
    prompt: str
    status: str  # pending, running, completed, failed
    assigned_node: Optional[str] = None
    result: Optional[str] = None
    created_at: float = 0.0
    completed_at: float = 0.0
    term: int = 0


class DistributedState:
    """分布式状态存储"""
    
    def __init__(self):
        self.lock = threading.RLock()
        
        # 节点信息
        self.nodes: Dict[str, NodeInfo] = {}
        self.leader_id: Optional[str] = None
        
        # 任务信息
        self.tasks: Dict[str, TaskInfo] = {}
        self.pending_tasks: List[str] = []
        self.completed_tasks: List[str] = []
        
        # 选举状态
        self.current_term: int = 0
        self.voted_for: Optional[str] = None
        self.last_log_index: int = 0
        self.last_log_term: int = 0
        
        # 日志
        self.log: List[Dict] = []
    
    def add_node(self, node: NodeInfo):
        """添加节点"""
        with self.lock:
            self.nodes[node.node_id] = node
    
    def remove_node(self, node_id: str):
        """移除节点"""
        with self.lock:
            if node_id in self.nodes:
                del self.nodes[node_id]
            if self.leader_id == node_id:
                self.leader_id = None
    
    def get_node(self, node_id: str) -> Optional[NodeInfo]:
        """获取节点"""
        with self.lock:
            return self.nodes.get(node_id)
    
    def get_active_nodes(self) -> List[NodeInfo]:
        """获取活跃节点"""
        with self.lock:
            now = time.time()
            return [
                node for node in self.nodes.values()
                if now - node.last_heartbeat < 30  # 30秒内有心跳
            ]
    
    def get_nodes_for_inference(self) -> List[NodeInfo]:
        """获取可用于推理的节点"""
        with self.lock:
            now = time.time()
            return [
                node for node in self.nodes.values()
                if (node.model_loaded and 
                    now - node.last_heartbeat < 30 and
                    node.active_tasks < node.max_workers)
            ]
    
    def set_leader(self, leader_id: str, term: int):
        """设置主节点"""
        with self.lock:
            self.leader_id = leader_id
            self.current_term = term
            if leader_id in self.nodes:
                self.nodes[leader_id].is_leader = True
                self.nodes[leader_id].state = NodeState.LEADER
    
    def add_task(self, task: TaskInfo):
        """添加任务"""
        with self.lock:
            self.tasks[task.task_id] = task
            if task.status == "pending":
                self.pending_tasks.append(task.task_id)
    
    def update_task(self, task_id: str, **kwargs):
        """更新任务"""
        with self.lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                for key, value in kwargs.items():
                    if hasattr(task, key):
                        setattr(task, key, value)
                
                if kwargs.get("status") == "completed":
                    if task_id in self.pending_tasks:
                        self.pending_tasks.remove(task_id)
                    self.completed_tasks.append(task_id)
    
    def get_next_task(self) -> Optional[TaskInfo]:
        """获取下一个待处理任务"""
        with self.lock:
            if self.pending_tasks:
                task_id = self.pending_tasks.pop(0)
                return self.tasks.get(task_id)
            return None
    
    def to_dict(self) -> Dict:
        """序列化为字典"""
        with self.lock:
            return {
                "nodes": {
                    nid: {
                        "node_id": n.node_id,
                        "node_name": n.node_name,
                        "host": n.host,
                        "port": n.port,
                        "state": n.state.value,
                        "is_leader": n.is_leader,
                        "last_heartbeat": n.last_heartbeat,
                        "model_loaded": n.model_loaded,
                        "model_name": n.model_name,
                        "available_memory": n.available_memory,
                        "cpu_cores": n.cpu_cores,
                        "active_tasks": n.active_tasks,
                        "max_workers": n.max_workers,
                    }
                    for nid, n in self.nodes.items()
                },
                "leader_id": self.leader_id,
                "current_term": self.current_term,
                "pending_tasks": len(self.pending_tasks),
                "completed_tasks": len(self.completed_tasks),
            }
    
    def from_dict(self, data: Dict):
        """从字典反序列化"""
        with self.lock:
            for nid, n in data.get("nodes", {}).items():
                self.nodes[nid] = NodeInfo(
                    node_id=n["node_id"],
                    node_name=n["node_name"],
                    host=n["host"],
                    port=n["port"],
                    state=NodeState(n["state"]),
                    is_leader=n["is_leader"],
                    last_heartbeat=n["last_heartbeat"],
                    model_loaded=n["model_loaded"],
                    model_name=n["model_name"],
                    available_memory=n["available_memory"],
                    cpu_cores=n["cpu_cores"],
                    active_tasks=n["active_tasks"],
                    max_workers=n["max_workers"],
                )
            
            self.leader_id = data.get("leader_id")
            self.current_term = data.get("current_term", 0)


# ==================== 网络通信 ====================

class P2PNetwork:
    """P2P网络通信"""
    
    def __init__(self, config: DecentralizedConfig, state: DistributedState):
        self.config = config
        self.state = state
        self.node_id = config.node_id
        
        # 已知节点
        self.known_nodes: Set[str] = set()
        self.node_connections: Dict[str, socketio.Client] = {}
        
        # 消息处理器
        self.message_handlers: Dict[MessageType, callable] = {}
        
        # Socket.IO服务器
        self.sio: Optional[socketio.AsyncServer] = None
        self.server_running = False
        
        # 客户端连接
        self.client_sio = socketio.Client(
            reconnection=True,
            reconnection_attempts=5,
            logger=False,
            engineio_logger=False
        )
    
    def register_handler(self, msg_type: MessageType, handler: callable):
        """注册消息处理器"""
        self.message_handlers[msg_type] = handler
    
    async def start_server(self):
        """启动P2P服务器"""
        from aiohttp import web
        import socketio as sio
        
        self.sio = sio.AsyncServer(
            cors_allowed_origins='*',
            async_mode='aiohttp'
        )
        app = web.Application()
        self.sio.attach(app)
        
        # 注册事件处理
        @self.sio.event
        async def connect(sid, environ):
            print(f"[P2P] 节点连接: {sid}")
        
        @self.sio.event
        async def disconnect(sid):
            print(f"[P2P] 节点断开: {sid}")
        
        @self.sio.on('message')
        async def on_message(sid, data):
            await self._handle_message(sid, data)
        
        # 启动服务器
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, self.config.host, self.config.port)
        await site.start()
        
        self.server_running = True
        print(f"[P2P] 服务器启动: {self.config.host}:{self.config.port}")
    
    async def _handle_message(self, sid: str, data: Dict):
        """处理接收到的消息"""
        try:
            msg_type = MessageType(data.get("type"))
            msg_data = data.get("data", {})
            
            if msg_type in self.message_handlers:
                handler = self.message_handlers[msg_type]
                result = await handler(msg_data)
                
                # 发送响应
                if result and data.get("request_id"):
                    await self.sio.emit('message', {
                        "type": msg_type.value + "_response",
                        "data": result,
                        "request_id": data["request_id"]
                    }, room=sid)
        except Exception as e:
            print(f"[P2P] 消息处理错误: {e}")
    
    def connect_to_node(self, host: str, port: int) -> bool:
        """连接到其他节点"""
        node_addr = f"{host}:{port}"
        
        if node_addr in self.node_connections:
            return True
        
        try:
            client = socketio.Client(
                reconnection=True,
                reconnection_attempts=3,
                logger=False,
                engineio_logger=False
            )
            
            client.connect(f"http://{host}:{port}", transports=['polling'])
            self.node_connections[node_addr] = client
            self.known_nodes.add(node_addr)
            
            print(f"[P2P] 已连接到节点: {node_addr}")
            return True
            
        except Exception as e:
            print(f"[P2P] 连接失败 {node_addr}: {e}")
            return False
    
    def send_message(self, node_addr: str, msg_type: MessageType, 
                     data: Dict, wait_response: bool = False) -> Optional[Dict]:
        """发送消息到节点"""
        if node_addr not in self.node_connections:
            return None
        
        client = self.node_connections[node_addr]
        request_id = str(uuid.uuid4())
        
        message = {
            "type": msg_type.value,
            "data": data,
            "request_id": request_id,
            "from_node": self.node_id
        }
        
        if wait_response:
            # 同步等待响应
            response_event = threading.Event()
            response_data = {"data": None}
            
            def on_response(data):
                response_data["data"] = data
                response_event.set()
            
            client.on('message', on_response)
            client.emit('message', message)
            
            if response_event.wait(timeout=5):
                return response_data["data"]
            return None
        else:
            client.emit('message', message)
            return None
    
    def broadcast(self, msg_type: MessageType, data: Dict):
        """广播消息到所有已知节点"""
        message = {
            "type": msg_type.value,
            "data": data,
            "from_node": self.node_id,
            "timestamp": time.time()
        }
        
        for node_addr, client in list(self.node_connections.items()):
            try:
                client.emit('message', message)
            except:
                del self.node_connections[node_addr]


# ==================== Raft共识协议 ====================

class RaftConsensus:
    """Raft共识协议实现"""
    
    def __init__(self, config: DecentralizedConfig, state: DistributedState, network: P2PNetwork):
        self.config = config
        self.state = state
        self.network = network
        self.node_id = config.node_id
        
        # 节点状态
        self.node_state = NodeState.FOLLOWER
        self.current_term = 0
        self.voted_for: Optional[str] = None
        
        # 选举相关
        self.last_heartbeat = time.time()
        self.election_timer: Optional[threading.Timer] = None
        self.votes_received: Set[str] = set()
        
        # 心跳
        self.heartbeat_thread: Optional[threading.Thread] = None
        
        # 回调
        self.on_become_leader: Optional[callable] = None
        self.on_become_follower: Optional[callable] = None
    
    def start(self):
        """启动共识协议"""
        # 注册消息处理器
        self.network.register_handler(
            MessageType.REQUEST_VOTE, 
            self._handle_vote_request
        )
        self.network.register_handler(
            MessageType.VOTE_RESPONSE,
            self._handle_vote_response
        )
        self.network.register_handler(
            MessageType.HEARTBEAT,
            self._handle_heartbeat
        )
        
        # 启动选举定时器
        self._reset_election_timer()
        
        print(f"[Raft] 共识协议启动, 初始状态: {self.node_state.value}")
    
    def _reset_election_timer(self):
        """重置选举定时器"""
        if self.election_timer:
            self.election_timer.cancel()
        
        # 随机超时时间，避免同时选举
        timeout = self.config.election_timeout + random.random() * 2
        
        self.election_timer = threading.Timer(timeout, self._start_election)
        self.election_timer.daemon = True
        self.election_timer.start()
    
    def _start_election(self):
        """开始选举"""
        if self.node_state == NodeState.LEADER:
            self._reset_election_timer()
            return
        
        print(f"[Raft] 开始选举, 任期 {self.current_term + 1}")
        
        # 转为候选人
        self.node_state = NodeState.CANDIDATE
        self.current_term += 1
        self.voted_for = self.node_id
        self.votes_received = {self.node_id}  # 投自己一票
        
        # 向所有节点请求投票
        vote_request = {
            "term": self.current_term,
            "candidate_id": self.node_id,
            "last_log_index": self.state.last_log_index,
            "last_log_term": self.state.last_log_term,
        }
        
        self.network.broadcast(MessageType.REQUEST_VOTE, vote_request)
        
        # 重置选举定时器
        self._reset_election_timer()
    
    async def _handle_vote_request(self, data: Dict) -> Dict:
        """处理投票请求"""
        term = data.get("term", 0)
        candidate_id = data.get("candidate_id")
        
        response = {
            "term": self.current_term,
            "vote_granted": False
        }
        
        # 如果请求的任期更高，更新自己的任期
        if term > self.current_term:
            self.current_term = term
            self.voted_for = None
            self.node_state = NodeState.FOLLOWER
        
        # 判断是否投票
        if term < self.current_term:
            return response
        
        if self.voted_for is None or self.voted_for == candidate_id:
            self.voted_for = candidate_id
            response["vote_granted"] = True
            self.last_heartbeat = time.time()
            self._reset_election_timer()
            print(f"[Raft] 投票给 {candidate_id}")
        
        return response
    
    async def _handle_vote_response(self, data: Dict):
        """处理投票响应"""
        if self.node_state != NodeState.CANDIDATE:
            return
        
        term = data.get("term", 0)
        vote_granted = data.get("vote_granted", False)
        voter_id = data.get("voter_id", "")
        
        if term > self.current_term:
            # 发现更高任期，转为follower
            self.current_term = term
            self.node_state = NodeState.FOLLOWER
            self.voted_for = None
            return
        
        if vote_granted:
            self.votes_received.add(voter_id)
            
            # 检查是否获得多数票
            total_nodes = len(self.state.nodes) + 1  # +1 是自己
            majority = total_nodes // 2 + 1
            
            if len(self.votes_received) >= majority:
                self._become_leader()
    
    async def _handle_heartbeat(self, data: Dict) -> Dict:
        """处理心跳"""
        term = data.get("term", 0)
        leader_id = data.get("leader_id")
        
        if term >= self.current_term:
            self.current_term = term
            self.node_state = NodeState.FOLLOWER
            self.state.set_leader(leader_id, term)
            self.last_heartbeat = time.time()
            self._reset_election_timer()
        
        return {
            "term": self.current_term,
            "node_id": self.node_id
        }
    
    def _become_leader(self):
        """成为主节点"""
        print(f"[Raft] 成为主节点! 任期 {self.current_term}")
        
        self.node_state = NodeState.LEADER
        self.state.set_leader(self.node_id, self.current_term)
        
        # 启动心跳线程
        if self.heartbeat_thread is None or not self.heartbeat_thread.is_alive():
            self.heartbeat_thread = threading.Thread(target=self._send_heartbeats, daemon=True)
            self.heartbeat_thread.start()
        
        # 回调
        if self.on_become_leader:
            self.on_become_leader()
    
    def _send_heartbeats(self):
        """发送心跳"""
        while self.node_state == NodeState.LEADER:
            heartbeat = {
                "term": self.current_term,
                "leader_id": self.node_id,
                "state": self.state.to_dict()
            }
            
            self.network.broadcast(MessageType.HEARTBEAT, heartbeat)
            time.sleep(self.config.heartbeat_interval)


# ==================== 去中心化节点 ====================

class DecentralizedNode:
    """去中心化节点"""
    
    def __init__(self, config: DecentralizedConfig):
        self.config = config
        self.node_id = config.node_id
        
        # 组件
        self.state = DistributedState()
        self.network = P2PNetwork(config, self.state)
        self.consensus = RaftConsensus(config, self.state, self.network)
        
        # 推理引擎
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        self.active_tasks = 0
        
        # 运行状态
        self.running = False
        
        # 注册回调
        self.consensus.on_become_leader = self._on_become_leader
        self.consensus.on_become_follower = self._on_become_follower
    
    def start(self):
        """启动节点"""
        print(f"\n{'='*60}")
        print(f"  分布式大模型推理系统 - 去中心化节点")
        print(f"{'='*60}")
        print(f"  节点ID: {self.node_id}")
        print(f"  节点名称: {self.config.node_name}")
        print(f"  监听地址: {self.config.host}:{self.config.port}")
        print(f"  模型: {self.config.model_name}")
        print(f"{'='*60}\n")
        
        self.running = True
        
        # 注册自己
        self._register_self()
        
        # 连接种子节点
        self._connect_to_seeds()
        
        # 启动共识协议
        self.consensus.start()
        
        # 启动节点发现
        self._start_discovery()
        
        # 启动任务处理
        self._start_task_processor()
        
        # 主循环
        try:
            while self.running:
                time.sleep(1)
                self._check_inference_readiness()
        except KeyboardInterrupt:
            self.stop()
    
    def _register_self(self):
        """注册自己"""
        memory = psutil.virtual_memory()
        
        self.state.add_node(NodeInfo(
            node_id=self.node_id,
            node_name=self.config.node_name,
            host=self.config.host,
            port=self.config.port,
            state=NodeState.FOLLOWER,
            last_heartbeat=time.time(),
            model_loaded=False,
            model_name=self.config.model_name,
            available_memory=int(memory.available / (1024**2)),
            cpu_cores=os.cpu_count() or 4,
            active_tasks=0,
            max_workers=self.config.max_workers
        ))
    
    def _connect_to_seeds(self):
        """连接到种子节点"""
        for seed in self.config.seed_nodes:
            try:
                host, port = seed.split(":")
                self.network.connect_to_node(host, int(port))
            except Exception as e:
                print(f"[发现] 连接种子节点失败 {seed}: {e}")
    
    def _start_discovery(self):
        """启动节点发现"""
        def discovery_loop():
            while self.running:
                # 广播发现消息
                self.network.broadcast(MessageType.DISCOVER, {
                    "node_id": self.node_id,
                    "node_name": self.config.node_name,
                    "host": self.config.host,
                    "port": self.config.port,
                })
                
                time.sleep(10)
        
        thread = threading.Thread(target=discovery_loop, daemon=True)
        thread.start()
    
    def _start_task_processor(self):
        """启动任务处理器"""
        def process_loop():
            while self.running:
                # 只有主节点分配任务
                if self.consensus.node_state == NodeState.LEADER:
                    self._assign_tasks()
                
                # 处理分配给自己的任务
                self._process_assigned_tasks()
                
                time.sleep(0.5)
        
        thread = threading.Thread(target=process_loop, daemon=True)
        thread.start()
    
    def _check_inference_readiness(self):
        """检查是否应该开启推理"""
        active_nodes = self.state.get_active_nodes()
        node_count = len(active_nodes)
        
        # 检查是否满足最少节点数
        if node_count >= self.config.min_nodes_for_inference:
            if not self.model_loaded:
                self._load_model()
        else:
            # 节点不足，卸载模型节省资源
            if self.model_loaded and self.active_tasks == 0:
                self._unload_model()
    
    def _load_model(self):
        """加载模型"""
        if not HAS_TORCH:
            print("[模型] PyTorch未安装，无法加载模型")
            return
        
        print(f"[模型] 加载模型: {self.config.model_name}")
        
        try:
            # 加载tokenizer（带重试）
            for retry in range(3):
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        self.config.model_name,
                        trust_remote_code=True
                    )
                    break
                except Exception as e:
                    if retry == 2:
                        raise
                    print(f"[模型] Tokenizer加载失败，重试 {retry+1}/3...")
                    import time
                    time.sleep(2)
            
            # 加载模型（带重试）
            for retry in range(3):
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.config.model_name,
                        torch_dtype=torch.float32,
                        trust_remote_code=True,
                        low_cpu_mem_usage=True
                    )
                    break
                except Exception as e:
                    if retry == 2:
                        raise
                    print(f"[模型] 模型加载失败，重试 {retry+1}/3...")
                    import time
                    time.sleep(5)
            self.model.eval()
            
            self.model_loaded = True
            
            # 更新状态
            node = self.state.get_node(self.node_id)
            if node:
                node.model_loaded = True
            
            print(f"[模型] 加载完成")
            
        except Exception as e:
            print(f"[模型] 加载失败: {e}")
    
    def _unload_model(self):
        """卸载模型"""
        if self.model:
            del self.model
            self.model = None
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        
        self.model_loaded = False
        
        # 更新状态
        node = self.state.get_node(self.node_id)
        if node:
            node.model_loaded = False
        
        import gc
        gc.collect()
        
        print(f"[模型] 已卸载")
    
    def _on_become_leader(self):
        """成为主节点回调"""
        print(f"[主节点] 成为集群主节点")
        
        # 主节点需要加载模型
        if not self.model_loaded:
            self._load_model()
    
    def _on_become_follower(self):
        """成为从节点回调"""
        print(f"[从节点] 成为集群从节点")
    
    def _assign_tasks(self):
        """分配任务（主节点执行）"""
        task = self.state.get_next_task()
        if not task:
            return
        
        # 找到可用节点
        available_nodes = self.state.get_nodes_for_inference()
        if not available_nodes:
            # 没有可用节点，任务放回队列
            self.state.pending_tasks.insert(0, task.task_id)
            return
        
        # 选择负载最低的节点
        best_node = min(available_nodes, key=lambda n: n.active_tasks)
        
        # 分配任务
        self.state.update_task(
            task.task_id,
            status="running",
            assigned_node=best_node.node_id
        )
        
        # 发送任务到节点
        node_addr = f"{best_node.host}:{best_node.port}"
        self.network.send_message(node_addr, MessageType.TASK_ASSIGN, {
            "task_id": task.task_id,
            "prompt": task.prompt
        })
        
        print(f"[任务] 分配 {task.task_id[:8]} 到 {best_node.node_name}")
    
    def _process_assigned_tasks(self):
        """处理分配给自己的任务"""
        # 检查是否有分配给自己的任务
        for task in list(self.state.tasks.values()):
            if (task.assigned_node == self.node_id and 
                task.status == "running" and
                self.model_loaded and
                self.active_tasks < self.config.max_workers):
                
                threading.Thread(
                    target=self._execute_task,
                    args=(task,),
                    daemon=True
                ).start()
    
    def _execute_task(self, task: TaskInfo):
        """执行推理任务"""
        self.active_tasks += 1
        
        try:
            # 推理
            inputs = self.tokenizer(task.prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=256,
                    temperature=0.7,
                    do_sample=True
                )
            
            result = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
            
            # 更新任务状态
            self.state.update_task(
                task.task_id,
                status="completed",
                result=result,
                completed_at=time.time()
            )
            
            print(f"[任务] 完成 {task.task_id[:8]}")
            
        except Exception as e:
            self.state.update_task(
                task.task_id,
                status="failed",
                result=str(e)
            )
            print(f"[任务] 失败 {task.task_id[:8]}: {e}")
        
        finally:
            self.active_tasks -= 1
    
    def submit_task(self, prompt: str) -> str:
        """提交任务"""
        task_id = str(uuid.uuid4())
        
        task = TaskInfo(
            task_id=task_id,
            prompt=prompt,
            status="pending",
            created_at=time.time(),
            term=self.state.current_term
        )
        
        self.state.add_task(task)
        
        print(f"[任务] 提交 {task_id[:8]}")
        return task_id
    
    def get_status(self) -> Dict:
        """获取状态"""
        return {
            "node_id": self.node_id,
            "node_name": self.config.node_name,
            "node_state": self.consensus.node_state.value,
            "is_leader": self.consensus.node_state == NodeState.LEADER,
            "leader_id": self.state.leader_id,
            "current_term": self.state.current_term,
            "model_loaded": self.model_loaded,
            "active_nodes": len(self.state.get_active_nodes()),
            "pending_tasks": len(self.state.pending_tasks),
            "active_tasks": self.active_tasks,
            "cluster_state": self.state.to_dict()
        }
    
    def stop(self):
        """停止节点"""
        print("\n[节点] 正在停止...")
        self.running = False
        self._unload_model()
        print("[节点] 已停止")


# ==================== 主函数 ====================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="分布式大模型推理 - 去中心化节点")
    
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", "-p", type=int, default=5000, help="监听端口")
    parser.add_argument("--name", "-n", default=None, help="节点名称")
    parser.add_argument("--model", "-m", default="Qwen/Qwen2.5-0.5B-Instruct", help="模型名称")
    parser.add_argument("--seeds", "-s", default="", help="种子节点列表，逗号分隔")
    parser.add_argument("--workers", "-w", type=int, default=2, help="并行工作线程")
    parser.add_argument("--min-nodes", type=int, default=1, help="最少节点数才开启推理")
    
    args = parser.parse_args()
    
    # 解析种子节点
    seed_nodes = []
    if args.seeds:
        seed_nodes = [s.strip() for s in args.seeds.split(",")]
    
    config = DecentralizedConfig(
        node_name=args.name,
        host=args.host,
        port=args.port,
        model_name=args.model,
        seed_nodes=seed_nodes,
        max_workers=args.workers,
        min_nodes_for_inference=args.min_nodes
    )
    
    node = DecentralizedNode(config)
    
    try:
        node.start()
    except KeyboardInterrupt:
        node.stop()


if __name__ == "__main__":
    main()
