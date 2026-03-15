#!/usr/bin/env python3
"""
分布式大模型推理系统 - 节点服务 (真实推理版本)
=====================================

这是真实可运行的版本，支持：
1. 真正加载模型（使用transformers）
2. 真正的推理执行
3. Pipeline并行（层间数据传递）
4. KV Cache管理

依赖安装:
    pip install socketio-client psutil torch transformers accelerate

使用方法:
    python node_service_real.py --server <服务器地址> --model <模型名称>
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
import pickle
import base64
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

try:
    import socketio
    import psutil
except ImportError:
    print("请安装依赖: pip install socketio-client psutil")
    sys.exit(1)

# 核心依赖检查
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("❌ 未安装torch/transformers，无法进行真实推理")
    print("请运行: pip install torch transformers accelerate")
    sys.exit(1)


@dataclass
class ModelShard:
    """模型分片 - 真实加载模型层"""
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
    config: Any = None
    layers: List[Any] = field(default_factory=list)
    load_time: Optional[float] = None
    device: str = "cpu"
    
    def load(self, base_model_name: str = None, model_path: str = None) -> bool:
        """真实加载模型分片"""
        start_time = time.time()
        
        try:
            print(f"📥 加载分片 {self.shard_id} (层 {self.layer_start}-{self.layer_end})...")
            
            # 检测设备
            if torch.cuda.is_available():
                self.device = "cuda"
                print(f"   使用GPU: {torch.cuda.get_device_name(0)}")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
                print("   使用Apple Silicon GPU")
            else:
                self.device = "cpu"
                print("   使用CPU")
            
            # 加载模型配置
            model_name = base_model_name or "Qwen/Qwen2.5-0.5B-Instruct"  # 使用小模型作为默认
            
            print(f"   加载模型配置: {model_name}")
            self.config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            
            # 加载tokenizer
            print(f"   加载Tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            
            # 加载完整模型（对于小模型）
            # 注意：真正的Pipeline并行需要更复杂的层切片
            print(f"   加载模型权重...")
            
            # 根据内存选择加载方式
            if self.size > 5:  # 大于5GB，使用量化
                print(f"   使用8-bit量化加载...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    load_in_8bit=True,
                    device_map="auto",
                    trust_remote_code=True
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                    device_map="auto" if self.device != "cpu" else None,
                    trust_remote_code=True
                )
                if self.device == "cpu":
                    self.model = self.model.to(self.device)
            
            self.model.eval()
            
            # 提取特定层（如果需要）
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
                total_layers = len(self.model.model.layers)
                print(f"   模型总层数: {total_layers}")
                
                # 对于Pipeline并行，我们标记这个分片负责的层
                # 实际推理时需要传递隐藏状态
                self.layers = list(self.model.model.layers)[self.layer_start:self.layer_end]
                print(f"   分片负责层: {self.layer_start} - {min(self.layer_end, total_layers)}")
            
            self.status = "ready"
            self.load_time = time.time() - start_time
            
            # 计算实际内存使用
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                print(f"   GPU内存使用: {allocated:.2f} GB")
            
            print(f"✅ 分片 {self.shard_id} 加载完成 ({self.load_time:.1f}s)")
            return True
            
        except Exception as e:
            print(f"❌ 加载分片失败: {e}")
            import traceback
            traceback.print_exc()
            self.status = "error"
            return False
    
    def unload(self):
        """卸载分片释放内存"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self.layers = []
        
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.status = "unloaded"
        print(f"📤 分片 {self.shard_id} 已卸载")
    
    def encode(self, text: str) -> torch.Tensor:
        """编码文本为token IDs"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer未加载")
        
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
        return inputs.input_ids.to(self.device)
    
    def decode(self, token_ids: torch.Tensor) -> str:
        """解码token IDs为文本"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer未加载")
        
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)


class PipelineExecutor:
    """Pipeline并行执行器 - 处理层间数据传递"""
    
    def __init__(self, shard: ModelShard):
        self.shard = shard
        self.kv_cache = None
        self.hidden_states_buffer = None
    
    def forward_layer_range(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple] = None,
        use_cache: bool = True
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        """
        执行特定层的forward pass
        
        对于Pipeline并行：
        - 接收上一节点的隐藏状态
        - 执行本节点负责的层
        - 输出隐藏状态给下一节点
        """
        if self.shard.model is None:
            raise ValueError("模型未加载")
        
        model = self.shard.model
        
        # 获取模型的基础模型（transformer部分）
        if hasattr(model, 'model'):
            base_model = model.model
        else:
            base_model = model
        
        # 获取嵌入层输出
        if hasattr(base_model, 'embed_tokens'):
            # 如果输入是token IDs，先进行嵌入
            if hidden_states.dtype == torch.long or hidden_states.dtype == torch.int:
                hidden_states = base_model.embed_tokens(hidden_states)
        
        # 执行指定范围的层
        layers = self.layers if hasattr(self, 'layers') and self.layers else []
        
        if not layers and hasattr(base_model, 'layers'):
            # 如果没有预提取层，从基础模型获取
            all_layers = list(base_model.layers)
            layers = all_layers[self.shard.layer_start:self.shard.layer_end]
        
        new_past_key_values = [] if use_cache and past_key_values else None
        
        for i, layer in enumerate(layers):
            layer_idx = self.shard.layer_start + i
            
            # 获取过去KV缓存（如果有）
            past_kv = past_key_values[i] if past_key_values and i < len(past_key_values) else None
            
            # 执行层forward
            layer_output = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_kv,
                use_cache=use_cache
            )
            
            hidden_states = layer_output[0]
            
            if use_cache and layer_output[1] is not None:
                new_past_key_values.append(layer_output[1])
        
        return hidden_states, tuple(new_past_key_values) if new_past_key_values else None
    
    def serialize_hidden_states(self, hidden_states: torch.Tensor) -> str:
        """序列化隐藏状态用于网络传输"""
        # 转换为numpy然后序列化
        numpy_array = hidden_states.detach().cpu().numpy()
        serialized = pickle.dumps(numpy_array)
        return base64.b64encode(serialized).decode('utf-8')
    
    def deserialize_hidden_states(self, serialized: str) -> torch.Tensor:
        """反序列化隐藏状态"""
        decoded = base64.b64decode(serialized.encode('utf-8'))
        numpy_array = pickle.loads(decoded)
        return torch.from_numpy(numpy_array).to(self.shard.device)


class RealInferenceEngine:
    """真实推理引擎"""
    
    def __init__(self, shard: ModelShard):
        self.shard = shard
        self.pipeline_executor = PipelineExecutor(shard)
        self.generation_config = {
            "max_new_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
            "pad_token_id": None,
            "eos_token_id": None
        }
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> Tuple[str, int, float]:
        """
        执行真实推理生成
        
        返回: (生成的文本, token数量, 耗时)
        """
        start_time = time.time()
        
        if self.shard.model is None:
            raise ValueError("模型未加载")
        
        # 编码输入
        inputs = self.shard.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        )
        
        input_ids = inputs.input_ids.to(self.shard.device)
        attention_mask = inputs.attention_mask.to(self.shard.device) if 'attention_mask' in inputs else None
        
        input_token_count = input_ids.shape[1]
        
        # 设置生成参数
        generation_config = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": temperature > 0,
            "pad_token_id": self.shard.tokenizer.pad_token_id or self.shard.tokenizer.eos_token_id,
            "eos_token_id": self.shard.tokenizer.eos_token_id,
            **kwargs
        }
        
        # 执行生成
        with torch.no_grad():
            # 使用模型的generate方法
            output_ids = self.shard.model.generate(
                input_ids,
                attention_mask=attention_mask,
                **generation_config
            )
        
        # 解码输出
        generated_text = self.shard.tokenizer.decode(
            output_ids[0][input_token_count:],
            skip_special_tokens=True
        )
        
        # 计算token数量
        output_token_count = output_ids.shape[1] - input_token_count
        
        latency = time.time() - start_time
        
        return generated_text, output_token_count, latency
    
    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs
    ):
        """
        流式生成（用于实时响应）
        """
        if self.shard.model is None:
            raise ValueError("模型未加载")
        
        # 编码输入
        inputs = self.shard.tokenizer(prompt, return_tensors="pt", truncation=True)
        input_ids = inputs.input_ids.to(self.shard.device)
        
        # 使用TextIteratorStreamer进行流式生成
        try:
            from transformers import TextIteratorStreamer
            streamer = TextIteratorStreamer(
                self.shard.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True
            )
            
            generation_kwargs = {
                "input_ids": input_ids,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "streamer": streamer,
                **kwargs
            }
            
            # 在单独的线程中运行生成
            thread = threading.Thread(target=self.shard.model.generate, kwargs=generation_kwargs)
            thread.start()
            
            for text in streamer:
                yield text
            
            thread.join()
            
        except ImportError:
            # 如果没有TextIteratorStreamer，回退到普通生成
            result, _, _ = self.generate(prompt, max_new_tokens, temperature, **kwargs)
            yield result


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
        """计算负载分数 (0-100)"""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # 如果使用GPU，也考虑GPU负载
        gpu_percent = 0
        if torch.cuda.is_available():
            try:
                gpu_percent = torch.cuda.utilization()
            except:
                pass
        
        load_score = (cpu_percent * 0.4 + memory_percent * 0.3 + gpu_percent * 0.3)
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


class RealNodeService:
    """真实节点服务 - 支持实际模型推理"""
    
    def __init__(self, server_url: str, name: Optional[str] = None, model_name: Optional[str] = None):
        self.server_url = server_url
        self.node_id = self._load_or_create_node_id()
        self.node_name = name or f"Node-{self.node_id[:8]}"
        self.model_name = model_name or "Qwen/Qwen2.5-0.5B-Instruct"  # 默认使用小模型
        self.shards: Dict[str, ModelShard] = {}
        self.inference_engines: Dict[str, RealInferenceEngine] = {}
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
        
        # 检测GPU
        gpu_info = ""
        if torch.cuda.is_available():
            gpu_info = f", GPU: {torch.cuda.get_device_name(0)}"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            gpu_info = ", GPU: Apple Silicon"
        
        return {
            "nodeId": self.node_id,
            "name": self.node_name,
            "os": f"{platform.system()} {platform.release()}{gpu_info}",
            "cpuCores": psutil.cpu_count(logical=True),
            "totalMemory": int(memory.total / (1024 * 1024)),  # MB
            "availableMemory": int(memory.available / (1024 * 1024)),  # MB
            "shards": [],
            "loadScore": 0,
            "parallelSlots": self.parallel_slots if hasattr(self, 'parallel_slots') else 2,
            "modelName": self.model_name
        }
    
    def _setup_events(self):
        """设置Socket.IO事件处理"""
        
        @self.sio.event
        def connect():
            print(f"✅ 已连接到服务器: {self.server_url}")
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
            print(f"📦 默认模型: {self.model_name}")
        
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
            threading.Thread(
                target=self._load_shard_real,
                args=(shard,),
                daemon=True
            ).start()
        
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
            threading.Thread(
                target=self._migrate_shard_real,
                args=(shard,),
                daemon=True
            ).start()
        
        @self.sio.event
        def shard_unload(data):
            """卸载分片"""
            shard_id = data.get('shardId')
            if shard_id in self.shards:
                print(f"📤 卸载分片: {shard_id}")
                self.shards[shard_id].unload()
                if shard_id in self.inference_engines:
                    del self.inference_engines[shard_id]
                del self.shards[shard_id]
        
        @self.sio.event
        def task_inference(data):
            """接收推理任务"""
            task_id = data.get('taskId')
            prompt = data.get('prompt')
            parallel_index = data.get('parallelIndex', 0)
            
            print(f"🎯 收到推理任务: {task_id[:8]}... (并行 {parallel_index+1})")
            
            # 提交到任务处理器
            self.processor.submit_task(
                self._execute_real_inference,
                data
            )
        
        @self.sio.event
        def pipeline_forward(data):
            """Pipeline并行 - 接收中间隐藏状态"""
            task_id = data.get('taskId')
            shard_id = data.get('shardId')
            hidden_states_serialized = data.get('hiddenStates')
            
            print(f"🔄 Pipeline转发: {task_id[:8]}... -> {shard_id}")
            
            # 执行层forward并传递给下一节点
            threading.Thread(
                target=self._execute_pipeline_forward,
                args=(task_id, shard_id, hidden_states_serialized),
                daemon=True
            ).start()
    
    def _load_shard_real(self, shard: ModelShard):
        """真实加载模型分片"""
        success = shard.load(base_model_name=self.model_name)
        
        if success:
            # 创建推理引擎
            self.inference_engines[shard.shard_id] = RealInferenceEngine(shard)
            
            self.sio.emit("shard:loaded", {
                "shardId": shard.shard_id,
                "modelName": self.model_name,
                "device": shard.device,
                "loadTime": shard.load_time
            })
            self.system_info["shards"] = list(self.shards.keys())
        else:
            self.sio.emit("shard:error", {
                "shardId": shard.shard_id,
                "error": "Failed to load shard"
            })
    
    def _migrate_shard_real(self, shard: ModelShard):
        """真实迁移分片"""
        print(f"🔄 迁移分片 {shard.shard_id}...")
        
        # 真实加载
        success = shard.load(base_model_name=self.model_name)
        
        if success:
            self.inference_engines[shard.shard_id] = RealInferenceEngine(shard)
            self.sio.emit("shard:migrated", {
                "shardId": shard.shard_id,
                "modelName": self.model_name
            })
            print(f"✅ 分片 {shard.shard_id} 迁移完成")
        else:
            self.sio.emit("shard:error", {
                "shardId": shard.shard_id,
                "error": "Migration failed"
            })
    
    def _execute_real_inference(self, task_data: Dict[str, Any]):
        """执行真实推理"""
        task_id = task_data.get("taskId")
        prompt = task_data.get("prompt")
        max_tokens = task_data.get("maxTokens", 512)
        temperature = task_data.get("temperature", 0.7)
        
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
            
            # 找到可用的推理引擎
            engine = None
            for shard_id, eng in self.inference_engines.items():
                if eng.shard.status == "ready":
                    engine = eng
                    break
            
            if engine is None:
                raise ValueError("没有可用的推理引擎")
            
            # 执行真实推理
            result, tokens, latency = engine.generate(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature
            )
            
            self.monitor.record_task(latency, True, tokens)
            
            self.sio.emit("inference:result", {
                "taskId": task_id,
                "status": "completed",
                "result": result,
                "tokens": tokens,
                "latency": latency,
                "modelName": self.model_name
            })
            
            print(f"✅ 任务完成: {task_id[:8]}... (延迟: {latency:.2f}s, tokens: {tokens})")
            
        except Exception as e:
            print(f"❌ 推理失败: {e}")
            import traceback
            traceback.print_exc()
            self.monitor.record_task(0, False)
            
            self.sio.emit("inference:result", {
                "taskId": task_id,
                "status": "failed",
                "error": str(e)
            })
        
        finally:
            self.active_slots -= 1
            
            status = "busy" if self.active_slots > 0 else "online"
            self.sio.emit("node:status", {
                "status": status,
                "availableMemory": int(psutil.virtual_memory().available / (1024 * 1024)),
                "loadScore": self.monitor.get_load_score(),
                "activeSlots": self.active_slots,
                "throughput": self.monitor.get_throughput()
            })
    
    def _execute_pipeline_forward(self, task_id: str, shard_id: str, hidden_states_serialized: str):
        """执行Pipeline并行的层forward"""
        try:
            if shard_id not in self.shards:
                raise ValueError(f"分片 {shard_id} 不存在")
            
            shard = self.shards[shard_id]
            executor = PipelineExecutor(shard)
            
            # 反序列化隐藏状态
            hidden_states = executor.deserialize_hidden_states(hidden_states_serialized)
            
            # 执行层forward
            output_hidden_states, new_kv_cache = executor.forward_layer_range(hidden_states)
            
            # 序列化输出
            output_serialized = executor.serialize_hidden_states(output_hidden_states)
            
            # 发送给下一节点或返回结果
            self.sio.emit("pipeline:result", {
                "taskId": task_id,
                "shardId": shard_id,
                "hiddenStates": output_serialized,
                "hasMore": shard.layer_end < 64  # 假设总层数
            })
            
        except Exception as e:
            print(f"❌ Pipeline forward失败: {e}")
            self.sio.emit("pipeline:error", {
                "taskId": task_id,
                "shardId": shard_id,
                "error": str(e)
            })
    
    def _heartbeat_loop(self):
        """心跳循环"""
        while self.running:
            try:
                if self.sio.connected:
                    memory = psutil.virtual_memory()
                    stats = self.monitor.get_stats()
                    
                    self.sio.emit("node:heartbeat", {
                        "availableMemory": int(memory.available / (1024 * 1024)),
                        "status": "busy" if self.active_slots > 0 else "online",
                        "loadScore": stats['loadScore'],
                        "throughput": stats['throughput'],
                        "shardCount": len(self.shards),
                        "activeSlots": self.active_slots,
                        "tasksCompleted": stats['tasksCompleted'],
                        "gpuMemory": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                    })
            except Exception as e:
                print(f"[心跳错误] {e}")
            
            interval = self.config.get('heartbeatInterval', 30000) / 1000
            time.sleep(interval)
    
    def start(self):
        """启动节点服务"""
        print(f"\n{'='*60}")
        print(f"  分布式大模型推理系统 - 节点服务 (真实推理版)")
        print(f"{'='*60}")
        print(f"  节点ID: {self.node_id}")
        print(f"  节点名称: {self.node_name}")
        print(f"  操作系统: {self.system_info['os']}")
        print(f"  CPU核心: {self.system_info['cpuCores']}")
        print(f"  总内存: {self.system_info['totalMemory'] // 1024} GB")
        print(f"  并行槽位: {self.parallel_slots}")
        print(f"  默认模型: {self.model_name}")
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
        description="分布式大模型推理系统 - 节点服务 (真实推理版)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 使用默认小模型
    python node_service_real.py --server https://your-server.com
    
    # 指定模型
    python node_service_real.py --server https://your-server.com --model Qwen/Qwen2.5-1.5B-Instruct
    
    # 使用本地模型
    python node_service_real.py --server https://your-server.com --model /path/to/local/model

推荐模型:
    - Qwen/Qwen2.5-0.5B-Instruct (约1GB, 适合4GB内存)
    - Qwen/Qwen2.5-1.5B-Instruct (约3GB, 适合8GB内存)
    - Qwen/Qwen2.5-3B-Instruct (约6GB, 适合16GB内存)
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
        "--model", "-m",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="模型名称或路径 (默认: Qwen/Qwen2.5-0.5B-Instruct)"
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
        help="手动指定并行槽位数"
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
    service = RealNodeService(
        server_url=server_url,
        name=args.name,
        model_name=args.model
    )
    
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
