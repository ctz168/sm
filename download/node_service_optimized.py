#!/usr/bin/env python3
"""
分布式大模型推理系统 - 优化版节点服务
=====================================

支持:
1. 模型权重本地缓存
2. 智能分片加载
3. CPU多线程优化
4. 内存管理优化
5. 批处理推理
"""

import os
import sys
import time
import json
import uuid
import argparse
import platform
import threading
import signal
import gc
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from pathlib import Path

try:
    import socketio
    import psutil
except ImportError as e:
    print(f"❌ 缺少依赖: {e}")
    sys.exit(1)

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("❌ 未安装torch/transformers")
    sys.exit(1)


# ==================== 配置 ====================

@dataclass
class NodeConfig:
    """节点配置"""
    server_url: str
    model_name: str
    node_name: str
    max_workers: int = 2
    cache_dir: str = ""
    memory_limit_gb: float = 0.0
    use_quantization: bool = False
    quantization_bits: int = 8
    
    def __post_init__(self):
        if not self.cache_dir:
            self.cache_dir = os.environ.get(
                'HF_HOME',
                os.path.expanduser('~/.cache/huggingface')
            )
        if self.memory_limit_gb == 0:
            self.memory_limit_gb = psutil.virtual_memory().total / (1024**3)


# ==================== 内存管理 ====================

class MemoryManager:
    """内存管理器"""
    
    def __init__(self, limit_gb: float):
        self.limit_gb = limit_gb
        self.used_gb = 0.0
        self.lock = threading.Lock()
    
    def allocate(self, size_gb: float) -> bool:
        """分配内存"""
        with self.lock:
            if self.used_gb + size_gb > self.limit_gb * 0.8:  # 80%阈值
                return False
            self.used_gb += size_gb
            return True
    
    def release(self, size_gb: float):
        """释放内存"""
        with self.lock:
            self.used_gb = max(0, self.used_gb - size_gb)
    
    def get_available(self) -> float:
        """获取可用内存"""
        with self.lock:
            return max(0, self.limit_gb * 0.8 - self.used_gb)
    
    def get_usage_percent(self) -> float:
        """获取内存使用百分比"""
        with self.lock:
            return (self.used_gb / self.limit_gb) * 100


# ==================== 模型缓存管理 ====================

class ModelCacheManager:
    """模型缓存管理器"""
    
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.hub_cache = self.cache_dir / "hub"
    
    def get_model_path(self, model_name: str) -> Path:
        """获取模型缓存路径"""
        model_dir = "models--" + model_name.replace('/', '--')
        return self.hub_cache / model_dir
    
    def is_cached(self, model_name: str) -> bool:
        """检查模型是否已缓存"""
        model_path = self.get_model_path(model_name)
        return model_path.exists()
    
    def get_cache_size(self, model_name: str) -> float:
        """获取模型缓存大小(GB)"""
        model_path = self.get_model_path(model_name)
        if not model_path.exists():
            return 0.0
        
        total_size = 0
        for root, _, files in os.walk(model_path):
            for f in files:
                total_size += os.path.getsize(os.path.join(root, f))
        
        return total_size / (1024**3)
    
    def estimate_model_size(self, model_name: str) -> float:
        """估算模型大小"""
        # 常见模型大小映射
        size_map = {
            "Qwen2.5-0.5B": 1.0,
            "Qwen2.5-1.5B": 3.0,
            "Qwen2.5-3B": 6.0,
            "Qwen2.5-7B": 14.0,
            "Qwen2.5-14B": 28.0,
            "Qwen2.5-32B": 64.0,
            "gpt2": 0.5,
            "gpt2-medium": 1.5,
            "gpt2-large": 3.0,
        }
        
        for key, size in size_map.items():
            if key.lower() in model_name.lower():
                return size
        
        # 默认估算
        if "0.5b" in model_name.lower():
            return 1.0
        elif "1.5b" in model_name.lower():
            return 3.0
        elif "3b" in model_name.lower():
            return 6.0
        elif "7b" in model_name.lower():
            return 14.0
        
        return 2.0  # 默认2GB


# ==================== 优化推理引擎 ====================

class OptimizedInferenceEngine:
    """优化的推理引擎"""
    
    def __init__(self, config: NodeConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = self._detect_device()
        self.loaded = False
        self.load_time = 0.0
        self.model_size_gb = 0.0
        self.memory_manager = MemoryManager(config.memory_limit_gb)
        self.cache_manager = ModelCacheManager(config.cache_dir)
        
        # 性能统计
        self.stats = {
            "total_requests": 0,
            "total_tokens": 0,
            "total_latency": 0.0,
            "batch_requests": 0,
        }
    
    def _detect_device(self) -> str:
        """检测最佳设备"""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _get_torch_dtype(self):
        """获取最佳数据类型"""
        if self.device == "cuda":
            return torch.float16
        elif self.device == "mps":
            return torch.float16
        else:
            # CPU使用float32，但可以考虑量化
            if self.config.use_quantization:
                return torch.float32  # 量化后
            return torch.float32
    
    def load(self) -> bool:
        """加载模型"""
        start_time = time.time()
        
        try:
            # 检查缓存
            is_cached = self.cache_manager.is_cached(self.config.model_name)
            estimated_size = self.cache_manager.estimate_model_size(self.config.model_name)
            
            print(f"📥 加载模型: {self.config.model_name}")
            print(f"   设备: {self.device}")
            print(f"   已缓存: {'是' if is_cached else '否'}")
            print(f"   预估大小: {estimated_size:.1f}GB")
            
            # 检查内存
            if not self.memory_manager.allocate(estimated_size * 1.5):
                print(f"❌ 内存不足，需要 {estimated_size * 1.5:.1f}GB")
                return False
            
            # 加载tokenizer
            print("   [1/3] 加载Tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True,
                cache_dir=self.config.cache_dir
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 加载模型
            print("   [2/3] 加载模型权重...")
            
            torch_dtype = self._get_torch_dtype()
            
            # 根据设备和配置选择加载方式
            if self.config.use_quantization and self.device == "cuda":
                # GPU量化加载
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name,
                    load_in_8bit=True if self.config.quantization_bits == 8 else False,
                    load_in_4bit=True if self.config.quantization_bits == 4 else False,
                    device_map="auto",
                    trust_remote_code=True,
                    cache_dir=self.config.cache_dir
                )
            elif self.device == "cuda":
                # GPU正常加载
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name,
                    torch_dtype=torch_dtype,
                    device_map="auto",
                    trust_remote_code=True,
                    cache_dir=self.config.cache_dir
                )
            else:
                # CPU加载
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name,
                    torch_dtype=torch_dtype,
                    trust_remote_code=True,
                    cache_dir=self.config.cache_dir,
                    low_cpu_mem_usage=True
                )
            
            # 设置为评估模式
            print("   [3/3] 初始化推理引擎...")
            self.model.eval()
            
            # 计算实际模型大小
            param_count = sum(p.numel() for p in self.model.parameters())
            self.model_size_gb = param_count * 4 / (1024**3)  # 假设float32
            
            self.load_time = time.time() - start_time
            self.loaded = True
            
            print(f"✅ 模型加载完成!")
            print(f"   参数量: {param_count / 1e9:.2f}B")
            print(f"   实际大小: {self.model_size_gb:.2f}GB")
            print(f"   加载时间: {self.load_time:.1f}s")
            
            return True
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def unload(self):
        """卸载模型"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 强制垃圾回收
        gc.collect()
        
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
        """执行推理"""
        if not self.loaded:
            raise RuntimeError("模型未加载")
        
        start_time = time.time()
        
        # 编码
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
            padding=True
        )
        
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
                use_cache=True,  # 启用KV缓存
                **kwargs
            )
        
        # 解码
        output_token_count = outputs.shape[1]
        new_tokens = output_token_count - input_token_count
        
        response = self.tokenizer.decode(
            outputs[0][input_token_count:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        latency = time.time() - start_time
        throughput = new_tokens / latency if latency > 0 else 0
        
        # 更新统计
        self.stats["total_requests"] += 1
        self.stats["total_tokens"] += new_tokens
        self.stats["total_latency"] += latency
        
        details = {
            "input_tokens": input_token_count,
            "output_tokens": output_token_count,
            "new_tokens": new_tokens,
            "latency": latency,
            "throughput": throughput,
            "device": self.device,
            "model_size_gb": self.model_size_gb,
        }
        
        return response, new_tokens, latency, details
    
    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 256,
        **kwargs
    ) -> List[Tuple[str, int, float, Dict]]:
        """批量推理"""
        if not self.loaded:
            raise RuntimeError("模型未加载")
        
        results = []
        start_time = time.time()
        
        # 批量编码
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
            padding=True
        )
        
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        
        # 批量生成
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
                **kwargs
            )
        
        # 解码每个结果
        for i, output in enumerate(outputs):
            input_len = (attention_mask[i] == 1).sum().item()
            new_tokens = output.shape[0] - input_len
            
            response = self.tokenizer.decode(
                output[input_len:],
                skip_special_tokens=True
            )
            
            latency = time.time() - start_time
            results.append((response, new_tokens, latency, {}))
        
        self.stats["batch_requests"] += 1
        
        return results
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        avg_latency = (
            self.stats["total_latency"] / self.stats["total_requests"]
            if self.stats["total_requests"] > 0 else 0
        )
        avg_throughput = (
            self.stats["total_tokens"] / self.stats["total_latency"]
            if self.stats["total_latency"] > 0 else 0
        )
        
        return {
            **self.stats,
            "avg_latency": avg_latency,
            "avg_throughput": avg_throughput,
            "memory_usage_percent": self.memory_manager.get_usage_percent(),
            "model_size_gb": self.model_size_gb,
        }


# ==================== 优化节点服务 ====================

class OptimizedNodeService:
    """优化的节点服务"""
    
    def __init__(self, config: NodeConfig):
        self.config = config
        self.node_id = self._load_or_create_node_id()
        
        # 推理引擎
        self.engine: Optional[OptimizedInferenceEngine] = None
        self.task_processor: Optional[ThreadPoolExecutor] = None
        
        # 状态
        self.running = False
        self.connected = False
        self.model_ready = False
        self.active_tasks = 0
        self.lock = threading.Lock()
        
        # Socket.IO
        self.sio = socketio.Client(
            reconnection=True,
            reconnection_attempts=10,
            reconnection_delay=1,
            reconnection_delay_max=60,
            logger=False,
            engineio_logger=False
        )
        
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
    
    def _get_system_info(self) -> Dict:
        memory = psutil.virtual_memory()
        cpu_count = os.cpu_count() or 4
        
        return {
            "nodeId": self.node_id,
            "name": self.config.node_name,
            "os": platform.system() + " " + platform.release(),
            "cpuCores": cpu_count,
            "totalMemory": int(memory.total / (1024**2)),
            "availableMemory": int(memory.available / (1024**2)),
            "modelName": self.config.model_name,
            "maxWorkers": self.config.max_workers,
            "device": self.engine.device if self.engine else "unknown",
            "useQuantization": self.config.use_quantization,
        }
    
    def _setup_events(self):
        @self.sio.event
        def connect():
            print(f"✅ 已连接到服务器")
            self.connected = True
            self._register()
        
        @self.sio.event
        def disconnect():
            print("❌ 与服务器断开连接")
            self.connected = False
        
        @self.sio.on('node:registered')
        def on_registered(data):
            print(f"✅ 节点已注册")
            threading.Thread(target=self._heartbeat_loop, daemon=True).start()
        
        @self.sio.on('task:inference')
        def on_task(data):
            self._handle_inference_task(data)
        
        @self.sio.on('task:batch')
        def on_batch(data):
            self._handle_batch_task(data)
    
    def _register(self):
        self.sio.emit('node:register', self._get_system_info())
    
    def _handle_inference_task(self, data: Dict):
        task_id = data.get('taskId')
        prompt = data.get('prompt', '')
        
        if not self.model_ready:
            self.sio.emit('inference:result', {
                'taskId': task_id,
                'status': 'failed',
                'error': '模型未就绪'
            })
            return
        
        # 输入验证
        if not prompt or len(prompt) > 10000:
            self.sio.emit('inference:result', {
                'taskId': task_id,
                'status': 'failed',
                'error': '无效输入'
            })
            return
        
        with self.lock:
            self.active_tasks += 1
        
        try:
            response, tokens, latency, details = self.engine.generate(
                prompt,
                max_new_tokens=data.get('maxTokens', 256),
                temperature=data.get('temperature', 0.7)
            )
            
            self.sio.emit('inference:result', {
                'taskId': task_id,
                'status': 'completed',
                'result': response,
                'tokens': tokens,
                'latency': latency,
                'throughput': details.get('throughput', 0),
            })
            
        except Exception as e:
            self.sio.emit('inference:result', {
                'taskId': task_id,
                'status': 'failed',
                'error': str(e)
            })
        
        finally:
            with self.lock:
                self.active_tasks -= 1
    
    def _handle_batch_task(self, data: Dict):
        batch_id = data.get('batchId')
        prompts = data.get('prompts', [])
        
        if not self.model_ready or not prompts:
            return
        
        try:
            results = self.engine.generate_batch(prompts)
            
            self.sio.emit('batch:result', {
                'batchId': batch_id,
                'status': 'completed',
                'results': [
                    {'response': r, 'tokens': t, 'latency': l}
                    for r, t, l, _ in results
                ]
            })
        except Exception as e:
            self.sio.emit('batch:result', {
                'batchId': batch_id,
                'status': 'failed',
                'error': str(e)
            })
    
    def _heartbeat_loop(self):
        while self.running and self.connected:
            try:
                memory = psutil.virtual_memory()
                stats = self.engine.get_stats() if self.engine else {}
                
                self.sio.emit('node:heartbeat', {
                    'availableMemory': int(memory.available / (1024**2)),
                    'status': 'busy' if self.active_tasks > 0 else 'online',
                    'loadScore': psutil.cpu_percent(),
                    'activeSlots': self.active_tasks,
                    'modelReady': self.model_ready,
                    'stats': stats,
                })
            except Exception as e:
                print(f"[心跳错误] {e}")
            
            time.sleep(10)
    
    def start(self):
        print(f"\n{'='*60}")
        print(f"  分布式大模型推理系统 - 优化节点")
        print(f"{'='*60}")
        print(f"  节点ID: {self.node_id}")
        print(f"  节点名称: {self.config.node_name}")
        print(f"  模型: {self.config.model_name}")
        print(f"  并行槽位: {self.config.max_workers}")
        print(f"  内存限制: {self.config.memory_limit_gb:.1f}GB")
        print(f"  量化: {'启用' if self.config.use_quantization else '禁用'}")
        print(f"{'='*60}\n")
        
        self.running = True
        
        # 创建推理引擎
        self.engine = OptimizedInferenceEngine(self.config)
        
        # 加载模型
        if not self.engine.load():
            print("❌ 模型加载失败")
            return
        
        self.model_ready = True
        
        # 创建任务处理器
        self.task_processor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        
        print("\n🚀 节点就绪\n")
        
        # 连接服务器
        while self.running:
            try:
                print(f"🔗 连接到 {self.config.server_url}...")
                self.sio.connect(
                    self.config.server_url,
                    transports=['polling', 'websocket'],
                    wait_timeout=10
                )
                self.sio.wait()
            except Exception as e:
                print(f"❌ 连接错误: {e}")
                if self.running:
                    print("⏳ 5秒后重试...")
                    time.sleep(5)
    
    def stop(self):
        print("\n🛑 正在停止...")
        self.running = False
        self.model_ready = False
        
        if self.engine:
            self.engine.unload()
        
        if self.task_processor:
            self.task_processor.shutdown(wait=True)
        
        if self.sio.connected:
            self.sio.disconnect()
        
        print("✅ 服务已停止")


def main():
    parser = argparse.ArgumentParser(description="分布式大模型推理 - 优化节点服务")
    
    parser.add_argument("--server", "-s", required=True, help="服务器地址")
    parser.add_argument("--model", "-m", default="Qwen/Qwen2.5-0.5B-Instruct", help="模型名称")
    parser.add_argument("--name", "-n", default=None, help="节点名称")
    parser.add_argument("--workers", "-w", type=int, default=2, help="并行工作线程")
    parser.add_argument("--cache-dir", default="", help="模型缓存目录")
    parser.add_argument("--memory-limit", type=float, default=0, help="内存限制(GB)")
    parser.add_argument("--quantize", "-q", action="store_true", help="启用量化")
    parser.add_argument("--quantize-bits", type=int, default=8, choices=[4, 8], help="量化位数")
    
    args = parser.parse_args()
    
    config = NodeConfig(
        server_url=args.server,
        model_name=args.model,
        node_name=args.name or f"Node-{uuid.uuid4().hex[:8]}",
        max_workers=args.workers,
        cache_dir=args.cache_dir,
        memory_limit_gb=args.memory_limit,
        use_quantization=args.quantize,
        quantization_bits=args.quantize_bits,
    )
    
    service = OptimizedNodeService(config)
    
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
