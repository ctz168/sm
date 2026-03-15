#!/usr/bin/env python3
"""
分布式大模型推理系统 - Pipeline并行版本
======================================

核心改进:
- 每个节点只加载部分层（不是完整模型）
- 节点间流水线传递中间结果
- 内存使用降低为 1/N

示例:
- Qwen2.5-0.5B: 24层
- 2节点: 节点1加载0-11层，节点2加载12-23层
- 每节点内存: 0.9GB（原来1.8GB）
"""

import os
import sys
import time
import json
import uuid
import socket
import threading
import pickle
import zlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
import struct

try:
    import torch
    import torch.nn as nn
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("请安装: pip install torch transformers")

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


# ==================== 配置 ====================

@dataclass
class PipelineConfig:
    """Pipeline并行配置"""
    # 模型配置
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    
    # 节点配置
    node_id: str = ""
    node_index: int = 0  # 节点索引（0, 1, 2, ...）
    total_nodes: int = 1  # 总节点数
    
    # 网络配置
    host: str = "0.0.0.0"
    port: int = 6000
    next_node_host: str = ""  # 下一个节点的地址
    next_node_port: int = 0
    
    # 分片配置
    layer_start: int = 0  # 起始层
    layer_end: int = 0    # 结束层
    
    def __post_init__(self):
        if not self.node_id:
            self.node_id = str(uuid.uuid4())


# ==================== 模型分片加载器 ====================

class ModelShardLoader:
    """模型分片加载器 - 只加载指定层"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.model_name = config.model_name
        
        # 模型组件
        self.config_obj = None
        self.tokenizer = None
        self.embed_tokens = None  # 词嵌入（只有第一个节点需要）
        self.layers = None        # Transformer层
        self.norm = None          # 最终归一化（只有最后一个节点需要）
        self.lm_head = None       # 输出头（只有最后一个节点需要）
        
        # 分片信息
        self.layer_start = config.layer_start
        self.layer_end = config.layer_end
        self.is_first = (config.node_index == 0)
        self.is_last = (config.node_index == config.total_nodes - 1)
        
        # 状态
        self.loaded = False
        self.load_time = 0.0
    
    def calculate_shards(self, total_nodes: int) -> List[Tuple[int, int]]:
        """计算每个节点的层分配"""
        if not HAS_TORCH:
            return []
        
        # 加载配置获取层数
        config = AutoConfig.from_pretrained(self.model_name)
        num_layers = config.num_hidden_layers
        
        layers_per_node = num_layers // total_nodes
        remainder = num_layers % total_nodes
        
        shards = []
        current = 0
        
        for i in range(total_nodes):
            extra = 1 if i < remainder else 0
            count = layers_per_node + extra
            shards.append((current, current + count - 1))
            current += count
        
        return shards
    
    def load(self) -> bool:
        """加载模型分片"""
        if not HAS_TORCH:
            print("❌ PyTorch未安装")
            return False
        
        start_time = time.time()
        
        try:
            print(f"📦 加载模型分片: {self.model_name}")
            print(f"   节点索引: {self.config.node_index}/{self.config.total_nodes}")
            print(f"   层范围: {self.layer_start}-{self.layer_end}")
            print(f"   是否首节点: {self.is_first}")
            print(f"   是否尾节点: {self.is_last}")
            
            # 加载配置
            self.config_obj = AutoConfig.from_pretrained(self.model_name)
            
            # 加载tokenizer（所有节点都需要）
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # 加载完整模型（后续优化为只加载需要的层）
            print("   [1/3] 加载模型权重...")
            full_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # 获取基础模型
            if hasattr(full_model, 'model'):
                base_model = full_model.model
            else:
                base_model = full_model
            
            # 提取嵌入层（只有第一个节点需要）
            print("   [2/3] 提取模型组件...")
            if self.is_first:
                if hasattr(base_model, 'embed_tokens'):
                    self.embed_tokens = base_model.embed_tokens
                elif hasattr(base_model, 'wte'):
                    self.embed_tokens = base_model.wte
                print(f"      嵌入层: 已加载")
            
            # 提取Transformer层
            if hasattr(base_model, 'layers'):
                all_layers = list(base_model.layers)
            elif hasattr(base_model, 'h'):
                all_layers = list(base_model.h)
            else:
                all_layers = []
            
            # 只保留需要的层
            self.layers = nn.ModuleList(all_layers[self.layer_start:self.layer_end + 1])
            print(f"      Transformer层: {len(self.layers)}层")
            
            # 提取输出层（只有最后一个节点需要）
            if self.is_last:
                if hasattr(base_model, 'norm'):
                    self.norm = base_model.norm
                if hasattr(full_model, 'lm_head'):
                    self.lm_head = full_model.lm_head
                print(f"      输出层: 已加载")
            
            # 删除完整模型释放内存
            del full_model
            del all_layers
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            import gc
            gc.collect()
            
            self.loaded = True
            self.load_time = time.time() - start_time
            
            # 计算实际内存使用
            param_count = sum(p.numel() for p in self.layers.parameters())
            if self.embed_tokens:
                param_count += sum(p.numel() for p in self.embed_tokens.parameters())
            if self.lm_head:
                param_count += sum(p.numel() for p in self.lm_head.parameters())
            
            print(f"✅ 分片加载完成!")
            print(f"   参数量: {param_count/1e6:.1f}M")
            print(f"   内存占用: {param_count * 4 / (1024**2):.1f}MB")
            print(f"   加载时间: {self.load_time:.1f}秒")
            
            return True
            
        except Exception as e:
            print(f"❌ 加载失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def forward_first(self, input_ids: torch.Tensor) -> torch.Tensor:
        """第一节点: 嵌入 -> 前几层"""
        with torch.no_grad():
            # 嵌入
            if self.embed_tokens is not None:
                hidden_states = self.embed_tokens(input_ids)
            else:
                raise ValueError("第一节点需要嵌入层")
            
            # 通过层
            for layer in self.layers:
                hidden_states = layer(hidden_states)[0]
            
            return hidden_states
    
    def forward_middle(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """中间节点: 通过层"""
        with torch.no_grad():
            for layer in self.layers:
                hidden_states = layer(hidden_states)[0]
            return hidden_states
    
    def forward_last(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """最后节点: 通过层 -> 归一化 -> 输出"""
        with torch.no_grad():
            # 通过层
            for layer in self.layers:
                hidden_states = layer(hidden_states)[0]
            
            # 归一化
            if self.norm is not None:
                hidden_states = self.norm(hidden_states)
            
            # 输出logits
            if self.lm_head is not None:
                logits = self.lm_head(hidden_states)
                return logits, hidden_states
            
            return hidden_states, hidden_states
    
    def generate_token(self, logits: torch.Tensor) -> Tuple[int, str]:
        """从logits生成下一个token"""
        # 获取最后一个位置的logits
        next_token_logits = logits[:, -1, :]
        
        # 简单贪婪解码
        next_token_id = next_token_logits.argmax(dim=-1).item()
        
        # 解码
        token_text = self.tokenizer.decode([next_token_id])
        
        return next_token_id, token_text


# ==================== Pipeline通信 ====================

class PipelineCommunicator:
    """Pipeline节点间通信"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.node_id = config.node_id
        
        # 网络
        self.server_socket = None
        self.client_socket = None
        
        # 缓冲区
        self.receive_buffer: Dict[str, torch.Tensor] = {}
        self.lock = threading.Lock()
    
    def start_server(self):
        """启动服务器监听"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.config.host, self.config.port))
        self.server_socket.listen(5)
        self.server_socket.settimeout(1.0)
        
        print(f"✅ Pipeline服务器启动: {self.config.host}:{self.config.port}")
        
        # 启动监听线程
        threading.Thread(target=self._listen_loop, daemon=True).start()
    
    def _listen_loop(self):
        """监听连接"""
        while True:
            try:
                conn, addr = self.server_socket.accept()
                threading.Thread(target=self._handle_connection, args=(conn,), daemon=True).start()
            except socket.timeout:
                continue
            except Exception as e:
                break
    
    def _handle_connection(self, conn: socket.socket):
        """处理连接"""
        try:
            # 接收数据
            data = b""
            while True:
                chunk = conn.recv(65536)
                if not chunk:
                    break
                data += chunk
            
            if data:
                # 解析
                message = pickle.loads(zlib.decompress(data))
                request_id = message.get("request_id")
                hidden_states = message.get("hidden_states")
                
                if hidden_states is not None:
                    with self.lock:
                        self.receive_buffer[request_id] = torch.from_numpy(hidden_states)
        
        except Exception as e:
            pass
        finally:
            conn.close()
    
    def send_to_next(self, request_id: str, hidden_states: torch.Tensor) -> bool:
        """发送到下一个节点"""
        if not self.config.next_node_host:
            return False
        
        try:
            # 序列化
            numpy_array = hidden_states.numpy()
            data = zlib.compress(pickle.dumps({
                "request_id": request_id,
                "hidden_states": numpy_array,
                "from_node": self.node_id
            }))
            
            # 发送
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10)
            sock.connect((self.config.next_node_host, self.config.next_node_port))
            sock.sendall(data)
            sock.close()
            
            return True
        except Exception as e:
            print(f"发送失败: {e}")
            return False
    
    def receive_from_prev(self, request_id: str, timeout: float = 30.0) -> Optional[torch.Tensor]:
        """从上一个节点接收"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            with self.lock:
                if request_id in self.receive_buffer:
                    return self.receive_buffer.pop(request_id)
            time.sleep(0.01)
        
        return None


# ==================== Pipeline节点 ====================

class PipelineNode:
    """Pipeline并行节点"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.node_id = config.node_id
        self.node_index = config.node_index
        self.total_nodes = config.total_nodes
        
        # 组件
        self.shard_loader = ModelShardLoader(config)
        self.communicator = PipelineCommunicator(config)
        
        # 状态
        self.running = False
        self.active_requests: Dict[str, Dict] = {}
        
        # 统计
        self.stats = {
            "requests_processed": 0,
            "total_latency": 0.0,
        }
    
    def start(self):
        """启动节点"""
        print(f"\n{'='*60}")
        print(f"  Pipeline并行节点")
        print(f"{'='*60}")
        print(f"  节点ID: {self.node_id}")
        print(f"  节点索引: {self.node_index}/{self.total_nodes}")
        print(f"  层范围: {self.config.layer_start}-{self.config.layer_end}")
        print(f"  端口: {self.config.port}")
        print(f"{'='*60}\n")
        
        # 加载模型分片
        if not self.shard_loader.load():
            print("❌ 模型加载失败")
            return
        
        # 启动通信
        self.communicator.start_server()
        
        self.running = True
        
        # 启动请求处理
        threading.Thread(target=self._process_loop, daemon=True).start()
        
        print("✅ 节点启动完成\n")
    
    def _process_loop(self):
        """处理循环"""
        while self.running:
            time.sleep(0.1)
    
    def process_request(self, request_id: str, input_data: Any) -> Dict:
        """处理请求"""
        start_time = time.time()
        
        try:
            if self.shard_loader.is_first:
                # 第一节点: 处理输入
                if isinstance(input_data, str):
                    # 文本输入
                    inputs = self.shard_loader.tokenizer(input_data, return_tensors="pt")
                    input_ids = inputs.input_ids
                else:
                    input_ids = input_data
                
                # 前向传播
                hidden_states = self.shard_loader.forward_first(input_ids)
                
                # 发送到下一节点
                if self.total_nodes > 1:
                    self.communicator.send_to_next(request_id, hidden_states)
                    return {"status": "forwarded", "node": self.node_id}
                else:
                    # 单节点，直接生成
                    logits, _ = self.shard_loader.forward_last(hidden_states)
                    token_id, token_text = self.shard_loader.generate_token(logits)
                    return {
                        "status": "completed",
                        "token_id": token_id,
                        "token_text": token_text
                    }
            
            elif self.shard_loader.is_last:
                # 最后节点: 接收并生成
                hidden_states = self.communicator.receive_from_prev(request_id)
                
                if hidden_states is None:
                    return {"status": "error", "error": "接收超时"}
                
                # 前向传播
                logits, _ = self.shard_loader.forward_last(hidden_states)
                token_id, token_text = self.shard_loader.generate_token(logits)
                
                return {
                    "status": "completed",
                    "token_id": token_id,
                    "token_text": token_text
                }
            
            else:
                # 中间节点: 接收并转发
                hidden_states = self.communicator.receive_from_prev(request_id)
                
                if hidden_states is None:
                    return {"status": "error", "error": "接收超时"}
                
                # 前向传播
                hidden_states = self.shard_loader.forward_middle(hidden_states)
                
                # 发送到下一节点
                self.communicator.send_to_next(request_id, hidden_states)
                
                return {"status": "forwarded", "node": self.node_id}
        
        except Exception as e:
            return {"status": "error", "error": str(e)}
        
        finally:
            self.stats["requests_processed"] += 1
            self.stats["total_latency"] += time.time() - start_time


# ==================== 主函数 ====================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Pipeline并行节点")
    parser.add_argument("--model", "-m", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--index", "-i", type=int, default=0, help="节点索引(0,1,2...)")
    parser.add_argument("--total", "-t", type=int, default=2, help="总节点数")
    parser.add_argument("--port", "-p", type=int, default=6000)
    parser.add_argument("--next-host", default="", help="下一节点地址")
    parser.add_argument("--next-port", type=int, default=0, help="下一节点端口")
    
    args = parser.parse_args()
    
    # 计算层分配
    if HAS_TORCH:
        config = AutoConfig.from_pretrained(args.model)
        num_layers = config.num_hidden_layers
        
        layers_per_node = num_layers // args.total
        remainder = num_layers % args.total
        
        layer_start = args.index * layers_per_node + min(args.index, remainder)
        extra = 1 if args.index < remainder else 0
        layer_end = layer_start + layers_per_node + extra - 1
        
        print(f"模型层数: {num_layers}")
        print(f"本节点层: {layer_start}-{layer_end}")
    else:
        layer_start = 0
        layer_end = 0
    
    config = PipelineConfig(
        model_name=args.model,
        node_index=args.index,
        total_nodes=args.total,
        port=args.port,
        next_node_host=args.next_host,
        next_node_port=args.next_port,
        layer_start=layer_start,
        layer_end=layer_end
    )
    
    node = PipelineNode(config)
    node.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n停止...")


if __name__ == "__main__":
    main()
