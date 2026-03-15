#!/usr/bin/env python3
"""
分布式大模型推理系统 - 模型分片与分布式计算优化
================================================

解决问题:
1. 模型权重存储位置
2. 网络带宽优化
3. 模型分片方案
4. CPU分布式计算优化

方案概述:
- 数据并行: 每个节点加载完整模型，处理不同请求
- 模型并行: 将模型按层分割到不同节点
- 流水线并行: 数据流水线处理
"""

import os
import sys
import time
import json
import hashlib
import threading
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import struct

try:
    import torch
    import numpy as np
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("警告: PyTorch未安装，部分功能不可用")


# ==================== 模型权重存储方案 ====================

@dataclass
class ModelWeightInfo:
    """模型权重信息"""
    name: str
    shape: Tuple[int, ...]
    dtype: str
    size_bytes: int
    layer_index: Optional[int] = None
    weight_type: str = "unknown"  # "embedding", "attention", "mlp", "output"


class ModelWeightAnalyzer:
    """模型权重分析器"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.weights: Dict[str, ModelWeightInfo] = {}
        self.total_size = 0
        self.layer_weights: Dict[int, List[str]] = {}
    
    def analyze_huggingface_model(self, model_name: str) -> Dict:
        """分析HuggingFace模型权重"""
        if not HAS_TORCH:
            return {"error": "PyTorch未安装"}
        
        try:
            from transformers import AutoModelForCausalLM, AutoConfig
            
            print(f"分析模型: {model_name}")
            
            # 加载配置
            config = AutoConfig.from_pretrained(model_name)
            
            # 获取模型信息
            info = {
                "model_name": model_name,
                "hidden_size": getattr(config, 'hidden_size', 0),
                "num_layers": getattr(config, 'num_hidden_layers', 0),
                "num_attention_heads": getattr(config, 'num_attention_heads', 0),
                "vocab_size": getattr(config, 'vocab_size', 0),
                "intermediate_size": getattr(config, 'intermediate_size', 0),
            }
            
            # 估算参数量
            if info['hidden_size'] > 0 and info['num_layers'] > 0:
                # Transformer参数估算
                embed_params = info['vocab_size'] * info['hidden_size']
                attention_params = 4 * info['hidden_size'] * info['hidden_size'] * info['num_layers']
                mlp_params = 2 * info['hidden_size'] * info['intermediate_size'] * info['num_layers']
                output_params = info['vocab_size'] * info['hidden_size']
                
                total_params = embed_params + attention_params + mlp_params + output_params
                info['estimated_params'] = total_params
                info['estimated_size_fp32'] = total_params * 4 / (1024**3)  # GB
                info['estimated_size_fp16'] = total_params * 2 / (1024**3)  # GB
            
            return info
            
        except Exception as e:
            return {"error": str(e)}
    
    def get_model_cache_path(self, model_name: str) -> str:
        """获取模型缓存路径"""
        hf_cache = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
        hub_cache = os.environ.get('HUGGINGFACE_HUB_CACHE', os.path.join(hf_cache, 'hub'))
        
        # 转换模型名称为目录名格式
        model_dir = "models--" + model_name.replace('/', '--')
        return os.path.join(hub_cache, model_dir)
    
    def calculate_shard_sizes(self, total_size_gb: float, num_shards: int) -> List[float]:
        """计算每个分片的大小"""
        shard_size = total_size_gb / num_shards
        return [shard_size] * num_shards


# ==================== 模型分片方案 ====================

@dataclass
class ModelShardConfig:
    """模型分片配置"""
    shard_id: int
    layer_start: int
    layer_end: int
    size_gb: float
    weights: List[str] = field(default_factory=list)
    node_id: Optional[str] = None


class ModelShardingStrategy:
    """模型分片策略"""
    
    def __init__(self, num_layers: int, hidden_size: int, vocab_size: int):
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
    
    def calculate_layer_memory(self) -> Dict[str, float]:
        """计算每层的内存占用"""
        # Embedding层
        embed_size = self.vocab_size * self.hidden_size * 2 / (1024**3)  # FP16, GB
        
        # 每个Transformer层
        # Self-attention: Q, K, V, O (4个矩阵)
        attention_size = 4 * self.hidden_size * self.hidden_size * 2 / (1024**3)
        
        # MLP: 通常有扩展
        mlp_size = 2 * self.hidden_size * (self.hidden_size * 4) * 2 / (1024**3)
        
        # LayerNorm等
        norm_size = 4 * self.hidden_size * 2 / (1024**3)
        
        layer_size = attention_size + mlp_size + norm_size
        
        # Output层 (通常与embedding共享权重)
        output_size = 0  # 共享权重
        
        return {
            "embedding": embed_size,
            "per_layer": layer_size,
            "output": output_size,
            "total": embed_size + layer_size * self.num_layers + output_size
        }
    
    def create_pipeline_shards(self, num_shards: int) -> List[ModelShardConfig]:
        """创建Pipeline并行分片"""
        memory_info = self.calculate_layer_memory()
        layer_size = memory_info["per_layer"]
        
        # 计算每个分片应该包含的层数
        layers_per_shard = self.num_layers // num_shards
        extra_layers = self.num_layers % num_shards
        
        shards = []
        current_layer = 0
        
        for i in range(num_shards):
            # 分配层数
            num_layers_this_shard = layers_per_shard
            if i < extra_layers:
                num_layers_this_shard += 1
            
            layer_start = current_layer
            layer_end = current_layer + num_layers_this_shard - 1
            
            # 计算大小
            size = num_layers_this_shard * layer_size
            if i == 0:
                size += memory_info["embedding"]  # 第一个分片包含embedding
            
            shards.append(ModelShardConfig(
                shard_id=i,
                layer_start=layer_start,
                layer_end=layer_end,
                size_gb=size
            ))
            
            current_layer = layer_end + 1
        
        return shards
    
    def create_tensor_parallel_shards(self, num_shards: int) -> List[ModelShardConfig]:
        """创建Tensor并行分片（将权重矩阵按列/行分割）"""
        # Tensor并行将注意力头和MLP分割到不同节点
        # 每个节点处理部分注意力头
        
        memory_info = self.calculate_layer_memory()
        
        shards = []
        for i in range(num_shards):
            shards.append(ModelShardConfig(
                shard_id=i,
                layer_start=0,
                layer_end=self.num_layers - 1,
                size_gb=memory_info["total"] / num_shards
            ))
        
        return shards


# ==================== CPU分布式计算优化 ====================

class CPUDistributedConfig:
    """CPU分布式计算配置"""
    
    def __init__(self):
        self.num_workers = os.cpu_count() or 4
        self.memory_limit_gb = self._get_system_memory_gb()
        self.inference_threads = max(1, self.num_workers // 2)
    
    def _get_system_memory_gb(self) -> float:
        """获取系统内存"""
        try:
            import psutil
            return psutil.virtual_memory().total / (1024**3)
        except:
            return 8.0  # 默认8GB
    
    def get_optimal_config(self, model_size_gb: float) -> Dict:
        """获取最优配置"""
        # 计算可以并行加载的模型数量
        models_per_node = int(self.memory_limit_gb / (model_size_gb * 1.5))  # 1.5x安全系数
        models_per_node = max(1, models_per_node)
        
        # 计算推理线程数
        inference_threads = max(1, self.num_workers // models_per_node)
        
        return {
            "num_workers": self.num_workers,
            "memory_gb": self.memory_limit_gb,
            "models_per_node": models_per_node,
            "inference_threads": inference_threads,
            "batch_size": self._calculate_optimal_batch_size(model_size_gb),
        }
    
    def _calculate_optimal_batch_size(self, model_size_gb: float) -> int:
        """计算最优批处理大小"""
        # 根据可用内存计算
        available_memory = self.memory_limit_gb - model_size_gb * 1.5
        if available_memory < 1:
            return 1
        elif available_memory < 4:
            return 4
        elif available_memory < 8:
            return 8
        else:
            return 16


# ==================== 网络优化 ====================

class NetworkOptimizer:
    """网络优化器"""
    
    @staticmethod
    def estimate_transfer_time(size_gb: float, bandwidth_mbps: float) -> float:
        """估算传输时间（秒）"""
        size_mbits = size_gb * 1024 * 8  # GB -> Mbits
        return size_mbits / bandwidth_mbps
    
    @staticmethod
    def calculate_optimal_shard_size(
        total_size_gb: float,
        bandwidth_mbps: float,
        max_transfer_time_sec: float = 60
    ) -> int:
        """计算最优分片大小"""
        # 最大可接受传输时间内的最大分片大小
        max_size_mbits = bandwidth_mbps * max_transfer_time_sec
        max_size_gb = max_size_mbits / (1024 * 8)
        
        # 计算需要的分片数
        num_shards = int(total_size_gb / max_size_gb) + 1
        
        return num_shards
    
    @staticmethod
    def get_compression_ratio() -> Dict[str, float]:
        """获取不同压缩算法的压缩比"""
        return {
            "none": 1.0,
            "gzip": 0.3,      # 约70%压缩
            "lz4": 0.5,       # 约50%压缩，更快
            "zstd": 0.35,     # 约65%压缩，平衡
            "quantization_8bit": 0.25,  # 8位量化
            "quantization_4bit": 0.125, # 4位量化
        }


# ==================== 综合方案 ====================

class DistributedInferencePlanner:
    """分布式推理规划器"""
    
    def __init__(self, model_name: str, num_nodes: int, node_memory_gb: List[float]):
        self.model_name = model_name
        self.num_nodes = num_nodes
        self.node_memory_gb = node_memory_gb
        self.analyzer = ModelWeightAnalyzer("")
        self.cpu_config = CPUDistributedConfig()
        self.network_optimizer = NetworkOptimizer()
    
    def create_plan(self) -> Dict:
        """创建分布式推理计划"""
        # 分析模型
        model_info = self.analyzer.analyze_huggingface_model(self.model_name)
        
        if "error" in model_info:
            return {"error": model_info["error"]}
        
        model_size_gb = model_info.get('estimated_size_fp16', 1.0)
        num_layers = model_info.get('num_layers', 0)
        hidden_size = model_info.get('hidden_size', 0)
        vocab_size = model_info.get('vocab_size', 0)
        
        # 创建分片策略
        sharding = ModelShardingStrategy(num_layers, hidden_size, vocab_size)
        
        # 决定最佳并行策略
        plan = {
            "model_info": model_info,
            "strategy": self._determine_strategy(model_size_gb),
            "shards": [],
            "node_assignments": [],
            "network_requirements": {},
            "cpu_optimization": self.cpu_config.get_optimal_config(model_size_gb),
        }
        
        # 根据策略创建分片
        if plan["strategy"] == "data_parallel":
            # 数据并行：每个节点加载完整模型
            plan["shards"] = [{
                "shard_id": i,
                "type": "full_model",
                "size_gb": model_size_gb,
                "node_memory_required": model_size_gb * 1.5,
            } for i in range(self.num_nodes)]
        
        elif plan["strategy"] == "pipeline_parallel":
            # Pipeline并行：按层分割
            shards = sharding.create_pipeline_shards(self.num_nodes)
            plan["shards"] = [
                {
                    "shard_id": s.shard_id,
                    "type": "pipeline",
                    "layers": f"{s.layer_start}-{s.layer_end}",
                    "size_gb": s.size_gb,
                }
                for s in shards
            ]
        
        elif plan["strategy"] == "tensor_parallel":
            # Tensor并行：按注意力头分割
            shards = sharding.create_tensor_parallel_shards(self.num_nodes)
            plan["shards"] = [
                {
                    "shard_id": s.shard_id,
                    "type": "tensor",
                    "size_gb": s.size_gb,
                }
                for s in shards
            ]
        
        # 网络需求
        plan["network_requirements"] = {
            "model_transfer_time": {
                "100Mbps": self.network_optimizer.estimate_transfer_time(model_size_gb, 100),
                "1Gbps": self.network_optimizer.estimate_transfer_time(model_size_gb, 1000),
                "10Gbps": self.network_optimizer.estimate_transfer_time(model_size_gb, 10000),
            },
            "compression_options": self.network_optimizer.get_compression_ratio(),
        }
        
        return plan
    
    def _determine_strategy(self, model_size_gb: float) -> str:
        """决定最佳并行策略"""
        avg_node_memory = sum(self.node_memory_gb) / len(self.node_memory_gb)
        
        # 如果每个节点都能装下完整模型，使用数据并行
        if avg_node_memory >= model_size_gb * 1.5:
            return "data_parallel"
        
        # 如果需要多个节点才能装下，使用Pipeline并行
        total_memory = sum(self.node_memory_gb)
        if total_memory >= model_size_gb * 1.5:
            return "pipeline_parallel"
        
        # 否则使用Tensor并行（更复杂但内存效率更高）
        return "tensor_parallel"


# ==================== 测试 ====================

def main():
    print("="*70)
    print("分布式大模型推理 - 模型分片与优化分析")
    print("="*70)
    
    # 测试不同模型
    models = [
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
    ]
    
    # 测试不同节点配置
    node_configs = [
        ("单节点", [8]),           # 8GB内存
        ("双节点", [8, 8]),        # 两个8GB节点
        ("四节点", [8, 8, 8, 8]),  # 四个8GB节点
        ("异构节点", [4, 8, 16]),  # 不同内存大小
    ]
    
    for model_name in models:
        print(f"\n{'='*70}")
        print(f"模型: {model_name}")
        print("="*70)
        
        for config_name, node_memory in node_configs:
            print(f"\n配置: {config_name} ({node_memory})")
            print("-"*50)
            
            planner = DistributedInferencePlanner(
                model_name=model_name,
                num_nodes=len(node_memory),
                node_memory_gb=node_memory
            )
            
            plan = planner.create_plan()
            
            if "error" in plan:
                print(f"  错误: {plan['error']}")
                continue
            
            print(f"  策略: {plan['strategy']}")
            print(f"  分片数: {len(plan['shards'])}")
            
            for shard in plan['shards']:
                print(f"    - 分片{shard['shard_id']}: {shard['type']}, {shard['size_gb']:.2f}GB")
            
            print(f"\n  CPU优化:")
            cpu = plan['cpu_optimization']
            print(f"    - 工作线程: {cpu['num_workers']}")
            print(f"    - 每节点模型数: {cpu['models_per_node']}")
            print(f"    - 推理线程: {cpu['inference_threads']}")
            print(f"    - 批处理大小: {cpu['batch_size']}")
            
            print(f"\n  网络需求:")
            net = plan['network_requirements']
            for bw, time_sec in net['model_transfer_time'].items():
                if time_sec < 60:
                    print(f"    - {bw}: {time_sec:.1f}秒")
                elif time_sec < 3600:
                    print(f"    - {bw}: {time_sec/60:.1f}分钟")
                else:
                    print(f"    - {bw}: {time_sec/3600:.1f}小时")


if __name__ == "__main__":
    main()
