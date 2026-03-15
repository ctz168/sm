#!/usr/bin/env python3
"""
分布式大模型推理系统 - 资源感知动态启停版本
============================================

核心功能:
1. 资源检测 - 实时监控内存、CPU、GPU
2. 资源评估 - 判断是否有足够资源运行模型
3. 动态启停 - 资源不足时自动停止，充足时自动启动
4. 集群协调 - 多节点时协调资源分配

工作流程:
1. 启动时检测资源
2. 评估是否能运行模型
3. 如果资源充足 → 启动模型服务
4. 如果资源不足 → 进入待机模式，定期检测
5. 运行中持续监控，资源不足时自动降级
"""

import os
import sys
import time
import json
import uuid
import socket
import threading
import signal
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
import subprocess

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("警告: psutil未安装，请运行: pip install psutil")

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("警告: torch/transformers未安装")


# ==================== 枚举定义 ====================

class ServiceState(Enum):
    """服务状态"""
    STANDBY = "standby"           # 待机模式（资源不足）
    STARTING = "starting"         # 启动中
    RUNNING = "running"           # 运行中
    DEGRADED = "degraded"         # 降级模式
    STOPPING = "stopping"         # 停止中
    STOPPED = "stopped"           # 已停止
    ERROR = "error"               # 错误


class ResourceType(Enum):
    """资源类型"""
    MEMORY = "memory"
    CPU = "cpu"
    GPU = "gpu"
    DISK = "disk"


# ==================== 配置 ====================

@dataclass
class ResourceConfig:
    """资源配置"""
    # 模型配置
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    model_memory_gb: float = 2.0  # 模型需要的内存
    
    # 资源阈值
    min_memory_gb: float = 4.0    # 最小内存要求
    min_cpu_percent: float = 20.0  # 最小CPU空闲百分比
    min_gpu_memory_gb: float = 0.0  # 最小GPU内存（0表示不要求）
    
    # 安全裕度
    memory_safety_margin: float = 1.5  # 内存安全系数（模型大小 × 此系数）
    cpu_safety_margin: float = 1.2     # CPU安全系数
    
    # 检测间隔
    check_interval: float = 10.0  # 资源检测间隔（秒）
    startup_delay: float = 5.0    # 启动延迟（秒）
    
    # 动态调整
    auto_start: bool = True       # 资源充足时自动启动
    auto_stop: bool = True        # 资源不足时自动停止
    cooldown_period: float = 60.0  # 冷却期（秒），防止频繁启停
    
    # 网络配置
    host: str = "0.0.0.0"
    port: int = 7000


# ==================== 资源检测器 ====================

class ResourceDetector:
    """资源检测器"""
    
    def __init__(self):
        self.history: deque = deque(maxlen=60)  # 保存最近60次检测记录
        self.last_check_time = 0.0
    
    def get_system_info(self) -> Dict:
        """获取系统信息"""
        info = {
            "timestamp": time.time(),
            "memory": self._get_memory_info(),
            "cpu": self._get_cpu_info(),
            "gpu": self._get_gpu_info(),
            "disk": self._get_disk_info(),
        }
        
        self.history.append(info)
        return info
    
    def _get_memory_info(self) -> Dict:
        """获取内存信息"""
        if not HAS_PSUTIL:
            return {
                "total_gb": 8.0,
                "available_gb": 4.0,
                "used_gb": 4.0,
                "percent": 50.0,
            }
        
        mem = psutil.virtual_memory()
        return {
            "total_gb": mem.total / (1024**3),
            "available_gb": mem.available / (1024**3),
            "used_gb": mem.used / (1024**3),
            "percent": mem.percent,
        }
    
    def _get_cpu_info(self) -> Dict:
        """获取CPU信息"""
        if not HAS_PSUTIL:
            return {
                "cores": 4,
                "usage_percent": 50.0,
                "idle_percent": 50.0,
                "load_avg": [1.0, 1.0, 1.0],
            }
        
        return {
            "cores": psutil.cpu_count() or 4,
            "usage_percent": psutil.cpu_percent(interval=0.1),
            "idle_percent": 100.0 - psutil.cpu_percent(interval=0.1),
            "load_avg": list(os.getloadavg()) if hasattr(os, 'getloadavg') else [1.0, 1.0, 1.0],
        }
    
    def _get_gpu_info(self) -> Dict:
        """获取GPU信息"""
        if not HAS_TORCH or not torch.cuda.is_available():
            return {
                "available": False,
                "count": 0,
                "total_memory_gb": 0.0,
                "free_memory_gb": 0.0,
                "used_memory_gb": 0.0,
            }
        
        try:
            device = torch.cuda.current_device()
            total = torch.cuda.get_device_properties(device).total_memory
            reserved = torch.cuda.memory_reserved(device)
            allocated = torch.cuda.memory_allocated(device)
            free = total - reserved
            
            return {
                "available": True,
                "count": torch.cuda.device_count(),
                "total_memory_gb": total / (1024**3),
                "free_memory_gb": free / (1024**3),
                "used_memory_gb": allocated / (1024**3),
                "device_name": torch.cuda.get_device_name(device),
            }
        except:
            return {
                "available": False,
                "count": 0,
                "total_memory_gb": 0.0,
                "free_memory_gb": 0.0,
                "used_memory_gb": 0.0,
            }
    
    def _get_disk_info(self) -> Dict:
        """获取磁盘信息"""
        if not HAS_PSUTIL:
            return {
                "total_gb": 100.0,
                "free_gb": 50.0,
                "percent": 50.0,
            }
        
        disk = psutil.disk_usage('/')
        return {
            "total_gb": disk.total / (1024**3),
            "free_gb": disk.free / (1024**3),
            "percent": disk.percent,
        }
    
    def get_average_usage(self, seconds: int = 60) -> Dict:
        """获取平均资源使用率"""
        if not self.history:
            return {}
        
        # 取最近N秒的数据
        cutoff = time.time() - seconds
        recent = [h for h in self.history if h["timestamp"] > cutoff]
        
        if not recent:
            return {}
        
        avg_memory = sum(h["memory"]["percent"] for h in recent) / len(recent)
        avg_cpu = sum(h["cpu"]["usage_percent"] for h in recent) / len(recent)
        
        return {
            "avg_memory_percent": avg_memory,
            "avg_cpu_percent": avg_cpu,
            "samples": len(recent),
        }


# ==================== 资源评估器 ====================

class ResourceEvaluator:
    """资源评估器"""
    
    def __init__(self, config: ResourceConfig):
        self.config = config
        self.detector = ResourceDetector()
    
    def can_run_model(self, model_memory_gb: Optional[float] = None) -> Tuple[bool, str, Dict]:
        """
        评估是否能运行模型
        
        返回: (是否可以运行, 原因, 详细信息)
        """
        if model_memory_gb is None:
            model_memory_gb = self.config.model_memory_gb
        
        # 获取当前资源
        info = self.detector.get_system_info()
        
        # 计算需要的内存（含安全裕度）
        required_memory = model_memory_gb * self.config.memory_safety_margin
        
        details = {
            "required_memory_gb": required_memory,
            "available_memory_gb": info["memory"]["available_gb"],
            "cpu_idle_percent": info["cpu"]["idle_percent"],
            "gpu_available": info["gpu"]["available"],
        }
        
        # 检查内存
        if info["memory"]["available_gb"] < required_memory:
            reason = f"内存不足: 需要{required_memory:.1f}GB, 可用{info['memory']['available_gb']:.1f}GB"
            return False, reason, details
        
        # 检查CPU
        if info["cpu"]["idle_percent"] < self.config.min_cpu_percent:
            reason = f"CPU空闲不足: 需要{self.config.min_cpu_percent}%, 当前{info['cpu']['idle_percent']:.1f}%"
            return False, reason, details
        
        # 检查GPU（如果要求）
        if self.config.min_gpu_memory_gb > 0 and info["gpu"]["available"]:
            if info["gpu"]["free_memory_gb"] < self.config.min_gpu_memory_gb:
                reason = f"GPU内存不足: 需要{self.config.min_gpu_memory_gb}GB, 可用{info['gpu']['free_memory_gb']:.1f}GB"
                return False, reason, details
        
        # 所有检查通过
        reason = f"资源充足: 内存{info['memory']['available_gb']:.1f}GB可用, CPU空闲{info['cpu']['idle_percent']:.1f}%"
        return True, reason, details
    
    def get_resource_score(self) -> float:
        """
        获取资源评分 (0-100)
        分数越高，资源越充足
        """
        info = self.detector.get_system_info()
        
        # 内存评分 (权重50%)
        memory_score = min(100, info["memory"]["available_gb"] / self.config.min_memory_gb * 50)
        
        # CPU评分 (权重30%)
        cpu_score = min(100, info["cpu"]["idle_percent"] / self.config.min_cpu_percent * 30)
        
        # GPU评分 (权重20%，如果没有GPU则为满分)
        if info["gpu"]["available"]:
            gpu_score = min(100, info["gpu"]["free_memory_gb"] / max(1, self.config.min_gpu_memory_gb) * 20)
        else:
            gpu_score = 20
        
        return memory_score + cpu_score + gpu_score
    
    def get_recommendation(self) -> Dict:
        """获取资源建议"""
        can_run, reason, details = self.can_run_model()
        score = self.get_resource_score()
        
        return {
            "can_run": can_run,
            "reason": reason,
            "score": score,
            "details": details,
            "recommendation": self._get_action_recommendation(can_run, score),
        }
    
    def _get_action_recommendation(self, can_run: bool, score: float) -> str:
        """获取行动建议"""
        if can_run:
            if score >= 80:
                return "资源非常充足，建议启动模型服务"
            elif score >= 60:
                return "资源充足，可以启动模型服务"
            else:
                return "资源勉强够用，建议谨慎启动"
        else:
            if score >= 40:
                return "资源接近阈值，建议等待资源释放"
            else:
                return "资源严重不足，建议关闭其他进程"


# ==================== 动态服务管理器 ====================

class DynamicServiceManager:
    """动态服务管理器"""
    
    def __init__(self, config: ResourceConfig):
        self.config = config
        self.evaluator = ResourceEvaluator(config)
        
        # 状态
        self.state = ServiceState.STANDBY
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        
        # 时间记录
        self.last_state_change = time.time()
        self.last_start_attempt = 0.0
        self.start_attempts = 0
        
        # 统计
        self.stats = {
            "total_starts": 0,
            "total_stops": 0,
            "total_requests": 0,
            "total_errors": 0,
            "uptime": 0.0,
        }
        
        # 锁
        self.lock = threading.Lock()
    
    def start_monitoring(self):
        """启动监控"""
        print("🔍 启动资源监控...")
        
        # 初始检测
        self._check_and_adjust()
        
        # 启动监控线程
        threading.Thread(target=self._monitor_loop, daemon=True).start()
    
    def _monitor_loop(self):
        """监控循环"""
        while True:
            try:
                self._check_and_adjust()
            except Exception as e:
                print(f"[监控] 错误: {e}")
            
            time.sleep(self.config.check_interval)
    
    def _check_and_adjust(self):
        """检测并调整服务状态"""
        with self.lock:
            recommendation = self.evaluator.get_recommendation()
            can_run = recommendation["can_run"]
            score = recommendation["score"]
            
            current_time = time.time()
            
            # 状态机
            if self.state == ServiceState.STANDBY:
                # 待机状态，检查是否可以启动
                if can_run and self.config.auto_start:
                    # 检查冷却期
                    if current_time - self.last_state_change > self.config.cooldown_period:
                        self._try_start_model()
            
            elif self.state == ServiceState.RUNNING:
                # 运行状态，检查是否需要停止
                if not can_run and self.config.auto_stop:
                    # 资源不足，需要降级
                    print(f"[资源] 检测到资源不足: {recommendation['reason']}")
                    self._stop_model()
            
            elif self.state == ServiceState.DEGRADED:
                # 降级状态，检查是否可以恢复
                if can_run:
                    print(f"[资源] 资源恢复: {recommendation['reason']}")
                    self._try_start_model()
            
            elif self.state == ServiceState.ERROR:
                # 错误状态，尝试恢复
                if can_run and current_time - self.last_state_change > 300:  # 5分钟后重试
                    self._try_start_model()
    
    def _try_start_model(self) -> bool:
        """尝试启动模型"""
        if not HAS_TORCH:
            print("[模型] PyTorch未安装，无法启动")
            return False
        
        current_time = time.time()
        
        # 检查启动频率
        if current_time - self.last_start_attempt < 60:
            return False
        
        self.last_start_attempt = current_time
        self.start_attempts += 1
        
        print(f"\n{'='*50}")
        print(f"  尝试启动模型 (第{self.start_attempts}次)")
        print(f"{'='*50}")
        
        # 再次检查资源
        can_run, reason, details = self.evaluator.can_run_model()
        if not can_run:
            print(f"[模型] 资源不足，无法启动: {reason}")
            return False
        
        try:
            self.state = ServiceState.STARTING
            self.last_state_change = time.time()
            
            print(f"[模型] 加载模型: {self.config.model_name}")
            start_time = time.time()
            
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True
            )
            
            # 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            self.model.eval()
            
            load_time = time.time() - start_time
            
            # 更新状态
            self.model_loaded = True
            self.state = ServiceState.RUNNING
            self.last_state_change = time.time()
            self.stats["total_starts"] += 1
            
            print(f"[模型] ✅ 启动成功! 加载时间: {load_time:.1f}秒")
            print(f"{'='*50}\n")
            
            return True
            
        except Exception as e:
            print(f"[模型] ❌ 启动失败: {e}")
            self.state = ServiceState.ERROR
            self.last_state_change = time.time()
            self.stats["total_errors"] += 1
            return False
    
    def _stop_model(self):
        """停止模型"""
        if not self.model_loaded:
            return
        
        print(f"\n[模型] 正在停止模型...")
        
        self.state = ServiceState.STOPPING
        self.last_state_change = time.time()
        
        try:
            # 卸载模型
            if self.model:
                del self.model
                self.model = None
            if self.tokenizer:
                del self.tokenizer
                self.tokenizer = None
            
            # 清理内存
            if HAS_TORCH and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            import gc
            gc.collect()
            
            self.model_loaded = False
            self.state = ServiceState.STANDBY
            self.last_state_change = time.time()
            self.stats["total_stops"] += 1
            
            print(f"[模型] ✅ 已停止，进入待机模式\n")
            
        except Exception as e:
            print(f"[模型] 停止失败: {e}")
            self.state = ServiceState.ERROR
    
    def get_status(self) -> Dict:
        """获取状态"""
        info = self.evaluator.detector.get_system_info()
        recommendation = self.evaluator.get_recommendation()
        
        return {
            "state": self.state.value,
            "model_loaded": self.model_loaded,
            "model_name": self.config.model_name,
            "resources": {
                "memory": info["memory"],
                "cpu": info["cpu"],
                "gpu": info["gpu"],
            },
            "recommendation": recommendation,
            "stats": self.stats,
            "uptime": time.time() - self.last_state_change if self.state == ServiceState.RUNNING else 0,
        }
    
    def process_request(self, prompt: str, max_tokens: int = 50) -> Dict:
        """处理请求"""
        self.stats["total_requests"] += 1
        
        if not self.model_loaded or self.state != ServiceState.RUNNING:
            return {
                "success": False,
                "error": f"服务未就绪，当前状态: {self.state.value}",
                "state": self.state.value,
            }
        
        try:
            import torch
            
            start_time = time.time()
            
            # 编码
            inputs = self.tokenizer(prompt, return_tensors="pt")
            
            # 推理
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=max_tokens,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # 解码
            latency = time.time() - start_time
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
            self.stats["total_errors"] += 1
            return {
                "success": False,
                "error": str(e),
            }


# ==================== HTTP服务 ====================

class ResourceAwareHTTPServer:
    """资源感知HTTP服务"""
    
    def __init__(self, config: ResourceConfig):
        self.config = config
        self.manager = DynamicServiceManager(config)
        self.server_socket = None
        self.running = False
    
    def start(self):
        """启动服务"""
        print(f"\n{'='*60}")
        print(f"  分布式大模型推理系统 - 资源感知版本")
        print(f"{'='*60}")
        print(f"  模型: {self.config.model_name}")
        print(f"  最小内存: {self.config.min_memory_gb}GB")
        print(f"  最小CPU空闲: {self.config.min_cpu_percent}%")
        print(f"  端口: {self.config.port}")
        print(f"{'='*60}\n")
        
        # 启动资源监控
        self.manager.start_monitoring()
        
        # 启动HTTP服务
        self._start_http_server()
        
        self.running = True
        print(f"✅ 服务启动完成\n")
    
    def _start_http_server(self):
        """启动HTTP服务器"""
        from http.server import HTTPServer, BaseHTTPRequestHandler
        
        class RequestHandler(BaseHTTPRequestHandler):
            server = self
            
            def do_GET(self):
                if self.path == '/health':
                    self._send_json({"status": "healthy"})
                elif self.path == '/status':
                    self._send_json(self.server.manager.get_status())
                elif self.path == '/resources':
                    info = self.server.manager.evaluator.detector.get_system_info()
                    self._send_json(info)
                else:
                    self.send_error(404)
            
            def do_POST(self):
                if self.path == '/inference':
                    content_length = int(self.headers.get('Content-Length', 0))
                    body = self.rfile.read(content_length)
                    
                    try:
                        data = json.loads(body)
                        prompt = data.get('prompt', '')
                        max_tokens = data.get('max_tokens', 50)
                        
                        result = self.server.manager.process_request(prompt, max_tokens)
                        self._send_json(result)
                    except Exception as e:
                        self._send_json({"success": False, "error": str(e)}, 500)
                
                elif self.path == '/start':
                    # 手动启动
                    success = self.server.manager._try_start_model()
                    self._send_json({"success": success})
                
                elif self.path == '/stop':
                    # 手动停止
                    self.server.manager._stop_model()
                    self._send_json({"success": True})
                
                else:
                    self.send_error(404)
            
            def _send_json(self, data, code=200):
                self.send_response(code)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(data, ensure_ascii=False).encode())
            
            def log_message(self, format, *args):
                pass  # 禁用日志
        
        # 启动服务器
        self.server_socket = HTTPServer(
            (self.config.host, self.config.port),
            RequestHandler
        )
        
        threading.Thread(target=self.server_socket.serve_forever, daemon=True).start()
    
    def stop(self):
        """停止服务"""
        self.running = False
        if self.server_socket:
            self.server_socket.shutdown()
        self.manager._stop_model()


# ==================== 主函数 ====================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="资源感知动态启停推理服务")
    
    parser.add_argument("--model", "-m", default="Qwen/Qwen2.5-0.5B-Instruct", help="模型名称")
    parser.add_argument("--port", "-p", type=int, default=7000, help="服务端口")
    parser.add_argument("--min-memory", type=float, default=4.0, help="最小内存(GB)")
    parser.add_argument("--min-cpu", type=float, default=20.0, help="最小CPU空闲(%)")
    parser.add_argument("--model-memory", type=float, default=2.0, help="模型内存需求(GB)")
    parser.add_argument("--no-auto-start", action="store_true", help="禁用自动启动")
    parser.add_argument("--no-auto-stop", action="store_true", help="禁用自动停止")
    
    args = parser.parse_args()
    
    config = ResourceConfig(
        model_name=args.model,
        port=args.port,
        min_memory_gb=args.min_memory,
        min_cpu_percent=args.min_cpu,
        model_memory_gb=args.model_memory,
        auto_start=not args.no_auto_start,
        auto_stop=not args.no_auto_stop,
    )
    
    server = ResourceAwareHTTPServer(config)
    
    def signal_handler(sig, frame):
        print("\n停止服务...")
        server.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    server.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.stop()


if __name__ == "__main__":
    main()
