#!/usr/bin/env python3
"""
分布式大模型推理系统 - 压力测试脚本
=====================================

测试场景:
1. 节点连接稳定性
2. 高并发请求
3. 节点故障恢复
4. 内存使用
5. 响应时间
"""

import sys
import os
import time
import json
import uuid
import random
import threading
import queue
import socket
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict

# 添加虚拟环境路径
sys.path.insert(0, '/home/z/.local/lib/python3.13/site-packages')

try:
    import socketio
    import psutil
except ImportError:
    print("安装依赖...")
    import subprocess
    subprocess.run(['pip', 'install', '--user', 'python-socketio', 'psutil'], check=True)
    import socketio
    import psutil


@dataclass
class TestResult:
    """测试结果"""
    name: str
    total: int = 0
    success: int = 0
    failed: int = 0
    errors: List[str] = field(default_factory=list)
    latencies: List[float] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0
    
    def add_success(self, latency: float):
        self.total += 1
        self.success += 1
        self.latencies.append(latency)
    
    def add_failure(self, error: str):
        self.total += 1
        self.failed += 1
        self.errors.append(error)
    
    def get_stats(self) -> Dict:
        if not self.latencies:
            return {
                "name": self.name,
                "total": self.total,
                "success": self.success,
                "failed": self.failed,
                "success_rate": 0,
                "avg_latency": 0,
                "min_latency": 0,
                "max_latency": 0,
                "p50": 0,
                "p95": 0,
                "p99": 0,
                "duration": self.end_time - self.start_time,
                "throughput": 0
            }
        
        sorted_latencies = sorted(self.latencies)
        n = len(sorted_latencies)
        
        return {
            "name": self.name,
            "total": self.total,
            "success": self.success,
            "failed": self.failed,
            "success_rate": (self.success / self.total * 100) if self.total > 0 else 0,
            "avg_latency": sum(sorted_latencies) / n,
            "min_latency": sorted_latencies[0],
            "max_latency": sorted_latencies[-1],
            "p50": sorted_latencies[int(n * 0.5)],
            "p95": sorted_latencies[int(n * 0.95)],
            "p99": sorted_latencies[int(n * 0.99)],
            "duration": self.end_time - self.start_time,
            "throughput": n / (self.end_time - self.start_time) if self.end_time > self.start_time else 0
        }


class StressTester:
    """压力测试器"""
    
    def __init__(self, server_url: str, api_url: str):
        self.server_url = server_url
        self.api_url = api_url
        self.results: Dict[str, TestResult] = {}
        self.nodes: List[socketio.Client] = []
        self.lock = threading.Lock()
    
    def print_header(self, title: str):
        """打印标题"""
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}\n")
    
    def test_health_check(self) -> TestResult:
        """测试健康检查端点"""
        result = TestResult(name="健康检查测试")
        result.start_time = time.time()
        
        endpoints = ['/health', '/ready', '/metrics']
        
        for endpoint in endpoints:
            for i in range(10):  # 每个端点测试10次
                try:
                    start = time.time()
                    with urllib.request.urlopen(f"{self.api_url}{endpoint}", timeout=5) as response:
                        latency = (time.time() - start) * 1000
                        if response.status == 200:
                            result.add_success(latency)
                        else:
                            result.add_failure(f"状态码: {response.status}")
                except Exception as e:
                    result.add_failure(str(e))
        
        result.end_time = time.time()
        return result
    
    def test_node_connection(self, num_nodes: int = 5, duration: int = 30) -> TestResult:
        """测试节点连接稳定性"""
        result = TestResult(name=f"节点连接测试 ({num_nodes}节点, {duration}秒)")
        result.start_time = time.time()
        
        connected_nodes = []
        connection_times = []
        
        def create_node(node_id: int):
            try:
                sio = socketio.Client(
                    reconnection=True,
                    reconnection_attempts=5,
                    reconnection_delay=1,
                    logger=False,
                    engineio_logger=False
                )
                
                connected = False
                registered = False
                
                @sio.event
                def connect():
                    nonlocal connected
                    connected = True
                    sio.emit('node:register', {
                        'nodeId': f'test-node-{node_id}',
                        'name': f'TestNode-{node_id}',
                        'os': 'Test OS',
                        'cpuCores': 4,
                        'totalMemory': 8192,
                        'availableMemory': 6144,
                        'parallelSlots': 2
                    })
                
                @sio.on('node:registered')
                def on_registered(data):
                    nonlocal registered
                    registered = True
                
                @sio.event
                def disconnect():
                    nonlocal connected
                    connected = False
                
                start = time.time()
                sio.connect(self.server_url, transports=['polling'], wait_timeout=10)
                latency = (time.time() - start) * 1000
                
                if connected and registered:
                    result.add_success(latency)
                    with self.lock:
                        connected_nodes.append(sio)
                    connection_times.append(latency)
                else:
                    result.add_failure("连接或注册失败")
                
            except Exception as e:
                result.add_failure(str(e))
        
        # 并发创建节点
        with ThreadPoolExecutor(max_workers=num_nodes) as executor:
            futures = [executor.submit(create_node, i) for i in range(num_nodes)]
            for future in as_completed(futures):
                pass
        
        # 保持连接一段时间
        start_wait = time.time()
        while time.time() - start_wait < duration:
            time.sleep(1)
            # 检查连接状态
            alive = sum(1 for sio in connected_nodes if sio.connected)
            print(f"  运行中... {time.time() - start_wait:.0f}s, 存活节点: {alive}/{num_nodes}")
        
        # 断开所有节点
        for sio in connected_nodes:
            try:
                sio.disconnect()
            except:
                pass
        
        result.end_time = time.time()
        return result
    
    def test_concurrent_requests(self, num_requests: int = 100, concurrency: int = 10) -> TestResult:
        """测试高并发请求"""
        result = TestResult(name=f"并发请求测试 ({num_requests}请求, {concurrency}并发)")
        result.start_time = time.time()
        
        def send_request(request_id: int):
            try:
                start = time.time()
                data = json.dumps({
                    "prompt": f"测试请求 #{request_id}: 请简单回复。",
                    "modelId": "test-model"
                }).encode()
                
                req = urllib.request.Request(
                    f"{self.api_url}/api/inference",
                    data=data,
                    headers={"Content-Type": "application/json"}
                )
                
                with urllib.request.urlopen(req, timeout=30) as response:
                    latency = (time.time() - start) * 1000
                    if response.status == 200:
                        result.add_success(latency)
                    else:
                        result.add_failure(f"状态码: {response.status}")
            except Exception as e:
                result.add_failure(str(e))
        
        # 并发发送请求
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(send_request, i) for i in range(num_requests)]
            completed = 0
            for future in as_completed(futures):
                completed += 1
                if completed % 10 == 0:
                    print(f"  进度: {completed}/{num_requests}")
        
        result.end_time = time.time()
        return result
    
    def test_rate_limiting(self, requests_per_burst: int = 150) -> TestResult:
        """测试速率限制"""
        result = TestResult(name=f"速率限制测试 ({requests_per_burst}请求)")
        result.start_time = time.time()
        
        for i in range(requests_per_burst):
            try:
                start = time.time()
                data = json.dumps({"prompt": "测试", "modelId": "test"}).encode()
                req = urllib.request.Request(
                    f"{self.api_url}/api/inference",
                    data=data,
                    headers={"Content-Type": "application/json"}
                )
                
                with urllib.request.urlopen(req, timeout=5) as response:
                    latency = (time.time() - start) * 1000
                    if response.status == 200:
                        result.add_success(latency)
                    elif response.status == 429:
                        result.add_failure("速率限制触发")
                    else:
                        result.add_failure(f"状态码: {response.status}")
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    result.add_failure("速率限制触发 (429)")
                else:
                    result.add_failure(f"HTTP错误: {e.code}")
            except Exception as e:
                result.add_failure(str(e))
        
        result.end_time = time.time()
        return result
    
    def test_node_failure_recovery(self, num_nodes: int = 3) -> TestResult:
        """测试节点故障恢复"""
        result = TestResult(name=f"节点故障恢复测试 ({num_nodes}节点)")
        result.start_time = time.time()
        
        # 创建节点
        nodes = []
        for i in range(num_nodes):
            try:
                sio = socketio.Client(logger=False, engineio_logger=False)
                
                @sio.event
                def connect():
                    sio.emit('node:register', {
                        'nodeId': f'fail-test-{i}',
                        'name': f'FailTest-{i}',
                        'os': 'Test',
                        'cpuCores': 4,
                        'totalMemory': 8192,
                        'parallelSlots': 2
                    })
                
                sio.connect(self.server_url, transports=['polling'], wait_timeout=10)
                nodes.append(sio)
                result.add_success(0)
            except Exception as e:
                result.add_failure(str(e))
        
        time.sleep(2)
        
        # 模拟节点故障 - 断开一半节点
        print(f"  断开 {num_nodes // 2} 个节点...")
        for i, sio in enumerate(nodes[:num_nodes // 2]):
            try:
                sio.disconnect()
            except:
                pass
        
        time.sleep(2)
        
        # 检查系统状态
        try:
            with urllib.request.urlopen(f"{self.api_url}/health", timeout=5) as response:
                health = json.loads(response.read().decode())
                print(f"  系统状态: {health['nodes']['online']}/{health['nodes']['total']} 节点在线")
        except Exception as e:
            result.add_failure(f"健康检查失败: {e}")
        
        # 清理剩余节点
        for sio in nodes:
            try:
                sio.disconnect()
            except:
                pass
        
        result.end_time = time.time()
        return result
    
    def test_memory_usage(self, duration: int = 30) -> TestResult:
        """测试内存使用"""
        result = TestResult(name=f"内存使用测试 ({duration}秒)")
        result.start_time = time.time()
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        memory_samples = [initial_memory]
        
        # 持续发送请求并监控内存
        start_time = time.time()
        request_count = 0
        
        while time.time() - start_time < duration:
            try:
                # 发送请求
                data = json.dumps({"prompt": "内存测试", "modelId": "test"}).encode()
                req = urllib.request.Request(
                    f"{self.api_url}/api/inference",
                    data=data,
                    headers={"Content-Type": "application/json"}
                )
                with urllib.request.urlopen(req, timeout=5):
                    request_count += 1
            except:
                pass
            
            # 采样内存
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_samples.append(current_memory)
            
            time.sleep(0.5)
        
        final_memory = memory_samples[-1]
        memory_growth = final_memory - initial_memory
        
        result.add_success(0)
        result.errors.append(f"初始内存: {initial_memory:.1f}MB")
        result.errors.append(f"最终内存: {final_memory:.1f}MB")
        result.errors.append(f"内存增长: {memory_growth:.1f}MB")
        result.errors.append(f"最大内存: {max(memory_samples):.1f}MB")
        result.errors.append(f"请求数: {request_count}")
        
        result.end_time = time.time()
        return result
    
    def run_all_tests(self):
        """运行所有测试"""
        self.print_header("分布式大模型推理系统 - 压力测试")
        
        print(f"服务器: {self.server_url}")
        print(f"API: {self.api_url}")
        print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 测试1: 健康检查
        self.print_header("测试1: 健康检查端点")
        self.results['health'] = self.test_health_check()
        self.print_result(self.results['health'])
        
        # 测试2: 节点连接
        self.print_header("测试2: 节点连接稳定性")
        self.results['connection'] = self.test_node_connection(num_nodes=5, duration=20)
        self.print_result(self.results['connection'])
        
        # 测试3: 并发请求
        self.print_header("测试3: 高并发请求")
        self.results['concurrent'] = self.test_concurrent_requests(num_requests=50, concurrency=10)
        self.print_result(self.results['concurrent'])
        
        # 测试4: 速率限制
        self.print_header("测试4: 速率限制")
        self.results['rate_limit'] = self.test_rate_limiting(requests_per_burst=120)
        self.print_result(self.results['rate_limit'])
        
        # 测试5: 节点故障恢复
        self.print_header("测试5: 节点故障恢复")
        self.results['failure'] = self.test_node_failure_recovery(num_nodes=4)
        self.print_result(self.results['failure'])
        
        # 测试6: 内存使用
        self.print_header("测试6: 内存使用")
        self.results['memory'] = self.test_memory_usage(duration=20)
        self.print_result(self.results['memory'])
        
        # 打印总结
        self.print_summary()
    
    def print_result(self, result: TestResult):
        """打印测试结果"""
        stats = result.get_stats()
        
        print(f"  总请求数: {stats['total']}")
        print(f"  成功: {stats['success']}, 失败: {stats['failed']}")
        print(f"  成功率: {stats['success_rate']:.1f}%")
        
        if stats['avg_latency'] > 0:
            print(f"  平均延迟: {stats['avg_latency']:.2f}ms")
            print(f"  P50: {stats['p50']:.2f}ms, P95: {stats['p95']:.2f}ms, P99: {stats['p99']:.2f}ms")
            print(f"  吞吐量: {stats['throughput']:.2f} 请求/秒")
        
        if result.errors:
            print(f"  错误信息:")
            for error in result.errors[:5]:
                print(f"    - {error}")
        
        print()
    
    def print_summary(self):
        """打印测试总结"""
        self.print_header("测试总结")
        
        total_tests = sum(r.total for r in self.results.values())
        total_success = sum(r.success for r in self.results.values())
        total_failed = sum(r.failed for r in self.results.values())
        
        print(f"  总测试数: {total_tests}")
        print(f"  成功: {total_success}")
        print(f"  失败: {total_failed}")
        print(f"  成功率: {total_success/total_tests*100:.1f}%" if total_tests > 0 else "  成功率: N/A")
        
        print("\n  各测试结果:")
        for name, result in self.results.items():
            stats = result.get_stats()
            status = "✅" if stats['success_rate'] >= 90 else "⚠️" if stats['success_rate'] >= 70 else "❌"
            print(f"    {status} {result.name}: {stats['success_rate']:.1f}%")
        
        print("\n" + "="*60)


def main():
    tester = StressTester(
        server_url="http://localhost:3003",
        api_url="http://localhost:3004"
    )
    
    try:
        tester.run_all_tests()
    except KeyboardInterrupt:
        print("\n\n测试被中断")
    except Exception as e:
        print(f"\n\n测试出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
