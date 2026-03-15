#!/usr/bin/env python3
"""
分布式大模型推理系统 - 测试节点
使用最小模型进行实际测试
"""

import os
import sys
import time
import uuid
import threading
import json

# 确保使用正确的依赖路径
sys.path.insert(0, '/home/z/.local/lib/python3.13/site-packages')

import socketio
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class TestNode:
    """测试节点"""
    
    def __init__(self, server_url, model_name="gpt2"):
        self.server_url = server_url
        self.model_name = model_name
        self.node_id = str(uuid.uuid4())
        self.name = f"Test-Node-{self.node_id[:8]}"
        self.model = None
        self.tokenizer = None
        self.connected = False
        self.shards = []
        self.tasks_completed = 0
        
        # 创建Socket.IO客户端
        self.sio = socketio.Client()
        self._setup_events()
    
    def _setup_events(self):
        @self.sio.event
        def connect():
            print(f"✅ 已连接到服务器")
            self.connected = True
            # 发送注册信息
            self.sio.emit("node:register", {
                "nodeId": self.node_id,
                "name": self.name,
                "os": "Linux Test",
                "cpuCores": 4,
                "totalMemory": 8192,
                "availableMemory": 6144,
                "parallelSlots": 2
            })
        
        @self.sio.event
        def disconnect():
            print("❌ 与服务器断开连接")
            self.connected = False
        
        @self.sio.on('node:registered')
        def on_registered(data):
            print(f"✅ 节点已注册: {data.get('nodeId', 'unknown')}")
        
        @self.sio.on('shard:assign')
        def on_shard_assign(data):
            print(f"📥 收到分片分配: {data.get('shardId', 'unknown')}")
            self.shards.append(data.get('shardId'))
            # 模拟加载完成
            self.sio.emit("shard:loaded", {"shardId": data.get('shardId')})
        
        @self.sio.on('task:inference')
        def on_task_inference(data):
            """接收推理任务"""
            print(f"🎯 收到推理任务: {data.get('taskId', 'unknown')[:8]}...")
            self._execute_inference(data)
        
        @self.sio.event
        def system_status(data):
            print(f"📊 系统状态更新")
    
    def load_model(self):
        """加载模型"""
        print(f"📥 加载模型: {self.model_name}")
        start = time.time()
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.model.eval()
        
        print(f"✅ 模型加载完成 ({time.time() - start:.1f}s)")
        return True
    
    def _execute_inference(self, data):
        """执行推理"""
        task_id = data.get("taskId")
        prompt = data.get("prompt", "")
        
        start = time.time()
        
        try:
            # 编码输入
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            
            # 生成
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.7,
                    do_sample=True
                )
            
            # 解码
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            latency = time.time() - start
            tokens = outputs.shape[1] - inputs.input_ids.shape[1]
            
            self.tasks_completed += 1
            
            # 发送结果
            self.sio.emit("inference:result", {
                "taskId": task_id,
                "status": "completed",
                "result": result,
                "tokens": tokens,
                "latency": latency
            })
            
            print(f"✅ 任务完成: {task_id[:8]}... (延迟: {latency:.2f}s, tokens: {tokens})")
            
        except Exception as e:
            print(f"❌ 推理失败: {e}")
            self.sio.emit("inference:result", {
                "taskId": task_id,
                "status": "failed",
                "error": str(e)
            })
    
    def connect(self):
        """连接到服务器"""
        print(f"🔗 连接到 {self.server_url}")
        self.sio.connect(self.server_url, transports=["websocket", "polling"])
    
    def disconnect(self):
        """断开连接"""
        if self.connected:
            self.sio.disconnect()
    
    def send_heartbeat(self):
        """发送心跳"""
        if self.connected:
            self.sio.emit("node:heartbeat", {
                "availableMemory": 6144,
                "status": "online",
                "loadScore": 20,
                "activeSlots": 0
            })
    
    def run_heartbeat_loop(self):
        """心跳循环"""
        while self.connected:
            try:
                self.send_heartbeat()
            except Exception as e:
                print(f"心跳错误: {e}")
            time.sleep(10)
    
    def wait_for_tasks(self, timeout=30):
        """等待任务处理"""
        start = time.time()
        while time.time() - start < timeout:
            if not self.connected:
                print("   节点已断开!")
                break
            time.sleep(1)
            if self.tasks_completed > 0:
                return True
        return False


def test_node_operations():
    """测试节点操作"""
    print("\n" + "="*60)
    print("  分布式大模型推理系统 - 节点测试")
    print("="*60 + "\n")
    
    server_url = "http://localhost:3003"
    
    # 创建节点
    print("【1】创建测试节点...")
    node = TestNode(server_url, "gpt2")
    
    # 加载模型
    print("\n【2】加载模型...")
    node.load_model()
    
    # 连接到服务器
    print("\n【3】连接到Orchestrator...")
    try:
        node.connect()
        time.sleep(2)
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        return
    
    # 启动心跳线程
    heartbeat_thread = threading.Thread(target=node.run_heartbeat_loop, daemon=True)
    heartbeat_thread.start()
    
    # 等待节点完全就绪
    print("\n【4】等待节点就绪...")
    time.sleep(3)  # 等待节点注册完成
    
    # 发送心跳确保节点状态更新
    node.send_heartbeat()
    time.sleep(1)
    
    # 检查节点状态
    print("\n【5】检查节点状态...")
    time.sleep(1)
    
    import urllib.request
    try:
        with urllib.request.urlopen("http://localhost:3004/api/status") as response:
            data = json.loads(response.read().decode())
            nodes = data.get("nodes", [])
            print(f"   当前节点数: {len(nodes)}")
            for n in nodes:
                print(f"   - {n.get('name')}: {n.get('status')}")
    except Exception as e:
        print(f"   获取状态失败: {e}")
    
    # 发送测试推理请求
    print("\n【6】发送测试推理请求...")
    
    # 先发送心跳确保节点状态
    node.send_heartbeat()
    time.sleep(0.5)
    
    try:
        req_data = json.dumps({
            "prompt": "Hello, this is a test of the distributed inference system.",
            "modelId": "test-model"
        }).encode()
        
        req = urllib.request.Request(
            "http://localhost:3004/api/inference",
            data=req_data,
            headers={"Content-Type": "application/json"}
        )
        
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode())
            task_id = result.get("taskId")
            print(f"   任务已创建: {task_id[:16]}...")
    except Exception as e:
        print(f"   创建任务失败: {e}")
    
    # 等待推理完成 - 增加等待时间
    print("\n【7】等待推理完成 (保持节点连接)...")
    for i in range(20):  # 等待20秒
        if not node.connected:
            print("   节点已断开!")
            break
        time.sleep(1)
        # 每秒发送心跳
        try:
            node.send_heartbeat()
        except:
            pass
        if node.tasks_completed > 0:
            print(f"   ✅ 任务已完成!")
            break
        if i % 5 == 4:
            print(f"   等待中... {i+1}/20")
    
    # 检查结果
    print("\n【8】检查最终状态...")
    try:
        with urllib.request.urlopen("http://localhost:3004/api/status") as response:
            data = json.loads(response.read().decode())
            nodes = data.get("nodes", [])
            metrics = data.get("metrics", {})
            print(f"   节点数: {len(nodes)}")
            print(f"   完成任务: {node.tasks_completed}")
            print(f"   系统并行度: {metrics.get('currentParallelism', 0)}/{metrics.get('maxParallelism', 0)}")
    except Exception as e:
        print(f"   获取状态失败: {e}")
    
    # 测试节点断开
    print("\n【8】测试节点断开...")
    node.disconnect()
    time.sleep(2)
    
    try:
        with urllib.request.urlopen("http://localhost:3004/api/status") as response:
            data = json.loads(response.read().decode())
            nodes = data.get("nodes", [])
            print(f"   断开后节点数: {len(nodes)}")
    except Exception as e:
        print(f"   获取状态失败: {e}")
    
    print("\n" + "="*60)
    print("  测试完成!")
    print("="*60)


if __name__ == "__main__":
    test_node_operations()
