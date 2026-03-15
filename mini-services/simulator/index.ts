/**
 * 模拟节点服务 - 用于演示和测试
 * 
 * 这个服务会模拟多个节点加入集群，演示系统的动态调度能力
 */

import { io, Socket } from 'socket.io-client';
import { v4 as uuidv4 } from 'uuid';

// 模拟节点配置
const SIMULATED_NODES = [
  { name: 'Windows-PC-4GB', os: 'Windows 11', cpuCores: 4, memory: 4096 },
  { name: 'MacBook-8GB', os: 'macOS Sonoma', cpuCores: 8, memory: 8192 },
  { name: 'Linux-Server-16GB', os: 'Ubuntu 22.04', cpuCores: 16, memory: 16384 },
  { name: 'Old-Laptop-4GB', os: 'Windows 10', cpuCores: 2, memory: 4096 },
];

interface SimulatedNode {
  nodeId: string;
  name: string;
  os: string;
  cpuCores: number;
  totalMemory: number;
  availableMemory: number;
  socket: Socket | null;
  shards: string[];
  status: string;
}

class SimulatedNodeService {
  private nodes: Map<string, SimulatedNode> = new Map();
  private serverUrl: string;
  private running: boolean = false;

  constructor(serverUrl: string) {
    this.serverUrl = serverUrl;
  }

  async start() {
    console.log('\n🎭 模拟节点服务启动');
    console.log(`   服务器: ${this.serverUrl}`);
    console.log(`   模拟节点数: ${SIMULATED_NODES.length}\n`);

    this.running = true;

    // 逐个启动节点，模拟真实场景
    for (let i = 0; i < SIMULATED_NODES.length; i++) {
      await new Promise(resolve => setTimeout(resolve, 2000)); // 2秒间隔
      await this.startNode(SIMULATED_NODES[i], i);
    }

    // 模拟心跳和状态变化
    this.startHeartbeatSimulation();

    console.log('\n✅ 所有模拟节点已启动');
    console.log('💡 提示: 按 Ctrl+C 停止所有节点\n');
  }

  private async startNode(config: typeof SIMULATED_NODES[0], index: number) {
    const nodeId = uuidv4();
    
    const node: SimulatedNode = {
      nodeId,
      name: config.name,
      os: config.os,
      cpuCores: config.cpuCores,
      totalMemory: config.memory,
      availableMemory: config.memory,
      socket: null,
      shards: [],
      status: 'connecting'
    };

    this.nodes.set(nodeId, node);

    console.log(`[${index + 1}/${SIMULATED_NODES.length}] 启动节点: ${node.name}`);
    console.log(`    OS: ${node.os}, CPU: ${node.cpuCores}核, 内存: ${node.totalMemory / 1024}GB`);

    return new Promise<void>((resolve) => {
      const socket = io(this.serverUrl, {
        transports: ['websocket', 'polling']
      });

      node.socket = socket;

      socket.on('connect', () => {
        console.log(`    ✅ 已连接`);
        
        // 注册节点
        socket.emit('node:register', {
          nodeId: node.nodeId,
          name: node.name,
          os: node.os,
          cpuCores: node.cpuCores,
          totalMemory: node.totalMemory,
          availableMemory: node.availableMemory
        });
      });

      socket.on('node:registered', (data: any) => {
        console.log(`    📝 已注册: ${data.nodeId.slice(0, 8)}...`);
        node.status = 'online';
        resolve();
      });

      socket.on('shard:assign', (data: any) => {
        console.log(`    📥 收到分片: ${data.shardId} (层 ${data.layerStart}-${data.layerEnd})`);
        node.shards.push(data.shardId);
        
        // 模拟加载延迟
        const loadTime = 1000 + Math.random() * 2000;
        setTimeout(() => {
          socket.emit('shard:loaded', { shardId: data.shardId });
          console.log(`    ✅ 分片加载完成: ${data.shardId}`);
          
          // 更新可用内存
          const usedMemory = data.size * 1024 * 1.1; // GB to MB + 10% overhead
          node.availableMemory = Math.max(512, node.availableMemory - usedMemory);
        }, loadTime);
      });

      socket.on('shard:migrate', (data: any) => {
        console.log(`    🔄 收到迁移请求: ${data.shardId}`);
        
        setTimeout(() => {
          node.shards.push(data.shardId);
          socket.emit('shard:migrated', { shardId: data.shardId });
          console.log(`    ✅ 迁移完成: ${data.shardId}`);
        }, 2000);
      });

      socket.on('task:inference', (data: any) => {
        console.log(`    🎯 收到推理任务: ${data.taskId.slice(0, 8)}...`);
        node.status = 'busy';

        // 模拟推理延迟
        const inferenceTime = 2000 + Math.random() * 3000;
        setTimeout(() => {
          const result = this.generateMockResponse(data.prompt, node);
          socket.emit('inference:result', {
            taskId: data.taskId,
            status: 'completed',
            result
          });
          node.status = 'online';
          console.log(`    ✅ 任务完成: ${data.taskId.slice(0, 8)}...`);
        }, inferenceTime);
      });

      socket.on('disconnect', () => {
        console.log(`    ❌ 断开连接: ${node.name}`);
        node.status = 'offline';
      });

      socket.on('connect_error', (error: Error) => {
        console.log(`    ⚠️ 连接错误: ${error.message}`);
      });
    });
  }

  private generateMockResponse(prompt: string, node: SimulatedNode): string {
    const responses = [
      `🤖 来自 ${node.name} 的响应\n\n您的输入: "${prompt.slice(0, 50)}..."\n\n这是一个模拟的分布式推理响应。系统正在使用 Pipeline 并行处理，您的请求被分配到了多个节点协同完成。\n\n节点信息:\n- 操作系统: ${node.os}\n- CPU核心: ${node.cpuCores}\n- 内存: ${node.totalMemory / 1024}GB\n- 承载分片: ${node.shards.length}个`,
      
      `✨ 分布式推理结果\n\n感谢您测试分布式大模型推理系统！\n\n您的查询已通过 ${node.shards.length} 个分片节点处理完成。系统采用了 Pipeline 并行技术，将模型按层切分到不同节点，实现了跨设备的协同推理。\n\n当前节点负载: ${node.cpuCores}核 CPU, ${(node.availableMemory / 1024).toFixed(1)}GB 可用内存`,
      
      `🎯 推理完成\n\n输入: "${prompt.slice(0, 100)}..."\n\n本响应由分布式推理系统生成。您的请求经过了多个计算节点的 Pipeline 处理，每个节点负责模型的不同层。\n\n系统特性:\n- 自动故障恢复\n- 动态负载均衡\n- 分片冗余存储\n- 跨平台支持`,
    ];

    return responses[Math.floor(Math.random() * responses.length)];
  }

  private startHeartbeatSimulation() {
    setInterval(() => {
      this.nodes.forEach((node) => {
        if (node.socket && node.socket.connected) {
          // 模拟内存波动
          const memoryVariation = (Math.random() - 0.5) * 200;
          node.availableMemory = Math.max(
            512,
            Math.min(node.totalMemory, node.availableMemory + memoryVariation)
          );

          // 模拟负载分数
          const loadScore = 10 + Math.random() * 30 + (node.status === 'busy' ? 30 : 0);

          node.socket.emit('node:heartbeat', {
            availableMemory: Math.round(node.availableMemory),
            status: node.status,
            loadScore: loadScore
          });
        }
      });
    }, 30000); // 每30秒发送心跳
  }

  async stop() {
    console.log('\n🛑 停止所有模拟节点...');
    this.running = false;
    
    this.nodes.forEach((node) => {
      if (node.socket) {
        node.socket.disconnect();
      }
    });

    console.log('✅ 所有模拟节点已停止');
  }
}

// 主程序
const serverUrl = process.argv[2] || 'http://localhost:3003';

const service = new SimulatedNodeService(serverUrl);

process.on('SIGINT', async () => {
  await service.stop();
  process.exit(0);
});

service.start().catch(console.error);
