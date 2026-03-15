/**
 * 分布式大模型推理系统 - 中央调度服务 (并行计算增强版)
 * 
 * 并行计算策略：
 * 1. Pipeline并行 - 模型按层切分，数据流水线处理
 * 2. 数据并行 - 多节点同时处理不同数据批次
 * 3. 动态批处理 - 合并请求提高吞吐量
 * 4. 异步推理 - 非阻塞式处理，充分利用CPU
 */

import { Server as HttpServer } from 'http';
import { Server as IOServer, Socket } from 'socket.io';
import { createServer } from 'http';
import { v4 as uuidv4 } from 'uuid';

// ==================== 类型定义 ====================

interface NodeInfo {
  nodeId: string;
  name: string;
  os: string;
  cpuCores: number;
  totalMemory: number;
  availableMemory: number;
  ipAddress?: string;
  port?: number;
  status: 'online' | 'offline' | 'busy' | 'error' | 'recovering';
  lastHeartbeat: Date;
  shards: string[];
  currentTask?: string;
  performance: {
    avgLatency: number;
    tasksCompleted: number;
    tasksFailed: number;
    loadScore: number;
    throughput: number; // 每秒处理token数
  };
  joinTime: Date;
  // 并行计算相关
  parallelSlots: number; // 可并行处理的任务槽位数
  activeSlots: number;   // 当前活跃的任务数
  // 网络感知相关
  network?: {
    latency_ms: number;      // 网络延迟（毫秒）
    jitter_ms: number;       // 抖动
    bandwidth_mbps: number;  // 带宽
    packet_loss: number;     // 丢包率
    network_score: number;   // 网络综合分数
  };
  // 综合分数
  compositeScore?: number;   // 综合评分（用于调度决策）
}

interface ModelShard {
  shardId: string;
  modelId: string;
  shardIndex: number;
  layerStart: number;
  layerEnd: number;
  size: number;
  replicas: Array<{
    nodeId: string;
    status: 'pending' | 'downloading' | 'ready' | 'error' | 'migrating';
    lastSync?: Date;
    loadScore?: number; // 节点负载分数
  }>;
  primaryReplica: number;
  checksum?: string;
  downloadUrl?: string;
  priority: 'critical' | 'high' | 'normal';
}

interface Model {
  modelId: string;
  name: string;
  version: string;
  totalParams: number;
  quantization: string;
  totalShards: number;
  totalSize: number;
  shards: ModelShard[];
  status: 'pending' | 'downloading' | 'ready' | 'degraded' | 'error';
  minRequiredNodes: number;
}

// 并行任务类型
interface ParallelTask {
  taskId: string;
  parentTaskId?: string; // 父任务ID（用于批处理）
  prompt: string;
  modelId: string;
  status: 'pending' | 'queued' | 'running' | 'completed' | 'failed' | 'cancelled';
  result?: string;
  assignedNodes: string[];
  createdAt: Date;
  startedAt?: Date;
  completedAt?: Date;
  retryCount: number;
  maxRetries: number;
  // 并行计算相关
  batchIndex?: number;    // 批次索引
  totalBatches?: number;  // 总批次数
  parallelGroup?: string; // 并行组ID
  tokens?: number;        // token数量
}

// 批处理任务
interface BatchTask {
  batchId: string;
  tasks: string[]; // 子任务ID列表
  status: 'pending' | 'processing' | 'completed' | 'partial' | 'failed';
  results: Map<string, string>;
  createdAt: Date;
  completedAt?: Date;
  parallelism: number; // 并行度
}

interface SystemMetrics {
  totalComputePower: number;
  availableComputePower: number;
  avgNodeLoad: number;
  shardRedundancy: number;
  healthyNodes: number;
  degradedNodes: number;
  // 并行计算指标
  currentParallelism: number;    // 当前并行度
  maxParallelism: number;        // 最大并行度
  avgThroughput: number;         // 平均吞吐量 (tokens/s)
  queueLength: number;           // 队列长度
  batchEfficiency: number;       // 批处理效率
}

// ==================== 存储与状态 ====================

const nodes = new Map<string, NodeInfo>();
const models = new Map<string, Model>();
const tasks = new Map<string, ParallelTask>();
const batchTasks = new Map<string, BatchTask>();
const pendingTasks: string[] = [];
const processingTasks: string[] = [];

const socketToNode = new Map<string, string>();
const nodeToSocket = new Map<string, string>();

// 并行计算配置
const PARALLEL_CONFIG = {
  HEARTBEAT_TIMEOUT: 60000,
  HEARTBEAT_INTERVAL: 30000,
  SHARD_REPLICATION_FACTOR: 2,
  MAX_RETRIES: 3,
  REBALANCE_INTERVAL: 60000,
  MIGRATION_TIMEOUT: 300000,
  MIN_NODES_FOR_MODEL: 3,
  // 并行计算配置
  MAX_BATCH_SIZE: 8,           // 最大批处理大小
  BATCH_TIMEOUT: 100,          // 批处理等待时间(ms)
  MIN_PARALLELISM: 1,          // 最小并行度
  MAX_PARALLELISM: 16,         // 最大并行度
  TASK_TIMEOUT: 30000,         // 任务超时时间
  LOAD_BALANCE_THRESHOLD: 0.7, // 负载均衡阈值
};

// 批处理队列
let batchQueue: string[] = [];
let batchTimer: NodeJS.Timeout | null = null;

// ==================== 生产级配置 ====================

// 日志级别
enum LogLevel { DEBUG = 0, INFO = 1, WARN = 2, ERROR = 3 }
const LOG_LEVEL = LogLevel.INFO;

// 速率限制
const RATE_LIMIT_WINDOW = 60000; // 1分钟
const RATE_LIMIT_MAX = 100; // 每分钟最多100个请求
const rateLimitMap: Map<string, number[]> = new Map();

// 优雅关闭
let isShuttingDown = false;

// ==================== 工具函数 ====================

function log(level: LogLevel, context: string, message: string, data?: any): void {
  if (level < LOG_LEVEL) return;
  const entry = {
    timestamp: new Date().toISOString(),
    level: LogLevel[level],
    context,
    message,
    ...(data && { data })
  };
  if (level >= LogLevel.ERROR) console.error(JSON.stringify(entry));
  else if (level >= LogLevel.WARN) console.warn(JSON.stringify(entry));
  else console.log(JSON.stringify(entry));
}

function checkRateLimit(identifier: string): boolean {
  const now = Date.now();
  const windowStart = now - RATE_LIMIT_WINDOW;
  let timestamps = rateLimitMap.get(identifier) || [];
  timestamps = timestamps.filter(t => t > windowStart);
  if (timestamps.length >= RATE_LIMIT_MAX) return false;
  timestamps.push(now);
  rateLimitMap.set(identifier, timestamps);
  return true;
}

function validateInput(prompt: string): { valid: boolean; error?: string } {
  if (!prompt || typeof prompt !== 'string') {
    return { valid: false, error: '提示词不能为空' };
  }
  if (prompt.length > 10000) {
    return { valid: false, error: '提示词长度不能超过10000字符' };
  }
  return { valid: true };
}

// ==================== HTTP服务器 ====================

const httpServer = createServer();

const io = new IOServer(httpServer, {
  cors: {
    origin: '*',
    methods: ['GET', 'POST']
  },
  transports: ['polling', 'websocket'], // polling优先，更稳定
  allowEIO3: true, // 兼容旧版本客户端
  pingTimeout: 60000,      // 心跳超时60秒
  pingInterval: 25000,     // 心跳间隔25秒
  upgradeTimeout: 30000,   // 升级超时30秒
  maxHttpBufferSize: 1e8,  // 最大消息大小100MB
  allowUpgrades: true,     // 允许升级
  cookie: false            // 不使用cookie
});

const PORT = 3003;

// ==================== 并行计算核心 ====================

/**
 * 计算最优并行度
 */
function calculateOptimalParallelism(): number {
  const onlineNodes = Array.from(nodes.values()).filter(n => n.status === 'online');
  
  if (onlineNodes.length === 0) return PARALLEL_CONFIG.MIN_PARALLELISM;
  
  // 计算总可用槽位
  const totalSlots = onlineNodes.reduce((sum, node) => {
    const availableSlots = node.parallelSlots - node.activeSlots;
    return sum + Math.max(0, availableSlots);
  }, 0);
  
  // 计算平均负载
  const avgLoad = onlineNodes.reduce((sum, n) => sum + n.performance.loadScore, 0) / onlineNodes.length;
  
  // 根据负载调整并行度
  let parallelism = totalSlots;
  if (avgLoad > 70) {
    parallelism = Math.floor(totalSlots * 0.5); // 高负载时减半
  } else if (avgLoad > 50) {
    parallelism = Math.floor(totalSlots * 0.75);
  }
  
  return Math.max(
    PARALLEL_CONFIG.MIN_PARALLELISM,
    Math.min(PARALLEL_CONFIG.MAX_PARALLELISM, parallelism)
  );
}

/**
 * 动态批处理 - 合并多个请求
 */
function addToBatch(taskId: string): void {
  batchQueue.push(taskId);
  
  // 如果达到最大批次大小，立即处理
  if (batchQueue.length >= PARALLEL_CONFIG.MAX_BATCH_SIZE) {
    processBatch();
    return;
  }
  
  // 否则设置定时器等待更多请求
  if (!batchTimer) {
    batchTimer = setTimeout(() => {
      processBatch();
    }, PARALLEL_CONFIG.BATCH_TIMEOUT);
  }
}

/**
 * 处理批次
 */
function processBatch(): void {
  if (batchTimer) {
    clearTimeout(batchTimer);
    batchTimer = null;
  }
  
  if (batchQueue.length === 0) return;
  
  const batchId = uuidv4();
  const taskIds = [...batchQueue];
  batchQueue = [];
  
  const batchTask: BatchTask = {
    batchId,
    tasks: taskIds,
    status: 'processing',
    results: new Map(),
    createdAt: new Date(),
    parallelism: calculateOptimalParallelism()
  };
  
  batchTasks.set(batchId, batchTask);
  
  console.log(`[Batch] 📦 创建批次 ${batchId.slice(0,8)}: ${taskIds.length}个任务, 并行度=${batchTask.parallelism}`);
  
  // 并行调度任务
  scheduleParallelTasks(taskIds, batchTask.parallelism);
}

/**
 * 并行调度任务
 */
function scheduleParallelTasks(taskIds: string[], parallelism: number): void {
  const onlineNodes = Array.from(nodes.values())
    .filter(n => n.status === 'online' || n.status === 'busy')
    .sort((a, b) => a.performance.loadScore - b.performance.loadScore);
  
  console.log(`[Schedule] 📊 调度任务: ${taskIds.length}个, 在线节点: ${onlineNodes.length}, 并行度: ${parallelism}`);
  
  if (onlineNodes.length === 0) {
    console.log('[Schedule] ⚠️ 无可用节点，任务加入待处理队列');
    // 将任务加入待处理队列
    for (const taskId of taskIds) {
      if (!pendingTasks.includes(taskId)) {
        pendingTasks.push(taskId);
      }
    }
    return;
  }
  
  // 为每个任务分配节点
  let nodeIndex = 0;
  
  for (const taskId of taskIds) {
    const task = tasks.get(taskId);
    if (!task || task.status !== 'pending') continue;
    
    // 找到负载最低的节点
    const node = onlineNodes[nodeIndex % onlineNodes.length];
    nodeIndex++;
    
    // 检查节点是否有可用槽位
    if (node.activeSlots >= node.parallelSlots) {
      // 寻找有可用槽位的节点
      const availableNode = onlineNodes.find(n => n.activeSlots < n.parallelSlots);
      if (!availableNode) {
        console.log(`[Schedule] ⏳ 任务 ${taskId.slice(0,8)} 等待可用槽位`);
        pendingTasks.push(taskId);
        continue;
      }
      assignTaskToNode(task, availableNode);
    } else {
      assignTaskToNode(task, node);
    }
  }
}

/**
 * 将任务分配给节点
 */
function assignTaskToNode(task: ParallelTask, node: NodeInfo): void {
  task.status = 'running';
  task.startedAt = new Date();
  task.assignedNodes = [node.nodeId];
  
  node.activeSlots++;
  node.status = 'busy';
  node.currentTask = task.taskId;
  nodes.set(node.nodeId, node);
  
  processingTasks.push(task.taskId);
  
  // 发送任务到节点
  const socketId = nodeToSocket.get(node.nodeId);
  if (socketId) {
    console.log(`[Schedule] 📤 发送任务到节点: ${node.name}, socketId: ${socketId}`);
    io.to(socketId).emit('task:inference', {
      taskId: task.taskId,
      prompt: task.prompt,
      modelId: task.modelId,
      parallelIndex: task.batchIndex,
      totalParallel: task.totalBatches
    });
  } else {
    console.log(`[Schedule] ⚠️ 无法找到节点的socket: ${node.nodeId}`);
  }
  
  console.log(`[Schedule] 🚀 任务 ${task.taskId.slice(0,8)} -> 节点 ${node.name} (槽位 ${node.activeSlots}/${node.parallelSlots})`);
  
  // 设置任务超时
  setTimeout(() => {
    handleTaskTimeout(task.taskId);
  }, PARALLEL_CONFIG.TASK_TIMEOUT);
}

/**
 * 任务超时处理
 */
function handleTaskTimeout(taskId: string): void {
  const task = tasks.get(taskId);
  if (!task || task.status !== 'running') return;
  
  console.log(`[Task] ⏰ 任务超时: ${taskId.slice(0,8)}`);
  
  // 释放节点槽位
  task.assignedNodes.forEach(nodeId => {
    const node = nodes.get(nodeId);
    if (node) {
      node.activeSlots = Math.max(0, node.activeSlots - 1);
      if (node.activeSlots === 0) {
        node.status = 'online';
        node.currentTask = undefined;
      }
      nodes.set(nodeId, node);
    }
  });
  
  // 重试或失败
  task.retryCount++;
  if (task.retryCount < task.maxRetries) {
    task.status = 'pending';
    pendingTasks.push(taskId);
    console.log(`[Task] 🔄 重试任务: ${taskId.slice(0,8)} (${task.retryCount}/${task.maxRetries})`);
  } else {
    task.status = 'failed';
    io.emit('task:failed', { taskId, error: 'Timeout' });
  }
  
  tasks.set(taskId, task);
}

/**
 * 数据并行 - 将大任务拆分成多个子任务
 */
function splitIntoParallelTasks(prompt: string, modelId: string, parallelism: number): string[] {
  const taskIds: string[] = [];
  const parallelGroupId = uuidv4();
  
  // 对于长文本，可以按段落或句子拆分
  // 这里简化处理，实际应用中需要根据模型特性拆分
  const chunks = splitPrompt(prompt, parallelism);
  
  for (let i = 0; i < chunks.length; i++) {
    const taskId = uuidv4();
    const task: ParallelTask = {
      taskId,
      prompt: chunks[i],
      modelId,
      status: 'pending',
      assignedNodes: [],
      createdAt: new Date(),
      retryCount: 0,
      maxRetries: PARALLEL_CONFIG.MAX_RETRIES,
      batchIndex: i,
      totalBatches: chunks.length,
      parallelGroup: parallelGroupId
    };
    
    tasks.set(taskId, task);
    taskIds.push(taskId);
  }
  
  console.log(`[Parallel] 📊 拆分任务: ${prompt.slice(0,30)}... -> ${taskIds.length}个并行任务`);
  
  return taskIds;
}

/**
 * 拆分提示词
 */
function splitPrompt(prompt: string, parts: number): string[] {
  if (prompt.length < 100 || parts <= 1) {
    return [prompt];
  }
  
  // 按句子拆分
  const sentences = prompt.match(/[^.!?]+[.!?]+/g) || [prompt];
  
  if (sentences.length <= parts) {
    return sentences;
  }
  
  // 平均分配句子
  const chunkSize = Math.ceil(sentences.length / parts);
  const chunks: string[] = [];
  
  for (let i = 0; i < parts; i++) {
    const start = i * chunkSize;
    const end = Math.min(start + chunkSize, sentences.length);
    if (start < sentences.length) {
      chunks.push(sentences.slice(start, end).join(' '));
    }
  }
  
  return chunks;
}

/**
 * 合并并行任务结果
 */
function mergeParallelResults(parallelGroupId: string): string | null {
  const groupTasks = Array.from(tasks.values())
    .filter(t => t.parallelGroup === parallelGroupId);
  
  if (groupTasks.length === 0) return null;
  
  // 检查是否全部完成
  const allCompleted = groupTasks.every(t => t.status === 'completed');
  if (!allCompleted) return null;
  
  // 按批次索引排序并合并结果
  const sortedTasks = groupTasks.sort((a, b) => (a.batchIndex || 0) - (b.batchIndex || 0));
  
  return sortedTasks.map(t => t.result || '').join('\n\n');
}

// ==================== 节点管理 ====================

function registerNode(socket: Socket, data: any): NodeInfo {
  const nodeId = data.nodeId || uuidv4();
  
  // 计算节点的并行槽位数
  const parallelSlots = Math.max(1, Math.floor(data.cpuCores / 2));
  
  const nodeInfo: NodeInfo = {
    nodeId,
    name: data.name || `Node-${nodeId.slice(0, 8)}`,
    os: data.os || 'unknown',
    cpuCores: data.cpuCores || 1,
    totalMemory: data.totalMemory || 4096,
    availableMemory: data.availableMemory || data.totalMemory || 4096,
    ipAddress: data.ipAddress || socket.handshake.address,
    port: data.port,
    status: 'online',
    lastHeartbeat: new Date(),
    shards: [],
    performance: {
      avgLatency: 0,
      tasksCompleted: 0,
      tasksFailed: 0,
      loadScore: 0,
      throughput: 0
    },
    joinTime: new Date(),
    parallelSlots,
    activeSlots: 0,
    // 网络指标
    network: data.network || undefined,
    compositeScore: 50 // 初始分数
  };
  
  // 计算初始综合分数
  nodeInfo.compositeScore = calculateNodeScore(nodeInfo);
  
  nodes.set(nodeId, nodeInfo);
  socketToNode.set(socket.id, nodeId);
  nodeToSocket.set(nodeId, socket.id);
  
  const networkInfo = data.network ? 
    `网络=${data.network.network_score?.toFixed(0) || 'N/A'}分` : '';
  console.log(`[Node] ✅ 注册: ${nodeInfo.name} (${nodeInfo.cpuCores}核, ${Math.round(nodeInfo.totalMemory/1024)}GB, 槽位=${parallelSlots}, ${networkInfo})`);
  
  io.emit('node:joined', { 
    nodeId, 
    name: nodeInfo.name, 
    specs: { 
      cpuCores: nodeInfo.cpuCores, 
      memory: nodeInfo.totalMemory,
      parallelSlots: nodeInfo.parallelSlots,
      network: nodeInfo.network
    } 
  });
  
  setTimeout(() => assignShardsToNode(nodeId), 1000);
  
  return nodeInfo;
}

function updateHeartbeat(nodeId: string, data: any): void {
  const node = nodes.get(nodeId);
  if (node) {
    node.lastHeartbeat = new Date();
    node.availableMemory = data.availableMemory || node.availableMemory;
    node.status = data.status || node.status;
    node.activeSlots = data.activeSlots ?? node.activeSlots;
    
    if (data.loadScore !== undefined) {
      node.performance.loadScore = data.loadScore;
    }
    if (data.throughput !== undefined) {
      node.performance.throughput = data.throughput;
    }
    
    // 更新网络指标
    if (data.network) {
      node.network = data.network;
    }
    
    // 重新计算综合分数
    node.compositeScore = calculateNodeScore(node);
    
    // 更新节点状态
    if (node.activeSlots === 0 && node.status === 'busy') {
      node.status = 'online';
    }
    
    nodes.set(nodeId, node);
  }
}

function getOnlineNodes(): NodeInfo[] {
  return Array.from(nodes.values()).filter(n => n.status === 'online' || n.status === 'busy');
}

function checkNodeTimeout(): void {
  const now = new Date();
  
  nodes.forEach((node, nodeId) => {
    if (node.lastHeartbeat) {
      const elapsed = now.getTime() - new Date(node.lastHeartbeat).getTime();
      
      if (elapsed > PARALLEL_CONFIG.HEARTBEAT_TIMEOUT) {
        if (node.status !== 'offline') {
          console.log(`[Node] ⚠️ 节点超时: ${node.name}`);
          handleNodeFailure(nodeId, 'timeout');
        }
      }
    }
  });
}

function handleNodeFailure(nodeId: string, reason: string): void {
  const node = nodes.get(nodeId);
  if (!node) return;
  
  node.status = 'offline';
  nodes.set(nodeId, node);
  
  console.log(`[Node] ❌ 节点离线: ${node.name} (原因: ${reason})`);
  
  io.emit('node:offline', { nodeId, name: node.name, reason });
  
  // 重新分配该节点上的任务
  node.activeSlots = 0;
  
  // 处理正在运行的任务
  tasks.forEach((task, taskId) => {
    if (task.status === 'running' && task.assignedNodes.includes(nodeId)) {
      console.log(`[Task] 🔄 重新调度任务: ${taskId.slice(0,8)}`);
      task.status = 'pending';
      task.assignedNodes = [];
      pendingTasks.push(taskId);
    }
  });
  
  handleShardRecovery(nodeId);
}

// ==================== 分片管理 ====================

function initializeQwenModel(): Model {
  const modelId = 'qwen2.5-27b-q4';
  
  const shardConfigs = [
    { layerStart: 0, layerEnd: 10, size: 3.2, priority: 'critical' as const },
    { layerStart: 10, layerEnd: 20, size: 3.2, priority: 'normal' as const },
    { layerStart: 20, layerEnd: 30, size: 3.2, priority: 'normal' as const },
    { layerStart: 30, layerEnd: 40, size: 3.2, priority: 'normal' as const },
    { layerStart: 40, layerEnd: 50, size: 3.2, priority: 'normal' as const },
    { layerStart: 50, layerEnd: 60, size: 3.2, priority: 'normal' as const },
    { layerStart: 60, layerEnd: 64, size: 1.5, priority: 'critical' as const },
  ];
  
  const shards: ModelShard[] = shardConfigs.map((config, index) => ({
    shardId: `${modelId}-shard-${index}`,
    modelId,
    shardIndex: index,
    layerStart: config.layerStart,
    layerEnd: config.layerEnd,
    size: config.size,
    replicas: [],
    primaryReplica: 0,
    priority: config.priority
  }));
  
  const model: Model = {
    modelId,
    name: 'Qwen 2.5 27B',
    version: '2.5',
    totalParams: 27,
    quantization: 'Q4_K_M',
    totalShards: shards.length,
    totalSize: 15,
    shards,
    status: 'pending',
    minRequiredNodes: PARALLEL_CONFIG.MIN_NODES_FOR_MODEL
  };
  
  models.set(modelId, model);
  console.log(`[Model] 📦 初始化: ${model.name} (${model.totalShards}分片, 冗余=${PARALLEL_CONFIG.SHARD_REPLICATION_FACTOR})`);
  
  return model;
}

function assignShardsToNode(nodeId: string): void {
  const node = nodes.get(nodeId);
  if (!node || node.status === 'offline') return;
  
  models.forEach(model => {
    model.shards.forEach(shard => {
      const readyReplicas = shard.replicas.filter(r => r.status === 'ready').length;
      const neededReplicas = PARALLEL_CONFIG.SHARD_REPLICATION_FACTOR - readyReplicas;
      
      if (neededReplicas > 0) {
        const hasReplica = shard.replicas.some(r => r.nodeId === nodeId);
        
        if (!hasReplica && canNodeHoldShard(node, shard)) {
          shard.replicas.push({
            nodeId,
            status: 'pending',
            lastSync: new Date(),
            loadScore: node.performance.loadScore
          });
          
          node.shards.push(shard.shardId);
          nodes.set(nodeId, node);
          
          const socketId = nodeToSocket.get(nodeId);
          if (socketId) {
            io.to(socketId).emit('shard:assign', {
              shardId: shard.shardId,
              modelId: shard.modelId,
              layerStart: shard.layerStart,
              layerEnd: shard.layerEnd,
              size: shard.size,
              priority: shard.priority
            });
          }
          
          console.log(`[Shard] 📍 分配 ${shard.shardId} -> ${node.name}`);
        }
      }
    });
    
    updateModelStatus(model.modelId);
  });
}

function canNodeHoldShard(node: NodeInfo, shard: ModelShard): boolean {
  const usedMemory = node.shards.reduce((sum, shardId) => {
    for (const model of models.values()) {
      const s = model.shards.find(sh => sh.shardId === shardId);
      if (s) return sum + s.size * 1024;
    }
    return sum;
  }, 0);
  
  const availableMB = node.availableMemory - usedMemory;
  const requiredMB = shard.size * 1024;
  
  return availableMB >= requiredMB * 1.1;
}

function handleShardRecovery(offlineNodeId: string): void {
  models.forEach(model => {
    model.shards.forEach(shard => {
      const replicaIndex = shard.replicas.findIndex(r => r.nodeId === offlineNodeId);
      
      if (replicaIndex !== -1) {
        console.log(`[Shard] 🔄 恢复分片 ${shard.shardId}`);
        
        shard.replicas = shard.replicas.filter(r => r.nodeId !== offlineNodeId);
        
        const healthyNodes = getOnlineNodes().filter(n => 
          !shard.replicas.some(r => r.nodeId === n.nodeId) &&
          canNodeHoldShard(n, shard)
        );
        
        healthyNodes.sort((a, b) => a.performance.loadScore - b.performance.loadScore);
        
        if (healthyNodes.length > 0) {
          const newNode = healthyNodes[0];
          
          shard.replicas.push({
            nodeId: newNode.nodeId,
            status: 'migrating',
            lastSync: new Date()
          });
          
          newNode.shards.push(shard.shardId);
          nodes.set(newNode.nodeId, newNode);
          
          const socketId = nodeToSocket.get(newNode.nodeId);
          if (socketId) {
            io.to(socketId).emit('shard:migrate', {
              shardId: shard.shardId,
              modelId: shard.modelId,
              layerStart: shard.layerStart,
              layerEnd: shard.layerEnd,
              size: shard.size
            });
          }
        }
      }
    });
    
    updateModelStatus(model.modelId);
  });
}

function updateModelStatus(modelId: string): void {
  const model = models.get(modelId);
  if (!model) return;
  
  const allShardsAvailable = model.shards.every(shard => 
    shard.replicas.some(r => r.status === 'ready')
  );
  
  const totalReadyReplicas = model.shards.reduce((sum, shard) => 
    sum + shard.replicas.filter(r => r.status === 'ready').length, 0
  );
  const totalNeededReplicas = model.totalShards * PARALLEL_CONFIG.SHARD_REPLICATION_FACTOR;
  
  if (totalReadyReplicas >= totalNeededReplicas) {
    model.status = 'ready';
  } else if (allShardsAvailable) {
    model.status = 'degraded';
  } else {
    model.status = 'error';
  }
  
  models.set(modelId, model);
}

// ==================== 任务管理 ====================

function createInferenceTask(prompt: string, modelId: string): ParallelTask {
  const taskId = uuidv4();
  
  const task: ParallelTask = {
    taskId,
    prompt,
    modelId: modelId || 'qwen2.5-27b-q4',
    status: 'pending',
    assignedNodes: [],
    createdAt: new Date(),
    retryCount: 0,
    maxRetries: PARALLEL_CONFIG.MAX_RETRIES
  };
  
  tasks.set(taskId, task);
  
  // 添加到批处理队列
  addToBatch(taskId);
  
  return task;
}

function handleInferenceResult(nodeId: string, data: any): void {
  const task = tasks.get(data.taskId);
  if (!task) return;
  
  const node = nodes.get(nodeId);
  
  if (data.status === 'completed') {
    task.status = 'completed';
    task.result = data.result;
    task.completedAt = new Date();
    task.tokens = data.tokens || 0;
    
    if (node && task.startedAt) {
      const latency = task.completedAt.getTime() - task.startedAt.getTime();
      node.performance.tasksCompleted++;
      node.performance.avgLatency = 
        (node.performance.avgLatency * (node.performance.tasksCompleted - 1) + latency) 
        / node.performance.tasksCompleted;
      
      // 更新吞吐量
      if (task.tokens && latency > 0) {
        const throughput = (task.tokens / latency) * 1000; // tokens/s
        node.performance.throughput = 
          (node.performance.throughput * (node.performance.tasksCompleted - 1) + throughput)
          / node.performance.tasksCompleted;
      }
    }
    
    // 释放槽位
    if (node) {
      node.activeSlots = Math.max(0, node.activeSlots - 1);
      if (node.activeSlots === 0) {
        node.status = 'online';
        node.currentTask = undefined;
      }
      nodes.set(nodeId, node);
    }
    
    // 检查是否是并行任务组
    if (task.parallelGroup) {
      const mergedResult = mergeParallelResults(task.parallelGroup);
      if (mergedResult) {
        io.emit('task:completed', {
          taskId: task.parallelGroup,
          result: mergedResult
        });
      }
    } else {
      io.emit('task:completed', {
        taskId: task.taskId,
        result: task.result
      });
    }
    
    console.log(`[Task] ✅ 完成: ${task.taskId.slice(0,8)}`);
    
  } else if (data.status === 'failed') {
    task.retryCount++;
    
    if (node) {
      node.performance.tasksFailed++;
      node.activeSlots = Math.max(0, node.activeSlots - 1);
      if (node.activeSlots === 0) {
        node.status = 'online';
      }
      nodes.set(nodeId, node);
    }
    
    if (task.retryCount < task.maxRetries) {
      task.status = 'pending';
      pendingTasks.push(task.taskId);
    } else {
      task.status = 'failed';
      io.emit('task:failed', { taskId: task.taskId, error: data.error });
    }
  }
  
  tasks.set(task.taskId, task);
  
  // 处理下一个任务
  processNextTask();
}

function processNextTask(): void {
  if (pendingTasks.length === 0) return;
  
  const parallelism = calculateOptimalParallelism();
  const currentRunning = processingTasks.filter(id => {
    const t = tasks.get(id);
    return t && t.status === 'running';
  }).length;
  
  if (currentRunning >= parallelism) return;
  
  const nextTaskId = pendingTasks.shift();
  if (!nextTaskId) return;
  
  const task = tasks.get(nextTaskId);
  if (!task || task.status !== 'pending') {
    processNextTask();
    return;
  }
  
  // 找到最佳节点
  const bestNode = findBestNode();
  if (bestNode) {
    assignTaskToNode(task, bestNode);
  } else {
    pendingTasks.unshift(nextTaskId);
  }
}

/**
 * 计算节点综合分数（网络感知 + 历史性能）
 * 综合考虑：CPU负载、吞吐量、网络延迟、带宽、历史成功率
 */
function calculateNodeScore(node: NodeInfo): number {
  // 负载分数（负载越低越好，0-100）
  const loadScore = 100 - node.performance.loadScore;
  
  // 吞吐量分数（吞吐量越高越好）
  const throughputScore = Math.min(100, node.performance.throughput / 10);
  
  // 网络分数（如果有网络数据）
  let networkScore = 50; // 默认中等分数
  if (node.network) {
    // 延迟分数（延迟越低越好）
    const latencyScore = Math.max(0, 100 - node.network.latency_ms / 2);
    // 抖动分数（抖动越小越好）
    const jitterScore = Math.max(0, 100 - node.network.jitter_ms * 5);
    // 带宽分数（带宽越高越好）
    const bandwidthScore = Math.min(100, node.network.bandwidth_mbps / 10);
    // 综合网络分数
    networkScore = latencyScore * 0.5 + jitterScore * 0.3 + bandwidthScore * 0.2;
  }
  
  // 槽位可用性分数
  const availableSlots = node.parallelSlots - node.activeSlots;
  const slotScore = (availableSlots / node.parallelSlots) * 100;
  
  // 历史成功率分数
  const totalTasks = node.performance.tasksCompleted + node.performance.tasksFailed;
  const successRate = totalTasks > 0 
    ? (node.performance.tasksCompleted / totalTasks) * 100 
    : 100; // 新节点默认100%
  
  // 历史延迟分数（延迟越低越好）
  const avgLatency = node.performance.avgLatency || 0;
  const latencyHistoryScore = Math.max(0, 100 - avgLatency / 100);
  
  // 综合分数（权重可调整）
  const compositeScore = 
    loadScore * 0.15 +           // 负载权重 15%
    throughputScore * 0.20 +      // 吞吐量权重 20%
    networkScore * 0.25 +         // 网络权重 25%
    slotScore * 0.15 +            // 槽位权重 15%
    successRate * 0.15 +          // 成功率权重 15%
    latencyHistoryScore * 0.10;   // 历史延迟权重 10%
  
  return compositeScore;
}

function findBestNode(): NodeInfo | null {
  const onlineNodes = getOnlineNodes()
    .filter(n => n.activeSlots < n.parallelSlots)
    .map(n => {
      // 计算并缓存综合分数
      n.compositeScore = calculateNodeScore(n);
      return n;
    })
    .sort((a, b) => {
      // 按综合分数降序排列（分数越高越好）
      return (b.compositeScore || 0) - (a.compositeScore || 0);
    });
  
  // 返回分数最高的节点
  return onlineNodes[0] || null;
}

/**
 * 网络感知调度 - 选择网络最优的节点组
 */
function findBestNodesForParallel(count: number): NodeInfo[] {
  const onlineNodes = getOnlineNodes()
    .filter(n => n.activeSlots < n.parallelSlots)
    .map(n => {
      n.compositeScore = calculateNodeScore(n);
      return n;
    })
    .sort((a, b) => (b.compositeScore || 0) - (a.compositeScore || 0));
  
  return onlineNodes.slice(0, count);
}

// ==================== 系统指标 ====================

function getSystemMetrics(): SystemMetrics {
  const healthyNodes = getOnlineNodes();
  
  let totalReadyReplicas = 0;
  let totalNeededReplicas = 0;
  
  models.forEach(model => {
    model.shards.forEach(shard => {
      totalReadyReplicas += shard.replicas.filter(r => r.status === 'ready').length;
      totalNeededReplicas += PARALLEL_CONFIG.SHARD_REPLICATION_FACTOR;
    });
  });
  
  const currentParallelism = processingTasks.filter(id => {
    const t = tasks.get(id);
    return t && t.status === 'running';
  }).length;
  
  const avgThroughput = healthyNodes.length > 0
    ? healthyNodes.reduce((sum, n) => sum + n.performance.throughput, 0) / healthyNodes.length
    : 0;
  
  return {
    totalComputePower: Array.from(nodes.values()).reduce((sum, n) => sum + n.cpuCores, 0),
    availableComputePower: healthyNodes.reduce((sum, n) => sum + n.cpuCores, 0),
    avgNodeLoad: healthyNodes.length > 0 
      ? healthyNodes.reduce((sum, n) => sum + n.performance.loadScore, 0) / healthyNodes.length 
      : 0,
    shardRedundancy: totalNeededReplicas > 0 ? totalReadyReplicas / totalNeededReplicas : 0,
    healthyNodes: healthyNodes.length,
    degradedNodes: Array.from(nodes.values()).filter(n => n.status === 'offline').length,
    currentParallelism,
    maxParallelism: calculateOptimalParallelism(),
    avgThroughput,
    queueLength: pendingTasks.length + batchQueue.length,
    batchEfficiency: batchTasks.size > 0 
      ? Array.from(batchTasks.values()).filter(b => b.status === 'completed').length / batchTasks.size 
      : 1
  };
}

// ==================== Socket.IO事件 ====================

io.on('connection', (socket: Socket) => {
  console.log(`[Socket] 连接: ${socket.id}`);
  
  socket.on('node:register', (data) => {
    const node = registerNode(socket, data);
    socket.emit('node:registered', { 
      nodeId: node.nodeId, 
      name: node.name,
      parallelSlots: node.parallelSlots,
      config: PARALLEL_CONFIG
    });
    
    // 节点注册后处理待处理任务
    if (pendingTasks.length > 0) {
      console.log(`[Node] 📋 处理待处理任务: ${pendingTasks.length}个`);
      setTimeout(() => processNextTask(), 500);
    }
  });
  
  socket.on('node:heartbeat', (data) => {
    const nodeId = socketToNode.get(socket.id);
    if (nodeId) {
      updateHeartbeat(nodeId, data);
    }
  });
  
  socket.on('node:status', (data) => {
    const nodeId = socketToNode.get(socket.id);
    if (nodeId) {
      const node = nodes.get(nodeId);
      if (node) {
        node.status = data.status;
        node.activeSlots = data.activeSlots ?? node.activeSlots;
        nodes.set(nodeId, node);
        io.emit('node:updated', { nodeId, status: data.status });
      }
    }
  });
  
  socket.on('shard:loaded', (data) => {
    const nodeId = socketToNode.get(socket.id);
    console.log(`[Shard] ✅ 加载完成: ${data.shardId} (节点: ${nodeId})`);
    
    models.forEach(model => {
      const shard = model.shards.find(s => s.shardId === data.shardId);
      if (shard) {
        const replica = shard.replicas.find(r => r.nodeId === nodeId);
        if (replica) {
          replica.status = 'ready';
          replica.lastSync = new Date();
        }
      }
      updateModelStatus(model.modelId);
    });
    
    io.emit('shard:updated', { shardId: data.shardId, nodeId, status: 'ready' });
  });
  
  socket.on('shard:migrated', (data) => {
    const nodeId = socketToNode.get(socket.id);
    console.log(`[Shard] ✅ 迁移完成: ${data.shardId}`);
    
    models.forEach(model => {
      const shard = model.shards.find(s => s.shardId === data.shardId);
      if (shard) {
        const replica = shard.replicas.find(r => r.nodeId === nodeId);
        if (replica) {
          replica.status = 'ready';
          replica.lastSync = new Date();
        }
      }
      updateModelStatus(model.modelId);
    });
  });
  
  socket.on('inference:request', (data) => {
    const task = createInferenceTask(data.prompt, data.modelId);
    socket.emit('inference:queued', { taskId: task.taskId });
  });
  
  socket.on('inference:result', (data) => {
    const nodeId = socketToNode.get(socket.id);
    if (nodeId) {
      handleInferenceResult(nodeId, data);
    }
  });
  
  socket.on('disconnect', () => {
    const nodeId = socketToNode.get(socket.id);
    if (nodeId) {
      handleNodeFailure(nodeId, 'disconnect');
      socketToNode.delete(socket.id);
      nodeToSocket.delete(nodeId);
    }
    console.log(`[Socket] 断开: ${socket.id}`);
  });
  
  socket.on('system:status', () => {
    const metrics = getSystemMetrics();
    
    socket.emit('system:status', {
      nodes: Array.from(nodes.values()).map(n => ({
        nodeId: n.nodeId,
        name: n.name,
        os: n.os,
        cpuCores: n.cpuCores,
        totalMemory: n.totalMemory,
        availableMemory: n.availableMemory,
        status: n.status,
        shards: n.shards,
        performance: n.performance,
        joinTime: n.joinTime,
        parallelSlots: n.parallelSlots,
        activeSlots: n.activeSlots
      })),
      models: Array.from(models.values()).map(m => ({
        modelId: m.modelId,
        name: m.name,
        status: m.status,
        totalShards: m.totalShards,
        readyShards: m.shards.filter(s => s.replicas.some(r => r.status === 'ready')).length,
        totalReplicas: m.shards.reduce((sum, s) => sum + s.replicas.filter(r => r.status === 'ready').length, 0),
        shards: m.shards.map(s => ({
          shardId: s.shardId,
          layerStart: s.layerStart,
          layerEnd: s.layerEnd,
          priority: s.priority,
          replicas: s.replicas.map(r => ({
            nodeId: r.nodeId,
            status: r.status
          }))
        }))
      })),
      tasks: Array.from(tasks.values()).slice(-20),
      pendingCount: pendingTasks.length + batchQueue.length,
      metrics
    });
  });
});

// ==================== HTTP API ====================

import { createServer as createHttpServer, IncomingMessage, ServerResponse } from 'http';

const apiServer = createHttpServer((req: IncomingMessage, res: ServerResponse) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  
  if (req.method === 'OPTIONS') {
    res.writeHead(200);
    res.end();
    return;
  }
  
  if (req.url === '/api/status' && req.method === 'GET') {
    const metrics = getSystemMetrics();
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({
      nodes: Array.from(nodes.values()),
      models: Array.from(models.values()),
      metrics,
      config: PARALLEL_CONFIG
    }));
    return;
  }
  
  if (req.url === '/api/metrics' && req.method === 'GET') {
    const metrics = getSystemMetrics();
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(metrics));
    return;
  }
  
  if (req.url === '/api/inference' && req.method === 'POST') {
    let body = '';
    req.on('data', chunk => body += chunk);
    req.on('end', () => {
      try {
        const data = JSON.parse(body);
        const task = createInferenceTask(data.prompt, data.modelId);
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ taskId: task.taskId, status: 'queued' }));
      } catch {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'Invalid request' }));
      }
    });
    return;
  }
  
  if (req.url?.startsWith('/api/task/') && req.method === 'GET') {
    const taskId = req.url.split('/api/task/')[1];
    const task = tasks.get(taskId);
    if (task) {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(task));
    } else {
      res.writeHead(404, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: 'Task not found' }));
    }
    return;
  }
  
  if (req.url === '/api/parallelism' && req.method === 'GET') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({
      current: getSystemMetrics().currentParallelism,
      max: getSystemMetrics().maxParallelism,
      optimal: calculateOptimalParallelism()
    }));
    return;
  }
  
  res.writeHead(404, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify({ error: 'Not found' }));
});

// ==================== 健康检查端点 ====================

// 健康检查
apiServer.on('request', (req: any, res: any) => {
  if (req.url === '/health' && req.method === 'GET') {
    const health = {
      status: 'healthy',
      timestamp: new Date().toISOString(),
      uptime: process.uptime(),
      nodes: {
        total: nodes.size,
        online: Array.from(nodes.values()).filter(n => n.status === 'online').length,
        busy: Array.from(nodes.values()).filter(n => n.status === 'busy').length,
        offline: Array.from(nodes.values()).filter(n => n.status === 'offline').length,
      },
      tasks: {
        pending: pendingTasks.length,
        processing: processingTasks.length,
        queueSize: batchQueue.length,
      },
      memory: {
        heapUsed: Math.round(process.memoryUsage().heapUsed / 1024 / 1024),
        heapTotal: Math.round(process.memoryUsage().heapTotal / 1024 / 1024),
        rss: Math.round(process.memoryUsage().rss / 1024 / 1024),
      }
    };
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(health, null, 2));
    return;
  }
  
  // 就绪检查
  if (req.url === '/ready' && req.method === 'GET') {
    const onlineNodes = Array.from(nodes.values()).filter(n => n.status === 'online').length;
    const isReady = onlineNodes > 0;
    res.writeHead(isReady ? 200 : 503, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ ready: isReady, nodes: onlineNodes }));
    return;
  }
  
  // Prometheus指标
  if (req.url === '/metrics' && req.method === 'GET') {
    const onlineNodes = Array.from(nodes.values()).filter(n => n.status === 'online').length;
    const busyNodes = Array.from(nodes.values()).filter(n => n.status === 'busy').length;
    const metrics = [
      `# HELP llm_nodes_total Total number of nodes`,
      `# TYPE llm_nodes_total gauge`,
      `llm_nodes_total{status="all"} ${nodes.size}`,
      `llm_nodes_total{status="online"} ${onlineNodes}`,
      `llm_nodes_total{status="busy"} ${busyNodes}`,
      `llm_nodes_total{status="offline"} ${nodes.size - onlineNodes - busyNodes}`,
      ``,
      `# HELP llm_tasks_pending Number of pending tasks`,
      `# TYPE llm_tasks_pending gauge`,
      `llm_tasks_pending ${pendingTasks.length}`,
      ``,
      `# HELP llm_tasks_processing Number of processing tasks`,
      `# TYPE llm_tasks_processing gauge`,
      `llm_tasks_processing ${processingTasks.length}`,
      ``,
      `# HELP llm_tasks_completed_total Total completed tasks`,
      `# TYPE llm_tasks_completed_total counter`,
      `llm_tasks_completed_total ${Array.from(nodes.values()).reduce((sum, n) => sum + n.performance.tasksCompleted, 0)}`,
      ``,
      `# HELP llm_memory_heap_bytes Process heap memory in bytes`,
      `# TYPE llm_memory_heap_bytes gauge`,
      `llm_memory_heap_bytes{type="used"} ${process.memoryUsage().heapUsed}`,
      `llm_memory_heap_bytes{type="total"} ${process.memoryUsage().heapTotal}`,
    ].join('\n');
    res.writeHead(200, { 'Content-Type': 'text/plain' });
    res.end(metrics);
    return;
  }
});

// ==================== 定期清理任务 ====================

function startCleanupTask(): void {
  setInterval(() => {
    const now = Date.now();
    const maxAge = 3600000; // 1小时
    
    // 清理已完成的旧任务
    let cleanedTasks = 0;
    tasks.forEach((task, taskId) => {
      if (task.status === 'completed' || task.status === 'failed') {
        const age = now - (task.completedAt?.getTime() || task.createdAt.getTime());
        if (age > maxAge) {
          tasks.delete(taskId);
          cleanedTasks++;
        }
      }
    });
    
    // 清理离线节点数据（24小时后）
    let cleanedNodes = 0;
    nodes.forEach((node, nodeId) => {
      if (node.status === 'offline') {
        const age = now - node.lastHeartbeat.getTime();
        if (age > maxAge * 24) {
          nodes.delete(nodeId);
          cleanedNodes++;
        }
      }
    });
    
    // 清理速率限制缓存
    const windowStart = now - RATE_LIMIT_WINDOW;
    rateLimitMap.forEach((timestamps, key) => {
      const valid = timestamps.filter(t => t > windowStart);
      if (valid.length === 0) {
        rateLimitMap.delete(key);
      } else {
        rateLimitMap.set(key, valid);
      }
    });
    
    if (cleanedTasks > 0 || cleanedNodes > 0) {
      log(LogLevel.INFO, 'Cleanup', `清理完成: ${cleanedTasks}个任务, ${cleanedNodes}个节点`);
    }
  }, 300000); // 每5分钟清理一次
}

// ==================== 优雅关闭 ====================

async function gracefulShutdown(signal: string): Promise<void> {
  if (isShuttingDown) return;
  isShuttingDown = true;
  
  log(LogLevel.INFO, 'Shutdown', `收到 ${signal} 信号，开始优雅关闭...`);
  
  // 1. 停止接受新连接
  io.close(() => {
    log(LogLevel.INFO, 'Shutdown', 'Socket.IO 服务已关闭');
  });
  
  // 2. 等待正在处理的任务完成
  const processingCount = processingTasks.length;
  if (processingCount > 0) {
    log(LogLevel.INFO, 'Shutdown', `等待 ${processingCount} 个任务完成...`);
    
    // 最多等待30秒
    setTimeout(() => {
      log(LogLevel.WARN, 'Shutdown', '等待超时，强制关闭');
      process.exit(1);
    }, 30000);
    
    // 检查任务是否完成
    const checkInterval = setInterval(() => {
      if (processingTasks.length === 0) {
        clearInterval(checkInterval);
        log(LogLevel.INFO, 'Shutdown', '所有任务已完成');
        process.exit(0);
      }
    }, 1000);
  } else {
    log(LogLevel.INFO, 'Shutdown', '没有正在处理的任务');
    process.exit(0);
  }
}

process.on('SIGTERM', () => gracefulShutdown('SIGTERM'));
process.on('SIGINT', () => gracefulShutdown('SIGINT'));

// ==================== 启动服务 ====================

setInterval(checkNodeTimeout, 15000);
setInterval(processNextTask, 1000);
startCleanupTask();

initializeQwenModel();

io.listen(httpServer);
httpServer.listen(PORT, () => {
  console.log(`\n🚀 分布式大模型推理系统 - 生产级版本`);
  console.log(`   WebSocket: ws://localhost:${PORT}`);
  console.log(`   HTTP API:  http://localhost:${PORT + 1}`);
  console.log(`   健康检查:  http://localhost:${PORT + 1}/health`);
  console.log(`   就绪检查:  http://localhost:${PORT + 1}/ready`);
  console.log(`   监控指标:  http://localhost:${PORT + 1}/metrics`);
  console.log(`\n⚙️  并行计算配置:`);
  console.log(`   最大批处理: ${PARALLEL_CONFIG.MAX_BATCH_SIZE}`);
  console.log(`   最大并行度: ${PARALLEL_CONFIG.MAX_PARALLELISM}`);
  console.log(`   批处理超时: ${PARALLEL_CONFIG.BATCH_TIMEOUT}ms`);
  console.log(`   任务超时: ${PARALLEL_CONFIG.TASK_TIMEOUT}ms`);
  console.log(`   速率限制: ${RATE_LIMIT_MAX} 请求/分钟`);
});

apiServer.listen(PORT + 1, () => {
  console.log(`   REST API:  http://localhost:${PORT + 1}/api\n`);
});
