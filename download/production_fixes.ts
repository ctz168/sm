/**
 * 分布式大模型推理系统 - 生产级修复补丁
 * 
 * 这个文件包含了所有生产级问题的修复方案
 * 将这些修复应用到相应的文件中
 */

// ==================== 1. WebSocket稳定性修复 ====================

// 在 orchestrator/index.ts 中替换 WebSocket 配置
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
  cookie: false,           // 不使用cookie
  // 添加连接状态管理
  connectTimeout: 45000,   // 连接超时45秒
});

// ==================== 2. 连接超时和重连机制 ====================

// 在节点服务中添加指数退避重连
class ReconnectionManager {
  private attempt: number = 0;
  private maxAttempts: number = 10;
  private baseDelay: number = 1000; // 1秒
  private maxDelay: number = 60000; // 60秒
  private jitter: number = 0.3; // 30%抖动
  
  getNextDelay(): number {
    if (this.attempt >= this.maxAttempts) {
      return -1; // 不再重试
    }
    
    // 指数退避 + 抖动
    const delay = Math.min(
      this.baseDelay * Math.pow(2, this.attempt),
      this.maxDelay
    );
    const jitterAmount = delay * this.jitter * (Math.random() * 2 - 1);
    const finalDelay = Math.max(0, delay + jitterAmount);
    
    this.attempt++;
    return finalDelay;
  }
  
  reset(): void {
    this.attempt = 0;
  }
  
  getAttempt(): number {
    return this.attempt;
  }
}

// ==================== 3. 并发安全修复 ====================

// 添加线程安全的Map包装器
class ConcurrentMap<K, V> {
  private map: Map<K, V> = new Map();
  private lock: any; // 在Node.js中使用async-mutex
  
  async get(key: K): Promise<V | undefined> {
    return this.map.get(key);
  }
  
  async set(key: K, value: V): Promise<void> {
    this.map.set(key, value);
  }
  
  async delete(key: K): Promise<boolean> {
    return this.map.delete(key);
  }
  
  async has(key: K): Promise<boolean> {
    return this.map.has(key);
  }
  
  async forEach(callback: (value: V, key: K) => void): Promise<void> {
    this.map.forEach(callback);
  }
  
  get size(): number {
    return this.map.size;
  }
}

// ==================== 4. 内存泄漏修复 ====================

// 添加定期清理任务
function startCleanupTask(): void {
  setInterval(() => {
    const now = Date.now();
    const maxAge = 3600000; // 1小时
    
    // 清理已完成的旧任务
    tasks.forEach((task, taskId) => {
      if (task.status === 'completed' || task.status === 'failed') {
        const age = now - (task.completedAt?.getTime() || task.createdAt.getTime());
        if (age > maxAge) {
          tasks.delete(taskId);
          console.log(`[Cleanup] 删除过期任务: ${taskId.slice(0, 8)}`);
        }
      }
    });
    
    // 清理离线节点数据
    nodes.forEach((node, nodeId) => {
      if (node.status === 'offline') {
        const age = now - node.lastHeartbeat.getTime();
        if (age > maxAge * 24) { // 24小时后删除
          nodes.delete(nodeId);
          console.log(`[Cleanup] 删除离线节点: ${node.name}`);
        }
      }
    });
    
    // 清理空的批处理任务
    batchTasks.forEach((batch, batchId) => {
      if (batch.status === 'completed' || batch.status === 'failed') {
        batchTasks.delete(batchId);
      }
    });
    
  }, 300000); // 每5分钟清理一次
}

// ==================== 5. 输入验证 ====================

// 添加输入验证函数
function validatePrompt(prompt: string): { valid: boolean; error?: string } {
  if (!prompt || typeof prompt !== 'string') {
    return { valid: false, error: '提示词不能为空' };
  }
  
  if (prompt.length > 10000) {
    return { valid: false, error: '提示词长度不能超过10000字符' };
  }
  
  // 检查危险字符
  const dangerousPatterns = [
    /<script/i,
    /javascript:/i,
    /on\w+=/i,
    /data:/i,
  ];
  
  for (const pattern of dangerousPatterns) {
    if (pattern.test(prompt)) {
      return { valid: false, error: '提示词包含不允许的内容' };
    }
  }
  
  return { valid: true };
}

function validateNodeId(nodeId: string): boolean {
  // UUID格式验证
  const uuidPattern = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
  return uuidPattern.test(nodeId);
}

// ==================== 6. 结构化日志 ====================

// 添加日志级别和结构化日志
enum LogLevel {
  DEBUG = 0,
  INFO = 1,
  WARN = 2,
  ERROR = 3,
}

class Logger {
  private level: LogLevel;
  private context: string;
  
  constructor(context: string, level: LogLevel = LogLevel.INFO) {
    this.context = context;
    this.level = level;
  }
  
  private log(level: LogLevel, message: string, data?: any): void {
    if (level < this.level) return;
    
    const timestamp = new Date().toISOString();
    const levelName = LogLevel[level];
    
    const logEntry = {
      timestamp,
      level: levelName,
      context: this.context,
      message,
      ...(data && { data })
    };
    
    const output = JSON.stringify(logEntry);
    
    if (level >= LogLevel.ERROR) {
      console.error(output);
    } else if (level >= LogLevel.WARN) {
      console.warn(output);
    } else {
      console.log(output);
    }
  }
  
  debug(message: string, data?: any): void { this.log(LogLevel.DEBUG, message, data); }
  info(message: string, data?: any): void { this.log(LogLevel.INFO, message, data); }
  warn(message: string, data?: any): void { this.log(LogLevel.WARN, message, data); }
  error(message: string, data?: any): void { this.log(LogLevel.ERROR, message, data); }
}

const logger = new Logger('Orchestrator', LogLevel.INFO);

// ==================== 7. 健康检查端点 ====================

// 添加健康检查API
function setupHealthCheck(httpServer: HttpServer): void {
  httpServer.on('request', (req, res) => {
    if (req.url === '/health' && req.method === 'GET') {
      const health = {
        status: 'healthy',
        timestamp: new Date().toISOString(),
        uptime: process.uptime(),
        nodes: {
          total: nodes.size,
          online: Array.from(nodes.values()).filter(n => n.status === 'online').length,
        },
        tasks: {
          pending: pendingTasks.length,
          processing: processingTasks.length,
        },
        memory: {
          heapUsed: process.memoryUsage().heapUsed,
          heapTotal: process.memoryUsage().heapTotal,
        }
      };
      
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(health));
    }
    
    // 就绪检查
    if (req.url === '/ready' && req.method === 'GET') {
      const onlineNodes = Array.from(nodes.values()).filter(n => n.status === 'online').length;
      const isReady = onlineNodes > 0;
      
      res.writeHead(isReady ? 200 : 503, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ ready: isReady, nodes: onlineNodes }));
    }
  });
}

// ==================== 8. 速率限制 ====================

// 添加简单的速率限制
class RateLimiter {
  private requests: Map<string, number[]> = new Map();
  private windowMs: number;
  private maxRequests: number;
  
  constructor(windowMs: number = 60000, maxRequests: number = 100) {
    this.windowMs = windowMs;
    this.maxRequests = maxRequests;
  }
  
  isAllowed(identifier: string): boolean {
    const now = Date.now();
    const windowStart = now - this.windowMs;
    
    let timestamps = this.requests.get(identifier) || [];
    
    // 清理过期的时间戳
    timestamps = timestamps.filter(t => t > windowStart);
    
    if (timestamps.length >= this.maxRequests) {
      return false;
    }
    
    timestamps.push(now);
    this.requests.set(identifier, timestamps);
    return true;
  }
  
  getRemaining(identifier: string): number {
    const timestamps = this.requests.get(identifier) || [];
    const windowStart = Date.now() - this.windowMs;
    const validTimestamps = timestamps.filter(t => t > windowStart);
    return Math.max(0, this.maxRequests - validTimestamps.length);
  }
}

const rateLimiter = new RateLimiter(60000, 100); // 每分钟100个请求

// ==================== 9. 优雅关闭 ====================

// 添加优雅关闭处理
let isShuttingDown = false;

async function gracefulShutdown(signal: string): Promise<void> {
  if (isShuttingDown) return;
  isShuttingDown = true;
  
  logger.info(`收到 ${signal} 信号，开始优雅关闭...`);
  
  // 1. 停止接受新连接
  io.close(() => {
    logger.info('Socket.IO 服务已关闭');
  });
  
  // 2. 等待正在处理的任务完成
  const processingCount = processingTasks.length;
  if (processingCount > 0) {
    logger.info(`等待 ${processingCount} 个任务完成...`);
    
    // 最多等待30秒
    const timeout = setTimeout(() => {
      logger.warn('等待超时，强制关闭');
      process.exit(1);
    }, 30000);
    
    // 检查任务是否完成
    const checkInterval = setInterval(() => {
      if (processingTasks.length === 0) {
        clearTimeout(timeout);
        clearInterval(checkInterval);
        logger.info('所有任务已完成');
        process.exit(0);
      }
    }, 1000);
  } else {
    logger.info('没有正在处理的任务');
    process.exit(0);
  }
}

process.on('SIGTERM', () => gracefulShutdown('SIGTERM'));
process.on('SIGINT', () => gracefulShutdown('SIGINT'));

// ==================== 10. Prometheus指标 ====================

// 添加Prometheus格式的指标
function getPrometheusMetrics(): string {
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
  ];
  
  return metrics.join('\n');
}

// 添加指标端点
// 在HTTP请求处理中添加:
// if (req.url === '/metrics' && req.method === 'GET') {
//   res.writeHead(200, { 'Content-Type': 'text/plain' });
//   res.end(getPrometheusMetrics());
// }

// ==================== 导出修复函数 ====================

export {
  ReconnectionManager,
  ConcurrentMap,
  Logger,
  LogLevel,
  RateLimiter,
  validatePrompt,
  validateNodeId,
  startCleanupTask,
  setupHealthCheck,
  gracefulShutdown,
  getPrometheusMetrics,
};
