 'use client'

import { useState, useEffect, useCallback } from 'react'
import dynamic from 'next/dynamic'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Textarea } from '@/components/ui/textarea'
import { Separator } from '@/components/ui/separator'
import { 
  Server, Cpu, HardDrive, Network, Activity, 
  Play, Pause, RefreshCw, Download, Terminal,
  CheckCircle, XCircle, Clock, Zap, Globe, AlertTriangle,
  ArrowRightLeft, TrendingUp, Shield
} from 'lucide-react'

// 类型定义
interface NodeInfo {
  nodeId: string
  name: string
  os: string
  cpuCores: number
  totalMemory: number
  availableMemory: number
  status: 'online' | 'offline' | 'busy' | 'error' | 'recovering'
  shards: string[]
  performance: {
    avgLatency: number
    tasksCompleted: number
    tasksFailed: number
    loadScore: number
  }
  joinTime: string
  // 并行计算相关
  parallelSlots?: number
  activeSlots?: number
}

interface ShardReplica {
  nodeId: string
  status: string
}

interface ShardInfo {
  shardId: string
  layerStart: number
  layerEnd: number
  priority: string
  replicas: ShardReplica[]
}

interface ModelInfo {
  modelId: string
  name: string
  status: string
  totalShards: number
  readyShards: number
  totalReplicas: number
  shards: ShardInfo[]
}

interface SystemMetrics {
  totalComputePower: number
  availableComputePower: number
  avgNodeLoad: number
  shardRedundancy: number
  healthyNodes: number
  degradedNodes: number
  // 并行计算指标
  currentParallelism: number
  maxParallelism: number
  avgThroughput: number
  queueLength: number
  batchEfficiency: number
}

interface SystemStatus {
  nodes: NodeInfo[]
  models: ModelInfo[]
  metrics: SystemMetrics
  pendingCount: number
}

// 动态导入socket.io-client类型
type SocketType = {
  on: (event: string, callback: (data: unknown) => void) => void
  emit: (event: string, data?: unknown) => void
  connected: boolean
  close: () => void
}

export default function Home() {
  const [socket, setSocket] = useState<SocketType | null>(null)
  const [connected, setConnected] = useState(false)
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null)
  const [nodes, setNodes] = useState<NodeInfo[]>([])
  const [models, setModels] = useState<ModelInfo[]>([])
  const [chatInput, setChatInput] = useState('')
  const [chatMessages, setChatMessages] = useState<Array<{role: string, content: string}>>([])
  const [currentTaskId, setCurrentTaskId] = useState<string | null>(null)
  const [isProcessing, setIsProcessing] = useState(false)
    const [logs, setLogs] = useState<Array<{time: string, type: string, message: string}>>([])

    // 添加日志
    const addLog = useCallback((type: string, message: string) => {
        const time = new Date().toLocaleTimeString()
        setLogs(prev => [...prev.slice(-50), { time, type, message }])
    }, [])

    // 连接WebSocket
    useEffect(() => {
        let newSocket: SocketType | null = null
        
        const initSocket = async () => {
            try {
                const { io } = await import('socket.io-client')
                
                newSocket = io('/?XTransformPort=3003', {
                    transports: ['websocket', 'polling']
                }) as SocketType

                newSocket.on('connect', () => {
                    setConnected(true)
                    addLog('info', '已连接到调度服务器')
                    newSocket?.emit('system:status')
                })

                newSocket.on('disconnect', () => {
                    setConnected(false)
                    addLog('warn', '与调度服务器断开连接')
                })

                newSocket.on('system:status', (status: unknown) => {
                    const s = status as SystemStatus
                    setSystemStatus(s)
                    setNodes(s.nodes || [])
                    setModels(s.models || [])
                })

                newSocket.on('node:joined', (data: unknown) => {
                    const d = data as { name: string }
                    addLog('success', `节点加入: ${d.name}`)
                    newSocket?.emit('system:status')
                })

                newSocket.on('node:offline', (data: unknown) => {
                    const d = data as { name: string; reason: string }
                    addLog('error', `节点离线: ${d.name} (${d.reason})`)
                    newSocket?.emit('system:status')
                })

                newSocket.on('node:recovered', (data: unknown) => {
                    const d = data as { name: string }
                    addLog('success', `节点恢复: ${d.name}`)
                    newSocket?.emit('system:status')
                })

                newSocket.on('node:warning', (data: unknown) => {
                    const d = data as { nodeId: string; reason: string }
                    addLog('warn', `节点警告: ${d.nodeId.slice(0,8)} (${d.reason})`)
                })

                newSocket.on('shard:updated', (data: unknown) => {
                    const d = data as { shardId: string; status: string }
                    addLog('info', `分片更新: ${d.shardId.slice(-3)} -> ${d.status}`)
                    newSocket?.emit('system:status')
                })

                newSocket.on('inference:queued', (data: unknown) => {
                    const d = data as { taskId: string }
                    setCurrentTaskId(d.taskId)
                    setIsProcessing(true)
                    addLog('info', `任务排队: ${d.taskId.slice(0, 8)}`)
                })

                newSocket.on('task:completed', (data: unknown) => {
                    const d = data as { result?: string }
                    setChatMessages(prev => [...prev, { role: 'assistant', content: d.result || '' }])
                    setIsProcessing(false)
                    setCurrentTaskId(null)
                    addLog('success', '任务完成')
                })

                newSocket.on('task:failed', (data: unknown) => {
                    const d = data as { error?: string }
                    addLog('error', `任务失败: ${d.error || 'unknown'}`)
                    setIsProcessing(false)
                    setCurrentTaskId(null)
                })

                setSocket(newSocket)
            } catch (error) {
                addLog('error', 'WebSocket初始化失败')
            }
        }
        
        initSocket()

        return () => {
            if (newSocket) {
                newSocket.close()
            }
        }
    }, [addLog])

    // 定时刷新状态
    useEffect(() => {
        const interval = setInterval(() => {
            if (socket && connected) {
                socket.emit('system:status')
            }
        }, 5000)
        return () => clearInterval(interval)
    }, [socket, connected])

    // 发送聊天消息
    const sendChat = useCallback(() => {
        if (!chatInput.trim() || !socket || isProcessing) return
        
        setChatMessages(prev => [...prev, { role: 'user', content: chatInput }])
        socket.emit('inference:request', { prompt: chatInput })
        setChatInput('')
    }, [chatInput, socket, isProcessing])

    // 刷新状态
    const refreshStatus = useCallback(() => {
        if (socket) {
            socket.emit('system:status')
            addLog('info', '刷新系统状态')
        }
    }, [socket, addLog])

    // 触发再平衡
    const triggerRebalance = useCallback(async () => {
        try {
            await fetch('/api/?XTransformPort=3004/api/rebalance', { method: 'POST' })
            addLog('info', '触发分片再平衡')
        } catch {
            addLog('error', '再平衡请求失败')
        }
    }, [addLog])

    // 模拟器状态
    const [simulatorStatus, setSimulatorStatus] = useState<'running' | 'stopped'>('stopped')

    // 启动/停止模拟节点
    const toggleSimulator = useCallback(async () => {
        try {
            const action = simulatorStatus === 'running' ? 'stop' : 'start'
            const response = await fetch('/api/simulator', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ action })
            })
            const data = await response.json()
            
            if (data.status === 'started') {
                setSimulatorStatus('running')
                addLog('success', '模拟器已启动，节点将陆续加入')
            } else if (data.status === 'stopped') {
                setSimulatorStatus('stopped')
                addLog('info', '模拟器已停止')
            } else if (data.status === 'already_running') {
                setSimulatorStatus('running')
                addLog('info', '模拟器已在运行中')
            }
        } catch {
            addLog('error', '模拟器控制失败')
        }
    }, [simulatorStatus, addLog])

    // 格式化内存大小
    const formatMemory = (mb: number) => {
        if (mb >= 1024) return `${(mb / 1024).toFixed(1)} GB`
        return `${mb} MB`
    }

    // 获取状态颜色
    const getStatusColor = (status: string) => {
        switch (status) {
            case 'online': return 'bg-green-500'
            case 'offline': return 'bg-gray-500'
            case 'busy': return 'bg-yellow-500'
            case 'error': return 'bg-red-500'
            case 'recovering': return 'bg-orange-500'
            case 'ready': return 'bg-green-500'
            case 'pending': return 'bg-yellow-500'
            case 'degraded': return 'bg-orange-500'
            case 'migrating': return 'bg-blue-500'
            default: return 'bg-gray-500'
        }
    }

    // 获取状态图标
    const getStatusIcon = (status: string) => {
        switch (status) {
            case 'online': return <CheckCircle className="w-4 h-4 text-green-500" />
            case 'offline': return <Server className="w-4 h-4 text-gray-500" />
            case 'busy': return <Activity className="w-4 h-4 text-yellow-500" />
            case 'error': return <AlertTriangle className="w-4 h-4 text-red-500" />
            case 'recovering': return <RefreshCw className="w-4 h-4 text-orange-500 animate-spin" />
            default: return <Server className="w-4 h-4 text-gray-500" />
        }
    }

    // 获取操作系统图标
    const getOsIcon = (os: string) => {
        const osLower = os.toLowerCase()
        if (osLower.includes('windows')) return '🪟'
        if (osLower.includes('mac') || osLower.includes('darwin')) return '🍎'
        if (osLower.includes('linux')) return '🐧'
        return '💻'
    }

    // 下载节点服务
    const downloadNodeService = () => {
        const link = document.createElement('a')
        link.href = '/api/download/node_service.py'
        link.download = 'node_service.py'
        link.click()
    }

    return (
        <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-white">
            {/* Header */}
            <header className="border-b border-slate-700 bg-slate-900/50 backdrop-blur-sm sticky top-0 z-50">
                <div className="container mx-auto px-4 py-4">
                    <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
                                <Globe className="w-6 h-6" />
                            </div>
                            <div>
                                <h1 className="text-xl font-bold">分布式大模型推理系统</h1>
                                <p className="text-sm text-slate-400">动态算力调度 · 自动故障恢复</p>
                            </div>
                        </div>
                        <div className="flex items-center gap-4">
                            <Badge variant={connected ? "default" : "secondary"} className="gap-1">
                                <span className={`w-2 h-2 rounded-full ${connected ? 'bg-green-500' : 'bg-red-500'} ${connected ? 'animate-pulse' : ''}`} />
                                {connected ? '已连接' : '未连接'}
                            </Badge>
                            <Button variant="outline" size="sm" onClick={refreshStatus}>
                                <RefreshCw className="w-4 h-4 mr-1" />
                                刷新
                            </Button>
                        </div>
                    </div>
                </div>
            </header>

            <main className="container mx-auto px-4 py-6">
                {/* Stats Overview */}
                <div className="grid grid-cols-2 md:grid-cols-8 gap-4 mb-6">
                    <Card className="bg-slate-800/50 border-slate-700">
                        <CardContent className="pt-4 pb-4">
                            <div className="flex items-center justify-between">
                                <div>
                                    <p className="text-xs text-slate-400">总节点</p>
                                    <p className="text-2xl font-bold">{systemStatus?.metrics.healthyNodes || 0}</p>
                                </div>
                                <Server className="w-8 h-8 text-blue-500" />
                            </div>
                        </CardContent>
                    </Card>
                    <Card className="bg-slate-800/50 border-slate-700">
                        <CardContent className="pt-4 pb-4">
                            <div className="flex items-center justify-between">
                                <div>
                                    <p className="text-xs text-slate-400">可用算力</p>
                                    <p className="text-2xl font-bold text-green-500">{systemStatus?.metrics.availableComputePower || 0}</p>
                                    <p className="text-xs text-slate-500">核心</p>
                                </div>
                                <Cpu className="w-8 h-8 text-green-500" />
                            </div>
                        </CardContent>
                    </Card>
                    <Card className="bg-slate-800/50 border-slate-700">
                        <CardContent className="pt-4 pb-4">
                            <div className="flex items-center justify-between">
                                <div>
                                    <p className="text-xs text-slate-400">并行度</p>
                                    <p className="text-2xl font-bold text-cyan-500">
                                        {systemStatus?.metrics.currentParallelism || 0}/{systemStatus?.metrics.maxParallelism || 0}
                                    </p>
                                </div>
                                <Activity className="w-8 h-8 text-cyan-500" />
                            </div>
                        </CardContent>
                    </Card>
                    <Card className="bg-slate-800/50 border-slate-700">
                        <CardContent className="pt-4 pb-4">
                            <div className="flex items-center justify-between">
                                <div>
                                    <p className="text-xs text-slate-400">吞吐量</p>
                                    <p className="text-2xl font-bold text-emerald-500">
                                        {(systemStatus?.metrics.avgThroughput || 0).toFixed(0)}
                                    </p>
                                    <p className="text-xs text-slate-500">tokens/s</p>
                                </div>
                                <TrendingUp className="w-8 h-8 text-emerald-500" />
                            </div>
                        </CardContent>
                    </Card>
                    <Card className="bg-slate-800/50 border-slate-700">
                        <CardContent className="pt-4 pb-4">
                            <div className="flex items-center justify-between">
                                <div>
                                    <p className="text-xs text-slate-400">平均负载</p>
                                    <p className="text-2xl font-bold text-yellow-500">{(systemStatus?.metrics.avgNodeLoad || 0).toFixed(0)}%</p>
                                </div>
                                <TrendingUp className="w-8 h-8 text-yellow-500" />
                            </div>
                        </CardContent>
                    </Card>
                    <Card className="bg-slate-800/50 border-slate-700">
                        <CardContent className="pt-4 pb-4">
                            <div className="flex items-center justify-between">
                                <div>
                                    <p className="text-xs text-slate-400">分片冗余</p>
                                    <p className="text-2xl font-bold text-purple-500">{((systemStatus?.metrics.shardRedundancy || 0) * 100).toFixed(0)}%</p>
                                </div>
                                <Shield className="w-8 h-8 text-purple-500" />
                            </div>
                        </CardContent>
                    </Card>
                    <Card className="bg-slate-800/50 border-slate-700">
                        <CardContent className="pt-4 pb-4">
                            <div className="flex items-center justify-between">
                                <div>
                                    <p className="text-xs text-slate-400">队列</p>
                                    <p className="text-2xl font-bold text-orange-500">{systemStatus?.metrics.queueLength || 0}</p>
                                </div>
                                <Clock className="w-8 h-8 text-orange-500" />
                            </div>
                        </CardContent>
                    </Card>
                    <Card className="bg-slate-800/50 border-slate-700">
                        <CardContent className="pt-4 pb-4">
                            <div className="flex items-center justify-between">
                                <div>
                                    <p className="text-xs text-slate-400">模型状态</p>
                                    <p className="text-2xl font-bold">{models[0]?.status || 'pending'}</p>
                                </div>
                                <Activity className="w-8 h-8 text-blue-500" />
                            </div>
                        </CardContent>
                    </Card>
                </div>

                {/* Main Content */}
                <Tabs defaultValue="nodes" className="space-y-4">
                    <TabsList className="bg-slate-800/50 border border-slate-700">
                        <TabsTrigger value="nodes" className="data-[state=active]:bg-slate-700">
                            <Server className="w-4 h-4 mr-2" />
                            节点管理
                        </TabsTrigger>
                        <TabsTrigger value="shards" className="data-[state=active]:bg-slate-700">
                            <HardDrive className="w-4 h-4 mr-2" />
                            分片分布
                        </TabsTrigger>
                        <TabsTrigger value="chat" className="data-[state=active]:bg-slate-700">
                            <Zap className="w-4 h-4 mr-2" />
                            推理测试
                        </TabsTrigger>
                        <TabsTrigger value="logs" className="data-[state=active]:bg-slate-700">
                            <Terminal className="w-4 h-4 mr-2" />
                            系统日志
                        </TabsTrigger>
                        <TabsTrigger value="deploy" className="data-[state=active]:bg-slate-700">
                            <Download className="w-4 h-4 mr-2" />
                            部署指南
                        </TabsTrigger>
                    </TabsList>

                    {/* Nodes Tab */}
                    <TabsContent value="nodes">
                        {/* 模拟器控制栏 */}
                        <Card className="bg-slate-800/50 border-slate-700 mb-4">
                            <CardContent className="py-4">
                                <div className="flex items-center justify-between">
                                    <div className="flex items-center gap-3">
                                        <div className={`w-3 h-3 rounded-full ${simulatorStatus === 'running' ? 'bg-green-500 animate-pulse' : 'bg-gray-500'}`} />
                                        <div>
                                            <p className="font-medium">模拟节点服务</p>
                                            <p className="text-sm text-slate-400">
                                                {simulatorStatus === 'running' 
                                                    ? '正在运行 - 4个模拟节点已加入集群' 
                                                    : '未运行 - 点击启动以测试系统功能'}
                                            </p>
                                        </div>
                                    </div>
                                    <div className="flex gap-2">
                                        <Button 
                                            onClick={toggleSimulator}
                                            variant={simulatorStatus === 'running' ? 'destructive' : 'default'}
                                        >
                                            {simulatorStatus === 'running' ? (
                                                <>
                                                    <span className="mr-2">⏹️</span>
                                                    停止模拟
                                                </>
                                            ) : (
                                                <>
                                                    <Play className="w-4 h-4 mr-2" />
                                                    启动模拟节点
                                                </>
                                            )}
                                        </Button>
                                        <Button variant="outline" onClick={triggerRebalance}>
                                            <ArrowRightLeft className="w-4 h-4 mr-2" />
                                            再平衡
                                        </Button>
                                    </div>
                                </div>
                            </CardContent>
                        </Card>
                        
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                            {nodes.length === 0 ? (
                                <Card className="bg-slate-800/50 border-slate-700 col-span-full">
                                    <CardContent className="pt-6 text-center">
                                        <Server className="w-16 h-16 mx-auto text-slate-600 mb-4" />
                                        <h3 className="text-lg font-semibold mb-2">暂无节点连接</h3>
                                        <p className="text-slate-400 mb-4">启动模拟节点或部署真实节点来加入集群</p>
                                        <div className="flex gap-2 justify-center">
                                            <Button onClick={toggleSimulator}>
                                                <Play className="w-4 h-4 mr-2" />
                                                启动模拟节点
                                            </Button>
                                            <Button variant="outline" onClick={() => {
                                                const deployTab = document.querySelector('[value="deploy"]') as HTMLElement
                                                deployTab?.click()
                                            }}>
                                                查看部署指南
                                            </Button>
                                        </div>
                                    </CardContent>
                                </Card>
                            ) : (
                                nodes.map((node) => (
                                    <Card key={node.nodeId} className="bg-slate-800/50 border-slate-700">
                                        <CardHeader className="pb-2">
                                            <div className="flex items-center justify-between">
                                                <CardTitle className="text-lg flex items-center gap-2">
                                                    <span className="text-2xl">{getOsIcon(node.os)}</span>
                                                    {node.name}
                                                </CardTitle>
                                                <div className="flex items-center gap-2">
                                                    {getStatusIcon(node.status)}
                                                    <Badge className={`${getStatusColor(node.status)} text-white text-xs`}>
                                                        {node.status}
                                                    </Badge>
                                                </div>
                                            </div>
                                            <CardDescription className="text-slate-400">
                                                {node.os} · {node.nodeId.slice(0, 8)}
                                            </CardDescription>
                                        </CardHeader>
                                        <CardContent>
                                            <div className="space-y-3">
                                                <div className="grid grid-cols-2 gap-2 text-sm">
                                                    <div>
                                                        <p className="text-slate-400">CPU核心</p>
                                                        <p className="font-medium">{node.cpuCores} 核</p>
                                                    </div>
                                                    <div>
                                                        <p className="text-slate-400">内存</p>
                                                        <p className="font-medium">{formatMemory(node.totalMemory)}</p>
                                                    </div>
                                                    <div>
                                                        <p className="text-slate-400">并行槽位</p>
                                                        <p className="font-medium text-cyan-400">
                                                            {node.activeSlots ?? 0}/{node.parallelSlots ?? 2}
                                                        </p>
                                                    </div>
                                                    <div>
                                                        <p className="text-slate-400">分片数</p>
                                                        <p className="font-medium">{node.shards.length} 个</p>
                                                    </div>
                                                </div>
                                                
                                                {/* 并行槽位进度 */}
                                                <div>
                                                    <div className="flex justify-between text-xs mb-1">
                                                        <span className="text-slate-400">并行槽位使用</span>
                                                        <span>{node.activeSlots ?? 0}/{node.parallelSlots ?? 2}</span>
                                                    </div>
                                                    <Progress 
                                                        value={((node.activeSlots ?? 0) / (node.parallelSlots ?? 2)) * 100} 
                                                        className="h-2"
                                                    />
                                                </div>
                                                
                                                <div>
                                                    <div className="flex justify-between text-xs mb-1">
                                                        <span className="text-slate-400">内存使用</span>
                                                        <span>{((1 - node.availableMemory / node.totalMemory) * 100).toFixed(0)}%</span>
                                                    </div>
                                                    <Progress 
                                                        value={(1 - node.availableMemory / node.totalMemory) * 100} 
                                                        className="h-2"
                                                    />
                                                </div>
                                                
                                                <div>
                                                    <div className="flex justify-between text-xs mb-1">
                                                        <span className="text-slate-400">节点负载</span>
                                                        <span>{node.performance.loadScore.toFixed(0)}%</span>
                                                    </div>
                                                    <Progress 
                                                        value={node.performance.loadScore} 
                                                        className={`h-2 ${node.performance.loadScore > 70 ? 'bg-red-900' : ''}`}
                                                    />
                                                </div>
                                                
                                                <Separator className="bg-slate-700" />
                                                
                                                <div className="grid grid-cols-2 gap-2 text-xs">
                                                    <div>
                                                        <p className="text-slate-500">完成任务</p>
                                                        <p className="text-green-400">{node.performance.tasksCompleted}</p>
                                                    </div>
                                                    <div>
                                                        <p className="text-slate-500">平均延迟</p>
                                                        <p className="text-blue-400">{node.performance.avgLatency.toFixed(1)}s</p>
                                                    </div>
                                                </div>
                                            </div>
                                        </CardContent>
                                    </Card>
                                ))
                            )}
                        </div>
                    </TabsContent>

                    {/* Shards Tab */}
                    <TabsContent value="shards">
                        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                            {/* 模型状态卡片 */}
                            <Card className="bg-slate-800/50 border-slate-700 lg:col-span-1">
                                <CardHeader>
                                    <CardTitle className="flex items-center gap-2">
                                        <HardDrive className="w-5 h-5 text-blue-500" />
                                        模型状态
                                    </CardTitle>
                                </CardHeader>
                                <CardContent>
                                    {models.map((model) => (
                                        <div key={model.modelId} className="space-y-4">
                                            <div className="flex items-center justify-between">
                                                <span className="font-medium">{model.name}</span>
                                                <Badge className={`${getStatusColor(model.status)} text-white`}>
                                                    {model.status}
                                                </Badge>
                                            </div>
                                            
                                            <div className="space-y-2">
                                                <div>
                                                    <div className="flex justify-between text-sm mb-1">
                                                        <span className="text-slate-400">分片就绪</span>
                                                        <span>{model.readyShards} / {model.totalShards}</span>
                                                    </div>
                                                    <Progress value={(model.readyShards / model.totalShards) * 100} className="h-2" />
                                                </div>
                                                
                                                <div>
                                                    <div className="flex justify-between text-sm mb-1">
                                                        <span className="text-slate-400">副本冗余</span>
                                                        <span>{model.totalReplicas} / {model.totalShards * 2}</span>
                                                    </div>
                                                    <Progress 
                                                        value={(model.totalReplicas / (model.totalShards * 2)) * 100} 
                                                        className="h-2" 
                                                    />
                                                </div>
                                            </div>
                                            
                                            <Button 
                                                variant="outline" 
                                                size="sm" 
                                                className="w-full"
                                                onClick={triggerRebalance}
                                            >
                                                <ArrowRightLeft className="w-4 h-4 mr-2" />
                                                触发再平衡
                                            </Button>
                                        </div>
                                    ))}
                                </CardContent>
                            </Card>
                            
                            {/* 分片分布详情 */}
                            <Card className="bg-slate-800/50 border-slate-700 lg:col-span-2">
                                <CardHeader>
                                    <CardTitle>分片分布详情</CardTitle>
                                    <CardDescription>
                                        每个分片的副本分布情况（目标：每个分片2个副本）
                                    </CardDescription>
                                </CardHeader>
                                <CardContent>
                                    <ScrollArea className="h-[400px]">
                                        <div className="space-y-3">
                                            {models.flatMap(m => m.shards).map((shard) => {
                                                const readyReplicas = shard.replicas.filter(r => r.status === 'ready').length
                                                const isHealthy = readyReplicas >= 2
                                                const isDegraded = readyReplicas === 1
                                                
                                                return (
                                                    <div 
                                                        key={shard.shardId} 
                                                        className={`p-3 rounded-lg border ${
                                                            isHealthy ? 'border-green-700 bg-green-900/20' :
                                                            isDegraded ? 'border-yellow-700 bg-yellow-900/20' :
                                                            'border-red-700 bg-red-900/20'
                                                        }`}
                                                    >
                                                        <div className="flex items-center justify-between mb-2">
                                                            <div className="flex items-center gap-2">
                                                                <span className="font-medium">{shard.shardId}</span>
                                                                {shard.priority === 'critical' && (
                                                                    <Badge variant="destructive" className="text-xs">关键</Badge>
                                                                )}
                                                            </div>
                                                            <div className="flex items-center gap-2">
                                                                {isHealthy ? (
                                                                    <CheckCircle className="w-4 h-4 text-green-500" />
                                                                ) : isDegraded ? (
                                                                    <AlertTriangle className="w-4 h-4 text-yellow-500" />
                                                                ) : (
                                                                    <AlertTriangle className="w-4 h-4 text-red-500" />
                                                                )}
                                                                <span className="text-sm">
                                                                    {readyReplicas}/2 副本
                                                                </span>
                                                            </div>
                                                        </div>
                                                        
                                                        <div className="text-sm text-slate-400 mb-2">
                                                            层 {shard.layerStart} - {shard.layerEnd}
                                                        </div>
                                                        
                                                        <div className="flex flex-wrap gap-2">
                                                            {shard.replicas.map((replica, idx) => {
                                                                const node = nodes.find(n => n.nodeId === replica.nodeId)
                                                                return (
                                                                    <Badge 
                                                                        key={idx}
                                                                        variant="outline"
                                                                        className={`${getStatusColor(replica.status)} border-current`}
                                                    >
                                                        {node?.name || replica.nodeId.slice(0,8)}: {replica.status}
                                                    </Badge>
                                                )
                                            })}
                                            {readyReplicas < 2 && (
                                                <Badge variant="outline" className="border-dashed border-slate-500">
                                                    + 待分配
                                                </Badge>
                                            )}
                                        </div>
                                    </div>
                                                )
                                            })}
                                        </div>
                                    </ScrollArea>
                                </CardContent>
                            </Card>
                        </div>
                    </TabsContent>

                    {/* Chat Tab */}
                    <TabsContent value="chat">
                        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                            <Card className="bg-slate-800/50 border-slate-700 lg:col-span-2">
                                <CardHeader>
                                    <CardTitle className="flex items-center gap-2">
                                        <Zap className="w-5 h-5 text-yellow-500" />
                                        推理测试
                                    </CardTitle>
                                    <CardDescription>
                                        测试分布式推理功能 - Qwen 2.5 27B (4-bit量化)
                                    </CardDescription>
                                </CardHeader>
                                <CardContent>
                                    <div className="space-y-4">
                                        <ScrollArea className="h-[400px] rounded-lg border border-slate-700 bg-slate-900/50 p-4">
                                            {chatMessages.length === 0 ? (
                                                <div className="text-center text-slate-500 py-8">
                                                    <Zap className="w-12 h-12 mx-auto mb-4 opacity-50" />
                                                    <p>输入消息开始测试分布式推理</p>
                                                    <p className="text-sm mt-2">系统会自动调度可用节点执行Pipeline并行推理</p>
                                                </div>
                                            ) : (
                                                <div className="space-y-4">
                                                    {chatMessages.map((msg, i) => (
                                                        <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                                                            <div className={`max-w-[80%] rounded-lg px-4 py-2 ${
                                                                msg.role === 'user' 
                                                                    ? 'bg-blue-600 text-white' 
                                                                    : 'bg-slate-700 text-white'
                                                            }`}>
                                                                <p className="whitespace-pre-wrap text-sm">{msg.content}</p>
                                                            </div>
                                                        </div>
                                                    ))}
                                                    {isProcessing && (
                                                        <div className="flex justify-start">
                                                            <div className="bg-slate-700 rounded-lg px-4 py-2">
                                                                <div className="flex items-center gap-2">
                                                                    <RefreshCw className="w-4 h-4 animate-spin" />
                                                                    <span className="text-sm">正在推理...</span>
                                                                </div>
                                                            </div>
                                                        </div>
                                                    )}
                                                </div>
                                            )}
                                        </ScrollArea>
                                        
                                        <div className="flex gap-2">
                                            <Textarea
                                                value={chatInput}
                                                onChange={(e) => setChatInput(e.target.value)}
                                                placeholder="输入您的消息..."
                                                className="bg-slate-900/50 border-slate-700"
                                                onKeyDown={(e) => {
                                                    if (e.key === 'Enter' && !e.shiftKey) {
                                                        e.preventDefault()
                                                        sendChat()
                                                    }
                                                }}
                                            />
                                            <Button 
                                                onClick={sendChat} 
                                                disabled={!chatInput.trim() || isProcessing || !connected}
                                                className="shrink-0"
                                            >
                                                <Play className="w-4 h-4" />
                                            </Button>
                                        </div>
                                    </div>
                                </CardContent>
                            </Card>

                            <Card className="bg-slate-800/50 border-slate-700">
                                <CardHeader>
                                    <CardTitle className="flex items-center gap-2">
                                        <Clock className="w-5 h-5 text-yellow-500" />
                                        任务状态
                                    </CardTitle>
                                </CardHeader>
                                <CardContent>
                                    <ScrollArea className="h-[400px]">
                                        {currentTaskId ? (
                                            <div className="space-y-4">
                                                <div className="p-3 rounded-lg bg-slate-700/50">
                                                    <div className="flex items-center gap-2 mb-2">
                                                        <RefreshCw className="w-4 h-4 animate-spin text-yellow-500" />
                                                        <span className="text-sm font-medium">正在执行</span>
                                                    </div>
                                                    <p className="text-xs text-slate-400 font-mono">
                                                        {currentTaskId}
                                                    </p>
                                                </div>
                                            </div>
                                        ) : (
                                            <div className="text-center text-slate-500 py-8">
                                                <Clock className="w-12 h-12 mx-auto mb-4 opacity-50" />
                                                <p>暂无进行中的任务</p>
                                            </div>
                                        )}
                                    </ScrollArea>
                                </CardContent>
                            </Card>
                        </div>
                    </TabsContent>

                    {/* Logs Tab */}
                    <TabsContent value="logs">
                        <Card className="bg-slate-800/50 border-slate-700">
                            <CardHeader>
                                <CardTitle className="flex items-center gap-2">
                                    <Terminal className="w-5 h-5 text-green-500" />
                                    系统日志
                                </CardTitle>
                                <CardDescription>
                                    实时显示系统事件和状态变化
                                </CardDescription>
                            </CardHeader>
                            <CardContent>
                                <ScrollArea className="h-[500px] rounded-lg border border-slate-700 bg-slate-900 p-4">
                                    {logs.length === 0 ? (
                                        <div className="text-center text-slate-500 py-8">
                                            <Terminal className="w-12 h-12 mx-auto mb-4 opacity-50" />
                                            <p>暂无日志</p>
                                        </div>
                                    ) : (
                                        <div className="space-y-1 font-mono text-sm">
                                            {logs.map((log, i) => (
                                                <div key={i} className={`flex gap-2 ${
                                                    log.type === 'error' ? 'text-red-400' :
                                                    log.type === 'warn' ? 'text-yellow-400' :
                                                    log.type === 'success' ? 'text-green-400' :
                                                    'text-slate-300'
                                                }`}>
                                                    <span className="text-slate-500">[{log.time}]</span>
                                                    <span className={
                                                        log.type === 'error' ? 'text-red-500' :
                                                        log.type === 'warn' ? 'text-yellow-500' :
                                                        log.type === 'success' ? 'text-green-500' :
                                                        'text-blue-500'
                                                    }>
                                                        [{log.type.toUpperCase()}]
                                                    </span>
                                                    <span>{log.message}</span>
                                                </div>
                                            ))}
                                        </div>
                                    )}
                                </ScrollArea>
                            </CardContent>
                        </Card>
                    </TabsContent>

                    {/* Deploy Tab */}
                    <TabsContent value="deploy">
                        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                            <Card className="bg-slate-800/50 border-slate-700">
                                <CardHeader>
                                    <CardTitle className="flex items-center gap-2">
                                        <Terminal className="w-5 h-5 text-green-500" />
                                        快速部署
                                    </CardTitle>
                                    <CardDescription>
                                        在您的旧电脑上运行以下命令来加入集群
                                    </CardDescription>
                                </CardHeader>
                                <CardContent className="space-y-4">
                                    <div className="space-y-2">
                                        <p className="text-sm font-medium">1. 安装Python依赖</p>
                                        <div className="bg-slate-900 rounded-lg p-3 font-mono text-sm overflow-x-auto">
                                            <code>pip install socketio-client psutil torch transformers</code>
                                        </div>
                                    </div>
                                    <div className="space-y-2">
                                        <p className="text-sm font-medium">2. 下载节点服务脚本</p>
                                        <Button className="w-full" variant="outline" onClick={downloadNodeService}>
                                            <Download className="w-4 h-4 mr-2" />
                                            下载 node_service.py
                                        </Button>
                                    </div>
                                    <div className="space-y-2">
                                        <p className="text-sm font-medium">3. 运行节点服务</p>
                                        <div className="bg-slate-900 rounded-lg p-3 font-mono text-sm overflow-x-auto">
                                            <code>python node_service.py --server YOUR_SERVER_URL</code>
                                        </div>
                                    </div>
                                </CardContent>
                            </Card>

                            <Card className="bg-slate-800/50 border-slate-700">
                                <CardHeader>
                                    <CardTitle className="flex items-center gap-2">
                                        <Network className="w-5 h-5 text-blue-500" />
                                        系统特性
                                    </CardTitle>
                                </CardHeader>
                                <CardContent>
                                    <div className="space-y-4 text-sm">
                                        <div className="bg-slate-900/50 rounded-lg p-4">
                                            <h4 className="font-semibold mb-2 text-blue-400">🔄 动态节点加入</h4>
                                            <p className="text-slate-400">
                                                新节点自动注册并获取分片分配，无需手动配置。系统自动检测节点内存大小，
                                                分配适量的模型分片。
                                            </p>
                                        </div>
                                        <div className="bg-slate-900/50 rounded-lg p-4">
                                            <h4 className="font-semibold mb-2 text-green-400">🛡️ 自动故障恢复</h4>
                                            <p className="text-slate-400">
                                                节点离线时，系统自动将其分片迁移到其他健康节点。每个分片保持2个副本，
                                                确保高可用性。
                                            </p>
                                        </div>
                                        <div className="bg-slate-900/50 rounded-lg p-4">
                                            <h4 className="font-semibold mb-2 text-purple-400">⚖️ 智能负载均衡</h4>
                                            <p className="text-slate-400">
                                                系统持续监控各节点负载，自动进行分片再平衡。任务优先分配给负载较低的节点。
                                            </p>
                                        </div>
                                        <div className="bg-slate-900/50 rounded-lg p-4">
                                            <h4 className="font-semibold mb-2 text-yellow-400">📊 实时监控</h4>
                                            <p className="text-slate-400">
                                                实时显示节点状态、分片分布、系统负载等关键指标。所有事件记录在系统日志中。
                                            </p>
                                        </div>
                                    </div>
                                </CardContent>
                            </Card>

                            <Card className="bg-slate-800/50 border-slate-700 lg:col-span-2">
                                <CardHeader>
                                    <CardTitle>故障恢复流程</CardTitle>
                                </CardHeader>
                                <CardContent>
                                    <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                                        <div className="text-center p-4">
                                            <div className="w-12 h-12 rounded-full bg-red-500/20 flex items-center justify-center mx-auto mb-3">
                                                <AlertTriangle className="w-6 h-6 text-red-500" />
                                            </div>
                                            <h4 className="font-medium mb-1">检测故障</h4>
                                            <p className="text-xs text-slate-400">心跳超时或连接断开</p>
                                        </div>
                                        <div className="text-center p-4">
                                            <div className="w-12 h-12 rounded-full bg-yellow-500/20 flex items-center justify-center mx-auto mb-3">
                                                <RefreshCw className="w-6 h-6 text-yellow-500" />
                                            </div>
                                            <h4 className="font-medium mb-1">标记离线</h4>
                                            <p className="text-xs text-slate-400">更新节点和分片状态</p>
                                        </div>
                                        <div className="text-center p-4">
                                            <div className="w-12 h-12 rounded-full bg-blue-500/20 flex items-center justify-center mx-auto mb-3">
                                                <ArrowRightLeft className="w-6 h-6 text-blue-500" />
                                            </div>
                                            <h4 className="font-medium mb-1">迁移分片</h4>
                                            <p className="text-xs text-slate-400">选择健康节点接收分片</p>
                                        </div>
                                        <div className="text-center p-4">
                                            <div className="w-12 h-12 rounded-full bg-green-500/20 flex items-center justify-center mx-auto mb-3">
                                                <CheckCircle className="w-6 h-6 text-green-500" />
                                            </div>
                                            <h4 className="font-medium mb-1">恢复服务</h4>
                                            <p className="text-xs text-slate-400">模型继续正常运行</p>
                                        </div>
                                    </div>
                                </CardContent>
                            </Card>
                        </div>
                    </TabsContent>
                </Tabs>
            </main>

            {/* Footer */}
            <footer className="border-t border-slate-700 bg-slate-900/50 mt-8">
                <div className="container mx-auto px-4 py-4">
                    <div className="flex items-center justify-between text-sm text-slate-400">
                        <span>分布式大模型推理系统 v2.0 - 动态算力调度版</span>
                        <div className="flex items-center gap-4">
                            <span>WebSocket: {connected ? '已连接' : '未连接'}</span>
                            <span>节点: {nodes.filter(n => n.status === 'online').length}/{nodes.length} 在线</span>
                        </div>
                    </div>
                </div>
            </footer>
        </div>
    )
}
