"use client"

import { useState } from "react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Checkbox } from "@/components/ui/checkbox"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import {
  CheckCircle,
  XCircle,
  Play,
  AlertTriangle,
  Key,
  FileKey,
  Terminal,
  Zap,
  Search,
  Upload,
  Wifi,
  Info,
} from "lucide-react"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"

interface TroubleshootingPageProps {
  language: "zh" | "en"
  t: any
}

// GPU基准值配置
const defaultGpuBenchmarks = {
  "RTX 3090": { p2p: 18, nccl: 7, bw: 20 },
  L40S: { p2p: 28, nccl: 9, bw: 20 },
  "RTX 4090": { p2p: 18, nccl: 7, bw: 20 },
  A100: { p2p: 420, nccl: 70, bw: 20 },
  A800: { p2p: 340, nccl: 55, bw: 20 },
  H100: { p2p: 700, nccl: 139, bw: 40 },
  H800: { p2p: 340, nccl: 65, bw: 47 },
  H20: { p2p: 700, nccl: 139, bw: 47 },
  H200: { p2p: 730, nccl: 145, bw: 54 },
}

// 检查项目配置
const checkItems = {
  zh: [
    {
      id: "nvbandwidthTest",
      label: "nvBandwidthTest",
      description: "测试CPU与GPU间内存拷贝带宽性能，使用nvbandwidth工具评估数据传输效率",
    },
    {
      id: "p2pBandwidthLatencyTest",
      label: "p2pBandwidthLatencyTest",
      description: "测试GPU间点对点通信带宽和延迟，评估多GPU协作性能",
    },
    { id: "ncclTests", label: "NCCL Tests", description: "测试NVIDIA集合通信库性能，评估分布式训练通信效率" },
    { id: "dcgmDiag", label: "DCGM Diagnostics", description: "NVIDIA数据中心GPU管理器诊断，检查GPU硬件健康状态" },
    { id: "ibCheck", label: "IB Check", description: "InfiniBand网络连接检查，确保高速网络通信正常" },
  ],
  en: [
    { id: "nvbandwidthTest", label: "nvBandwidthTest", description: "Test CPU-GPU memory copy bandwidth performance" },
    {
      id: "p2pBandwidthLatencyTest",
      label: "p2pBandwidthLatencyTest",
      description: "Test GPU peer-to-peer communication bandwidth and latency",
    },
    { id: "ncclTests", label: "NCCL Tests", description: "Test NVIDIA Collective Communications Library performance" },
    { id: "dcgmDiag", label: "DCGM Diagnostics", description: "NVIDIA Data Center GPU Manager diagnostics" },
    { id: "ibCheck", label: "IB Check", description: "InfiniBand network connection check" },
  ],
}

// DCGM级别配置
const dcgmLevels = [
  { value: 1, label: "Level 1", description: { zh: "快速检查（秒级）", en: "Quick check (seconds)" } },
  { value: 2, label: "Level 2", description: { zh: "标准检查（<2分钟）", en: "Standard check (<2 minutes)" } },
  { value: 3, label: "Level 3", description: { zh: "详细检查（<30分钟）", en: "Detailed check (<30 minutes)" } },
  { value: 4, label: "Level 4", description: { zh: "全面检查（1-2小时）", en: "Comprehensive check (1-2 hours)" } },
]

export default function TroubleshootingPage({ language, t }: TroubleshootingPageProps) {
  // SSH配置状态
  const [sshAuthType, setSshAuthType] = useState<"password" | "privatekey">("password")
  const [sshPassword, setSshPassword] = useState("")
  const [sshPrivateKey, setSshPrivateKey] = useState("")
  const [sshUsername, setSshUsername] = useState("root")
  const [sshPort, setSshPort] = useState("22")

  // 节点配置
  const [nodeHost, setNodeHost] = useState("")
  const [nodes, setNodes] = useState<{ host: string; status: string }[]>([])

  // 连接测试状态
  const [connectivityStatus, setConnectivityStatus] = useState<{
    tested: boolean
    connected: boolean
    message: string
  }>({ tested: false, connected: false, message: "" })

  // 命令检查状态
  const [commandStatus, setCommandStatus] = useState<{
    tested: boolean
    nvidiaSmi: boolean
    dcgmi: boolean
    nvcc: boolean
    messages: string[]
  }>({ tested: false, nvidiaSmi: false, dcgmi: false, nvcc: false, messages: [] })

  // 健康检查状态
  const [selectedCheckItems, setSelectedCheckItems] = useState<string[]>([
    "nvbandwidthTest",
    "p2pBandwidthLatencyTest",
    "ncclTests",
    "dcgmDiag",
    "ibCheck",
  ])
  const [dcgmLevel, setDcgmLevel] = useState(2)
  const [healthCheckRunning, setHealthCheckRunning] = useState(false)
  const [healthCheckResults, setHealthCheckResults] = useState<any[]>([])

  // GPU基准值
  const [gpuBenchmarks, setGpuBenchmarks] = useState(defaultGpuBenchmarks)

  // 测试SSH连接
  const testConnectivity = async () => {
    if (!nodeHost) {
      setConnectivityStatus({
        tested: true,
        connected: false,
        message: language === "zh" ? "请输入节点地址" : "Please enter node host",
      })
      return
    }

    setConnectivityStatus({
      tested: false,
      connected: false,
      message: language === "zh" ? "正在测试连接..." : "Testing connection...",
    })

    // 模拟API调用
    setTimeout(() => {
      const success = Math.random() > 0.2
      setConnectivityStatus({
        tested: true,
        connected: success,
        message: success
          ? language === "zh"
            ? "SSH连接成功"
            : "SSH connection successful"
          : language === "zh"
            ? "SSH连接失败，请检查配置"
            : "SSH connection failed, please check configuration",
      })

      if (success) {
        setNodes((prev) => {
          const exists = prev.find((n) => n.host === nodeHost)
          if (!exists) {
            return [...prev, { host: nodeHost, status: "connected" }]
          }
          return prev
        })
      }
    }, 2000)
  }

  // 检查命令可用性
  const checkCommands = async () => {
    if (!connectivityStatus.connected) {
      setCommandStatus({
        tested: true,
        nvidiaSmi: false,
        dcgmi: false,
        nvcc: false,
        messages: [language === "zh" ? "请先测试SSH连接" : "Please test SSH connection first"],
      })
      return
    }

    setCommandStatus({
      tested: false,
      nvidiaSmi: false,
      dcgmi: false,
      nvcc: false,
      messages: [language === "zh" ? "正在检查命令..." : "Checking commands..."],
    })

    // 模拟API调用
    setTimeout(() => {
      const nvidiaSmi = Math.random() > 0.1
      const dcgmi = Math.random() > 0.3
      const nvcc = Math.random() > 0.2

      const messages: string[] = []

      if (!nvidiaSmi) {
        messages.push(
          language === "zh"
            ? "⚠️ nvidia-smi 命令不可用，请安装 NVIDIA 驱动"
            : "⚠️ nvidia-smi not available, please install NVIDIA driver",
        )
      }

      if (!dcgmi) {
        messages.push(
          language === "zh"
            ? "⚠️ dcgmi 命令不可用，请安装 datacenter-gpu-manager 包"
            : "⚠️ dcgmi not available, please install datacenter-gpu-manager package",
        )
      }

      if (!nvcc) {
        messages.push(
          language === "zh"
            ? "⚠️ nvcc 命令不可用，请安装 CUDA 驱动（路径: /usr/local/cuda/bin/nvcc）"
            : "⚠️ nvcc not available, please install CUDA driver (path: /usr/local/cuda/bin/nvcc)",
        )
      }

      if (messages.length === 0) {
        messages.push(
          language === "zh"
            ? "✅ 所有命令检查通过，可以开始健康检查"
            : "✅ All commands available, ready for health check",
        )
      }

      setCommandStatus({
        tested: true,
        nvidiaSmi,
        dcgmi,
        nvcc,
        messages,
      })
    }, 1500)
  }

  // 开始健康检查
  const startHealthCheck = async () => {
    if (!commandStatus.tested || !commandStatus.nvidiaSmi) {
      alert(language === "zh" ? "请先完成连接测试和命令检查" : "Please complete connectivity and command checks first")
      return
    }

    setHealthCheckRunning(true)

    // 模拟健康检查
    setTimeout(() => {
      const mockResults = [
        {
          nodeHost,
          gpuType: "H200",
          nvbandwidthTest: "54.9 GB/s",
          p2pBandwidthLatencyTest: "736.40 GB/s",
          ncclTests: "150.946 GB/s",
          dcgmDiag: commandStatus.dcgmi ? "Pass" : "Skipped",
          ibCheck: "Pass",
          timestamp: new Date().toISOString(),
        },
      ]

      setHealthCheckResults(mockResults)
      setHealthCheckRunning(false)
    }, 3000)
  }

  const texts = {
    zh: {
      title: "自检专区",
      subtitle: "通过SSH连接进行节点健康检查",
      sshConfig: "SSH配置",
      sshConfigDesc: "配置SSH连接信息以访问目标节点",
      authMethod: "认证方式",
      password: "密码",
      privateKey: "私钥",
      nodeHost: "节点地址",
      nodeHostPlaceholder: "输入节点IP或主机名",
      username: "用户名",
      port: "端口",
      sshPassword: "SSH密码",
      sshPasswordPlaceholder: "输入SSH密码",
      sshPrivateKey: "SSH私钥",
      sshPrivateKeyPlaceholder: "粘贴私钥内容或上传文件",
      uploadKey: "上传私钥文件",
      testConnection: "测试连接",
      testing: "测试中...",
      commandCheck: "命令检测",
      commandCheckDesc: "检查必需命令是否可用",
      checkCommands: "检测命令",
      checking: "检测中...",
      healthCheck: "健康检查",
      healthCheckDesc: "选择检查项目并开始节点健康检查",
      selectItems: "选择检查项目",
      dcgmLevel: "DCGM诊断级别",
      startCheck: "开始健康检查",
      checking: "检查中...",
      results: "检查结果",
      noResults: "暂无检查结果",
      benchmarkValues: "GPU性能基准值",
      benchmarkDesc: "各GPU型号的性能基准参考值",
      testItemsDesc: "检查项目说明",
      connectedNodes: "已连接节点",
      noConnectedNodes: "暂无已连接的节点",
      removeNode: "移除",
    },
    en: {
      title: "Self-Check Zone",
      subtitle: "Perform node health checks via SSH connection",
      sshConfig: "SSH Configuration",
      sshConfigDesc: "Configure SSH connection to access target nodes",
      authMethod: "Authentication Method",
      password: "Password",
      privateKey: "Private Key",
      nodeHost: "Node Host",
      nodeHostPlaceholder: "Enter node IP or hostname",
      username: "Username",
      port: "Port",
      sshPassword: "SSH Password",
      sshPasswordPlaceholder: "Enter SSH password",
      sshPrivateKey: "SSH Private Key",
      sshPrivateKeyPlaceholder: "Paste private key content or upload file",
      uploadKey: "Upload Private Key",
      testConnection: "Test Connection",
      testing: "Testing...",
      commandCheck: "Command Check",
      commandCheckDesc: "Check if required commands are available",
      checkCommands: "Check Commands",
      checking: "Checking...",
      healthCheck: "Health Check",
      healthCheckDesc: "Select check items and start node health check",
      selectItems: "Select Check Items",
      dcgmLevel: "DCGM Diagnostic Level",
      startCheck: "Start Health Check",
      checking: "Checking...",
      results: "Check Results",
      noResults: "No results yet",
      benchmarkValues: "GPU Performance Benchmarks",
      benchmarkDesc: "Performance benchmark reference values for GPU models",
      testItemsDesc: "Check Items Description",
      connectedNodes: "Connected Nodes",
      noConnectedNodes: "No connected nodes yet",
      removeNode: "Remove",
    },
  }

  const currentTexts = texts[language]
  const currentCheckItems = checkItems[language]

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="space-y-2">
        <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-500 via-purple-500 to-cyan-500 bg-clip-text text-transparent">
          {currentTexts.title}
        </h1>
        <p className="text-muted-foreground">{currentTexts.subtitle}</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column - SSH Config and Command Check */}
        <div className="lg:col-span-1 space-y-6">
          {/* SSH Configuration */}
          <Card className="tech-card">
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Key className="w-5 h-5 text-tech-blue" />
                <span>{currentTexts.sshConfig}</span>
              </CardTitle>
              <CardDescription>{currentTexts.sshConfigDesc}</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Node Host */}
              <div className="space-y-2">
                <Label>{currentTexts.nodeHost}</Label>
                <Input
                  placeholder={currentTexts.nodeHostPlaceholder}
                  value={nodeHost}
                  onChange={(e) => setNodeHost(e.target.value)}
                  className="tech-input"
                />
              </div>

              {/* Username and Port */}
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label>{currentTexts.username}</Label>
                  <Input value={sshUsername} onChange={(e) => setSshUsername(e.target.value)} className="tech-input" />
                </div>
                <div className="space-y-2">
                  <Label>{currentTexts.port}</Label>
                  <Input value={sshPort} onChange={(e) => setSshPort(e.target.value)} className="tech-input" />
                </div>
              </div>

              {/* Auth Method Tabs */}
              <div className="space-y-2">
                <Label>{currentTexts.authMethod}</Label>
                <Tabs value={sshAuthType} onValueChange={(v) => setSshAuthType(v as any)}>
                  <TabsList className="grid w-full grid-cols-2">
                    <TabsTrigger value="password">
                      <Key className="w-4 h-4 mr-2" />
                      {currentTexts.password}
                    </TabsTrigger>
                    <TabsTrigger value="privatekey">
                      <FileKey className="w-4 h-4 mr-2" />
                      {currentTexts.privateKey}
                    </TabsTrigger>
                  </TabsList>
                  <TabsContent value="password" className="space-y-2">
                    <Input
                      type="password"
                      placeholder={currentTexts.sshPasswordPlaceholder}
                      value={sshPassword}
                      onChange={(e) => setSshPassword(e.target.value)}
                      className="tech-input"
                    />
                  </TabsContent>
                  <TabsContent value="privatekey" className="space-y-2">
                    <Textarea
                      placeholder={currentTexts.sshPrivateKeyPlaceholder}
                      value={sshPrivateKey}
                      onChange={(e) => setSshPrivateKey(e.target.value)}
                      className="tech-input min-h-[120px] font-mono text-sm"
                    />
                    <Button variant="outline" size="sm" className="w-full bg-transparent">
                      <Upload className="w-4 h-4 mr-2" />
                      {currentTexts.uploadKey}
                    </Button>
                  </TabsContent>
                </Tabs>
              </div>

              {/* Test Connection Button */}
              <Button
                onClick={testConnectivity}
                className="w-full tech-button"
                disabled={!nodeHost || (connectivityStatus.tested && !connectivityStatus.connected)}
              >
                <Wifi className="w-4 h-4 mr-2" />
                {connectivityStatus.tested && !connectivityStatus.connected
                  ? currentTexts.testConnection
                  : currentTexts.testing}
              </Button>

              {/* Connection Status */}
              {connectivityStatus.tested && (
                <Alert variant={connectivityStatus.connected ? "default" : "destructive"}>
                  {connectivityStatus.connected ? <CheckCircle className="h-4 w-4" /> : <XCircle className="h-4 w-4" />}
                  <AlertDescription>{connectivityStatus.message}</AlertDescription>
                </Alert>
              )}
            </CardContent>
          </Card>

          {/* Command Check */}
          <Card className="tech-card">
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Terminal className="w-5 h-5 text-tech-green" />
                <span>{currentTexts.commandCheck}</span>
              </CardTitle>
              <CardDescription>{currentTexts.commandCheckDesc}</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <Button
                onClick={checkCommands}
                className="w-full tech-button"
                disabled={!connectivityStatus.connected || commandStatus.tested}
              >
                <Search className="w-4 h-4 mr-2" />
                {commandStatus.tested ? currentTexts.checkCommands : currentTexts.checking}
              </Button>

              {/* Command Status */}
              {commandStatus.tested && (
                <div className="space-y-2">
                  <div className="flex items-center justify-between p-2 bg-secondary/30 rounded-lg">
                    <span className="text-sm font-mono">nvidia-smi</span>
                    {commandStatus.nvidiaSmi ? (
                      <CheckCircle className="w-4 h-4 text-tech-green" />
                    ) : (
                      <XCircle className="w-4 h-4 text-tech-red" />
                    )}
                  </div>
                  <div className="flex items-center justify-between p-2 bg-secondary/30 rounded-lg">
                    <span className="text-sm font-mono">dcgmi</span>
                    {commandStatus.dcgmi ? (
                      <CheckCircle className="w-4 h-4 text-tech-green" />
                    ) : (
                      <XCircle className="w-4 h-4 text-tech-red" />
                    )}
                  </div>
                  <div className="flex items-center justify-between p-2 bg-secondary/30 rounded-lg">
                    <span className="text-sm font-mono">nvcc</span>
                    {commandStatus.nvcc ? (
                      <CheckCircle className="w-4 h-4 text-tech-green" />
                    ) : (
                      <XCircle className="w-4 h-4 text-tech-red" />
                    )}
                  </div>

                  {/* Messages */}
                  <div className="space-y-2 mt-4">
                    {commandStatus.messages.map((msg, idx) => (
                      <Alert key={idx} variant={msg.includes("✅") ? "default" : "destructive"}>
                        {msg.includes("✅") ? (
                          <CheckCircle className="h-4 w-4" />
                        ) : (
                          <AlertTriangle className="h-4 w-4" />
                        )}
                        <AlertDescription className="text-xs">{msg}</AlertDescription>
                      </Alert>
                    ))}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Right Column - Health Check and Results */}
        <div className="lg:col-span-2 space-y-6">
          {/* Health Check Configuration */}
          <Card className="tech-card">
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Zap className="w-5 h-5 text-tech-yellow" />
                <span>{currentTexts.healthCheck}</span>
              </CardTitle>
              <CardDescription>{currentTexts.healthCheckDesc}</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Select Check Items */}
              <div className="space-y-3">
                <Label className="text-base font-semibold">{currentTexts.selectItems}</Label>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  {currentCheckItems.map((item) => (
                    <div key={item.id} className="flex items-start space-x-3 p-3 bg-secondary/30 rounded-lg">
                      <Checkbox
                        id={item.id}
                        checked={selectedCheckItems.includes(item.id)}
                        onCheckedChange={(checked) => {
                          if (checked) {
                            setSelectedCheckItems([...selectedCheckItems, item.id])
                          } else {
                            setSelectedCheckItems(selectedCheckItems.filter((i) => i !== item.id))
                          }
                        }}
                        className="mt-1 burnin-checkbox"
                      />
                      <div className="space-y-1 flex-1">
                        <Label htmlFor={item.id} className="text-sm font-semibold cursor-pointer">
                          {item.label}
                        </Label>
                        <p className="text-xs text-muted-foreground">{item.description}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* DCGM Level */}
              <div className="space-y-2">
                <Label>{currentTexts.dcgmLevel}</Label>
                <Select value={dcgmLevel.toString()} onValueChange={(v) => setDcgmLevel(Number.parseInt(v))}>
                  <SelectTrigger className="tech-input">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {dcgmLevels.map((level) => (
                      <SelectItem key={level.value} value={level.value.toString()}>
                        {level.label} - {level.description[language]}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {/* Start Check Button */}
              <Button
                onClick={startHealthCheck}
                className="w-full tech-button"
                disabled={
                  !commandStatus.tested ||
                  !commandStatus.nvidiaSmi ||
                  healthCheckRunning ||
                  selectedCheckItems.length === 0
                }
              >
                <Play className="w-4 h-4 mr-2" />
                {healthCheckRunning ? currentTexts.checking : currentTexts.startCheck}
              </Button>
            </CardContent>
          </Card>

          {/* Results */}
          {healthCheckResults.length > 0 && (
            <Card className="tech-card">
              <CardHeader>
                <CardTitle>{currentTexts.results}</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="overflow-x-auto">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>{language === "zh" ? "节点" : "Node"}</TableHead>
                        <TableHead>{language === "zh" ? "GPU类型" : "GPU Type"}</TableHead>
                        {selectedCheckItems.includes("nvbandwidthTest") && <TableHead>nvBandwidth</TableHead>}
                        {selectedCheckItems.includes("p2pBandwidthLatencyTest") && <TableHead>P2P</TableHead>}
                        {selectedCheckItems.includes("ncclTests") && <TableHead>NCCL</TableHead>}
                        {selectedCheckItems.includes("dcgmDiag") && <TableHead>DCGM</TableHead>}
                        {selectedCheckItems.includes("ibCheck") && <TableHead>IB</TableHead>}
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {healthCheckResults.map((result, idx) => (
                        <TableRow key={idx}>
                          <TableCell className="font-mono">{result.nodeHost}</TableCell>
                          <TableCell>
                            <Badge variant="outline">{result.gpuType}</Badge>
                          </TableCell>
                          {selectedCheckItems.includes("nvbandwidthTest") && (
                            <TableCell className="text-tech-green">{result.nvbandwidthTest}</TableCell>
                          )}
                          {selectedCheckItems.includes("p2pBandwidthLatencyTest") && (
                            <TableCell className="text-tech-green">{result.p2pBandwidthLatencyTest}</TableCell>
                          )}
                          {selectedCheckItems.includes("ncclTests") && (
                            <TableCell className="text-tech-green">{result.ncclTests}</TableCell>
                          )}
                          {selectedCheckItems.includes("dcgmDiag") && (
                            <TableCell>
                              {result.dcgmDiag === "Pass" ? (
                                <Badge variant="default" className="bg-tech-green">
                                  Pass
                                </Badge>
                              ) : (
                                <Badge variant="secondary">Skipped</Badge>
                              )}
                            </TableCell>
                          )}
                          {selectedCheckItems.includes("ibCheck") && (
                            <TableCell>
                              <Badge variant="default" className="bg-tech-green">
                                Pass
                              </Badge>
                            </TableCell>
                          )}
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </div>

      {/* Bottom Section - Benchmark Values and Test Descriptions */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* GPU Benchmark Values */}
        <Card className="tech-card">
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Info className="w-5 h-5 text-tech-cyan" />
              <span>{currentTexts.benchmarkValues}</span>
            </CardTitle>
            <CardDescription>{currentTexts.benchmarkDesc}</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>{language === "zh" ? "GPU型号" : "GPU Model"}</TableHead>
                    <TableHead>P2P (GB/s)</TableHead>
                    <TableHead>NCCL (GB/s)</TableHead>
                    <TableHead>BW (GB/s)</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {Object.entries(gpuBenchmarks).map(([model, values]) => (
                    <TableRow key={model}>
                      <TableCell className="font-semibold">{model}</TableCell>
                      <TableCell>{values.p2p}</TableCell>
                      <TableCell>{values.nccl}</TableCell>
                      <TableCell>{values.bw}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          </CardContent>
        </Card>

        {/* Test Items Description */}
        <Card className="tech-card">
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Info className="w-5 h-5 text-tech-purple" />
              <span>{currentTexts.testItemsDesc}</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {currentCheckItems.map((item) => (
                <div key={item.id} className="p-3 bg-secondary/30 rounded-lg space-y-1">
                  <h4 className="font-semibold text-sm text-tech-blue">{item.label}</h4>
                  <p className="text-xs text-muted-foreground">{item.description}</p>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
