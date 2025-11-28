"use client"

import type React from "react"
import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Card } from "@/components/ui/card"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Badge } from "@/components/ui/badge"
import { Checkbox } from "@/components/ui/checkbox"
import {
  AlertCircle,
  CheckCircle2,
  Eye,
  EyeOff,
  FileText,
  Key,
  ListChecks,
  Loader2,
  Monitor,
  Play,
  Plus,
  Server,
  Terminal,
  Trash2,
  Upload,
  XCircle,
} from "lucide-react"
import JSZip from "jszip"
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { useToast } from "@/components/ui/use-toast"

type AuthMethod = "password" | "privateKey"
type CommandStatus = "idle" | "checking" | "available" | "missing"

interface SSHConnectionPayload {
  host: string
  port: number
  username: string
  auth: {
    type: "password" | "privateKey"
    value: string
    passphrase?: string
  }
  sudoPassword?: string
}

interface SSHTestResult {
  hostname: string
  gpus: string[]
  driverVersion?: string
}

interface SelectedNode {
  id: string
  alias: string
  host: string
  port: string
  username: string
  authMethod: AuthMethod
  password?: string
  privateKey?: string
  sudoPassword?: string
  summary?: SSHTestResult
}

interface JobMetric {
  status?: string
  value?: number
  unit?: string
  benchmark?: number
  passed?: boolean
  message?: string
}

interface JobNodeStatus {
  nodeId: string
  alias: string
  host: string
  status: string
  gpuType?: string
  results?: Record<string, JobMetric>
  executionLog?: string
  completedAt?: string
}

interface JobDetail {
  jobId: string
  status: string
  nodes: JobNodeStatus[]
}

interface JobNodeResult {
  id: string
  jobId: string
  alias: string
  hostname: string
  host: string
  gpuType?: string
  nvbandwidth?: JobMetric
  p2p?: JobMetric
  nccl?: JobMetric
  dcgm?: JobMetric
  ib?: JobMetric
  status: string
  executionLog?: string
  completedAt?: string
}

type BenchmarkMap = Record<string, { p2p: number; nccl: number; bw: number }>

// 动态获取API地址：优先使用环境变量，否则根据当前页面地址推断
function normalizeBaseUrl(url: string): string {
  if (!url) return url
  // 确保没有多余的结尾斜杠（保留单个 "/"）
  if (url.length > 1 && url.endsWith("/")) {
    return url.slice(0, -1)
  }
  return url
}

function getApiBaseUrl(): string {
  // 1. 优先使用构建时注入的环境变量
  if (process.env.NEXT_PUBLIC_GHX_API) {
    return normalizeBaseUrl(process.env.NEXT_PUBLIC_GHX_API)
  }

  // 2. 浏览器运行时：使用当前 origin 并固定走 /api，方便通过反向代理统一转发
  if (typeof window !== "undefined") {
    const origin = window.location.origin
    // 保持本地开发场景兼容：如果是 localhost，仍然可以访问 5000 端口
    if (origin.includes("localhost") || origin.includes("127.0.0.1")) {
      return "http://localhost:5000"
    }
    return `${origin}/api`
  }

  // 3. SSR 或未知环境，默认指向 /api，由上层代理处理
  return "/api"
}

const sanitizeToken = (value: string, fallback: string) => {
  const cleaned = value.replace(/[^0-9a-zA-Z]/g, "")
  return cleaned.length > 0 ? cleaned.toLowerCase() : fallback
}

const extractGpuModel = (summary?: SSHTestResult) => {
  if (!summary?.gpus?.length) return ""
  const first = summary.gpus[0]
  const match = first.match(/NVIDIA\s+([A-Za-z0-9-]+)/i)
  if (match?.[1]) return match[1].toLowerCase()
  const parts = first.split(":")
  return sanitizeToken(parts.pop()?.trim() || "", "gpu")
}

const formatJobName = (node: SelectedNode) => {
  const gpu = extractGpuModel(node.summary) || sanitizeToken(node.alias, "node")
  const hostToken = sanitizeToken(node.host, "host")
  const now = new Date()
  const pad = (num: number) => num.toString().padStart(2, "0")
  const timestamp = `${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}${pad(now.getHours())}${pad(
    now.getMinutes(),
  )}`
  return `job-${gpu}-${hostToken}-${timestamp}`
}

const downloadTextFile = (filename: string, content: string) => {
  const blob = new Blob([content], { type: "text/plain;charset=utf-8" })
  downloadBlob(filename, blob)
}

const downloadBlob = (filename: string, blob: Blob) => {
  const url = URL.createObjectURL(blob)
  const link = document.createElement("a")
  link.href = url
  link.download = filename
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
  URL.revokeObjectURL(url)
}

const API_BASE_URL = getApiBaseUrl()
const LANGUAGE_STORAGE_KEY = "ghx-language"

async function apiRequest<T>(path: string, options: RequestInit = {}): Promise<T> {
  const base = API_BASE_URL
  const url =
    base.endsWith("/") && path.startsWith("/")
      ? `${base.slice(0, -1)}${path}`
      : `${base}${path}`
  console.debug("[API Request]", options.method || "GET", url)
  
  const response = await fetch(url, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
    ...options,
  })

  const payload = await response.json().catch(() => ({}))
  if (!response.ok || payload.success === false) {
    const errorMsg = payload.message || `HTTP ${response.status}: ${response.statusText}`
    console.error("[API Error]", url, errorMsg, payload)
    throw new Error(errorMsg)
  }
  return payload.data as T
}

// 动画logo组件
function AnimatedLogo() {
  return (
    <div className="flex items-center gap-3">
      <div className="flex items-center justify-center w-12 h-12 bg-gradient-to-br from-blue-500/20 to-purple-500/20 rounded-lg border border-blue-500/30">
        <Monitor className="w-6 h-6 text-blue-400" />
      </div>
      <div className="flex flex-col">
        <div className="flex items-center gap-2">
          <h1 className="text-xl font-bold bg-gradient-to-r from-blue-400 via-cyan-400 to-teal-400 bg-clip-text text-transparent">
            GHealthX
          </h1>
          <div className="flex gap-1">
            {[0, 1, 2].map((i) => (
              <div
                key={i}
                className="w-1.5 h-1.5 rounded-full bg-cyan-400"
                style={{
                  animation: `pulse 1.5s ease-in-out ${i * 0.2}s infinite`,
                }}
              />
            ))}
          </div>
        </div>
        <div className="flex items-center gap-2 mt-0.5">
          <span className="text-xs text-cyan-400/80">GPU Health Expert</span>
          <Badge variant="outline" className="text-[10px] px-1.5 py-0 border-green-500/50 text-green-400">
            LIVE
          </Badge>
        </div>
      </div>
    </div>
  )
}

// 检查项配置
const CHECK_ITEMS = [
  {
    id: "nvbandwidth",
    name: { zh: "nvBandwidthTest", en: "nvBandwidthTest" },
    description: {
      zh: "测试CPU与GPU间内存拷贝带宽性能，使用nvbandwidth工具评估数据传输效率",
      en: "Measure CPU↔GPU memory copy bandwidth with nvbandwidth to evaluate throughput",
    },
  },
  {
    id: "p2p",
    name: { zh: "p2pBandwidthLatencyTest", en: "p2pBandwidthLatencyTest" },
    description: {
      zh: "测试GPU间点对点通信带宽和延迟，评估多GPU协作性能",
      en: "Test GPU peer-to-peer bandwidth and latency to gauge multi-GPU collaboration",
    },
  },
  {
    id: "nccl",
    name: { zh: "NCCL测试", en: "NCCL Test" },
    description: {
      zh: "测试NVIDIA集合通信库性能，评估分布式训练通信效率",
      en: "Benchmark NVIDIA NCCL collective communication performance",
    },
  },
  {
    id: "dcgm",
    name: { zh: "DCGM诊断", en: "DCGM Diagnostics" },
    description: {
      zh: "NVIDIA数据中心GPU管理器诊断，检查GPU硬件健康状态",
      en: "Run NVIDIA DCGM diagnostics to assess GPU health",
    },
  },
  {
    id: "ib",
    name: { zh: "IB检查", en: "IB Check" },
    description: {
      zh: "InfiniBand网络连接检查，确保高速网络通信正常",
      en: "Inspect InfiniBand networking to ensure high-speed connectivity",
    },
  },
]

const DEFAULT_GPU_BENCHMARKS: BenchmarkMap = {
  "RTX 3090": { p2p: 18, nccl: 7, bw: 20 },
  L40S: { p2p: 28, nccl: 9, bw: 20 },
  "RTX 4090": { p2p: 18, nccl: 7, bw: 20 },
  A100: { p2p: 420, nccl: 70, bw: 20 },
  A800: { p2p: 340, nccl: 55, bw: 20 },
  H100: { p2p: 700, nccl: 139, bw: 40 },
  H800: { p2p: 340, nccl: 65, bw: 47 },
  H200: { p2p: 730, nccl: 145, bw: 54 },
}

// 核心命令配置
const CORE_COMMANDS = [
  { name: "nvidia-smi", description: { zh: "NVIDIA GPU驱动管理工具", en: "NVIDIA GPU management utility" } },
  {
    name: "dcgmi",
    description: { zh: "NVIDIA数据中心GPU管理器", en: "NVIDIA Data Center GPU Manager" },
    package: "datacenter-gpu-manager",
  },
  {
    name: "/usr/local/cuda/bin/nvcc",
    description: { zh: "CUDA编译器", en: "CUDA compiler" },
    package: "CUDA Toolkit",
  },
  {
    name: "ibstat",
    description: { zh: "InfiniBand状态查询工具", en: "InfiniBand status tool" },
    package: "OFED IB",
  },
]

const generateId = () => (crypto?.randomUUID ? crypto.randomUUID() : Math.random().toString(36).slice(2))

const METRIC_LABELS = {
  zh: { notRun: "未执行", error: "错误", skipped: "已跳过", benchmark: "基准值" },
  en: { notRun: "Not run", error: "Error", skipped: "Skipped", benchmark: "Benchmark" },
}

const STATUS_LABELS = {
  zh: { passed: "通过", running: "执行中", failed: "异常" },
  en: { passed: "Passed", running: "Running", failed: "Abnormal" },
}

const renderMetric = (metric?: JobMetric, unitHint = "GB/s", lang: "zh" | "en" = "zh") => {
  const labels = METRIC_LABELS[lang]
  if (!metric) {
    return <span className="text-slate-500 text-xs">{labels.notRun}</span>
  }
  if (metric.status === "error") {
    return <span className="text-red-400 text-xs">{metric.message || labels.error}</span>
  }
  if (metric.status === "skipped") {
    return <span className="text-slate-400 text-xs">{labels.skipped}</span>
  }
  if (typeof metric.value !== "number") {
    return <span className="text-slate-300 text-xs">{metric.status || "--"}</span>
  }
  const valueText = `${metric.value.toFixed(1)} ${metric.unit || unitHint}`
  const passed = metric.passed !== false && (metric.status === "passed" || (metric.benchmark && metric.value >= metric.benchmark))
  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center gap-2">
        <span className="text-white">{valueText}</span>
        {passed && <CheckCircle2 className="w-4 h-4 text-green-400 flex-shrink-0" />}
        {!passed && metric.status === "failed" && <XCircle className="w-4 h-4 text-red-400 flex-shrink-0" />}
      </div>
      {metric.benchmark && (
        <span className="text-xs text-slate-400">
          {labels.benchmark}: {metric.benchmark} {metric.unit || unitHint}
        </span>
      )}
    </div>
  )
}

const renderStatusBadge = (status: string, lang: "zh" | "en" = "zh") => {
  const labels = STATUS_LABELS[lang]
  const normalized = status?.toLowerCase()
  if (normalized === "passed" || normalized === "success" || normalized === "completed") {
    return (
      <Badge className="bg-green-500/20 text-green-400 border border-green-500/40">
        <CheckCircle2 className="w-3 h-3 mr-1" />
        {labels.passed}
      </Badge>
    )
  }
  if (normalized === "running") {
    return (
      <Badge className="bg-blue-500/20 text-blue-400 border border-blue-500/40">
        <Loader2 className="w-3 h-3 mr-1 animate-spin" />
        {labels.running}
      </Badge>
    )
  }
  return (
    <Badge className="bg-red-500/20 text-red-400 border border-red-500/40">
      <XCircle className="w-3 h-3 mr-1" />
      {labels.failed}
    </Badge>
  )
}

const flattenJobNodes = (job: JobDetail): JobNodeResult[] => {
  return (job.nodes || []).map((node) => ({
    id: `${job.jobId}-${node.nodeId}`,
    jobId: job.jobId,
    alias: node.alias || node.host,
    hostname: node.alias || node.host,
    host: node.host,
    gpuType: node.gpuType,
    nvbandwidth: node.results?.nvbandwidth,
    p2p: node.results?.p2p,
    nccl: node.results?.nccl,
    dcgm: node.results?.dcgm,
    ib: node.results?.ib,
    status: node.status,
    executionLog: node.executionLog,
    completedAt: node.completedAt,
  }))
}

export default function BareMetal() {
  const { toast } = useToast()
  const [language, setLanguage] = useState<"zh" | "en">("zh")
  const [authMethod, setAuthMethod] = useState<AuthMethod>("password")
  const [sshConfig, setSshConfig] = useState({
    alias: "",
    host: "",
    username: "root",
    port: "22",
    password: "",
    privateKey: "",
    sudoPassword: "",
  })
  const [privateKeyFileName, setPrivateKeyFileName] = useState<string>("")
  const [showPassword, setShowPassword] = useState(false)
  const [showSudoPassword, setShowSudoPassword] = useState(false)
  const [sshStatus, setSshStatus] = useState<"idle" | "testing" | "success" | "error">("idle")
  const [commandStatus, setCommandStatus] = useState<Record<string, CommandStatus>>(
    Object.fromEntries(CORE_COMMANDS.map((cmd) => [cmd.name, "idle"])),
  )
  const [selectedItems, setSelectedItems] = useState<string[]>(["nvbandwidth", "p2p", "nccl", "dcgm", "ib"])
  const [dcgmLevel, setDcgmLevel] = useState("2")
  const [isTestingSSH, setIsTestingSSH] = useState(false)
  const [isCheckingCommands, setIsCheckingCommands] = useState(false)
  const [isRunningTests, setIsRunningTests] = useState(false)
  const [selectedNodes, setSelectedNodes] = useState<SelectedNode[]>([])
  const [selectedNodeIds, setSelectedNodeIds] = useState<Set<string>>(new Set())
  const [currentPage, setCurrentPage] = useState(1)
  const [pageSize] = useState(10)
  const [lastTestDetails, setLastTestDetails] = useState<SSHTestResult | null>(null)

  // 从 localStorage 加载节点列表
  useEffect(() => {
    const saved = localStorage.getItem("ghx-baremetal-selected-nodes")
    if (saved) {
      try {
        const nodes = JSON.parse(saved) as SelectedNode[]
        setSelectedNodes(nodes)
      } catch (error) {
        console.error("Failed to load saved nodes:", error)
      }
    }
  }, [])

  // 保存节点列表到 localStorage（包含密码和私钥，用于持久化）
  useEffect(() => {
    if (selectedNodes.length > 0) {
      localStorage.setItem("ghx-baremetal-selected-nodes", JSON.stringify(selectedNodes))
    } else {
      localStorage.removeItem("ghx-baremetal-selected-nodes")
    }
  }, [selectedNodes])
  const [currentJobId, setCurrentJobId] = useState<string | null>(null)
  const [currentJob, setCurrentJob] = useState<JobDetail | null>(null)
  const [isPollingJob, setIsPollingJob] = useState(false)
  const [testResults, setTestResults] = useState<JobNodeResult[]>([])
  const [selectedResultIds, setSelectedResultIds] = useState<Set<string>>(new Set())
  const [resultsPage, setResultsPage] = useState(1)
  const [resultsPageSize] = useState(10)
  const [isExportingLogs, setIsExportingLogs] = useState(false)
  const tr = useCallback((zh: string, en: string) => (language === "zh" ? zh : en), [language])
  useEffect(() => {
    if (typeof window === "undefined") return
    const saved = localStorage.getItem(LANGUAGE_STORAGE_KEY)
    if (saved === "zh" || saved === "en") {
      setLanguage(saved)
    }
  }, [])
  const [logViewer, setLogViewer] = useState({ open: false, content: "", title: "" })

  // 从 localStorage 加载测试结果
  useEffect(() => {
    const saved = localStorage.getItem("ghx-baremetal-test-results")
    if (saved) {
      try {
        const results = JSON.parse(saved) as JobNodeResult[]
        setTestResults(results)
      } catch (error) {
        console.error("Failed to load saved test results:", error)
      }
    }
  }, [])

  // 保存测试结果到 localStorage
  useEffect(() => {
    if (testResults.length > 0) {
      localStorage.setItem("ghx-baremetal-test-results", JSON.stringify(testResults))
    } else {
      localStorage.removeItem("ghx-baremetal-test-results")
    }
  }, [testResults])
  const pollErrorOnceRef = useRef(false)
  const [gpuBenchmarks, setGpuBenchmarks] = useState<BenchmarkMap>(DEFAULT_GPU_BENCHMARKS)
  const benchmarkEntries = useMemo(() => Object.entries(gpuBenchmarks), [gpuBenchmarks])

  const allCommandsAvailable = useMemo(
    () => CORE_COMMANDS.every((cmd) => commandStatus[cmd.name] === "available"),
    [commandStatus],
  )

  const canRunTests = selectedNodes.length > 0 && selectedItems.length > 0 && !isRunningTests && !isPollingJob

  const buildConnectionPayload = (): SSHConnectionPayload => {
    if (!sshConfig.host) {
      throw new Error(tr("请填写节点地址", "Please enter the node address"))
    }
    if (!sshConfig.username) {
      throw new Error(tr("请填写用户名", "Please enter the username"))
    }
    const port = Number(sshConfig.port) || 22
    const authValue = authMethod === "password" ? sshConfig.password : sshConfig.privateKey
    if (!authValue) {
      throw new Error(
        authMethod === "password" ? tr("请输入密码", "Please enter password") : tr("请输入私钥", "Please enter private key"),
      )
    }
    return {
      host: sshConfig.host,
      port,
      username: sshConfig.username,
      auth: {
        type: authMethod,
        value: authValue,
      },
      sudoPassword: sshConfig.sudoPassword || (authMethod === "password" ? sshConfig.password : undefined),
    }
  }

  const handleTestSSH = async () => {
    try {
    setIsTestingSSH(true)
    setSshStatus("testing")
      const connection = buildConnectionPayload()
      const data = await apiRequest<SSHTestResult>("/api/ssh/test-connection", {
        method: "POST",
        body: JSON.stringify({ connection }),
      })
      setSshStatus("success")
      setLastTestDetails(data)
      toast({
        title: tr("SSH连接成功", "SSH connection succeeded"),
        description: `${tr("主机", "Host")}: ${data.hostname || connection.host}`,
      })
    } catch (error) {
      setSshStatus("error")
      toast({
        title: tr("SSH连接失败", "SSH connection failed"),
        description: (error as Error).message,
        variant: "destructive",
      })
    } finally {
    setIsTestingSSH(false)
    }
  }

  const handleCheckCommands = async () => {
    try {
    setIsCheckingCommands(true)
      const connection = buildConnectionPayload()
      const data = await apiRequest<{ commands: Record<string, boolean> }>("/api/ssh/check-commands", {
        method: "POST",
        body: JSON.stringify({ connection, commands: CORE_COMMANDS.map((cmd) => cmd.name) }),
      })
      const nextStatus: Record<string, CommandStatus> = {}
    for (const cmd of CORE_COMMANDS) {
        nextStatus[cmd.name] = data.commands[cmd.name] ? "available" : "missing"
      }
      setCommandStatus(nextStatus)
    } catch (error) {
      toast({
        title: tr("命令检查失败", "Command check failed"),
        description: (error as Error).message,
        variant: "destructive",
      })
    } finally {
      setIsCheckingCommands(false)
    }
  }

  const handlePrivateKeyUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      const reader = new FileReader()
      reader.onload = (event) => {
        const content = event.target?.result as string
        setSshConfig((prev) => ({ ...prev, privateKey: content }))
        setPrivateKeyFileName(file.name)
      }
      reader.readAsText(file)
    }
  }

  const handleAddNode = () => {
    if (sshStatus !== "success") {
      toast({
        title: tr("请先通过SSH测试", "Please run the SSH test first"),
        description: tr("完成连接测试后才能添加节点", "Add nodes after a successful connection test"),
        variant: "destructive",
      })
      return
    }
    try {
      const connection = buildConnectionPayload()
      // 检查是否已存在相同地址的节点
      const exists = selectedNodes.some(
        (node) => node.host === connection.host && node.port === String(connection.port),
      )
      if (exists) {
        toast({
          title: tr("节点已存在", "Node already exists"),
          description: tr("该节点地址已在待检查列表中", "This host is already in the pending list"),
          variant: "destructive",
        })
        return
      }
      const newNode: SelectedNode = {
        id: generateId(),
        alias: sshConfig.alias || connection.host,
        host: connection.host,
        port: String(connection.port),
        username: connection.username,
        authMethod,
        password: authMethod === "password" ? sshConfig.password : undefined,
        privateKey: authMethod === "privateKey" ? sshConfig.privateKey : undefined,
        sudoPassword: sshConfig.sudoPassword || (authMethod === "password" ? sshConfig.password : undefined),
        summary: lastTestDetails || undefined,
      }
      setSelectedNodes((prev) => [newNode, ...prev])
      toast({
        title: tr("节点已加入待检查列表", "Node added to pending list"),
        description: newNode.alias,
      })
    } catch (error) {
      toast({
        title: tr("无法添加节点", "Unable to add node"),
        description: (error as Error).message,
        variant: "destructive",
      })
    }
  }

  const handleRemoveNode = (id: string) => {
    setSelectedNodes((prev) => prev.filter((node) => node.id !== id))
    setSelectedNodeIds((prev) => {
      const next = new Set(prev)
      next.delete(id)
      return next
    })
  }

  const handleRemoveSelectedNodes = () => {
    if (selectedNodeIds.size === 0) {
      toast({ title: tr("请选择要移除的节点", "Please select nodes to remove"), variant: "destructive" })
      return
    }
    const count = selectedNodeIds.size
    setSelectedNodes((prev) => prev.filter((node) => !selectedNodeIds.has(node.id)))
    setSelectedNodeIds(new Set())
    toast({
      title: tr("已移除选中节点", "Selected nodes removed"),
      description: `${tr("共移除", "Removed")} ${count} ${tr("个节点", "nodes")}`,
    })
  }

  const handleClearNodes = () => {
    setSelectedNodes([])
    setSelectedNodeIds(new Set())
  }

  const handleToggleNodeSelection = (id: string) => {
    setSelectedNodeIds((prev) => {
      const next = new Set(prev)
      if (next.has(id)) {
        next.delete(id)
      } else {
        next.add(id)
      }
      return next
    })
  }

  const handleSelectAll = () => {
    const paginatedNodes = selectedNodes.slice((currentPage - 1) * pageSize, currentPage * pageSize)
    const allSelected = paginatedNodes.every((node) => selectedNodeIds.has(node.id))
    setSelectedNodeIds((prev) => {
      const next = new Set(prev)
      if (allSelected) {
        // 取消全选当前页
        paginatedNodes.forEach((node) => next.delete(node.id))
      } else {
        // 全选当前页
        paginatedNodes.forEach((node) => next.add(node.id))
    }
      return next
    })
  }

  // 分页计算
  const totalPages = Math.ceil(selectedNodes.length / pageSize)
  const paginatedNodes = selectedNodes.slice((currentPage - 1) * pageSize, currentPage * pageSize)
  const currentPageAllSelected =
    paginatedNodes.length > 0 && paginatedNodes.every((node) => selectedNodeIds.has(node.id))

  // 测试结果分页计算
  const resultsTotalPages = Math.ceil(testResults.length / resultsPageSize)
  const paginatedResults = testResults.slice(
    (resultsPage - 1) * resultsPageSize,
    resultsPage * resultsPageSize,
  )

  const formatLogTimestamp = (value?: string) => {
    if (!value) return "--"
    try {
      return new Date(value).toISOString().replace(/[:]/g, "").split(".")[0]
    } catch {
      return "--"
    }
  }

  const handleExportResult = (result: JobNodeResult) => {
    if (!result.executionLog) {
      toast({
        title: tr("暂无日志可导出", "No logs available for export"),
        description: tr("该节点还未生成执行日志", "This node has not produced execution logs yet"),
        variant: "destructive",
      })
      return
    }
    const filename = `${result.alias || result.hostname}-log-${formatLogTimestamp(result.completedAt)}.txt`
    downloadTextFile(filename, result.executionLog)
  }

  const handleExportSelectedResults = async () => {
    const selected = testResults.filter((item) => selectedResultIds.has(item.id))
    if (!selected.length) {
      toast({ title: tr("请选择需要导出的节点", "Please select nodes to export"), variant: "destructive" })
      return
    }
    setIsExportingLogs(true)
    try {
      const zip = new JSZip()
      selected.forEach((item, index) => {
        const filename = `${item.alias || item.hostname || "node"}-${index + 1}-${formatLogTimestamp(item.completedAt)}.txt`
        const header = [
          `${tr("节点", "Node")}: ${item.alias || item.hostname} (${item.host})`,
          `${tr("完成时间", "Completed at")}: ${item.completedAt ? new Date(item.completedAt).toLocaleString(language === "zh" ? "zh-CN" : "en-US") : "--"}`,
          "",
        ].join("\n")
        zip.file(filename, `${header}${item.executionLog || tr("暂无日志", "No logs")}`)
      })
      const blob = await zip.generateAsync({ type: "blob" })
      const zipName = `ghx-logs-${new Date().toISOString().replace(/[:]/g, "").split(".")[0]}.zip`
      downloadBlob(zipName, blob)
      toast({ title: tr("已导出", "Exported"), description: `${selected.length} ${tr("条日志", "logs")}` })
    } catch (error) {
      console.error(error)
      toast({ title: tr("导出失败", "Export failed"), description: (error as Error).message, variant: "destructive" })
    } finally {
      setIsExportingLogs(false)
    }
  }

  const handleRunTests = async () => {
    // 如果有选中的节点，只运行选中的；否则运行所有节点
    const nodesToRun = selectedNodeIds.size > 0 
      ? selectedNodes.filter((node) => selectedNodeIds.has(node.id))
      : selectedNodes
    
    if (nodesToRun.length === 0) {
      toast({
        title: tr("请选择节点", "Please select nodes"),
        description: tr("请至少选择一个节点后再发起任务", "Select at least one node before starting a job"),
        variant: "destructive",
      })
      return
    }
    if (!selectedItems.length) {
      toast({
        title: tr("请选择检查项", "Please select test items"),
        description: tr("至少选择一个检测项目", "Select at least one test item"),
        variant: "destructive",
      })
      return
    }
    try {
    setIsRunningTests(true)
      setIsPollingJob(true)
      const nodesPayload = nodesToRun.map((node) => {
        const auth =
          node.authMethod === "password"
            ? { type: "password", value: node.password || "" }
            : { type: "privateKey", value: node.privateKey || "" }
        if (!auth.value) {
          throw new Error(tr("节点", "Node") + ` ${node.alias} ` + tr("缺少认证信息", "lacks authentication info"))
        }
        return {
          alias: node.alias,
          host: node.host,
          port: Number(node.port) || 22,
          username: node.username,
          auth,
          sudoPassword: node.sudoPassword || (node.authMethod === "password" ? node.password : undefined),
        }
      })
      const payload = {
        jobName: formatJobName(nodesToRun[0]),
        nodes: nodesPayload,
        tests: selectedItems,
        dcgmLevel: Number(dcgmLevel),
      }
      const data = await apiRequest<{ jobId: string }>("/api/gpu-inspection/create-job", {
        method: "POST",
        body: JSON.stringify(payload),
      })
      setCurrentJobId(data.jobId)
      setCurrentJob({
        jobId: data.jobId,
        status: "pending",
        nodes: nodesToRun.map((node) => ({
          nodeId: node.id,
          alias: node.alias,
          host: node.host,
          status: "pending",
        })),
      })
      pollErrorOnceRef.current = false
      toast({
        title: tr("任务已创建", "Job created"),
        description: `Job ID: ${data.jobId}`,
      })
    } catch (error) {
      setIsPollingJob(false)
      toast({
        title: tr("创建任务失败", "Failed to create job"),
        description: (error as Error).message,
        variant: "destructive",
      })
    } finally {
    setIsRunningTests(false)
    }
  }

  useEffect(() => {
    if (!currentJobId) return
    let cancelled = false

    const poll = async () => {
      if (cancelled) return
      try {
        const data = await apiRequest<JobDetail>(`/api/gpu-inspection/job/${currentJobId}`)
        if (cancelled) return
        setCurrentJob(data)
        if (data.status === "completed" || data.status === "failed") {
          setIsPollingJob(false)
          const flattened = flattenJobNodes(data)
          if (flattened.length) {
            setTestResults((prev) => {
              const nextKeys = new Set(flattened.map((item) => item.id))
              const remaining = prev.filter((item) => !nextKeys.has(item.id))
              const merged = [...flattened, ...remaining]
              // 按完成时间倒序排列
              return merged.sort((a, b) => {
                const timeA = a.completedAt ? new Date(a.completedAt).getTime() : 0
                const timeB = b.completedAt ? new Date(b.completedAt).getTime() : 0
                return timeB - timeA
              })
            })
          }
          toast({
            title: data.status === "completed" ? tr("诊断完成", "Diagnostics finished") : tr("诊断失败", "Diagnostics failed"),
            description: `${tr("任务", "Job")} ${data.jobId} ${tr("已结束", "has finished")}`,
            variant: data.status === "completed" ? "default" : "destructive",
          })
          return
        }
      } catch (error) {
        if (!pollErrorOnceRef.current) {
          toast({
            title: tr("获取任务状态失败", "Failed to get job status"),
            description: (error as Error).message,
            variant: "destructive",
          })
          pollErrorOnceRef.current = true
        }
      }
      setTimeout(poll, 4000)
  }

    poll()

    return () => {
      cancelled = true
    }
  }, [currentJobId, toast])

  const toggleLanguage = () => {
    setLanguage((prev) => {
      const next = prev === "zh" ? "en" : "zh"
      if (typeof window !== "undefined") {
        localStorage.setItem(LANGUAGE_STORAGE_KEY, next)
      }
      return next
    })
  }

  useEffect(() => {
    apiRequest<{ benchmarks: BenchmarkMap }>("/api/config/gpu-benchmarks")
      .then((data) => {
        if (data?.benchmarks) {
          setGpuBenchmarks(data.benchmarks)
        }
      })
      .catch((error) => {
        toast({
          title: tr("获取GPU基准值失败", "Failed to fetch GPU benchmarks"),
          description: (error as Error).message,
          variant: "destructive",
        })
      })
  }, [toast])

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-blue-950 to-slate-950">
      <header className="border-b border-blue-500/20 bg-slate-950/50 backdrop-blur-xl">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <AnimatedLogo />
            <Button
              variant="outline"
              size="sm"
              onClick={toggleLanguage}
              className="border-blue-500/30 text-blue-400 hover:bg-blue-500/10 bg-transparent"
            >
              🌐 {language === "zh" ? "中文" : "English"}
            </Button>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-6 py-8">
        <div className="mb-8">
          <h2 className="text-2xl font-bold text-white mb-2">{tr("裸金属专用版", "Bare-metal Edition")}</h2>
          <p className="text-sm text-slate-400">{tr("通过SSH连接进行节点健康检查", "Run GPU health checks via SSH access")}</p>
        </div>

        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6 mb-8">
          <Card className="bg-slate-900/50 border-blue-500/20 p-6">
            <div className="flex items-center gap-2 mb-6">
              <Key className="w-5 h-5 text-cyan-400" />
              <h3 className="text-lg font-semibold text-white">{tr("SSH配置", "SSH Configuration")}</h3>
            </div>
            <p className="text-sm text-slate-400 mb-6">
              {tr("配置SSH连接信息以访问目标节点", "Configure SSH access to reach target nodes")}
            </p>

            <div className="space-y-4">
              <div>
                <Label className="text-slate-300">
                  {tr("节点别名", "Node Alias")}{" "}
                  <span className="text-xs text-slate-500">{tr("(可选)", "(Optional)")}</span>
                </Label>
                <Input
                  value={sshConfig.alias}
                  onChange={(e) => setSshConfig((prev) => ({ ...prev, alias: e.target.value }))}
                  placeholder={tr("如：gpu-node-01（留空则使用节点地址）", "e.g. gpu-node-01 (leave blank to use host)")}
                  className="bg-slate-800/50 border-slate-700 text-white mt-2"
                />
              </div>

              <div>
                <Label className="text-slate-300">{tr("节点地址", "Host Address")}</Label>
                <Input
                  value={sshConfig.host}
                  onChange={(e) => setSshConfig((prev) => ({ ...prev, host: e.target.value }))}
                  placeholder={tr("输入节点IP或域名", "Enter host IP or domain")}
                  className="bg-slate-800/50 border-slate-700 text-white mt-2"
                />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Label className="text-slate-300">{tr("用户名", "Username")}</Label>
                  <Input
                    value={sshConfig.username}
                    onChange={(e) => setSshConfig((prev) => ({ ...prev, username: e.target.value }))}
                    className="bg-slate-800/50 border-slate-700 text-white mt-2"
                  />
                </div>
                <div>
                  <Label className="text-slate-300">{tr("端口", "Port")}</Label>
                  <Input
                    value={sshConfig.port}
                    onChange={(e) => setSshConfig((prev) => ({ ...prev, port: e.target.value }))}
                    className="bg-slate-800/50 border-slate-700 text-white mt-2"
                  />
                </div>
              </div>

              <div>
                <Label className="text-slate-300 mb-3 block">{tr("认证方式", "Authentication")}</Label>
                <Tabs value={authMethod} onValueChange={(v) => setAuthMethod(v as AuthMethod)}>
                  <TabsList className="grid w-full grid-cols-2 bg-slate-800/50">
                    <TabsTrigger value="password">{tr("密码", "Password")}</TabsTrigger>
                    <TabsTrigger value="privateKey">{tr("私钥", "Private Key")}</TabsTrigger>
                  </TabsList>
                </Tabs>
              </div>

              {authMethod === "password" ? (
                <div>
                  <Label className="text-slate-300">{tr("SSH密码", "SSH Password")}</Label>
                  <div className="relative mt-2">
                  <Input
                      type={showPassword ? "text" : "password"}
                    value={sshConfig.password}
                      onChange={(e) => setSshConfig((prev) => ({ ...prev, password: e.target.value }))}
                    placeholder={tr("输入SSH密码", "Enter SSH password")}
                      className="bg-slate-800/50 border-slate-700 text-white pr-10"
                  />
                    <button
                      type="button"
                      onClick={() => setShowPassword(!showPassword)}
                      className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-200 transition-colors"
                      aria-label={showPassword ? tr("隐藏密码", "Hide password") : tr("显示密码", "Show password")}
                    >
                      {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                    </button>
                  </div>
                </div>
              ) : (
                <div>
                  <Label className="text-slate-300">{tr("私钥内容", "Private Key Content")}</Label>
                  <div className="space-y-3 mt-2">
                      <label className="flex-1">
                        <div className="flex items-center gap-2 bg-slate-800/50 border border-slate-700 rounded-md p-3 cursor-pointer hover:border-blue-500/50 transition-colors">
                          <Upload className="w-4 h-4 text-cyan-400" />
                          <span className="text-sm text-slate-300">
                            {privateKeyFileName || tr("选择私钥文件", "Select private key file")}
                          </span>
                        </div>
                      <input type="file" accept=".pem,.key,.txt" onChange={handlePrivateKeyUpload} className="hidden" />
                      </label>
                    <div className="text-xs text-slate-500 text-center">{tr("或", "or")}</div>
                    <textarea
                      value={sshConfig.privateKey}
                      onChange={(e) => setSshConfig((prev) => ({ ...prev, privateKey: e.target.value }))}
                      placeholder={tr("粘贴私钥内容...", "Paste private key content...")}
                      className="w-full h-32 bg-slate-800/50 border border-slate-700 text-white rounded-md p-3 text-sm font-mono resize-none focus:outline-none focus:ring-2 focus:ring-blue-500/50"
                    />
                  </div>
                </div>
              )}

              {sshConfig.username !== "root" && (
                <div>
                  <Label className="text-slate-300 flex items-center gap-2">
                    {tr("sudo密码", "sudo Password")}
                    <span className="text-xs text-slate-500">
                      {tr("(留空则默认使用SSH密码或免密sudo)", "(Leave blank to reuse SSH password or sudo-free)")}
                    </span>
                  </Label>
                  <div className="relative mt-2">
                    <Input
                      type={showSudoPassword ? "text" : "password"}
                      value={sshConfig.sudoPassword}
                      onChange={(e) => setSshConfig((prev) => ({ ...prev, sudoPassword: e.target.value }))}
                      placeholder={tr("输入sudo密码或留空", "Enter sudo password or leave blank")}
                      className="bg-slate-800/50 border-slate-700 text-white pr-10"
                    />
                    <button
                      type="button"
                      onClick={() => setShowSudoPassword(!showSudoPassword)}
                      className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-200 transition-colors"
                      aria-label={showSudoPassword ? tr("隐藏密码", "Hide password") : tr("显示密码", "Show password")}
                    >
                      {showSudoPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                    </button>
                  </div>
                </div>
              )}

              <Button
                onClick={handleTestSSH}
                disabled={!sshConfig.host || isTestingSSH}
                className="w-full bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-500 hover:to-cyan-500"
              >
                {isTestingSSH ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    {tr("测试中...", "Testing...")}
                  </>
                ) : (
                  <>
                    <Terminal className="w-4 h-4 mr-2" />
                    {tr("测试连接", "Test Connection")}
                  </>
                )}
              </Button>

              {sshStatus === "success" && (
                <div className="space-y-3 rounded-lg border border-green-500/30 bg-green-500/5 p-3 text-sm text-green-200">
                  <div className="flex items-center gap-2">
                  <CheckCircle2 className="w-4 h-4" />
                    <span>{tr("SSH连接成功", "SSH connection succeeded")}</span>
                  </div>
                  {lastTestDetails && (
                    <>
                      <div>
                        {tr("主机名：", "Hostname: ")}
                        {lastTestDetails.hostname || sshConfig.host}
                      </div>
                      <div className="text-xs text-green-100/70">
                        {tr("GPU：", "GPU: ")}
                        {lastTestDetails.gpus?.length ? lastTestDetails.gpus.slice(0, 2).join(" / ") : tr("未知", "Unknown")}
                      </div>
                    </>
                  )}
                </div>
              )}
              {sshStatus === "error" && (
                <div className="flex items-center gap-2 text-red-400 text-sm bg-red-500/10 p-3 rounded-lg">
                  <XCircle className="w-4 h-4" />
                  <span>{tr("SSH连接失败，请检查配置", "SSH connection failed, please check configuration")}</span>
                </div>
              )}

              <Button
                variant="outline"
                onClick={handleAddNode}
                disabled={sshStatus !== "success"}
                className="w-full border-blue-500/40 text-blue-300 hover:bg-blue-500/10"
              >
                <Plus className="w-4 h-4 mr-2" />
                {tr("加入待检查节点", "Add to pending nodes")}
              </Button>
            </div>
          </Card>

          <Card className="bg-slate-900/50 border-blue-500/20 p-6">
            <div className="flex items-center gap-2 mb-6">
              <Terminal className="w-5 h-5 text-purple-400" />
              <h3 className="text-lg font-semibold text-white">{tr("命令检测", "Command Detection")}</h3>
            </div>
            <p className="text-sm text-slate-400 mb-2">
              {tr("检查目标节点关键命令是否可用", "Verify whether required commands exist on the target node")}
            </p>
            <p className="text-xs text-slate-500 mb-6">
              {tr("点击下方按钮自动检测所有命令，无需手动选择", "Click the button below to check all commands automatically")}
            </p>

            <div className="space-y-3 mb-6">
              {CORE_COMMANDS.map((cmd) => (
                <div
                  key={cmd.name}
                  className="flex items-start gap-3 p-3 rounded-lg bg-slate-800/30 border border-slate-700/50"
                >
                  <div className="mt-0.5 flex-shrink-0">
                    {commandStatus[cmd.name] === "idle" && (
                      <div className="w-5 h-5 rounded-full border-2 border-slate-600" title={tr("未检测", "Not checked")} />
                    )}
                    {commandStatus[cmd.name] === "checking" && (
                      <Loader2
                        className="w-5 h-5 text-blue-400 animate-spin"
                        title={tr("检测中...", "Checking...")}
                      />
                    )}
                    {commandStatus[cmd.name] === "available" && (
                      <CheckCircle2 className="w-5 h-5 text-green-400" title={tr("命令可用", "Available")} />
                    )}
                    {commandStatus[cmd.name] === "missing" && (
                      <XCircle className="w-5 h-5 text-red-400" title={tr("命令缺失", "Missing")} />
                    )}
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-mono text-white truncate">{cmd.name}</p>
                    <p className="text-xs text-slate-400 mt-0.5">{cmd.description[language]}</p>
                    {commandStatus[cmd.name] === "missing" && cmd.package && (
                      <div className="flex items-start gap-1 mt-2 text-xs text-orange-400">
                        <AlertCircle className="w-3 h-3 mt-0.5 flex-shrink-0" />
                        <span>{tr("请安装", "Please install")}: {cmd.package}</span>
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>

            <Button
              onClick={handleCheckCommands}
              disabled={sshStatus !== "success" || isCheckingCommands}
              className="w-full bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-500 hover:to-pink-500 disabled:opacity-50"
            >
              {isCheckingCommands ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  {tr("检测中...", "Checking...")}
                </>
              ) : (
                <>
                  <Terminal className="w-4 h-4 mr-2" />
                  {tr("检测命令", "Check Commands")}
                </>
              )}
            </Button>
            {sshStatus !== "success" && (
              <p className="text-xs text-slate-500 mt-2 text-center">
                {tr("请先完成SSH连接测试", "Please complete the SSH test first")}
              </p>
            )}
          </Card>
        </div>

        <Card className="bg-slate-900/50 border-blue-500/20 p-6 mb-8">
          <div className="flex items-center gap-2 mb-6">
            <Server className="w-5 h-5 text-cyan-400" />
            <h3 className="text-lg font-semibold text-white">{tr("待检查节点", "Pending Nodes")}</h3>
            <Badge variant="outline" className="ml-2 border-blue-500/40 text-blue-200">
              {tr("共", "Total")} {selectedNodes.length} {tr("台", "nodes")}
            </Badge>
            {selectedNodeIds.size > 0 && (
              <Badge variant="outline" className="ml-2 border-orange-500/40 text-orange-200">
                {tr("已选", "Selected")} {selectedNodeIds.size} {tr("台", "nodes")}
              </Badge>
            )}
            <div className="ml-auto flex gap-2">
              {selectedNodeIds.size > 0 && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleRemoveSelectedNodes}
                  className="border-red-500/40 text-red-300 hover:bg-red-500/10"
                >
                  <Trash2 className="w-4 h-4 mr-1" />
                  {tr("移除选中", "Remove selected")} ({selectedNodeIds.size})
                </Button>
              )}
              <Button variant="outline" size="sm" onClick={handleClearNodes} disabled={!selectedNodes.length}>
                <Trash2 className="w-4 h-4 mr-1" />
                {tr("清空", "Clear")}
              </Button>
            </div>
          </div>

          {selectedNodes.length === 0 ? (
            <div className="text-center py-8 text-slate-400">
              {tr('暂无节点，请完成SSH测试后点击"加入待检查节点"', 'No nodes yet. Complete the SSH test, then click "Add to pending nodes".')}
            </div>
          ) : (
            <>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-slate-800 text-slate-300">
                      <th className="py-3 px-4 text-left w-12">
                        <Checkbox
                          checked={currentPageAllSelected}
                          onCheckedChange={handleSelectAll}
                          className="border-slate-600"
                        />
                      </th>
                      <th className="py-3 px-4 text-left">{tr("别名", "Alias")}</th>
                      <th className="py-3 px-4 text-left">{tr("地址", "Address")}</th>
                      <th className="py-3 px-4 text-left">{tr("用户名", "Username")}</th>
                      <th className="py-3 px-4 text-left">{tr("GPU信息", "GPU Info")}</th>
                      <th className="py-3 px-4 text-left">{tr("操作", "Actions")}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {paginatedNodes.map((node) => (
                      <tr key={node.id} className="border-b border-slate-800 hover:bg-slate-800/30">
                        <td className="py-3 px-4">
                          <Checkbox
                            checked={selectedNodeIds.has(node.id)}
                            onCheckedChange={() => handleToggleNodeSelection(node.id)}
                            className="border-slate-600"
                          />
                        </td>
                        <td className="py-3 px-4 text-white">{node.alias}</td>
                        <td className="py-3 px-4 text-slate-300">
                          {node.host}:{node.port}
                        </td>
                        <td className="py-3 px-4 text-slate-300">{node.username}</td>
                        <td className="py-3 px-4 text-slate-300">
                          {node.summary?.gpus?.length ? node.summary.gpus[0] : tr("待检测", "Pending")}
                        </td>
                        <td className="py-3 px-4">
                          <Button
                            variant="ghost"
                            size="sm"
                            className="text-red-300 hover:text-red-200"
                            onClick={() => handleRemoveNode(node.id)}
                          >
                            <Trash2 className="w-4 h-4 mr-1" />
                            {tr("移除", "Remove")}
                          </Button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {totalPages > 1 && (
                <div className="flex items-center justify-between mt-4 pt-4 border-t border-slate-800">
                  <div className="text-sm text-slate-400">
                    {tr("显示", "Showing")} {(currentPage - 1) * pageSize + 1} - {Math.min(currentPage * pageSize, selectedNodes.length)}{" "}
                    {tr("条，共", "of")} {selectedNodes.length} {tr("条", "records")}
                  </div>
                  <div className="flex items-center gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
                      disabled={currentPage === 1}
                      className="border-slate-600"
                    >
                      {tr("上一页", "Previous")}
                    </Button>
                    <div className="text-sm text-slate-300">
                      {tr("第", "Page")} {currentPage} / {totalPages} {tr("页", "")}
                    </div>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
                      disabled={currentPage === totalPages}
                      className="border-slate-600"
                    >
                      {tr("下一页", "Next")}
                    </Button>
                  </div>
                </div>
              )}
            </>
          )}
        </Card>

        <Card className="bg-slate-900/50 border-blue-500/20 p-6 mb-8">
          <div className="flex items-center gap-2 mb-6">
            <Play className="w-5 h-5 text-green-400" />
            <h3 className="text-lg font-semibold text-white">{tr("健康检查", "Health Checks")}</h3>
          </div>
          <p className="text-sm text-slate-400 mb-6">
            {tr("选择检查项目并开始节点健康检查", "Select test items and start the health inspection")}
          </p>

          <div className="mb-6">
          <Label className="text-slate-300 mb-4 block">{tr("选择检查项目", "Select test items")}</Label>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {CHECK_ITEMS.map((item) => (
                <label
                  key={item.id}
                  className="flex items-start gap-3 p-4 rounded-lg bg-slate-800/30 border border-slate-700/50 cursor-pointer hover:border-blue-500/50 transition-colors"
                >
                  <Checkbox
                    checked={selectedItems.includes(item.id)}
                    onCheckedChange={(checked) => {
                      if (checked) {
                        setSelectedItems((prev) => [...prev, item.id])
                      } else {
                        setSelectedItems((prev) => prev.filter((id) => id !== item.id))
                      }
                    }}
                    className="mt-1"
                  />
                  <div className="flex-1">
                    <p className="text-sm font-medium text-white flex items-center gap-2">
                      {item.name[language]}
                    </p>
                    <p className="text-xs text-slate-400 mt-1">{item.description[language]}</p>
                  </div>
                </label>
              ))}
            </div>
          </div>

          <div className="mb-6">
            <Label className="text-slate-300 mb-3 block">{tr("DCGM诊断级别", "DCGM diagnostic level")}</Label>
            <select
              value={dcgmLevel}
              onChange={(e) => setDcgmLevel(e.target.value)}
              className="w-full bg-slate-800/50 border border-slate-700 text-white rounded-md px-3 py-2 text-sm"
            >
              <option value="1">{tr("Level 1 - 快速检查 (<1分钟)", "Level 1 - Quick scan (<1 min)")}</option>
              <option value="2">{tr("Level 2 - 标准检查 (~2分钟)", "Level 2 - Standard (~2 min)")}</option>
              <option value="3">{tr("Level 3 - 深度检查 (~30分钟)", "Level 3 - Deep (~30 min)")}</option>
              <option value="4">{tr("Level 4 - 完整检查 (1-2小时)", "Level 4 - Full (1-2 hrs)")}</option>
            </select>
          </div>

          <Button
            onClick={handleRunTests}
            disabled={!canRunTests}
            className="w-full bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-500 hover:to-emerald-500 disabled:opacity-50"
            size="lg"
          >
            {isRunningTests || isPollingJob ? (
              <>
                <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                {tr("诊断进行中...", "Diagnostics running...")}
              </>
            ) : (
              <>
                <Play className="w-5 h-5 mr-2" />
                {tr("开始健康检查", "Start Health Check")}
              </>
            )}
          </Button>

          {currentJob && (
            <div className="mt-6 rounded-lg border border-slate-800 bg-slate-900/60 p-4 text-sm text-slate-300">
              <div className="flex items-center justify-between flex-wrap gap-3">
                <div>
                  <p className="text-white font-semibold">{tr("正在执行的任务", "Running task")}</p>
                  <p className="text-xs text-slate-400">
                    {tr("任务 ID", "Job ID")}: {currentJob.jobId}
                  </p>
                </div>
                <div>{renderStatusBadge(currentJob.status, language)}</div>
              </div>

              <div className="mt-4 grid md:grid-cols-2 gap-3">
                {currentJob.nodes?.map((node) => (
                  <div key={node.nodeId} className="rounded border border-slate-800 p-3 bg-slate-900/50">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-white font-medium">{node.alias || node.host}</p>
                        <p className="text-xs text-slate-500">
                          GPU: {node.gpuType || tr("未知", "Unknown")}
                        </p>
                      </div>
                      {renderStatusBadge(node.status, language)}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </Card>

        <Card className="bg-slate-900/50 border-blue-500/20 p-6 mb-8">
          <div className="flex items-center gap-2 mb-6">
            <ListChecks className="w-5 h-5 text-cyan-400" />
            <h3 className="text-lg font-semibold text-white">{tr("测试结果", "Test Results")}</h3>
            <Badge variant="outline" className="ml-2 border-blue-500/40 text-blue-200">
              {testResults.length} {tr("条", "records")}
            </Badge>
            {selectedResultIds.size > 0 && (
              <Badge variant="outline" className="ml-2 border-orange-500/40 text-orange-200">
                {tr("已选", "Selected")} {selectedResultIds.size} {tr("条", "records")}
              </Badge>
            )}
            <div className="ml-auto flex gap-2">
              {selectedResultIds.size > 0 && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleExportSelectedResults}
                  className="border-blue-500/40 text-blue-300 hover:bg-blue-500/10"
                  disabled={isExportingLogs}
                >
                  {isExportingLogs ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      {tr("导出中...", "Exporting...")}
                    </>
                  ) : (
                    <>
                      {tr("导出选中", "Export selected")} ({selectedResultIds.size})
                    </>
                  )}
                </Button>
              )}
              {selectedResultIds.size > 0 && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => {
                    const count = selectedResultIds.size
                    setTestResults((prev) => prev.filter((item) => !selectedResultIds.has(item.id)))
                    setSelectedResultIds(new Set())
                    toast({
                      title: tr("已删除选中结果", "Selected results removed"),
                      description: `${tr("共删除", "Removed")} ${count} ${tr("条结果", "records")}`,
                    })
                  }}
                  className="border-red-500/40 text-red-300 hover:bg-red-500/10"
                >
                  <Trash2 className="w-4 h-4 mr-1" />
                  {tr("删除选中", "Delete selected")} ({selectedResultIds.size})
                </Button>
              )}
              <Button
                variant="outline"
                size="sm"
                onClick={() => {
                  setTestResults([])
                  setSelectedResultIds(new Set())
                  toast({ title: tr("已清空所有结果", "All results cleared") })
                }}
                disabled={!testResults.length}
              >
                <Trash2 className="w-4 h-4 mr-1" />
                {tr("清空", "Clear")}
              </Button>
            </div>
          </div>

          {testResults.length === 0 ? (
            <div className="text-center py-8 text-slate-400">
              {tr("暂无测试结果，请配置SSH并运行健康检查", "No test results yet. Configure SSH and run a health check.")}
            </div>
          ) : (
            <>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-slate-700">
                      <th className="text-left py-3 px-4 w-12">
                        <Checkbox
                          checked={
                            paginatedResults.length > 0 &&
                            paginatedResults.every((item) => selectedResultIds.has(item.id))
                          }
                          onCheckedChange={() => {
                            const allSelected = paginatedResults.every((item) => selectedResultIds.has(item.id))
                            setSelectedResultIds((prev) => {
                              const next = new Set(prev)
                              if (allSelected) {
                                paginatedResults.forEach((item) => next.delete(item.id))
                              } else {
                                paginatedResults.forEach((item) => next.add(item.id))
                              }
                              return next
                            })
                          }}
                          className="border-slate-600"
                        />
                      </th>
                      <th className="text-left py-3 px-4 text-slate-300 font-medium">{tr("节点别名", "Alias")}</th>
                      <th className="text-left py-3 px-4 text-slate-300 font-medium">{tr("节点地址", "Address")}</th>
                  <th className="text-left py-3 px-4 text-slate-300 font-medium">{tr("GPU类型", "GPU Type")}</th>
                  <th className="text-left py-3 px-4 text-slate-300 font-medium">nvBandwidthTest</th>
                  <th className="text-left py-3 px-4 text-slate-300 font-medium">p2pBandwidthLatencyTest</th>
                  <th className="text-left py-3 px-4 text-slate-300 font-medium">{tr("NCCL测试", "NCCL Test")}</th>
                  <th className="text-left py-3 px-4 text-slate-300 font-medium">{tr("DCGM诊断", "DCGM Diagnostics")}</th>
                  <th className="text-left py-3 px-4 text-slate-300 font-medium">{tr("IB检查", "IB Check")}</th>
                      <th className="text-left py-3 px-4 text-slate-300 font-medium">{tr("状态", "Status")}</th>
                  <th className="text-left py-3 px-4 text-slate-300 font-medium">{tr("执行日志", "Execution Log")}</th>
                  <th className="text-left py-3 px-4 text-slate-300 font-medium">{tr("完成时间", "Completed At")}</th>
                </tr>
              </thead>
              <tbody>
                    {paginatedResults.map((result) => (
                      <tr key={result.id} className="border-b border-slate-800 hover:bg-slate-800/30">
                      <td className="py-3 px-4">
                          <Checkbox
                            checked={selectedResultIds.has(result.id)}
                            onCheckedChange={() => {
                              setSelectedResultIds((prev) => {
                                const next = new Set(prev)
                                if (next.has(result.id)) {
                                  next.delete(result.id)
                                } else {
                                  next.add(result.id)
                                }
                                return next
                              })
                            }}
                            className="border-slate-600"
                          />
                      </td>
                        <td className="py-3 px-4 text-white font-medium">{result.alias || result.hostname}</td>
                        <td className="py-3 px-4 text-slate-300 text-xs">{result.host}</td>
                        <td className="py-3 px-4 text-white">{result.gpuType || "Unknown"}</td>
                        <td className="py-3 px-4">{renderMetric(result.nvbandwidth, "GB/s", language)}</td>
                        <td className="py-3 px-4">{renderMetric(result.p2p, "GB/s", language)}</td>
                        <td className="py-3 px-4">{renderMetric(result.nccl, "GB/s", language)}</td>
                      <td className="py-3 px-4">
                          {result.dcgm
                            ? renderStatusBadge(result.dcgm.status || (result.dcgm.passed ? "passed" : "failed"), language)
                            : renderMetric(undefined, "GB/s", language)}
                      </td>
                      <td className="py-3 px-4">
                          {result.ib
                            ? renderStatusBadge(result.ib.status || (result.ib.passed ? "passed" : "failed"), language)
                            : renderMetric(undefined, "GB/s", language)}
                      </td>
                        <td className="py-3 px-4">{renderStatusBadge(result.status, language)}</td>
                      <td className="py-3 px-4">
                        <Button
                          size="sm"
                          variant="outline"
                          className="border-blue-500/30 text-blue-400 bg-transparent hover:bg-blue-500/10"
                            disabled={!result.executionLog}
                            onClick={() =>
                              setLogViewer({
                                open: true,
                                content: result.executionLog || "",
                              title: `${result.alias || result.hostname} ${tr("日志", "Logs")}`,
                              })
                            }
                        >
                          <FileText className="w-3 h-3 mr-1" />
                          {tr("查看日志", "View log")}
                        </Button>
                          <Button
                            size="sm"
                            variant="outline"
                            className="ml-2 border-blue-500/30 text-blue-400 bg-transparent hover:bg-blue-500/10"
                            disabled={!result.executionLog}
                            onClick={() => handleExportResult(result)}
                          >
                            {tr("导出", "Export")}
                        </Button>
                      </td>
                        <td className="py-3 px-4 text-slate-300 text-xs whitespace-nowrap">
                          {result.completedAt ? new Date(result.completedAt).toLocaleString("zh-CN") : "--"}
                    </td>
                  </tr>
                    ))}
              </tbody>
            </table>
          </div>
              {resultsTotalPages > 1 && (
                <div className="flex items-center justify-between mt-4 pt-4 border-t border-slate-800">
                  <div className="text-sm text-slate-400">
                    {tr("显示", "Showing")} {(resultsPage - 1) * resultsPageSize + 1} -{" "}
                    {Math.min(resultsPage * resultsPageSize, testResults.length)} {tr("条，共", "of")} {testResults.length}{" "}
                    {tr("条", "records")}
                  </div>
                  <div className="flex items-center gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setResultsPage((p) => Math.max(1, p - 1))}
                      disabled={resultsPage === 1}
                      className="border-slate-600"
                    >
                      {tr("上一页", "Previous")}
                    </Button>
                    <div className="text-sm text-slate-300">
                      {tr("第", "Page")} {resultsPage} / {resultsTotalPages} {tr("页", "")}
                    </div>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setResultsPage((p) => Math.min(resultsTotalPages, p + 1))}
                      disabled={resultsPage === resultsTotalPages}
                      className="border-slate-600"
                    >
                      {tr("下一页", "Next")}
                    </Button>
                  </div>
                </div>
              )}
            </>
          )}
        </Card>

        <Card className="bg-slate-900/50 border-blue-500/20 p-6 mb-8">
          <div className="flex items-center gap-2 mb-4">
            <Monitor className="w-5 h-5 text-cyan-400" />
            <h3 className="text-lg font-semibold text-white">{tr("GPU性能基准值", "GPU Benchmark Reference")}</h3>
          </div>
          <p className="text-sm text-slate-400 mb-6">
            {tr("各GPU型号的性能基准值对照表", "Benchmark reference for different GPU models")}
          </p>

          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-slate-700">
                  <th className="text-left py-3 px-4 text-slate-300 font-medium">{tr("GPU型号", "GPU Model")}</th>
                  <th className="text-left py-3 px-4 text-slate-300 font-medium">P2P (GB/s)</th>
                  <th className="text-left py-3 px-4 text-slate-300 font-medium">NCCL (GB/s)</th>
                  <th className="text-left py-3 px-4 text-slate-300 font-medium">BW (GB/s)</th>
                </tr>
              </thead>
              <tbody>
                {benchmarkEntries.map(([model, values]) => (
                  <tr key={model} className="border-b border-slate-800">
                    <td className="py-3 px-4 text-white font-medium">{model}</td>
                    <td className="py-3 px-4 text-slate-300">{values.p2p}</td>
                    <td className="py-3 px-4 text-slate-300">{values.nccl}</td>
                    <td className="py-3 px-4 text-slate-300">{values.bw}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>

        <Card className="bg-slate-900/50 border-blue-500/20 p-6">
          <div className="flex items-center gap-2 mb-4">
            <FileText className="w-5 h-5 text-purple-400" />
            <h3 className="text-lg font-semibold text-white">{tr("检查项目说明", "Test Item Descriptions")}</h3>
          </div>

          <div className="space-y-4">
            {CHECK_ITEMS.map((item) => (
              <div key={item.id} className="border-l-2 border-blue-500/50 pl-4">
                <p className="text-sm font-semibold text-white mb-1">{item.name[language]}</p>
                <p className="text-xs text-slate-400">{item.description[language]}</p>
              </div>
            ))}
          </div>
        </Card>
      </main>

      <Dialog open={logViewer.open} onOpenChange={(open) => setLogViewer((prev) => ({ ...prev, open }))}>
        <DialogContent className="max-w-3xl bg-slate-950 text-white border border-slate-800">
          <DialogHeader>
            <DialogTitle>{logViewer.title}</DialogTitle>
          </DialogHeader>
          <pre className="max-h-[60vh] overflow-auto whitespace-pre-wrap text-xs bg-slate-900/70 rounded-md p-4 border border-slate-800 text-slate-200">
            {logViewer.content || tr("暂无日志", "No logs")}
          </pre>
        </DialogContent>
      </Dialog>

      <style jsx global>{`
        @keyframes pulse {
          0%,
          100% {
            opacity: 1;
            transform: scale(1);
          }
          50% {
            opacity: 0.5;
            transform: scale(0.8);
          }
        }
      `}</style>
    </div>
  )
}
