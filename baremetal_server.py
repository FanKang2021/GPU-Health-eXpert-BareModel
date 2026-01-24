#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GHX Bare-Metal Orchestrator
通过SSH在裸金属节点上运行GPU健康检查
1. 支持SSH连接测试与基础命令检查
2. 通过后台Job执行nvbandwidth/p2p/nccl/dcgm/ib检查
3. 将nvbandwidth、p2pBandwidthLatencyTest上传到目标主机的/tmp/ghx目录执行；上传nccl.tgz和nccl-tests.tgz源码到目标主机并在本地编译（适配CUDA和GLIBC版本）
"""
from __future__ import annotations

import io
import json
import logging
import os
import re
import shutil
import tarfile
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any

import paramiko
from flask import Flask, jsonify, request
from flask_cors import CORS

# -----------------------------------------------------------------------------
# 基础配置
# -----------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
# 支持通过环境变量指定资产目录（Docker 容器内使用）
ASSET_DIR = Path(os.getenv("GHX_ASSET_DIR", str(BASE_DIR)))
ASSETS = {
    "nvbandwidth": ASSET_DIR / "nvbandwidth",
    "p2p": ASSET_DIR / "p2pBandwidthLatencyTest",
    "nccl": ASSET_DIR / "nccl.tgz",
    "nccl_tests": ASSET_DIR / "nccl-tests.tgz",
    "ib_check": ASSET_DIR / "ib_health_check.sh",
}

for name, path in ASSETS.items():
    if not path.exists():
        logging.warning("Asset %s not found at %s", name, path)

FALLBACK_GPU_BENCHMARKS = {
    "RTX 3090": {"p2p": 18, "nccl": 7, "bw": 20},
    "L40S": {"p2p": 28, "nccl": 9, "bw": 20},
    "RTX 4090": {"p2p": 18, "nccl": 7, "bw": 20},
    "A100": {"p2p": 420, "nccl": 70, "bw": 20},
    "A800": {"p2p": 340, "nccl": 55, "bw": 20},
    "H100": {"p2p": 700, "nccl": 139, "bw": 40},
    "H800": {"p2p": 340, "nccl": 65, "bw": 47},
    "H200": {"p2p": 730, "nccl": 145, "bw": 54},
}

BENCHMARK_FILE = os.getenv("GPU_BENCHMARK_FILE", str(BASE_DIR / "config" / "gpu-benchmarks.json"))

# 初始化日志（需要在 load_gpu_benchmarks 之前）
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ghx-baremetal")

# 记录资产目录配置
logger.info("资产目录配置: BASE_DIR=%s, ASSET_DIR=%s (GHX_ASSET_DIR=%s)", 
           BASE_DIR, ASSET_DIR, os.getenv("GHX_ASSET_DIR", "未设置"))
for name, path in ASSETS.items():
    if path.exists():
        logger.debug("Asset %s found at %s", name, path)

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})


@app.before_request
def log_request_info():
    """记录所有API请求"""
    logger.info(
        "Request: %s %s from %s",
        request.method,
        request.path,
        request.remote_addr,
    )
    if request.is_json:
        # 对于敏感信息（如密码），只记录部分内容
        payload = request.get_json(silent=True) or {}
        safe_payload = {}
        for key, value in payload.items():
            if isinstance(value, dict):
                safe_value = {}
                for k, v in value.items():
                    if k in ("password", "value", "privateKey") and isinstance(v, str):
                        safe_value[k] = f"{v[:3]}***" if len(v) > 3 else "***"
                    else:
                        safe_value[k] = v
                safe_payload[key] = safe_value
            else:
                safe_payload[key] = value
        logger.debug("Request payload: %s", json.dumps(safe_payload, ensure_ascii=False))


@app.after_request
def log_response_info(response):
    """记录API响应状态"""
    logger.info(
        "Response: %s %s -> %s",
        request.method,
        request.path,
        response.status_code,
    )
    return response


def load_gpu_benchmarks() -> Dict[str, Dict[str, float]]:
    path = Path(BENCHMARK_FILE)
    if not path.exists():
        logger.warning("GPU benchmark file %s not found, using fallback defaults", path)
        return dict(FALLBACK_GPU_BENCHMARKS)
    try:
        with open(path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
            logger.info("Loaded GPU benchmarks from %s", path)
            return data
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Failed to load GPU benchmarks from %s: %s. Using fallback.", path, exc)
        return dict(FALLBACK_GPU_BENCHMARKS)


GPU_BENCHMARKS = load_gpu_benchmarks()

# -----------------------------------------------------------------------------
# 工具函数
# -----------------------------------------------------------------------------


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_response(success: bool, data: Any = None, message: str = "", status: int = 200):
    payload = {"success": success, "message": message, "data": data, "timestamp": utc_now()}
    return jsonify(payload), status


def normalize_gpu_name(raw: str) -> str:
    if not raw:
        return "Unknown"
    cleaned = raw.strip()
    for key in GPU_BENCHMARKS:
        if key.lower().replace(" ", "") in cleaned.lower().replace(" ", ""):
            return key
    return cleaned


def ensure_payload_fields(payload: Dict[str, Any], fields: List[str]):
    missing = [field for field in fields if field not in payload or payload[field] in (None, "")]
    if missing:
        raise ValueError(f"缺少必填字段: {', '.join(missing)}")


def wrap_bash(command: str) -> str:
    safe = command.replace("'", "'\"'\"'")
    return f"bash -lc 'set -euo pipefail; {safe}'"


def load_private_key(key_str: str, passphrase: Optional[str] = None) -> paramiko.PKey:
    """加载SSH私钥，支持多种格式（PEM、OpenSSH等）"""
    if not key_str or not key_str.strip():
        raise ValueError("私钥内容为空")
    
    key_str = key_str.strip()
    last_exc: Optional[Exception] = None
    errors: List[str] = []
    
    # 检测是否为 OpenSSH 格式（以 "-----BEGIN OPENSSH PRIVATE KEY-----" 开头）
    is_openssh_format = key_str.startswith("-----BEGIN OPENSSH PRIVATE KEY-----")
    
    # 尝试不同的加载方式
    key_types = [paramiko.RSAKey, paramiko.ECDSAKey, paramiko.Ed25519Key, paramiko.DSSKey]
    
    for key_cls in key_types:
        # 方法1: 使用 StringIO
        try:
            key_file = io.StringIO(key_str)
            key_file.seek(0)
            key = key_cls.from_private_key(key_file, password=passphrase)
            logger.debug("成功使用 %s (StringIO) 加载私钥", key_cls.__name__)
            return key
        except Exception as exc:  # pylint: disable=broad-except
            errors.append(f"{key_cls.__name__}(StringIO): {str(exc)}")
            last_exc = exc
        
        # 方法2: 使用 BytesIO
        try:
            key_file = io.BytesIO(key_str.encode('utf-8'))
            key_file.seek(0)
            key = key_cls.from_private_key(key_file, password=passphrase)
            logger.debug("成功使用 %s (BytesIO) 加载私钥", key_cls.__name__)
            return key
        except Exception as exc:  # pylint: disable=broad-except
            errors.append(f"{key_cls.__name__}(BytesIO): {str(exc)}")
            last_exc = exc
    
    # 如果所有方法都失败，尝试使用 paramiko 的通用加载方法
    if is_openssh_format:
        try:
            # OpenSSH 格式可能需要特殊处理
            key_file = io.BytesIO(key_str.encode('utf-8'))
            key_file.seek(0)
            # 尝试使用 Ed25519Key（OpenSSH 格式常用）
            key = paramiko.Ed25519Key.from_private_key(key_file, password=passphrase)
            logger.debug("成功使用 Ed25519Key 加载 OpenSSH 格式私钥")
            return key
        except Exception as exc:  # pylint: disable=broad-except
            errors.append(f"Ed25519Key(OpenSSH): {str(exc)}")
            last_exc = exc
    
    # 记录所有错误
    logger.error("私钥加载失败，尝试的方法: %s", "; ".join(errors))
    raise ValueError(f"无法解析私钥。私钥格式: {'OpenSSH' if is_openssh_format else 'PEM'}。最后错误: {last_exc}")


# -----------------------------------------------------------------------------
# SSH Session
# -----------------------------------------------------------------------------


@dataclass
class SSHCommandResult:
    command: str
    exit_code: int
    stdout: str
    stderr: str


class SSHSession:
    """封装Paramiko连接，提供上传和执行命令的能力"""

    def __init__(self, connection: Dict[str, Any]):
        self.connection = connection
        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.username = connection.get("username", "root")
        auth = connection.get("auth", {})
        sudo_password = connection.get("sudoPassword")
        if not sudo_password and auth.get("type") == "password":
            sudo_password = auth.get("value")
        self.need_sudo = self.username != "root"
        self.sudo_password = sudo_password
        self._sftp = None

    def __enter__(self):
        auth = self.connection.get("auth", {})
        kwargs = {
            "hostname": self.connection["host"],
            "port": int(self.connection.get("port", 22)),
            "username": self.connection["username"],
            "timeout": self.connection.get("timeout", 15),
            "allow_agent": False,
            "look_for_keys": False,
        }
        if auth.get("type") == "password":
            kwargs["password"] = auth.get("value")
        elif auth.get("type") == "privateKey":
            kwargs["pkey"] = load_private_key(auth.get("value", ""), auth.get("passphrase"))
        else:
            raise ValueError("认证方式不支持")

        self.client.connect(**kwargs)
        self._sftp = None  # 延迟初始化
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._sftp:
            self._sftp.close()
        self.client.close()

    @property
    def sftp(self):
        """延迟初始化SFTP，只在需要时打开"""
        if self._sftp is None:
            self._sftp = self.client.open_sftp()
        return self._sftp

    def run(self, command: str, timeout: int = 300, require_root: bool = False) -> SSHCommandResult:
        wrapped = wrap_bash(command)
        if require_root and self.need_sudo:
            sudo_prefix = "sudo -S -p ''" if self.sudo_password else "sudo -n"
            wrapped = wrapped.replace("bash -lc", f"{sudo_prefix} bash -lc", 1)
        stdin, stdout, stderr = self.client.exec_command(wrapped, timeout=timeout)
        if require_root and self.need_sudo and self.sudo_password:
            stdin.write(self.sudo_password + "\n")
            stdin.flush()
        stdout_str = stdout.read().decode("utf-8", errors="ignore")
        stderr_str = stderr.read().decode("utf-8", errors="ignore")
        exit_code = stdout.channel.recv_exit_status()
        return SSHCommandResult(command=command, exit_code=exit_code, stdout=stdout_str, stderr=stderr_str)

    def upload(self, local_path: Path, remote_path: str, executable: bool = False):
        remote_dir = Path(remote_path).parent.as_posix()
        self.run(f"mkdir -p {remote_dir}")
        posix_local = str(local_path)
        self.sftp.put(posix_local, remote_path)
        if executable:
            self.run(f"chmod +x {remote_path}", require_root=self.need_sudo)
    
    def upload_directory(self, local_dir: Path, remote_dir: str):
        """递归上传整个目录到远程"""
        self.run(f"mkdir -p {remote_dir}")
        for root, dirs, files in os.walk(local_dir):
            # 计算相对路径
            rel_root = Path(root).relative_to(local_dir)
            remote_root = f"{remote_dir}/{rel_root.as_posix()}" if rel_root != Path('.') else remote_dir
            
            # 创建远程目录
            if rel_root != Path('.'):
                self.run(f"mkdir -p {remote_root}")
            
            # 上传文件
            for file in files:
                local_file = Path(root) / file
                remote_file = f"{remote_root}/{file}"
                self.sftp.put(str(local_file), remote_file)
                # 如果是可执行文件，设置执行权限
                if os.access(local_file, os.X_OK):
                    self.run(f"chmod +x {remote_file}", require_root=self.need_sudo)


# -----------------------------------------------------------------------------
# 解析函数
# -----------------------------------------------------------------------------


def parse_nvbandwidth(output: str) -> float:
    values: List[float] = []
    for line in output.splitlines():
        line = line.strip()
        if not line or not line[0].isdigit():
            continue
        parts = line.split()
        for chunk in parts[1:]:
            try:
                value = float(chunk)
                if 10 <= value <= 1200:
                    values.append(value)
            except ValueError:
                break
    return min(values) if values else 0.0


def parse_p2p(output: str) -> float:
    collecting = False
    values: List[float] = []
    row_count = 0
    for line in output.splitlines():
        if "Bidirectional P2P=Enabled Bandwidth Matrix" in line:
            collecting = True
            row_count = 0
            continue
        if "P2P=Disabled Latency Matrix" in line:
            break
        if collecting:
            stripped = line.strip()
            if not stripped:
                continue
            parts = line.split()
            # 跳过矩阵顶部的列标题（例如 "D\D 0 1 2 ..."）
            if not parts[0].isdigit():
                continue
            row_idx = row_count
            row_count += 1
            parts = line.split()
            if len(parts) <= 1:
                continue
            for col_idx, value_str in enumerate(parts[1:]):
                try:
                    value = float(value_str)
                    # 对角线 (row == col) 的值跳过
                    if value > 0 and row_idx != col_idx:
                        values.append(value)
                except ValueError:
                    continue
    if not values:
        return 0.0
    return min(values)


def parse_nccl(output: str) -> float:
    for line in output.splitlines():
        if "Avg bus bandwidth" in line:
            parts = line.split()
            for chunk in parts:
                try:
                    return float(chunk)
                except ValueError:
                    continue
    return 0.0


# -----------------------------------------------------------------------------
# Job执行器
# -----------------------------------------------------------------------------


class RemoteNodeRunner:
    def __init__(self, node_meta: Dict[str, Any], tests: List[str], dcgm_level: int, connection: Dict[str, Any], cancelled_flag: Optional[threading.Event] = None):
        self.node_meta = node_meta
        self.tests = tests
        self.dcgm_level = dcgm_level
        self.connection = connection
        self.remote_dir = "/tmp/ghx"
        self.logs: List[str] = []
        self.session: Optional[SSHSession] = None
        self.cancelled = cancelled_flag or threading.Event()

    def log(self, message: str):
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"{timestamp} - {message}"
        self.logs.append(entry)
        host = self.node_meta.get("host", "unknown")
        port = self.node_meta.get("port", 22)
        node_display = f"{host}:{port}" if port != 22 else host
        logger.info("[%s] %s", node_display, message)

    def benchmark_for(self, metric: str) -> Optional[float]:
        gpu_type = self.node_meta.get("gpuType")
        if not gpu_type:
            return None
        return GPU_BENCHMARKS.get(gpu_type, {}).get(metric)

    def execute(self) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        if self.cancelled.is_set():
            self.log("任务已被取消，停止执行")
            return {
                "results": {},
                "overallStatus": "cancelled",
                "executionLog": "\n".join(self.logs),
                "gpuType": self.node_meta.get("gpuType", "Unknown"),
            }
        
        with SSHSession(self.connection) as session:
            self.session = session
            self.log("SSH连接已建立")
            
            if self.cancelled.is_set():
                self.log("任务已被取消，停止执行")
                return {
                    "results": {},
                    "overallStatus": "cancelled",
                    "executionLog": "\n".join(self.logs),
                    "gpuType": self.node_meta.get("gpuType", "Unknown"),
                }
            
            session.run(f"mkdir -p {self.remote_dir}")

            gpu_info = self._query_gpu_info()
            self.node_meta["gpuType"] = gpu_info["model"]
            self.node_meta["gpuList"] = gpu_info["list"]

            if self.cancelled.is_set():
                self.log("任务已被取消，停止执行")
                return {
                    "results": {},
                    "overallStatus": "cancelled",
                    "executionLog": "\n".join(self.logs),
                    "gpuType": self.node_meta.get("gpuType", "Unknown"),
                }

            if "nvbandwidth" in self.tests:
                if self.cancelled.is_set():
                    self.log("任务已被取消，停止执行nvbandwidth测试")
                    return {
                        "results": results,
                        "overallStatus": "cancelled",
                        "executionLog": "\n".join(self.logs),
                        "gpuType": self.node_meta.get("gpuType", "Unknown"),
                    }
                result = self._run_nvbandwidth()
                results["nvbandwidth"] = result
                if result.get("rawOutput"):
                    self.log(f"nvbandwidth命令输出:\n{result['rawOutput']}")
            if "p2p" in self.tests:
                if self.cancelled.is_set():
                    self.log("任务已被取消，停止执行p2p测试")
                    return {
                        "results": results,
                        "overallStatus": "cancelled",
                        "executionLog": "\n".join(self.logs),
                        "gpuType": self.node_meta.get("gpuType", "Unknown"),
                    }
                result = self._run_p2p()
                results["p2p"] = result
                if result.get("rawOutput"):
                    self.log(f"p2pBandwidthLatencyTest命令输出:\n{result['rawOutput']}")
            if "nccl" in self.tests:
                if self.cancelled.is_set():
                    self.log("任务已被取消，停止执行nccl测试")
                    return {
                        "results": results,
                        "overallStatus": "cancelled",
                        "executionLog": "\n".join(self.logs),
                        "gpuType": self.node_meta.get("gpuType", "Unknown"),
                    }
                result = self._run_nccl_tests()
                results["nccl"] = result
                if result.get("rawOutput"):
                    self.log(f"NCCL测试命令输出:\n{result['rawOutput']}")
            if "dcgm" in self.tests:
                if self.cancelled.is_set():
                    self.log("任务已被取消，停止执行dcgm测试")
                    return {
                        "results": results,
                        "overallStatus": "cancelled",
                        "executionLog": "\n".join(self.logs),
                        "gpuType": self.node_meta.get("gpuType", "Unknown"),
                    }
                result = self._run_dcgm_diag()
                results["dcgm"] = result
                if result.get("rawOutput"):
                    self.log(f"DCGM诊断命令输出:\n{result['rawOutput']}")
            if "ib" in self.tests:
                if self.cancelled.is_set():
                    self.log("任务已被取消，停止执行ib测试")
                    return {
                        "results": results,
                        "overallStatus": "cancelled",
                        "executionLog": "\n".join(self.logs),
                        "gpuType": self.node_meta.get("gpuType", "Unknown"),
                    }
                result = self._run_ib_check()
                results["ib"] = result
                if result.get("rawOutput"):
                    self.log(f"IB检查命令输出:\n{result['rawOutput']}")

        if self.cancelled.is_set():
            self.log("任务已被取消")
            return {
                "results": results,
                "overallStatus": "cancelled",
                "executionLog": "\n".join(self.logs),
                "gpuType": self.node_meta.get("gpuType", "Unknown"),
            }

        overall_pass = all(
            res.get("status") in ("passed", "skipped")
            for res in results.values()
        )
        return {
            "results": results,
            "overallStatus": "passed" if overall_pass else "failed",
            "executionLog": "\n".join(self.logs),
            "gpuType": self.node_meta.get("gpuType", "Unknown"),
        }

    def _query_gpu_info(self) -> Dict[str, Any]:
        gpu_cmd = self.session.run("nvidia-smi -L || true")
        gpu_lines = [line.strip() for line in gpu_cmd.stdout.splitlines() if line.strip()]
        primary_gpu = gpu_lines[0] if gpu_lines else "Unknown"
        short_name = normalize_gpu_name(primary_gpu)
        self.log(f"检测到GPU: {short_name}")
        return {"model": short_name, "list": gpu_lines}

    def _upload_asset(self, key: str, remote_name: str, executable: bool = True):
        local_path = ASSETS[key]
        if not local_path.exists():
            raise FileNotFoundError(f"缺少{key}资源 {local_path}")
        remote_path = f"{self.remote_dir}/{remote_name}"
        self.session.upload(local_path, remote_path, executable=executable)
        self.log(f"上传资源 {key} -> {remote_path}")
        return remote_path

    def _run_nvbandwidth(self) -> Dict[str, Any]:
        try:
            remote_bin = self._upload_asset("nvbandwidth", "nvbandwidth")
            h2d = self.session.run(
                f"cd {self.remote_dir} && {remote_bin} -t host_to_device_memcpy_ce",
                timeout=600,
                require_root=True,
            )
            d2h = self.session.run(
                f"cd {self.remote_dir} && {remote_bin} -t device_to_host_memcpy_ce",
                timeout=600,
                require_root=True,
            )
            if h2d.exit_code != 0 or d2h.exit_code != 0:
                raise RuntimeError(f"nvbandwidth命令执行失败: H2D={h2d.exit_code}, D2H={d2h.exit_code}")
            h2d_value = parse_nvbandwidth(h2d.stdout)
            d2h_value = parse_nvbandwidth(d2h.stdout)
            valid_values = [v for v in (h2d_value, d2h_value) if v > 0]
            if not valid_values:
                raise RuntimeError("nvbandwidth未解析到有效结果")
            value = min(valid_values)
            benchmark = self.benchmark_for("bw")
            passed = benchmark is None or value >= benchmark
            self.log(f"nvbandwidth测试完成: {value:.1f} GB/s")
            return {
                "status": "passed" if passed else "failed",
                "value": value,
                "unit": "GB/s",
                "benchmark": benchmark,
                "passed": passed,
                "details": {"h2d": h2d_value, "d2h": d2h_value},
                "rawOutput": f"{h2d.stdout}\n{d2h.stdout}",
            }
        except Exception as exc:  # pylint: disable=broad-except
            self.log(f"nvbandwidth测试失败: {exc}")
            return {"status": "error", "message": str(exc)}

    def _run_p2p(self) -> Dict[str, Any]:
        try:
            remote_bin = self._upload_asset("p2p", "p2pBandwidthLatencyTest")
            result = self.session.run(f"cd {self.remote_dir} && {remote_bin}", timeout=900, require_root=True)
            if result.exit_code != 0:
                raise RuntimeError(result.stderr or "p2pBandwidthLatencyTest 执行失败")
            value = parse_p2p(result.stdout)
            if value <= 0:
                raise RuntimeError("P2P测试未解析到有效带宽")
            benchmark = self.benchmark_for("p2p")
            passed = benchmark is None or value >= benchmark
            self.log(f"P2P测试完成: {value:.1f} GB/s")
            return {
                "status": "passed" if passed else "failed",
                "value": value,
                "unit": "GB/s",
                "benchmark": benchmark,
                "passed": passed,
                "rawOutput": result.stdout,
            }
        except Exception as exc:  # pylint: disable=broad-except
            self.log(f"P2P测试失败: {exc}")
            return {"status": "error", "message": str(exc)}

    def _run_nccl_tests(self) -> Dict[str, Any]:
        try:
            # 获取实际GPU数量
            gpu_count = len(self.node_meta.get("gpuList", []))
            if gpu_count == 0:
                raise RuntimeError("未检测到GPU，无法运行NCCL测试")
            
            # 检查nccl-tests是否已编译
            check_res = self.session.run(f"[ -f {self.remote_dir}/nccl-tests/build/all_reduce_perf ] && echo OK || echo MISSING")
            if check_res.stdout.strip() == "OK":
                self.log("nccl-tests 已存在，跳过编译")
            else:
                # 上传并编译 nccl 和 nccl-tests
                nccl_tgz = ASSETS["nccl"]
                nccl_tests_tgz = ASSETS["nccl_tests"]
                
                if not nccl_tgz.exists():
                    raise FileNotFoundError(f"nccl.tgz 不存在: {nccl_tgz}")
                if not nccl_tests_tgz.exists():
                    raise FileNotFoundError(f"nccl-tests.tgz 不存在: {nccl_tests_tgz}")
                
                remote_nccl_tgz = f"{self.remote_dir}/nccl.tgz"
                remote_nccl_tests_tgz = f"{self.remote_dir}/nccl-tests.tgz"
                remote_nccl_dir = f"{self.remote_dir}/nccl"
                remote_nccl_tests_dir = f"{self.remote_dir}/nccl-tests"
                
                # 上传压缩包
                self.log("上传 nccl.tgz 和 nccl-tests.tgz 到远程节点")
                self.session.upload(nccl_tgz, remote_nccl_tgz)
                self.session.upload(nccl_tests_tgz, remote_nccl_tests_tgz)
                
                # 编译 nccl 和 nccl-tests
                self.log("在远程节点编译 nccl 和 nccl-tests")
                compile_script = f"""
set -e
# 清理旧目录
rm -rf {remote_nccl_dir} {remote_nccl_tests_dir}

# 解压 nccl
echo "解压 nccl.tgz..."
tar -xzf {remote_nccl_tgz} -C {self.remote_dir}
rm -f {remote_nccl_tgz}

# 编译 nccl
echo "编译 nccl..."
cd {remote_nccl_dir}
make -j$(nproc) CUDA_HOME=/usr/local/cuda 2>&1 | tee /tmp/nccl_build.log
if [ $? -ne 0 ]; then
    echo "错误: nccl 编译失败"
    cat /tmp/nccl_build.log
    exit 1
fi

# 设置 NCCL_HOME
export NCCL_HOME={remote_nccl_dir}

# 解压 nccl-tests
echo "解压 nccl-tests.tgz..."
tar -xzf {remote_nccl_tests_tgz} -C {self.remote_dir}
rm -f {remote_nccl_tests_tgz}

# 编译 nccl-tests
echo "编译 nccl-tests..."
cd {remote_nccl_tests_dir}
make -j$(nproc) CUDA_HOME=/usr/local/cuda NCCL_HOME=$NCCL_HOME 2>&1 | tee /tmp/nccl_tests_build.log
if [ $? -ne 0 ]; then
    echo "错误: nccl-tests 编译失败"
    cat /tmp/nccl_tests_build.log
    exit 1
fi

# 验证文件是否存在
if [ ! -f {remote_nccl_tests_dir}/build/all_reduce_perf ]; then
    echo "错误: {remote_nccl_tests_dir}/build/all_reduce_perf 不存在"
    exit 1
fi

chmod +x {remote_nccl_tests_dir}/build/all_reduce_perf
echo "编译完成"
"""
                compile_result = self.session.run(compile_script, timeout=600, require_root=True)
                if compile_result.exit_code != 0:
                    raise RuntimeError(f"编译失败: {compile_result.stderr or compile_result.stdout}")
            
            # 运行 NCCL 测试
            self.log("运行 NCCL 测试")
            test_script = f"""
{self.remote_dir}/nccl-tests/build/all_reduce_perf -b 1024 -e 1G -f 2 -g {gpu_count}
"""
            result = self.session.run(test_script, timeout=600, require_root=True)
            if result.exit_code != 0:
                raise RuntimeError(result.stderr or "nccl-tests 执行失败")
            value = parse_nccl(result.stdout)
            if value <= 0:
                raise RuntimeError("NCCL测试未解析到有效结果")
            benchmark = self.benchmark_for("nccl")
            passed = benchmark is None or value >= benchmark
            self.log(f"NCCL测试完成: {value:.1f} GB/s")
            return {
                "status": "passed" if passed else "failed",
                "value": value,
                "unit": "GB/s",
                "benchmark": benchmark,
                "passed": passed,
                "rawOutput": result.stdout,
            }
        except Exception as exc:  # pylint: disable=broad-except
            self.log(f"NCCL测试失败: {exc}")
            return {"status": "error", "message": str(exc)}

    def _run_dcgm_diag(self) -> Dict[str, Any]:
        try:
            cmd = f"dcgmi diag -r {self.dcgm_level}"
            result = self.session.run(cmd, timeout=1800, require_root=True)
            passed = result.exit_code == 0
            status = "passed" if passed else "failed"
            self.log(f"DCGM诊断完成，状态: {status}")
            return {
                "status": status,
                "passed": passed,
                "level": self.dcgm_level,
                "rawOutput": result.stdout or result.stderr,
            }
        except Exception as exc:  # pylint: disable=broad-except
            self.log(f"DCGM诊断失败: {exc}")
            return {"status": "error", "message": str(exc)}

    def _run_ib_check(self) -> Dict[str, Any]:
        try:
            remote_script = self._upload_asset("ib_check", "ib_health_check.sh")
            # 确保脚本可执行
            self.session.run(f"chmod +x {remote_script}", require_root=True)
            cmd = (
                f"cd {self.remote_dir} && "
                "export TERM=xterm; "
                "export PATH=\"/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/ib_health_check:$PATH\"; "
                f"{remote_script}"
            )
            result = self.session.run(cmd, timeout=900, require_root=True)
            output = (result.stdout or "") + (result.stderr or "")
            passed = result.exit_code == 0 and "通过模块: 10/10" in output
            status = "passed" if passed else "failed"
            self.log(f"IB检查完成，状态: {status}")
            if not passed and "通过模块: 10/10" in output:
                status = "passed"
            return {
                "status": status,
                "passed": status == "passed",
                "rawOutput": output.strip() or result.stderr or result.stdout,
            }
        except Exception as exc:  # pylint: disable=broad-except
            self.log(f"IB检查失败: {exc}")
            return {"status": "error", "message": str(exc)}


# -----------------------------------------------------------------------------
# Job存储
# -----------------------------------------------------------------------------

jobs: Dict[str, Dict[str, Any]] = {}
jobs_lock = threading.Lock()


def sanitize_job(job: Dict[str, Any]) -> Dict[str, Any]:
    # 创建副本，移除不能序列化的对象
    job_copy = {}
    for key, value in job.items():
        if key == "cancelled":
            # 将 Event 对象转换为布尔值
            if isinstance(value, threading.Event):
                job_copy[key] = value.is_set()
            else:
                job_copy[key] = value
        elif key == "nodes":
            # 处理节点列表
            nodes_copy = []
            for node in value:
                node_copy = {}
                for node_key, node_value in node.items():
                    if node_key != "_connection":
                        node_copy[node_key] = node_value
                nodes_copy.append(node_copy)
            job_copy[key] = nodes_copy
        else:
            job_copy[key] = value
    return job_copy


def start_job_worker(job_id: str):
    thread = threading.Thread(target=run_job, args=(job_id,), daemon=True)
    thread.start()


def run_node_check(node: Dict[str, Any], tests: List[str], dcgm_level: str, cancelled_flag: Optional[threading.Event] = None):
    """在单个节点上执行健康检查（用于并发执行）"""
    node["status"] = "running"
    node["startedAt"] = utc_now()
    connection = node.get("_connection")
    runner = RemoteNodeRunner(node, tests, dcgm_level, connection, cancelled_flag)
    try:
        node_result = runner.execute()
        node.update(node_result)
        node["status"] = node_result["overallStatus"]
        node["completedAt"] = utc_now()
    except Exception as exc:  # pylint: disable=broad-except
        host = node.get("host", "unknown")
        port = node.get("port", 22)
        node_display = f"{host}:{port}" if port != 22 else host
        logger.exception("节点 %s 执行失败: %s", node_display, exc)
        node["status"] = "failed"
        node["executionLog"] = "\n".join(runner.logs + [f"异常: {exc}"])
    finally:
        node.pop("_connection", None)
    return node


def run_job(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return
        job["status"] = "running"
        job["updatedAt"] = utc_now()
        # 创建取消标志
        if "cancelled" not in job:
            job["cancelled"] = threading.Event()
        cancelled_flag = job["cancelled"]
        nodes = job["nodes"]
        tests = job["tests"]
        dcgm_level = job["dcgmLevel"]

    # 并发执行所有节点的检查
    # 使用线程池，最大并发数等于节点数量（或限制为合理值，如10）
    max_workers = min(len(nodes), 10)
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有节点的检查任务
            future_to_node = {
                executor.submit(run_node_check, node, tests, dcgm_level, cancelled_flag): node
                for node in nodes
            }
        
            # 等待所有任务完成
            for future in as_completed(future_to_node):
                # 如果已取消，立即更新状态并退出，不等待剩余任务
                if cancelled_flag.is_set():
                    logger.info("任务 %s 已被取消，立即更新状态为 cancelled", job_id)
                    # 立即更新状态为 cancelled
                    with jobs_lock:
                        job = jobs.get(job_id)
                        if job:
                            job["status"] = "cancelled"
                            job["updatedAt"] = utc_now()
                            # 更新所有未完成的节点状态
                            for node in job["nodes"]:
                                if node["status"] in ("running", "cancelling"):
                                    node["status"] = "cancelled"
                                    if not node.get("completedAt"):
                                        node["completedAt"] = utc_now()
                    # 不再等待剩余任务，直接返回
                    return
                
                try:
                    future.result()
                except Exception as exc:  # pylint: disable=broad-except
                    node = future_to_node[future]
                    host = node.get("host", "unknown")
                    port = node.get("port", 22)
                    node_display = f"{host}:{port}" if port != 22 else host
                    logger.exception("节点 %s 执行异常: %s", node_display, exc)
                    if node["status"] == "running":
                        node["status"] = "failed"
                        node["executionLog"] = f"执行异常: {exc}"
    finally:
        pass

    # 所有节点完成后，更新任务状态
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return
        job["updatedAt"] = utc_now()
        if cancelled_flag.is_set():
            # 如果已经取消，确保状态是 cancelled
            job["status"] = "cancelled"
            # 更新所有未完成的节点状态（从 cancelling 或 running 转为 cancelled）
            for node in job["nodes"]:
                if node["status"] in ("running", "cancelling"):
                    node["status"] = "cancelled"
                    if not node.get("completedAt"):
                        node["completedAt"] = utc_now()
        else:
            job["status"] = (
                "completed"
                if all(node["status"] == "passed" for node in job["nodes"])
                else "failed"
            )
    


# -----------------------------------------------------------------------------
# API 路由
# -----------------------------------------------------------------------------


@app.route("/api/ssh/test-connection", methods=["POST"])
def api_test_connection():
    try:
        payload = request.get_json(force=True)
        if not payload:
            raise ValueError("请求体为空")
        connection = payload.get("connection", {})
        if not connection:
            raise ValueError("connection字段为空")
        logger.info("测试SSH连接: %s@%s:%s", connection.get("username"), connection.get("host"), connection.get("port", 22))
        ensure_payload_fields(connection, ["host", "username", "auth"])
        
        # 验证认证信息
        auth = connection.get("auth", {})
        if auth.get("type") == "privateKey":
            key_value = auth.get("value", "")
            if not key_value or not key_value.strip():
                raise ValueError("私钥内容为空")
            # 尝试预加载私钥以验证格式
            try:
                load_private_key(key_value, auth.get("passphrase"))
                logger.debug("私钥格式验证通过")
            except Exception as key_exc:
                logger.error("私钥格式验证失败: %s", key_exc, exc_info=True)
                raise ValueError(f"私钥格式错误: {key_exc}") from key_exc
        
        with SSHSession(connection) as session:
            hostname_res = session.run("hostname")
            gpu_res = session.run("nvidia-smi -L || true")
            driver_res = session.run("nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1 || true")
            # 获取内网IP（默认路由的出口IP）
            internal_ip_res = session.run("ip route get 1.1.1.1 2>/dev/null | grep -oP 'src \\K[0-9.]+' | head -n 1 || hostname -I | awk '{print $1}'")
        gpu_lines = [line.strip() for line in gpu_res.stdout.splitlines() if line.strip()]
        gpu_count = len(gpu_lines)
        gpu_model = normalize_gpu_name(gpu_lines[0]) if gpu_lines else "Unknown"
        internal_ip = internal_ip_res.stdout.strip() if internal_ip_res.stdout.strip() else None
        data = {
            "hostname": hostname_res.stdout.strip(),
            "gpus": gpu_lines,  # 保留完整列表用于兼容
            "gpuModel": gpu_model,  # GPU型号
            "gpuCount": gpu_count,  # GPU数量
            "driverVersion": driver_res.stdout.strip(),
            "internalIp": internal_ip,  # 内网IP
        }
        logger.info("SSH连接测试成功: %s, 内网IP: %s", data.get("hostname"), internal_ip)
        return json_response(True, data=data, message="SSH连接成功")
    except Exception as exc:  # pylint: disable=broad-except
        error_msg = str(exc)
        logger.error("SSH连接测试失败: %s", error_msg, exc_info=True)
        return json_response(False, message=error_msg, status=400)


def extract_cuda_version(nvcc_output: str) -> str:
    """从 nvcc --version 输出中提取 CUDA 版本号"""
    import re
    # 匹配 "release X.Y" 或 "V X.Y.Z"
    match = re.search(r'release\s+(\d+\.\d+)', nvcc_output)
    if match:
        return match.group(1)
    match = re.search(r'V(\d+\.\d+)', nvcc_output)
    if match:
        return match.group(1)
    logger.warning("extract_cuda_version: 未能从输出中提取CUDA版本")
    return ""


def extract_nccl_version(apt_output: str, package_name: str) -> str:
    """从 apt list 输出中提取 NCCL 包版本"""
    import re
    lines = apt_output.splitlines()
    logger.debug("extract_nccl_version: 查找包 %s, 输入行数=%d", package_name, len(lines))
    
    for idx, line in enumerate(lines):
        # 跳过警告行
        line_stripped = line.strip()
        if line_stripped.startswith("WARNING:"):
            continue
        
        # 检查包名和 [installed] 标记
        has_package = package_name in line_stripped
        # 使用关键字匹配方法：检查 "installed" 是否在方括号之间
        has_installed = ("installed" in line_stripped.lower() and 
                        "[" in line_stripped and "]" in line_stripped)
        
        logger.debug("extract_nccl_version: 行[%d]: package_name(%s) in line=%s, '[installed]' in line=%s", 
                   idx, package_name, has_package, has_installed)
        
        if has_package and has_installed:
            # 格式: libnccl2/unknown,now 2.26.2-1+cuda12.8 amd64 [installed,upgradable to: 2.27.3-1+cuda12.9]
            # 只匹配 [installed] 之前的内容，避免匹配到 upgradable to 后的版本
            installed_part = line_stripped.split("[installed]")[0].strip()
            # 格式: libnccl2/unknown,now 2.26.2-1+cuda12.8 amd64
            # 匹配版本号格式: 数字.数字.数字-数字+cuda数字.数字
            import re
            match = re.search(r'(\d+\.\d+\.\d+)-\d+\+cuda(\d+\.\d+)', installed_part)
            if match:
                cuda_version = match.group(2)
                logger.debug("extract_nccl_version: 提取到CUDA版本: %s (完整匹配: %s)", cuda_version, match.group(0))
                return cuda_version
            else:
                # 尝试更宽松的匹配
                match2 = re.search(r'cuda(\d+\.\d+)', installed_part)
                if match2:
                    cuda_version = match2.group(1)
                    logger.debug("extract_nccl_version: 通过宽松模式提取到CUDA版本: %s", cuda_version)
                    return cuda_version
                else:
                    logger.warning("extract_nccl_version: 行匹配但版本提取失败: %s", installed_part)
    
    logger.warning("extract_nccl_version: 未找到包 %s 的已安装版本", package_name)
    return ""


@app.route("/api/ssh/check-commands", methods=["POST"])
def api_check_commands():
    try:
        payload = request.get_json(force=True)
        connection = payload.get("connection", {})
        commands = payload.get("commands", [])
        ensure_payload_fields(connection, ["host", "username", "auth"])
        if not commands:
            raise ValueError("commands不能为空")
        results = {}
        versions = {}
        
        with SSHSession(connection) as session:
            for cmd in commands:
                # 检查是否是包名（libnccl2, libnccl-dev）
                if cmd in ("libnccl2", "libnccl-dev"):
                    # 使用更简单的命令，直接检查包名和 [installed] 标记
                    # apt list 需要 root 权限
                    check_cmd = f"apt list --installed 2>/dev/null | grep -E '^{cmd}/'"
                    res = session.run(check_cmd, require_root=True)
                    output = res.stdout.strip()
                    # 检查输出中是否包含 [installed]
                    # 使用关键字匹配方法：检查 "installed" 是否在方括号之间
                    has_installed = False
                    if output:
                        has_installed = ("installed" in output.lower() and 
                                        "[" in output and "]" in output and
                                        output.find("[") < output.find("installed", output.find("[")) < output.find("]", output.find("[")))
                    logger.debug("包检测 %s: 输出长度=%d, 包含[installed]=%s", cmd, len(output), has_installed)
                    results[cmd] = has_installed
                # 检查 nvidia_peermem 内核模块是否加载
                elif cmd == "nvidia_peermem":
                    check_cmd = "lsmod | grep nvidia_peermem"
                    res = session.run(check_cmd)
                    # 有输出说明模块已加载
                    results[cmd] = bool(res.stdout.strip())
                # 检查 nouveau 驱动是否已卸载（应该没有输出才是通过）
                elif cmd == "nouveau_unloaded":
                    check_cmd = "lsmod | grep nouveau"
                    res = session.run(check_cmd)
                    # 没有输出说明已卸载
                    results[cmd] = not bool(res.stdout.strip())
                # 检查 ACS 是否已关闭（所有都应该是减号，不能有 + 号）
                elif cmd == "acsctl_disabled":
                    # 检查 lspci 输出中 ACSCtl 行（需要 root 权限才能看到详细信息）
                    check_cmd = "sudo lspci -vvv 2>/dev/null | grep -i acsctl || lspci -vvv 2>/dev/null | grep -i acsctl"
                    res = session.run(check_cmd, require_root=True)
                    acsctl_output = res.stdout.strip()
                    if acsctl_output:
                        # 检查是否有任何 + 号（如 SrcValid+ TransBlk+ 等）
                        # 有 + 号表示 ACS 未完全关闭
                        has_plus = '+' in acsctl_output
                        results[cmd] = not has_plus
                    else:
                        # 没有 ACSCtl 输出，可能设备不支持 ACS，视为通过
                        results[cmd] = True
                # 检查 nvidia-fabricmanager 服务是否激活
                elif cmd == "nvidia_fabricmanager_active":
                    check_cmd = "systemctl is-active nvidia-fabricmanager.service 2>/dev/null || echo inactive"
                    res = session.run(check_cmd)
                    results[cmd] = res.stdout.strip() == "active"
                # 检查 ulimit max locked memory 是否为 unlimited
                # 注意：必须以root权限检查，因为测试是以root权限运行的
                elif cmd == "ulimit_max_locked_memory":
                    check_cmd = "ulimit -a 2>/dev/null"
                    res = session.run(check_cmd, require_root=True)
                    # 解析 ulimit -a 输出，查找 max locked memory 行
                    value = None
                    matched_line = None
                    for line in res.stdout.splitlines():
                        line_lower = line.lower()
                        if "max locked memory" in line_lower:
                            matched_line = line
                            # 格式: "max locked memory           (kbytes, -l) 264176236"
                            # 或者: "max locked memory           (kbytes, -l) unlimited"
                            # 直接取最后一列（split() 会处理多个空格）
                            parts = line.split()
                            if parts:
                                value = parts[-1].strip().lower()
                            break
                    is_unlimited = (value == "unlimited") if value else False
                    logger.debug("ulimit_max_locked_memory检查(以root权限): 原始行='%s', 提取值='%s', 是否unlimited=%s, 结果=%s", 
                               matched_line, value, is_unlimited, "通过" if is_unlimited else "失败")
                    results[cmd] = is_unlimited
                # 检查 ulimit max memory size 是否为 unlimited
                # 注意：必须以root权限检查，因为测试是以root权限运行的
                elif cmd == "ulimit_max_memory_size":
                    check_cmd = "ulimit -a 2>/dev/null"
                    res = session.run(check_cmd, require_root=True)
                    # 解析 ulimit -a 输出，查找 max memory size 行
                    value = None
                    matched_line = None
                    for line in res.stdout.splitlines():
                        line_lower = line.lower()
                        if "max memory size" in line_lower:
                            matched_line = line
                            # 格式: "max memory size             (kbytes, -m) unlimited"
                            # 直接取最后一列（split() 会处理多个空格）
                            parts = line.split()
                            if parts:
                                value = parts[-1].strip().lower()
                            break
                    is_unlimited = (value == "unlimited") if value else False
                    logger.debug("ulimit_max_memory_size检查(以root权限): 原始行='%s', 提取值='%s', 是否unlimited=%s, 结果=%s", 
                               matched_line, value, is_unlimited, "通过" if is_unlimited else "失败")
                    results[cmd] = is_unlimited
                elif "/" in cmd:
                    check_cmd = f"[ -x {cmd} ] && echo OK || echo MISSING"
                    res = session.run(check_cmd)
                    results[cmd] = res.stdout.strip() == "OK"
                else:
                    check_cmd = f"command -v {cmd} >/dev/null 2>&1 && echo OK || echo MISSING"
                    res = session.run(check_cmd)
                    results[cmd] = res.stdout.strip() == "OK"
            
            # 获取版本信息用于比对
            nvcc_res = session.run("/usr/local/cuda/bin/nvcc --version 2>/dev/null || true")
            # apt list 需要 root 权限
            apt_res = session.run("apt list --installed 2>/dev/null | grep -E '^libnccl' || true", require_root=True)
            
            nvcc_version = extract_cuda_version(nvcc_res.stdout)
            libnccl2_version = extract_nccl_version(apt_res.stdout, "libnccl2")
            libnccl_dev_version = extract_nccl_version(apt_res.stdout, "libnccl-dev")
            
            version_match = bool(
                nvcc_version and 
                libnccl2_version and 
                libnccl_dev_version and
                nvcc_version == libnccl2_version == libnccl_dev_version
            )
            logger.debug("版本检查: nvcc=%s, libnccl2=%s, libnccl-dev=%s, 匹配=%s", 
                       nvcc_version, libnccl2_version, libnccl_dev_version, version_match)
            
            versions = {
                "nvcc": nvcc_version,
                "libnccl2": libnccl2_version,
                "libncclDev": libnccl_dev_version,
                "versionMatch": version_match
            }
        
        return json_response(True, data={"commands": results, "versions": versions}, message="命令检查完成")
    except Exception as exc:  # pylint: disable-broad-except
        logger.exception("命令检查失败: %s", exc)
        return json_response(False, message=str(exc), status=400)


@app.route("/api/config/gpu-benchmarks", methods=["GET"])
def api_get_gpu_benchmarks():
    return json_response(
        True,
        data={
            "benchmarks": GPU_BENCHMARKS,
            "source": BENCHMARK_FILE,
        },
    )


@app.route("/api/gpu-inspection/create-job", methods=["POST"])
def api_create_job():
    try:
        payload = request.get_json(force=True)
        nodes_payload = payload.get("nodes", [])
        tests = payload.get("tests", [])
        dcgm_level = int(payload.get("dcgmLevel", 2))
        if not nodes_payload:
            raise ValueError("nodes不能为空")
        if not tests:
            raise ValueError("tests不能为空")

        job_id = payload.get("jobName") or f"manual-{uuid.uuid4().hex[:8]}"
        job = {
            "jobId": job_id,
            "jobName": payload.get("jobName") or job_id,
            "createdAt": utc_now(),
            "updatedAt": utc_now(),
            "status": "pending",
            "tests": tests,
            "dcgmLevel": dcgm_level,
            "nodes": [],
            "cancelled": threading.Event(),
        }

        for node_payload in nodes_payload:
            ensure_payload_fields(node_payload, ["host", "username", "auth"])
            auth_payload = node_payload.get("auth")
            if not isinstance(auth_payload, dict) or not auth_payload.get("type"):
                raise ValueError(f"节点 {node_payload.get('host')} 缺少有效的认证信息")
            node_entry = {
                "nodeId": uuid.uuid4().hex,
                "host": node_payload["host"],
                "port": node_payload.get("port", 22),
                "username": node_payload["username"],
                "status": "pending",
                "_connection": {
                    "host": node_payload["host"],
                    "port": node_payload.get("port", 22),
                    "username": node_payload["username"],
                    "auth": auth_payload,
                    "sudoPassword": node_payload.get("sudoPassword"),
                },
            }
            job["nodes"].append(node_entry)

        with jobs_lock:
            jobs[job_id] = job

        start_job_worker(job_id)
        return json_response(True, data={"jobId": job_id}, message="Job已创建")
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("创建Job失败: %s", exc)
        return json_response(False, message=str(exc), status=400)


@app.route("/api/gpu-inspection/job/<job_id>", methods=["GET"])
def api_get_job(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return json_response(False, message="未找到Job", status=404)
        return json_response(True, data=sanitize_job(job))


@app.route("/api/gpu-inspection/jobs", methods=["GET"])
def api_list_jobs():
    with jobs_lock:
        data = [sanitize_job(job) for job in jobs.values()]
    return json_response(True, data=data)


@app.route("/api/gpu-inspection/setup-ssh-trust", methods=["POST"])
def api_setup_ssh_trust():
    """配置多节点间SSH免密互信"""
    try:
        payload = request.get_json(force=True)
        nodes = payload.get("nodes", [])  # 节点连接信息列表
        
        if len(nodes) < 2:
            raise ValueError("至少需要2个节点来配置SSH互信")
        
        results = []
        node_info = []  # 存储每个节点的信息：{connection, internal_ip, pubkey, display_name}
        
        # 第一步：收集所有节点的公钥和内网IP
        logger.info("开始收集 %d 个节点的SSH公钥和内网IP", len(nodes))
        for idx, node in enumerate(nodes):
            host = node.get("host")
            port = node.get("port", 22)
            display_name = f"{host}:{port}"
            
            try:
                with SSHSession(node) as session:
                    # 确保 .ssh 目录存在
                    session.run("mkdir -p /root/.ssh && chmod 700 /root/.ssh", require_root=True)
                    
                    # 获取内网IP
                    internal_ip_res = session.run("ip route get 1.1.1.1 2>/dev/null | grep -oP 'src \\K[0-9.]+' | head -n 1 || hostname -I | awk '{print $1}'")
                    internal_ip = internal_ip_res.stdout.strip()
                    if not internal_ip:
                        raise RuntimeError("无法获取内网IP")
                    
                    # 检查是否已有密钥，没有则生成
                    check_key = session.run("[ -f /root/.ssh/id_rsa ] && echo EXISTS || echo MISSING", require_root=True)
                    if check_key.stdout.strip() == "MISSING":
                        logger.info("为节点 %s (内网IP: %s) 生成SSH密钥对", display_name, internal_ip)
                        session.run("ssh-keygen -t rsa -b 2048 -f /root/.ssh/id_rsa -N '' -q", require_root=True)
                    
                    # 获取公钥
                    pubkey_result = session.run("cat /root/.ssh/id_rsa.pub", require_root=True)
                    if pubkey_result.exit_code == 0 and pubkey_result.stdout.strip():
                        node_info.append({
                            "connection": node,
                            "internal_ip": internal_ip,
                            "pubkey": pubkey_result.stdout.strip(),
                            "display_name": display_name,
                            "idx": idx,
                        })
                        results.append({"host": display_name, "internalIp": internal_ip, "status": "pubkey_collected", "message": f"公钥已收集 (内网: {internal_ip})"})
                        logger.info("节点 %s 公钥已收集，内网IP: %s", display_name, internal_ip)
                    else:
                        results.append({"host": display_name, "status": "error", "message": "无法获取公钥"})
            except Exception as exc:
                logger.error("收集节点 %s 公钥失败: %s", display_name, exc)
                results.append({"host": display_name, "status": "error", "message": str(exc)})
        
        if len(node_info) < 2:
            return json_response(False, message=f"成功收集的公钥数量不足({len(node_info)}个)，无法配置互信", data={"results": results}, status=400)
        
        # 第二步：将所有公钥分发到所有节点
        logger.info("开始分发公钥到 %d 个节点", len(node_info))
        authorized_keys_content = "\n".join([n["pubkey"] for n in node_info])
        all_internal_ips = [n["internal_ip"] for n in node_info]
        
        for info in node_info:
            display_name = info["display_name"]
            try:
                with SSHSession(info["connection"]) as session:
                    # 写入 authorized_keys
                    escaped_content = authorized_keys_content.replace("'", "'\\''")
                    session.run(f"echo '{escaped_content}' > /root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys", require_root=True)
                    
                    # 配置 ssh_config 禁用 StrictHostKeyChecking
                    session.run("grep -q 'StrictHostKeyChecking' /etc/ssh/ssh_config || echo 'StrictHostKeyChecking no' >> /etc/ssh/ssh_config", require_root=True)
                    
                    # 预填充 known_hosts（使用内网IP扫描所有节点）
                    all_ips_str = " ".join(all_internal_ips)
                    session.run(f"ssh-keyscan -t rsa {all_ips_str} >> /root/.ssh/known_hosts 2>/dev/null; sort -u /root/.ssh/known_hosts -o /root/.ssh/known_hosts", require_root=True)
                    
                    # 更新结果
                    for r in results:
                        if r["host"] == display_name:
                            r["status"] = "success"
                            r["message"] = f"SSH互信配置完成 (内网: {info['internal_ip']})"
                            break
                    logger.info("节点 %s SSH互信配置完成", display_name)
            except Exception as exc:
                logger.error("配置节点 %s SSH互信失败: %s", display_name, exc)
                for r in results:
                    if r["host"] == display_name:
                        r["status"] = "error"
                        r["message"] = f"分发公钥失败: {exc}"
                        break
        
        # 第三步：从第一个节点测试到其他节点的SSH连接
        logger.info("开始从第一个节点测试到其他节点的SSH连接")
        first_node_info = node_info[0]
        test_failures = []
        
        for info in node_info[1:]:  # 从第二个节点开始测试
            display_name = info["display_name"]
            target_internal_ip = info["internal_ip"]
            try:
                with SSHSession(first_node_info["connection"]) as session:
                    # 从第一个节点SSH连接到目标节点，测试连接是否成功
                    test_cmd = f"ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o BatchMode=yes {target_internal_ip} 'echo SSH_TEST_OK' 2>&1"
                    test_result = session.run(test_cmd, timeout=15, require_root=True)
                    
                    if test_result.exit_code == 0 and "SSH_TEST_OK" in test_result.stdout:
                        logger.info("从第一个节点到节点 %s (内网IP: %s) SSH连接测试成功", display_name, target_internal_ip)
                    else:
                        error_msg = test_result.stderr or test_result.stdout or "SSH连接测试失败"
                        logger.error("从第一个节点到节点 %s (内网IP: %s) SSH连接测试失败: %s", display_name, target_internal_ip, error_msg)
                        test_failures.append((display_name, target_internal_ip, error_msg))
                        # 提取关键错误信息（去除冗余前缀）
                        clean_error = error_msg.strip()
                        if "Permission denied" in clean_error:
                            # 提取关键部分，如 "Permission denied (publickey)"
                            if "(publickey)" in clean_error:
                                clean_error = "Permission denied (publickey) - 公钥认证失败，请检查SSH配置"
                            elif "(password)" in clean_error:
                                clean_error = "Permission denied (password) - 密码认证失败"
                            else:
                                clean_error = "Permission denied - 权限被拒绝"
                        # 更新结果状态为警告
                        for r in results:
                            if r["host"] == display_name:
                                r["status"] = "warning"
                                r["message"] = f"连接测试失败: {clean_error}"
                                break
            except Exception as exc:
                logger.exception("测试从第一个节点到节点 %s 的SSH连接时发生异常: %s", display_name, exc)
                error_msg = str(exc)
                test_failures.append((display_name, target_internal_ip, error_msg))
                # 更新结果状态为警告
                for r in results:
                    if r["host"] == display_name:
                        r["status"] = "warning"
                        r["message"] = f"连接测试异常: {error_msg}"
                        break
        
        success_count = sum(1 for r in results if r["status"] == "success")
        warning_count = sum(1 for r in results if r["status"] == "warning")
        
        # 如果有连接测试失败，返回警告信息
        if test_failures:
            failure_details = "; ".join([f"{name}({ip}): {err}" for name, ip, err in test_failures])
            warning_msg = f"SSH互信配置完成，但部分节点连接测试失败: {failure_details}"
            logger.warning(warning_msg)
            return json_response(
                True,
                data={"results": results, "successCount": success_count, "warningCount": warning_count, "totalCount": len(nodes), "testFailures": test_failures},
                message=warning_msg
            )
        
        return json_response(
            True,
            data={"results": results, "successCount": success_count, "totalCount": len(nodes)},
            message=f"SSH互信配置完成: {success_count}/{len(nodes)} 个节点成功"
        )
    except Exception as exc:
        logger.exception("配置SSH互信失败: %s", exc)
        return json_response(False, message=str(exc), status=400)


# 多机测试任务存储（类似 jobs，但结构更简单）
multi_node_tests: Dict[str, Dict[str, Any]] = {}
multi_node_tests_lock = threading.Lock()


def run_multi_node_nccl_task(test_id: str, payload: Dict[str, Any]):
    """在后台线程中执行多机NCCL测试"""
    try:
        with multi_node_tests_lock:
            test = multi_node_tests.get(test_id)
            if not test:
                return
            test["status"] = "running"
            test["startedAt"] = utc_now()
        
        hosts = payload.get("hosts", [])
        hostfile_content = payload.get("hostfileContent")
        mpi_params = payload.get("mpiParams", {})
        connection = payload.get("connection")
        
        # 解析hosts
        if hostfile_content:
            host_list = [h.strip() for h in hostfile_content.strip().split('\n') if h.strip()]
        elif hosts:
            host_list = hosts
        else:
            raise ValueError("请提供hosts列表或hostfile内容")
        
        if len(host_list) < 2:
            raise ValueError("多机测试至少需要2个节点")
        
        np_count = len(host_list)
        
        # 构建mpirun命令
        mpi_cmd_parts = [
            "mpirun",
            f"-np {np_count}",
            "--allow-run-as-root",
            "-N 1",
        ]
        
        if hostfile_content:
            mpi_cmd_parts.append("-hostfile /tmp/ghx/hostfile")
        else:
            mpi_cmd_parts.append(f"-host {','.join(host_list)}")
        
        # 添加用户自定义参数
        if mpi_params.get("btl_tcp_if"):
            mpi_cmd_parts.append(f"--mca btl_tcp_if_include {mpi_params['btl_tcp_if']}")
            mpi_cmd_parts.append(f"--mca oob_tcp_if_include {mpi_params['btl_tcp_if']}")
        
        if mpi_params.get("nccl_socket_ifname"):
            mpi_cmd_parts.append(f"-x NCCL_SOCKET_IFNAME={mpi_params['nccl_socket_ifname']}")
        
        if mpi_params.get("nccl_ib_hca"):
            mpi_cmd_parts.append(f"-x NCCL_IB_HCA={mpi_params['nccl_ib_hca']}")
        
        if mpi_params.get("ucx_net_devices"):
            mpi_cmd_parts.append(f"-x UCX_NET_DEVICES={mpi_params['ucx_net_devices']}")
        
        if mpi_params.get("nccl_ib_qps"):
            mpi_cmd_parts.append(f"-x NCCL_IB_QPS_PER_CONNECTION={mpi_params['nccl_ib_qps']}")
        
        if mpi_params.get("nccl_pxn_disable") is not None:
            mpi_cmd_parts.append(f"-x NCCL_PXN_DISABLE={mpi_params['nccl_pxn_disable']}")
        
        if mpi_params.get("nccl_min_nchannels"):
            mpi_cmd_parts.append(f"-x NCCL_MIN_NCHANNELS={mpi_params['nccl_min_nchannels']}")
        
        if mpi_params.get("nccl_nvls_enable") is not None:
            mpi_cmd_parts.append(f"-x NCCL_NVLS_ENABLE={mpi_params['nccl_nvls_enable']}")
        
        if mpi_params.get("sharp_relaxed_ordering"):
            mpi_cmd_parts.append("-x SHARP_COLL_ENABLE_PCI_RELAXED_ORDERING=1")
        
        if mpi_params.get("extra"):
            mpi_cmd_parts.append(mpi_params['extra'])
        
        gpu_count = mpi_params.get("gpuPerNode", 8)
        mpi_cmd_parts.append(f"/tmp/ghx/nccl-tests/build/all_reduce_perf -b 128M -e 16G -f 2 -g {gpu_count}")
        
        mpi_command = " \\\n".join(mpi_cmd_parts)
        
        # 连接主节点执行
        with SSHSession(connection) as session:
            session.run("mkdir -p /tmp/ghx")
            
            if hostfile_content:
                hostfile_path = "/tmp/ghx/hostfile"
                session.run(f"cat > {hostfile_path} << 'EOF'\n{hostfile_content}\nEOF")
            
            # 检查主节点nccl-tests是否存在，不存在则上传并编译
            check_res = session.run("[ -f /tmp/ghx/nccl-tests/build/all_reduce_perf ] && echo OK || echo MISSING")
            if check_res.stdout.strip() != "OK":
                logger.info("主节点 nccl-tests 不存在，开始上传源码并编译")
                
                nccl_tgz = ASSETS["nccl"]
                nccl_tests_tgz = ASSETS["nccl_tests"]
                
                if not nccl_tgz.exists():
                    raise FileNotFoundError(f"nccl.tgz 文件不存在: {nccl_tgz}")
                if not nccl_tests_tgz.exists():
                    raise FileNotFoundError(f"nccl-tests.tgz 文件不存在: {nccl_tests_tgz}")
                
                remote_nccl_tgz = "/tmp/ghx/nccl.tgz"
                remote_nccl_tests_tgz = "/tmp/ghx/nccl-tests.tgz"
                session.upload(nccl_tgz, remote_nccl_tgz)
                session.upload(nccl_tests_tgz, remote_nccl_tests_tgz)
                
                compile_script = """
set -e
rm -rf /tmp/ghx/nccl /tmp/ghx/nccl-tests
echo "解压 nccl.tgz..."
tar -xzf /tmp/ghx/nccl.tgz -C /tmp/ghx
rm -f /tmp/ghx/nccl.tgz
echo "编译 nccl..."
cd /tmp/ghx/nccl
make -j$(nproc) CUDA_HOME=/usr/local/cuda 2>&1 | tee /tmp/nccl_build.log
if [ $? -ne 0 ]; then
    echo "错误: nccl 编译失败"
    cat /tmp/nccl_build.log
    exit 1
fi
export NCCL_HOME=/tmp/ghx/nccl
echo "解压 nccl-tests.tgz..."
tar -xzf /tmp/ghx/nccl-tests.tgz -C /tmp/ghx
rm -f /tmp/ghx/nccl-tests.tgz
echo "编译 nccl-tests..."
cd /tmp/ghx/nccl-tests
make -j$(nproc) CUDA_HOME=/usr/local/cuda NCCL_HOME=$NCCL_HOME 2>&1 | tee /tmp/nccl_tests_build.log
if [ $? -ne 0 ]; then
    echo "错误: nccl-tests 编译失败"
    cat /tmp/nccl_tests_build.log
    exit 1
fi
if [ ! -f /tmp/ghx/nccl-tests/build/all_reduce_perf ]; then
    echo "错误: /tmp/ghx/nccl-tests/build/all_reduce_perf 不存在"
    exit 1
fi
chmod +x /tmp/ghx/nccl-tests/build/all_reduce_perf
echo "编译完成"
"""
                compile_result = session.run(compile_script, timeout=600, require_root=True)
                if compile_result.exit_code != 0:
                    raise RuntimeError(f"编译失败: {compile_result.stderr or compile_result.stdout}")
                
                check_res = session.run("[ -f /tmp/ghx/nccl-tests/build/all_reduce_perf ] && echo OK || echo MISSING")
                if check_res.stdout.strip() != "OK":
                    raise RuntimeError("nccl-tests 编译后仍未找到 all_reduce_perf")
            
            # 为所有其他节点上传源码并编译
            master_host = host_list[0]
            other_hosts = host_list[1:]
            
            if other_hosts:
                logger.info("开始为其他 %d 个节点并发上传源码并编译 nccl-tests", len(other_hosts))
                
                nccl_tgz = ASSETS["nccl"]
                nccl_tests_tgz = ASSETS["nccl_tests"]
                
                temp_nccl_path = "/tmp/ghx/nccl.tgz"
                temp_nccl_tests_path = "/tmp/ghx/nccl-tests.tgz"
                session.upload(nccl_tgz, temp_nccl_path)
                session.upload(nccl_tests_tgz, temp_nccl_tests_path)
                
                def upload_and_compile_node(host: str) -> tuple[str, bool, str]:
                    try:
                        # 先检查节点是否已有 nccl-tests
                        check_script = f"""
ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 {host} "[ -f /tmp/ghx/nccl-tests/build/all_reduce_perf ] && echo OK || echo MISSING" 2>/dev/null || echo "MISSING"
"""
                        check_result = session.run(check_script, timeout=30, require_root=True)
                        if check_result.stdout.strip() == "OK":
                            logger.info("节点 %s 已存在 nccl-tests，跳过编译", host)
                            return (host, True, "")
                        
                        logger.info("开始为节点 %s 上传源码并编译 nccl-tests", host)
                        upload_and_compile_script = f"""
set -e
echo "上传源码到节点 {host}..."
ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 {host} "mkdir -p /tmp/ghx" || exit 1
scp -o StrictHostKeyChecking=no -o ConnectTimeout=10 {temp_nccl_path} {host}:/tmp/ghx/nccl.tgz || exit 1
scp -o StrictHostKeyChecking=no -o ConnectTimeout=10 {temp_nccl_tests_path} {host}:/tmp/ghx/nccl-tests.tgz || exit 1
ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 {host} << 'REMOTE_SCRIPT'
set -e
rm -rf /tmp/ghx/nccl /tmp/ghx/nccl-tests
echo "解压 nccl.tgz..."
tar -xzf /tmp/ghx/nccl.tgz -C /tmp/ghx
rm -f /tmp/ghx/nccl.tgz
echo "编译 nccl..."
cd /tmp/ghx/nccl
make -j$(nproc) CUDA_HOME=/usr/local/cuda 2>&1 | tee /tmp/nccl_build.log
if [ $? -ne 0 ]; then
    echo "错误: nccl 编译失败"
    cat /tmp/nccl_build.log
    exit 1
fi
export NCCL_HOME=/tmp/ghx/nccl
echo "解压 nccl-tests.tgz..."
tar -xzf /tmp/ghx/nccl-tests.tgz -C /tmp/ghx
rm -f /tmp/ghx/nccl-tests.tgz
echo "编译 nccl-tests..."
cd /tmp/ghx/nccl-tests
make -j$(nproc) CUDA_HOME=/usr/local/cuda NCCL_HOME=$NCCL_HOME 2>&1 | tee /tmp/nccl_tests_build.log
if [ $? -ne 0 ]; then
    echo "错误: nccl-tests 编译失败"
    cat /tmp/nccl_tests_build.log
    exit 1
fi
if [ ! -f /tmp/ghx/nccl-tests/build/all_reduce_perf ]; then
    echo "错误: /tmp/ghx/nccl-tests/build/all_reduce_perf 不存在"
    exit 1
fi
chmod +x /tmp/ghx/nccl-tests/build/all_reduce_perf
echo "节点 {host} 编译完成"
REMOTE_SCRIPT
if [ $? -eq 0 ]; then
    echo "节点 {host} 编译成功"
else
    echo "节点 {host} 编译失败"
    exit 1
fi
"""
                        compile_result = session.run(upload_and_compile_script, timeout=600, require_root=True)
                        if compile_result.exit_code != 0:
                            error_msg = compile_result.stderr or compile_result.stdout or "未知错误"
                            logger.error("节点 %s 编译失败: %s", host, error_msg)
                            return (host, False, error_msg)
                        else:
                            logger.info("节点 %s 编译成功", host)
                            return (host, True, "")
                    except Exception as exc:
                        error_msg = str(exc)
                        logger.exception("节点 %s 编译异常: %s", host, error_msg)
                        return (host, False, error_msg)
                
                failed_hosts = []
                with ThreadPoolExecutor(max_workers=min(len(other_hosts), 10)) as executor:
                    future_to_host = {executor.submit(upload_and_compile_node, host): host for host in other_hosts}
                    for future in as_completed(future_to_host):
                        host = future_to_host[future]
                        try:
                            result_host, success, error_msg = future.result()
                            if not success:
                                failed_hosts.append((result_host, error_msg))
                        except Exception as exc:
                            logger.exception("节点 %s 任务执行异常: %s", host, exc)
                            failed_hosts.append((host, str(exc)))
                
                session.run(f"rm -f {temp_nccl_path} {temp_nccl_tests_path}", require_root=True)
                
                if failed_hosts:
                    error_msg = f"以下节点编译失败: {', '.join([h for h, _ in failed_hosts])}\n请确保：\n1. SSH免密已配置\n2. 节点之间网络连通\n3. 节点有足够的编译工具"
                    logger.error("部分节点编译失败: %s", ', '.join([h for h, _ in failed_hosts]))
                    raise RuntimeError(error_msg)
                
                logger.info("所有节点 nccl-tests 编译完成")
            
            # 执行mpirun命令
            logger.info("执行多机NCCL测试: %s", mpi_command)
            result = session.run(mpi_command, timeout=1800, require_root=True)
            
            # 解析结果
            value = parse_nccl(result.stdout)
            
            with multi_node_tests_lock:
                test = multi_node_tests.get(test_id)
                if test:
                    test["status"] = "completed"
                    test["completedAt"] = utc_now()
                    test["result"] = {
                        "command": mpi_command,
                        "hosts": host_list,
                        "nodeCount": np_count,
                        "exitCode": result.exit_code,
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                        "bandwidth": value if value > 0 else None,
                        "passed": result.exit_code == 0,
                    }
    except Exception as exc:
        logger.exception("多机NCCL测试失败: %s", exc)
        with multi_node_tests_lock:
            test = multi_node_tests.get(test_id)
            if test:
                test["status"] = "failed"
                test["completedAt"] = utc_now()
                test["error"] = str(exc)


@app.route("/api/gpu-inspection/multi-node-nccl", methods=["POST"])
def api_multi_node_nccl():
    """多机mpirun NCCL测试（异步）"""
    try:
        payload = request.get_json(force=True)
        connection = payload.get("connection")
        
        if not connection:
            raise ValueError("缺少主节点SSH连接信息")
        
        # 创建测试任务ID
        test_id = str(uuid.uuid4())
        
        # 创建测试任务记录
        with multi_node_tests_lock:
            multi_node_tests[test_id] = {
                "id": test_id,
                "status": "pending",
                "createdAt": utc_now(),
                "payload": payload,
            }
        
        # 在后台线程中执行
        thread = threading.Thread(target=run_multi_node_nccl_task, args=(test_id, payload), daemon=True)
        thread.start()
        
        # 立即返回任务ID
        return json_response(True, data={"testId": test_id}, message="多机NCCL测试已启动")
    except Exception as exc:
        logger.exception("启动多机NCCL测试失败: %s", exc)
        return json_response(False, message=str(exc), status=400)


@app.route("/api/gpu-inspection/multi-node-nccl/<test_id>", methods=["GET"])
def api_get_multi_node_nccl_status(test_id: str):
    """查询多机NCCL测试状态"""
    with multi_node_tests_lock:
        test = multi_node_tests.get(test_id)
        if not test:
            return json_response(False, message="未找到测试任务", status=404)
        
        result_data = {
            "testId": test_id,
            "status": test["status"],
            "createdAt": test["createdAt"],
        }
        
        if test.get("startedAt"):
            result_data["startedAt"] = test["startedAt"]
        if test.get("completedAt"):
            result_data["completedAt"] = test["completedAt"]
        if test.get("result"):
            result_data["result"] = test["result"]
        if test.get("error"):
            result_data["error"] = test["error"]
        
        return json_response(True, data=result_data)


@app.route("/api/gpu-inspection/stop-job/<job_id>", methods=["POST"])
def api_stop_job(job_id: str):
    """停止正在运行的健康检查任务"""
    try:
        with jobs_lock:
            job = jobs.get(job_id)
            if not job:
                return json_response(False, message="未找到Job", status=404)
            
            if job["status"] not in ("pending", "running", "cancelling"):
                return json_response(False, message=f"任务状态为 {job['status']}，无法停止", status=400)
            
            # 设置取消标志
            if "cancelled" in job:
                job["cancelled"].set()
            else:
                job["cancelled"] = threading.Event()
                job["cancelled"].set()
            
            # 如果任务状态是 running，立即更新为 cancelled（不再使用 cancelling 中间状态）
            # 这样可以避免前端一直显示"取消中"
            if job["status"] == "running":
                job["status"] = "cancelled"
                job["updatedAt"] = utc_now()
                # 更新所有运行中的节点状态为 cancelled
                for node in job.get("nodes", []):
                    if node["status"] == "running":
                        node["status"] = "cancelled"
                        if not node.get("completedAt"):
                            node["completedAt"] = utc_now()
                logger.info("任务 %s 已立即标记为取消，共 %d 个节点", job_id, len([n for n in job.get("nodes", []) if n["status"] == "cancelled"]))
            elif job["status"] == "cancelling":
                # 如果已经是 cancelling，直接更新为 cancelled
                job["status"] = "cancelled"
                job["updatedAt"] = utc_now()
                for node in job.get("nodes", []):
                    if node["status"] == "cancelling":
                        node["status"] = "cancelled"
                        if not node.get("completedAt"):
                            node["completedAt"] = utc_now()
                logger.info("任务 %s 从 cancelling 更新为 cancelled", job_id)
            else:
                # pending 状态，更新为 cancelling（任务还没开始）
                job["status"] = "cancelling"
                job["updatedAt"] = utc_now()
                logger.info("任务 %s 已标记为取消（pending状态）", job_id)
        
        return json_response(True, data={"jobId": job_id}, message="任务停止请求已发送")
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("停止任务失败: %s", exc)
        return json_response(False, message=str(exc), status=500)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

