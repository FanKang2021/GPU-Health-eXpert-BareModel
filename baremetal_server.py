#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GHX Bare-Metal Orchestrator
通过SSH在裸金属节点上运行GPU健康检查
1. 支持SSH连接测试与基础命令检查
2. 通过后台Job执行nvbandwidth/p2p/nccl/dcgm/ib检查
3. 将nccl.tgz、nccl-tests.tgz、nvbandwidth、p2pBandwidthLatencyTest上传到目标主机的/tmp执行
"""
from __future__ import annotations

import io
import json
import logging
import os
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
ASSETS = {
    "nvbandwidth": BASE_DIR / "nvbandwidth",
    "p2p": BASE_DIR / "p2pBandwidthLatencyTest",
    "nccl": BASE_DIR / "nccl.tgz",
    "nccl_tests": BASE_DIR / "nccl-tests.tgz",
    "ib_check": BASE_DIR / "assets" / "ib_health_check.sh",
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
)
logger = logging.getLogger("ghx-baremetal")

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
        self.sftp = None

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
        self.sftp = self.client.open_sftp()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.sftp:
            self.sftp.close()
        self.client.close()

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
        self.remote_dir = f"/tmp/ghx/{node_meta['nodeId']}"
        self.logs: List[str] = []
        self.session: Optional[SSHSession] = None
        self.cancelled = cancelled_flag or threading.Event()

    def log(self, message: str):
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"{timestamp} - {message}"
        self.logs.append(entry)
        logger.info("[%s] %s", self.node_meta["alias"], message)

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
            
            self._upload_asset("nccl", "nccl.tgz", executable=False)
            self._upload_asset("nccl_tests", "nccl-tests.tgz", executable=False)
            script = f"""
cd {self.remote_dir}
rm -rf nccl nccl-tests
mkdir -p nccl nccl-tests
tar -xzf nccl.tgz -C nccl --strip-components=1
tar -xzf nccl-tests.tgz -C nccl-tests --strip-components=1
cd nccl
make -j$(nproc)
cd ../nccl-tests
make -j$(nproc) NCCL_HOME={self.remote_dir}/nccl/build
if [ ! -x build/all_reduce_perf ]; then chmod +x build/all_reduce_perf; fi
./build/all_reduce_perf -b 1024 -e 1G -f 2 -g {gpu_count}
"""
            result = self.session.run(script, timeout=3600, require_root=True)
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
        logger.exception("节点 %s 执行失败: %s", node["alias"], exc)
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
                logger.exception("节点 %s 执行异常: %s", node.get("alias"), exc)
                if node["status"] == "running":
                    node["status"] = "failed"
                    node["executionLog"] = f"执行异常: {exc}"

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
        gpu_lines = [line.strip() for line in gpu_res.stdout.splitlines() if line.strip()]
        gpu_count = len(gpu_lines)
        gpu_model = normalize_gpu_name(gpu_lines[0]) if gpu_lines else "Unknown"
        data = {
            "hostname": hostname_res.stdout.strip(),
            "gpus": gpu_lines,  # 保留完整列表用于兼容
            "gpuModel": gpu_model,  # GPU型号
            "gpuCount": gpu_count,  # GPU数量
            "driverVersion": driver_res.stdout.strip(),
        }
        logger.info("SSH连接测试成功: %s", data.get("hostname"))
        return json_response(True, data=data, message="SSH连接成功")
    except Exception as exc:  # pylint: disable=broad-except
        error_msg = str(exc)
        logger.error("SSH连接测试失败: %s", error_msg, exc_info=True)
        return json_response(False, message=error_msg, status=400)


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
        with SSHSession(connection) as session:
            for cmd in commands:
                if "/" in cmd:
                    check_cmd = f"[ -x {cmd} ] && echo OK || echo MISSING"
                else:
                    check_cmd = f"command -v {cmd} >/dev/null 2>&1 && echo OK || echo MISSING"
                res = session.run(check_cmd)
                results[cmd] = res.stdout.strip() == "OK"
        return json_response(True, data={"commands": results}, message="命令检查完成")
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
                "alias": node_payload.get("alias") or node_payload["host"],
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

