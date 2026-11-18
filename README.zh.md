# GHX BareModel — 裸金属 GPU 健康检查专家

[English](README.md) | **简体中文**

通过普通 SSH 对裸金属 NVIDIA 服务器做无代理（agentless）GPU 健康检查。无需 Agent、无需 Kubernetes、不依赖堡垒机——指向你的节点，即可得到带宽、NCCL、DCGM、InfiniBand 检测结论，并按 GPU 型号基准值自动判定通过/失败。

由两个解耦部分组成：

1. **前端（Next.js）**：交互式控制台——SSH 配置、批量节点管理、创建任务、实时进度、结果表格与日志导出。支持 English / 中文界面、亮色 / 暗色主题。
2. **后端（Flask + Paramiko）**：经 SSH 上传工具，在目标节点执行 `nvbandwidth` / `nccl-tests` / `dcgmi` / IB 与 PCIe 诊断，解析输出并与 GPU 基准值比对。

---

## 核心特性

- **无代理、纯 SSH**：类似 Ansible 的分发执行模型；目标节点只需 SSH 可达且装有 NVIDIA 驱动。
- **工具自包含**：自动上传 `nvbandwidth`、`nccl.tgz`、`nccl-tests.tgz`（可选 `nccl-rdma-sharp-plugins.tgz`）到目标节点 `/tmp/ghx`，就地编译 NCCL/NCCL-tests 以适配节点 CUDA/glibc。带宽项依次执行 `nvbandwidth -t 2/3/6/7`（H2D、D2H、GPU 间双向 D2D read/write CE）。
- **NVIDIA 预检查**：`nvidia-smi` 缺失的节点直接拒绝并给出明确提示。
- **基准值判定**：按 GPU 型号的参考值（`config/gpu-benchmarks.json`，支持热更新）把原始数字变成通过/失败结论。
- **批量任务**：多节点并发检查、实时进度、逐节点日志、取消支持、ZIP/日志导出。节点占用冲突保护（HTTP 409），避免两个任务同时对同一节点跑基准。
- **多机 NCCL（`mpirun`）**：可配置 `NCCL_SOCKET_IFNAME`、`NCCL_IB_HCA`、`UCX_TLS`、`NCCL_CROSS_NIC`、`UCX_NET_DEVICES`、QPS、每节点 GPU 数、调试级别与额外参数。**SHARP** 由前端勾选开启（注入 `SHARP_COLL_ENABLE`、`SHARP_LOG_LEVEL`、`NCCL_COLLNET_ENABLE`）；仅勾选时才分发/编译 SHARP 插件，若已有 NCCL 构建仅缺插件则复用、不重复编译。
- **生产级后端**：gunicorn（单 worker、线程数可配）保证进程内任务状态一致；并发上限经环境变量调节。
- **前后端解耦 REST API**：前端经 `/api/*` 调用后端，两侧可独立扩展或替换。

---

## 部署要求

### 控制端（运行 GHX 的机器）

| 组件 | 要求 |
| ---- | ---- |
| Python | ≥ 3.9 |
| Node.js | ≥ 18（Docker 构建推荐 22.x） |
| pnpm | 9.x |
| Docker / Docker Compose | 可选，容器化部署 |

### 目标节点

- NVIDIA GPU 且驱动已装（`nvidia-smi` 可用）。
- CUDA 工具链（`nvcc`）用于就地编译 NCCL；`libnccl2`/`libnccl-dev` 版本会与 `nvcc` 交叉校验。
- 可选：DCGM（`dcgmi`）诊断、InfiniBand（`ibstat`）检查、Open MPI 多机测试。
- `/tmp/ghx` 可写（或配置无密码 sudo；普通用户自动回退 `~/.ghx-bare`）。

仓库已内置资产：`nvbandwidth`、`nccl.tgz`、`nccl-tests.tgz`、`nccl-rdma-sharp-plugins.tgz`、`gpuIBPcieDiag.sh`。可用 `GHX_ASSET_DIR` 指定其他目录。

---

## 快速开始

### 方式一：Docker Compose（推荐）

```bash
# 建议启用 BuildKit：export DOCKER_BUILDKIT=1（PowerShell：$env:DOCKER_BUILDKIT=1）
docker compose up -d --build
```

- 前端控制台：http://localhost:3000
- 后端 API：http://localhost:5000

前端通过 Next.js rewrites 将 `/api/*` 转发到后端容器（`GHX_BACKEND_ORIGIN` 可覆盖目标地址）。

> **自定义基准值**：编辑 `config/gpu-benchmarks.json`（已挂载进容器），重启后端容器即可生效。

> **中国大陆网络**：取消 `docker-compose.yml` 中 `args` 注释块（npmmirror 镜像 + 公共 DNS），或传 `--build-arg NPM_REGISTRY=https://registry.npmmirror.com --build-arg BUILD_DNS1=223.5.5.5 --build-arg BUILD_DNS2=114.114.114.114`。后端 pip 镜像：`--build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple`。

### 方式二：源码运行

**后端（Flask）**

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 开发（threaded Flask 服务器）：
python baremetal_server.py

# 生产（gunicorn，必须单 worker——任务状态为进程内内存态）：
gunicorn -w 1 -k gthread --threads 32 -b 0.0.0.0:5000 \
  --timeout 120 --limit-request-field_size 65536 --limit-request-fields 0 \
  baremetal_server:app
```

**前端（Next.js）**

```bash
pnpm install
pnpm dev            # http://localhost:3000（开发模式自动指向 http://localhost:5000）

# 生产：
pnpm build && pnpm start
```

非 localhost 部署：保持同源 `/api` 转发（将 `GHX_BACKEND_ORIGIN` 指向你的后端地址），或构建期注入显式 API 地址：`NEXT_PUBLIC_GHX_API=https://your-backend.example.com`。

---

## 配置项说明

| 变量 | 侧 | 默认值 | 说明 |
| ---- | ---- | ---- | ---- |
| `PORT` | 后端 | `5000` | API 监听端口 |
| `GHX_THREADS` | 后端 | `32` | gunicorn HTTP 并发线程数 |
| `GHX_MAX_CONCURRENCY` | 后端 | `32`（compose 中 `100`） | 单任务内节点并发上限 |
| `GHX_ASSET_DIR` | 后端 | 项目根目录 | `nvbandwidth` / `*.tgz` / 诊断脚本所在目录 |
| `GHX_REMOTE_DIR` | 后端 | 自动 | 远端首选工作目录；`/tmp/ghx` 不可写时自动回退 `~/.ghx-bare` |
| `GPU_BENCHMARK_FILE` | 后端 | `config/gpu-benchmarks.json` | 基准值 JSON（挂载即可热更新） |
| `GHX_BACKEND_ORIGIN` | 前端 | `http://backend:5000` | `/api/*` 服务端转发目标 |
| `NEXT_PUBLIC_GHX_API` | 前端（构建期） | 空 | 显式 API 地址；空 = 同源 / localhost 自动探测 |
| `NPM_REGISTRY` / `BUILD_DNS1` / `BUILD_DNS2` | 前端构建参数 | npmjs / 未设置 | npm 镜像与可选的构建容器 DNS 覆盖 |
| `PIP_INDEX_URL` | 后端构建参数 | pypi.org | pip 镜像源 |

---

## 项目结构

```
.
├── app/                        # Next.js 应用入口（布局、全局样式）
├── components/                 # UI 组件（含裸金属控制台页面）
├── config/gpu-benchmarks.json  # 按型号基准值（支持热更新）
├── baremetal_server.py         # Flask 编排后端（SSH、任务、解析、REST API）
├── command_check.sh            # 远端命令/库可用性探测
├── gpuIBPcieDiag.sh            # IB / PCIe 健康诊断脚本
├── nvbandwidth                 # NVIDIA 带宽基准预编译二进制
├── nccl.tgz / nccl-tests.tgz   # NCCL 与测试源码（目标节点编译）
├── nccl-rdma-sharp-plugins.tgz # 可选 SHARP 插件源码
├── Dockerfile.ghx-backend      # 后端镜像（gunicorn）
├── Dockerfile.ghx-dashboard    # 前端镜像（Next.js standalone）
├── docker-compose.yml
└── requirements.txt
```

---

## 调试与日志

- **多机 `LD_LIBRARY_PATH: unbound variable`**：远端命令在 `set -u` 下执行；后端已改为先 `export LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH:-}"` 再 `mpirun -x LD_LIBRARY_PATH`。
- **SHARP `autogen.sh: Permission denied`**：解压包内脚本可能无执行位，构建使用 `bash autogen.sh` / `bash ./configure`。
- **`[Errno 13] Permission denied`**（nvbandwidth/IB）：多为远端 `/tmp/ghx` 对当前 SSH 用户不可写、或后端容器无法读取本地资产。后端会回退 `~/.ghx-bare`；也可设置 `GHX_REMOTE_DIR`。IB 脚本经 `bash` 执行，避免无执行位问题。
- **多机上传 `[Errno 2] No such file`**：第一个节点需能创建 `/tmp/ghx`（当前 SSH 用户可写，或配置无密码 sudo 供 `mkdir`/`chown`）。
- **多机与单机编译目录**：多机固定使用 `/tmp/ghx`；单机普通用户用 `~/.ghx-bare`，但若已存在 `/tmp/ghx/.../all_reduce_perf` 构建则直接复用。
- **`dcgmi diag` 报 `API version mismatch (-12)`**：节点 DCGM/libdcgm 与驱动不匹配，需安装与 NVIDIA 驱动对应的 DCGM 版本。
- **nvbandwidth 解析**：H2D/D2H 与 D2D 使用不同矩阵解析；前端结果表分两行分别对照 `h2d_d2h` 与 `d2d` 基准值。
- **构建期 `getaddrinfo EAI_AGAIN`**：容器 DNS 问题——使用 `BUILD_DNS*` 参数、配置 Docker daemon DNS，或尝试 `docker build --network=host`。
- **反代后 431**：转发的大 cookie 超出默认请求头上限；提供的 gunicorn/Node 配置已将上限抬到 64KB。

---

## 安全提示

- 界面中输入的 SSH 凭据会保存在**浏览器 localStorage** 以便复用。请仅在可信机器/网络使用，优先密钥认证，用完清理节点列表。
- GHX **不带内置鉴权**。请勿直接暴露到公网，置于你自己的访问控制之后（VPN、反代鉴权等）。
- 任务以你配置的 SSH 用户权限在目标节点执行命令。

## 许可证

[MIT](LICENSE)

欢迎提交 Issue/PR——例如任务结果持久化、鉴权集成、更多 GPU 厂商支持等。