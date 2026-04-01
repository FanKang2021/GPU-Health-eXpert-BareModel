# GHX Bare-Metal GPU Health eXpert

这个仓库提供了一个通过 **SSH** 在裸金属服务器上执行 GPU 健康检查的完整方案。整体由两部分组成：

1. **前端（Next.js）**：交互式控制台，可配置 SSH、批量管理节点、创建任务、实时查看结果及日志。
2. **后端（Flask + Paramiko）**：负责通过 SSH 上传工具、执行 `nvbandwidth` / `nccl-tests` / `dcgmi` / `ibstat`，并对结果进行解析比对。

---

## 目录

1. [核心特性](#核心特性)
2. [部署要求](#部署要求)
3. [快速开始](#快速开始)
   - [方式一：Docker 部署](#方式一docker-部署)
   - [方式二：二进制/源码运行](#方式二二进制源码运行)
4. [配置项说明](#配置项说明)
5. [项目结构](#项目结构)
6. [调试与日志](#调试与日志)

---

## 核心特性

- 通过 SSH 直接接入裸金属主机，类似 Ansible 的分发执行方式。
- 自动上传 `nvbandwidth`、`nccl.tgz`、`nccl-tests.tgz` 等依赖，在目标节点 `/tmp/ghx` 目录下解压并编译后使用；带宽项依次执行 `nvbandwidth -t 2/3/6/7`（H2D、D2H、GPU 间双向 D2D read/write CE）。
- 支持多节点批量任务、实时进度查看、日志下载。
- 提供 GPU 基准值判定，自动给出通过/失败结论。
- 多机 NCCL（`mpirun`）：默认 TCP/NCCL 套接字网卡为 `bond0`，支持 `NCCL_CROSS_NIC`、`UCX_TLS`、`NCCL_DEBUG` 等；**SHARP** 由前端勾选统一开启（注入 `SHARP_COLL_ENABLE`、`SHARP_LOG_LEVEL`、`NCCL_COLLNET_ENABLE`）。**仅勾选 SHARP 时**才会上传并编译 `nccl-rdma-sharp-plugins`（需仓库提供 `nccl-rdma-sharp-plugins.tgz`），未勾选可显著缩短多机首次编译时间。若先前未勾选 SHARP 已在各节点编好 `nccl`/`nccl-tests`，后续再勾选时检测到**仅缺插件**：主节点与从节点只分发、编译 SHARP 插件，**不会**删除重建已有 `nccl`/`nccl-tests`。多机 `mpirun` 默认可执行文件路径为 `/tmp/ghx/nccl-tests/build/all_reduce_perf`（与本工具上传编译目录一致）；若各节点已预装至 `/opt`，请在界面中改为 `/opt/nccl-tests/build/all_reduce_perf`。
- 纯前后端解耦，前端通过 REST API 调用后端，可按需扩展。

---

## 部署要求

### 基础依赖

| 组件 | 要求 |
| ---- | ---- |
| Node.js | ≥ 18（推荐 20+） |
| pnpm | ≥ 8 |
| Python | ≥ 3.9 |
| Docker / Docker Compose | 可选，用于容器化部署 |

### 后端运行时要求

后端需要能够访问以下文件（仓库已提供）：

- `nvbandwidth`
- `nccl.tgz`（NCCL 源码压缩包，将在目标节点编译）
- `nccl-tests.tgz`（NCCL Tests 源码压缩包，将在目标节点编译）

所有文件将上传到目标节点的 `/tmp/ghx` 目录，`nccl` 和 `nccl-tests` 会在目标节点解压并编译；多机场景下从主节点向其余节点为**串行**分发编译，以避免单 SSH 连接上并发会话过多触发 `ChannelException`。

若放在其他目录，可通过环境变量 `GHX_ASSET_DIR` 指定并挂载到容器或运行目录。

---

## 快速开始

### 方式一：Docker 部署

此方式适合快速打包交付。下面示例使用多容器方案（可根据实际拆/并）：

```bash
# 建议启用 BuildKit（消除 legacy builder 提示，缓存更合理）
# Windows PowerShell: $env:DOCKER_BUILDKIT=1
# Linux/macOS: export DOCKER_BUILDKIT=1

# 1. 构建前端镜像（可按需覆盖 DNS / registry）
docker build -f Dockerfile.ghx-dashboard -t ghx-frontend \
  --build-arg NPM_REGISTRY=https://registry.npmmirror.com \
  --build-arg BUILD_DNS1=223.5.5.5 \
  --build-arg BUILD_DNS2=114.114.114.114 \
  .

# 2. 构建后端镜像
docker build -f Dockerfile.ghx-backend -t ghx-baremetal-backend .
```

> **构建期 `getaddrinfo EAI_AGAIN registry.npmmirror.com`**：多为容器 DNS。本仓库在安装依赖的同一 `RUN` 内写入国内公共 DNS，并重试/降并发。仍失败可：① `NPM_REGISTRY` 改为 `https://mirrors.cloud.tencent.com/npm/`；② 在 Docker Desktop 的 daemon.json 配置 `"dns": ["223.5.5.5","114.114.114.114"]`；③ Linux 可试 `docker build --network=host`。根布局使用 **geist** 本地字体，不访问 `fonts.googleapis.com`。

#### 使用 docker-compose

项目已包含 `docker-compose.yml` 配置文件，可直接使用：

启动：

```bash
docker compose up -d
```

访问前端：http://localhost:3000  
后端 API：http://localhost:5000

> **自定义基准值**：编辑仓库内 `config/gpu-benchmarks.json` 或挂载自定义文件到容器的 `/app/config/gpu-benchmarks.json`，然后重启容器即可生效。

> 若使用单一容器，也可以在镜像内同时安装 Node + Python，但建议前后端分离部署。

---

### 方式二：二进制/源码运行

#### 后端（Flask）

```bash
# 1. 创建虚拟环境
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 运行
python baremetal_server.py
```

默认监听 `0.0.0.0:5000`。可通过以下环境变量调整：

| 变量 | 默认值 | 说明 |
| ---- | ---- | ---- |
| `PORT` | 5000 | API 端口 |
| `GHX_ASSET_DIR` | 项目根目录 | 指定工具/压缩包所在目录 |
| `GHX_REMOTE_DIR` | （空，自动探测可写目录） | 单节点 Job 上传 `nvbandwidth`/IB 脚本的远程路径；默认识别 `/tmp/ghx` 不可写时改用 `~/.ghx-bare` |
| `GPU_BENCHMARK_FILE` | `config/gpu-benchmarks.json` | GPU基准值配置文件，支持通过 Docker Volume 热更新（重启容器即可生效） |

#### 前端（Next.js）

```bash
# 1. 安装依赖
pnpm install

# 2. 设置后端地址（可在 .env.local 或 shell 中）
export NEXT_PUBLIC_GHX_API=http://localhost:5000

# 3. 开发模式
pnpm dev

# 或构建生产包
pnpm build
pnpm start
```

访问 http://localhost:3000 进入控制台。

---

## 配置项说明

| 配置 | 位置 | 描述 |
| ---- | ---- | ---- |
| `NEXT_PUBLIC_GHX_API` | 前端环境变量 | 指向后端 REST API 地址 |
| `PORT` | 后端环境变量 | Flask 服务监听端口 |
| `GHX_ASSET_DIR` | 后端环境变量 | 指向 `nvbandwidth` 等资产所在目录 |
| `GHX_REMOTE_DIR` | 后端环境变量 | 可选，指定 SSH 目标上首选工作目录（默认可写性检测失败时自动用 `~/.ghx-bare`） |
| `GPU_BENCHMARK_FILE` | 后端环境变量 | GPU 基准值 JSON 文件路径，可通过挂载文件热更新 |

---

## 项目结构

```
ghx-bare/
├── app/                        # Next.js 应用入口
├── components/                 # 主要 UI 组件（包含裸金属控制台页面）
├── baremetal_server.py         # Flask 后端入口
├── requirements.txt            # Python 依赖
├── Dockerfile.ghx-dashboard    # 前端 Dockerfile
├── nvbandwidth / nccl*.tgz  # 执行所需资产
└── README.md                   # 本文件
```

---

## 调试与日志

- 多机 NCCL：远端命令在 `set -u` 下执行；若曾出现 `LD_LIBRARY_PATH: unbound variable`，后端已改为先 `export LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH:-}"` 再调用 `mpirun -x LD_LIBRARY_PATH`。
- SHARP 插件：解压包内脚本可能无执行位，构建使用 `bash autogen.sh` / `bash ./configure`，避免 `./autogen.sh: Permission denied`。
- nvbandwidth / IB 出现 `[Errno 13] Permission denied`：多为远端 `/tmp/ghx` 对当前 SSH 用户不可写、或后端容器无法读本地 `nvbandwidth` 资产。后端会尝试 `~/.ghx-bare`；也可设置 `GHX_REMOTE_DIR`。IB 脚本已改为 `bash ib_health_check.sh`，避免解压后无执行位。
- 多机 MPI NCCL 固定使用主节点 **`/tmp/ghx`**（常以 root/sudo 编译，目录属主多为 root）；单机 Job 若以普通用户连接、不能写 `/tmp/ghx`，会上传到 **`~/.ghx-bare`**。若节点上已有 **`/tmp/ghx/.../all_reduce_perf`**（例如刚跑过多机任务），单节点 NCCL 会**复用该构建**，不再在 home 下重复编译。
- 多机任务**第一个节点**会经 SFTP 上传 `nccl.tgz`：需能创建 `/tmp/ghx` 且**当前 SSH 用户可写**，或配置**无密码 sudo**（后端会 `sudo mkdir`/`sudo chown`）。若仅 `mkdir` 未成功仍上传，会报 **`[Errno 2] No such file`**。
- `dcgmi diag` 报 `API version mismatch (-12)`：节点上的 `dcgmi`/libdcgm 与驱动 DCGM 版本不一致，需安装与当前 NVIDIA 驱动匹配的 Data Center GPU Manager（或使用发行版/驱动自带的 `dcgmi`）。
- nvbandwidth：H2D/D2H 与 GPU D2D 使用不同矩阵解析；前端结果表分两行展示并分别对照 `config/gpu-benchmarks.json` 中的 `h2d_d2h` 与 `d2d`。
- 后端默认输出到标准输出，可自行接入 `gunicorn` 或 supervisor。
- 前端建议使用 `pnpm dev` 进行实时调试。
- 若 `next lint` 出现 “Converting circular structure to JSON”，请升级 `next`/`eslint-config-next` 或根据具体报错调整 `.eslintrc`。

---

如需扩展（如存储任务结果到数据库、集成鉴权、支持跳板机等），欢迎提交 Issue/PR。祝使用愉快！

