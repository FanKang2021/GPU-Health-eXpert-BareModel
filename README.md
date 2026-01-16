# GHX Bare-Metal GPU Health eXpert

这个仓库提供了一个通过 **SSH** 在裸金属服务器上执行 GPU 健康检查的完整方案。整体由两部分组成：

1. **前端（Next.js）**：交互式控制台，可配置 SSH、批量管理节点、创建任务、实时查看结果及日志。
2. **后端（Flask + Paramiko）**：负责通过 SSH 上传工具、执行 `nvbandwidth` / `p2pBandwidthLatencyTest` / `nccl-tests` / `dcgmi` / `ibstat`，并对结果进行解析比对。

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
- 自动上传 `nvbandwidth`、`p2pBandwidthLatencyTest`、`nccl-tests.tgz`（预编译版本）等依赖，nccl-tests 将解压到 `/opt/nccl-tests` 直接使用。
- 支持多节点批量任务、实时进度查看、日志下载。
- 提供 GPU 基准值判定，自动给出通过/失败结论。
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
- `p2pBandwidthLatencyTest`
- `nccl-tests.tgz`（预编译版本，将自动解压到 `/opt/nccl-tests`）

若放在其他目录，可通过环境变量 `GHX_ASSET_DIR` 指定并挂载到容器或运行目录。

---

## 快速开始

### 方式一：Docker 部署

此方式适合快速打包交付。下面示例使用多容器方案（可根据实际拆/并）：

```bash
# 1. 构建前端镜像
docker build -f Dockerfile.ghx-dashboard -t ghx-frontend .

# 2. 构建后端镜像
docker build -f Dockerfile.ghx-backend -t ghx-baremetal-backend .
```

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
├── nvbandwidth / p2pBandwidthLatencyTest / nccl*.tgz  # 执行所需资产
└── README.md                   # 本文件
```

---

## 调试与日志

- 后端默认输出到标准输出，可自行接入 `gunicorn` 或 supervisor。
- 前端建议使用 `pnpm dev` 进行实时调试。
- 若 `next lint` 出现 “Converting circular structure to JSON”，请升级 `next`/`eslint-config-next` 或根据具体报错调整 `.eslintrc`。

---

如需扩展（如存储任务结果到数据库、集成鉴权、支持跳板机等），欢迎提交 Issue/PR。祝使用愉快！

