# GHX BareModel — GPU Health eXpert for Bare-Metal Servers

**English** | [简体中文](README.zh.md)

Agentless GPU health checking for bare-metal NVIDIA servers over plain SSH. No agents, no Kubernetes, no bastion dependency — point it at your nodes and get bandwidth, NCCL, DCGM and InfiniBand verdicts against per-model benchmarks.

Two loosely coupled parts:

1. **Dashboard (Next.js)** — interactive console: SSH configuration, batch node management, job creation, live progress, result tables and log export. English / 中文 UI, light / dark theme.
2. **Orchestrator (Flask + Paramiko)** — uploads tooling over SSH, runs `nvbandwidth` / `nccl-tests` / `dcgmi` / IB & PCIe diagnostics on the targets, parses output and compares it with GPU benchmark references.

---

## Features

- **Agentless, SSH-only**: works like an Ansible-style push execution model; targets only need SSH access and an NVIDIA driver.
- **Self-contained tooling**: uploads `nvbandwidth`, `nccl.tgz`, `nccl-tests.tgz` (and optionally `nccl-rdma-sharp-plugins.tgz`) to `/tmp/ghx` on each target, compiles NCCL/NCCL-tests there to match the node's CUDA/glibc. Bandwidth tests run `nvbandwidth -t 2/3/6/7` (H2D, D2H, bidirectional D2D read/write CE).
- **NVIDIA pre-flight check**: nodes without `nvidia-smi` are rejected up front with a clear message.
- **Benchmark verdicts**: per-GPU-model reference values (`config/gpu-benchmarks.json`, hot-reloadable) turn raw numbers into pass/fail.
- **Batch jobs**: multi-node concurrent checks with live progress, per-node logs, cancel support and ZIP/log export. Node-occupancy conflict guard (HTTP 409) prevents two jobs from benchmarking the same node simultaneously.
- **Multi-node NCCL (`mpirun`)**: configurable `NCCL_SOCKET_IFNAME`, `NCCL_IB_HCA`, `UCX_TLS`, `NCCL_CROSS_NIC`, `UCX_NET_DEVICES`, QPS, GPUs-per-node, debug level and extra flags. **SHARP** is opt-in from the UI (injects `SHARP_COLL_ENABLE`, `SHARP_LOG_LEVEL`, `NCCL_COLLNET_ENABLE`); the SHARP plugin is only distributed/compiled when enabled, and an existing NCCL build is reused when only the plugin is missing.
- **Production-ready backend**: gunicorn (single worker, configurable thread pool) keeps in-memory job state consistent; concurrency tunable via environment variables.
- **Decoupled REST API**: the frontend talks to the backend over `/api/*`; extend or replace either side independently.

---

## Requirements

### Control host (where GHX runs)

| Component | Requirement |
| ---- | ---- |
| Python | ≥ 3.9 |
| Node.js | ≥ 18 (22.x recommended for Docker builds) |
| pnpm | 9.x |
| Docker / Docker Compose | optional, for containerized deployment |

### Target nodes

- NVIDIA GPU with driver installed (`nvidia-smi` must work).
- CUDA toolkit (`nvcc`) for on-node NCCL compilation; `libnccl2`/`libnccl-dev` versions are cross-checked against `nvcc`.
- Optional: DCGM (`dcgmi`) for diagnostics, InfiniBand (`ibstat`) for IB checks, Open MPI for multi-node tests.
- Writable `/tmp/ghx` (or passwordless sudo; falls back to `~/.ghx-bare` for non-root users).

Bundled assets (already in this repo): `nvbandwidth`, `nccl.tgz`, `nccl-tests.tgz`, `nccl-rdma-sharp-plugins.tgz`, `gpuIBPcieDiag.sh`. Override their location with `GHX_ASSET_DIR`.

---

## Quick Start

### Option 1: Docker Compose (recommended)

```bash
# BuildKit recommended: export DOCKER_BUILDKIT=1  (PowerShell: $env:DOCKER_BUILDKIT=1)
docker compose up -d --build
```

- Dashboard: http://localhost:3000
- Backend API: http://localhost:5000

The frontend proxies `/api/*` to the backend container via Next.js rewrites (`GHX_BACKEND_ORIGIN` overrides the target).

> **Custom benchmarks**: edit `config/gpu-benchmarks.json` (mounted into the container) and restart the backend container.

> **China mainland networks**: uncomment the `args` block in `docker-compose.yml` (npmmirror registry + public DNS), or pass `--build-arg NPM_REGISTRY=https://registry.npmmirror.com --build-arg BUILD_DNS1=223.5.5.5 --build-arg BUILD_DNS2=114.114.114.114`. Backend pip mirror: `--build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple`.

### Option 2: Run from source

**Backend (Flask)**

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Development (threaded Flask server):
python baremetal_server.py

# Production (gunicorn, single worker — job state is in-process memory):
gunicorn -w 1 -k gthread --threads 32 -b 0.0.0.0:5000 \
  --timeout 120 --limit-request-field_size 65536 --limit-request-fields 0 \
  baremetal_server:app
```

**Frontend (Next.js)**

```bash
pnpm install
pnpm dev            # http://localhost:3000 (auto-targets http://localhost:5000 in dev)

# Production:
pnpm build && pnpm start
```

For non-localhost deployments either keep the same-origin `/api` rewrite (set `GHX_BACKEND_ORIGIN` to your backend URL) or bake an explicit API base at build time: `NEXT_PUBLIC_GHX_API=https://your-backend.example.com`.

---

## Configuration

| Variable | Side | Default | Description |
| ---- | ---- | ---- | ---- |
| `PORT` | backend | `5000` | API listen port |
| `GHX_THREADS` | backend | `32` | gunicorn HTTP worker threads |
| `GHX_MAX_CONCURRENCY` | backend | `32` (compose: `100`) | Max nodes checked concurrently inside one job |
| `GHX_ASSET_DIR` | backend | repo root | Directory holding `nvbandwidth` / `*.tgz` / diag script |
| `GHX_REMOTE_DIR` | backend | auto | Preferred remote work dir; auto-falls back to `~/.ghx-bare` when `/tmp/ghx` is not writable |
| `GPU_BENCHMARK_FILE` | backend | `config/gpu-benchmarks.json` | Benchmark reference JSON (mount to hot-update) |
| `GHX_BACKEND_ORIGIN` | frontend | `http://backend:5000` | Server-side rewrite target for `/api/*` |
| `NEXT_PUBLIC_GHX_API` | frontend (build-time) | empty | Explicit API base URL; empty = same origin / localhost auto-detect |
| `NPM_REGISTRY` / `BUILD_DNS1` / `BUILD_DNS2` | frontend build args | npmjs / unset | Registry mirror and optional build-container DNS override |
| `PIP_INDEX_URL` | backend build arg | pypi.org | pip index mirror |

---

## Project Structure

```
.
├── app/                        # Next.js app entry (layout, global styles)
├── components/                 # UI components incl. the bare-metal console page
├── config/gpu-benchmarks.json  # Per-model benchmark references (hot-reloadable)
├── baremetal_server.py         # Flask orchestrator (SSH, jobs, parsing, REST API)
├── command_check.sh            # Remote command/library availability probe
├── gpuIBPcieDiag.sh            # IB / PCIe health diagnostic script
├── nvbandwidth                 # Prebuilt NVIDIA bandwidth benchmark
├── nccl.tgz / nccl-tests.tgz   # NCCL & tests sources (compiled on targets)
├── nccl-rdma-sharp-plugins.tgz # Optional SHARP plugin sources
├── Dockerfile.ghx-backend      # Backend image (gunicorn)
├── Dockerfile.ghx-dashboard    # Frontend image (Next.js standalone)
├── docker-compose.yml
└── requirements.txt
```

---

## Troubleshooting

- **`LD_LIBRARY_PATH: unbound variable`** in multi-node runs: remote commands execute under `set -u`; the backend exports `LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH:-}"` before `mpirun -x LD_LIBRARY_PATH`.
- **SHARP `Permission denied` on `autogen.sh`**: archive scripts may lack the exec bit; builds invoke `bash autogen.sh` / `bash ./configure`.
- **`[Errno 13] Permission denied`** (nvbandwidth/IB): remote `/tmp/ghx` not writable by the SSH user, or the backend container cannot read local assets. The backend falls back to `~/.ghx-bare`; set `GHX_REMOTE_DIR` to force a location. The IB script runs via `bash` to avoid missing exec bits.
- **`[Errno 2] No such file`** on multi-node upload: the first node must be able to create `/tmp/ghx` (writable by the SSH user, or passwordless sudo for `mkdir`/`chown`).
- **Multi-node vs single-node builds**: multi-node always uses `/tmp/ghx`; single-node jobs as non-root use `~/.ghx-bare` but reuse an existing `/tmp/ghx/.../all_reduce_perf` build when present.
- **`dcgmi diag` → `API version mismatch (-12)`**: node DCGM/libdcgm does not match the driver; install the DCGM release matching your NVIDIA driver.
- **nvbandwidth parsing**: H2D/D2H and D2D use different matrix parsers; the results table shows both rows against `h2d_d2h` and `d2d` benchmarks respectively.
- **Build-time `getaddrinfo EAI_AGAIN`**: container DNS issue — use the `BUILD_DNS*` args, configure Docker daemon DNS, or try `docker build --network=host`.
- **431 responses behind a reverse proxy**: large forwarded cookies exceed default header limits; the provided gunicorn/Node settings raise the limit to 64 KB.

---

## Security Notes

- SSH credentials entered in the UI are persisted in your **browser localStorage** for convenience. Use GHX only on trusted machines/networks, prefer key-based auth, and clear the node list when done.
- GHX has **no built-in authentication**. Do not expose it directly to the public internet; put it behind your own access control (VPN, reverse-proxy auth, etc.).
- Jobs execute commands on target nodes with the privileges of the SSH user you configure.

## License

[MIT](LICENSE)

Issues and PRs are welcome — e.g. persistent job storage, authentication integrations, additional GPU vendors.