#!/bin/bash

# =============================================================================
# GPU/NCCL/MPI 关键命令与库检测脚本
# 检测目标节点上的关键命令、库、内核模块、服务及系统优化配置
# 作者: Grissom
# 版本：1.0
# =============================================================================

# 版本信息
VERSION="1.0"
SCRIPT_NAME="Command Check"

# 默认选项
QUIET_MODE=false
SUMMARY_ONLY=false
USE_COLOR=true
JSON_OUTPUT=false

# 颜色定义
if [[ "$USE_COLOR" == true ]]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    NC='\033[0m' # No Color
else
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    NC=''
fi

# 显示帮助信息
show_help() {
    cat << EOF
GPU/NCCL/MPI 关键命令与库检测脚本 v${VERSION}

用法: $0 [选项]

选项:
  -h, --help      显示此帮助信息
  -v, --version   显示版本信息
  -q, --quiet     静默模式 (仅输出错误和警告)
  -s, --summary   仅显示摘要信息
  -j, --json      以 JSON 格式输出结果
  --no-color      禁用彩色输出

功能:
  • 检测 NVIDIA GPU 驱动管理工具 (nvidia-smi)
  • 检测 NVIDIA 数据中心 GPU 管理器 (dcgmi)
  • 检测 CUDA 编译器 (/usr/local/cuda/bin/nvcc)
  • 检测 InfiniBand 状态查询工具 (ibstat)
  • 检测 MPI 运行命令 (mpirun)
  • 检测 NCCL 运行时库和开发库 (libnccl2, libnccl-dev)
  • 检测 nvidia_peermem 内核模块是否加载
  • 检测 nouveau 开源驱动是否已卸载
  • 检测 ACS 控制是否已关闭 (P2P 优化)
  • 检测 IOMMU 是否已关闭 (GPU 性能优化)
  • 检测 NVIDIA Fabric Manager 服务是否已激活
  • 检测 ulimit max locked memory / max memory size 是否为 unlimited
  • 比对 nvcc、libnccl2、libnccl-dev 的 CUDA 版本是否一致

使用场景示例:
  sudo $0                    # 新环境部署后的全面检查
  sudo $0 -q                 # 监控脚本中的异常检测
  sudo $0 -j                 # 输出 JSON 格式供其他程序解析

注意:
  • 此脚本部分检查需要 root 权限运行
  • 脚本仅进行检查和提供建议，不会修改系统配置
  • 静默模式下无输出表示系统状态良好

EOF
}

# 显示版本信息
show_version() {
    echo "$SCRIPT_NAME v$VERSION"
}

# 解析命令行参数
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -v|--version)
                show_version
                exit 0
                ;;
            -q|--quiet)
                QUIET_MODE=true
                shift
                ;;
            -s|--summary)
                SUMMARY_ONLY=true
                shift
                ;;
            -j|--json)
                JSON_OUTPUT=true
                USE_COLOR=false
                shift
                ;;
            --no-color)
                USE_COLOR=false
                shift
                ;;
            *)
                echo "未知选项: $1" >&2
                show_help
                exit 1
                ;;
        esac
    done

    # 重新初始化颜色
    if [[ "$USE_COLOR" == true ]]; then
        RED='\033[0;31m'
        GREEN='\033[0;32m'
        YELLOW='\033[1;33m'
        BLUE='\033[0;34m'
        NC='\033[0m'
    else
        RED=''
        GREEN=''
        YELLOW=''
        BLUE=''
        NC=''
    fi
}

# 输出带颜色的结果
print_status() {
    local name="$1"
    local status="$2"
    local desc="$3"
    local detail="${4:-}"

    if [[ "$JSON_OUTPUT" == true ]]; then
        return
    fi

    if [[ "$QUIET_MODE" == true && "$status" == "PASS" ]]; then
        return
    fi

    local status_str
    case "$status" in
        PASS)
            status_str="${GREEN}[PASS]${NC}"
            ;;
        FAIL)
            status_str="${RED}[FAIL]${NC}"
            ;;
        WARN)
            status_str="${YELLOW}[WARN]${NC}"
            ;;
        *)
            status_str="${YELLOW}[INFO]${NC}"
            ;;
    esac

    printf "%-45s %s  %s\n" "$name" "$status_str" "$desc"
    if [[ -n "$detail" && "$QUIET_MODE" == false ]]; then
        printf "    %s\n" "$detail"
    fi
}

# 检查普通命令是否存在
check_command() {
    local cmd="$1"
    local desc="$2"

    if command -v "$cmd" >/dev/null 2>&1; then
        print_status "$cmd" "PASS" "$desc"
        return 0
    else
        print_status "$cmd" "FAIL" "$desc" "命令未找到，请安装相应软件包"
        return 1
    fi
}

# 检查指定路径的文件是否存在且可执行
check_path_command() {
    local cmd="$1"
    local desc="$2"

    if [[ -x "$cmd" ]]; then
        print_status "$cmd" "PASS" "$desc"
        return 0
    else
        print_status "$cmd" "FAIL" "$desc" "文件不存在或不可执行，请安装 CUDA Toolkit"
        return 1
    fi
}

# 检查 apt 包是否已安装
check_apt_package() {
    local pkg="$1"
    local desc="$2"

    if ! command -v apt >/dev/null 2>&1; then
        print_status "$pkg" "WARN" "$desc" "非 apt 系统，无法检测 Debian/Ubuntu 包"
        return 2
    fi

    local output
    output=$(apt list --installed 2>/dev/null | grep -E "^${pkg}/" || true)
    if [[ -n "$output" && "$output" == *"installed"* ]]; then
        print_status "$pkg" "PASS" "$desc"
        return 0
    else
        print_status "$pkg" "FAIL" "$desc" "包未安装，请执行 apt install $pkg"
        return 1
    fi
}

# 检查内核模块是否加载
check_module_loaded() {
    local module="$1"
    local desc="$2"

    if lsmod 2>/dev/null | grep -q "^${module}\b"; then
        print_status "$module" "PASS" "$desc"
        return 0
    else
        print_status "$module" "FAIL" "$desc" "内核模块未加载，请执行 modprobe $module"
        return 1
    fi
}

# 检查内核模块是否未加载
check_module_unloaded() {
    local module="$1"
    local desc="$2"

    if ! lsmod 2>/dev/null | grep -q "^${module}\b"; then
        print_status "${module}_unloaded" "PASS" "$desc"
        return 0
    else
        print_status "${module}_unloaded" "FAIL" "$desc" "内核模块仍加载，请执行 rmmod $module 并在黑名单中禁用"
        return 1
    fi
}

# 检查 ACS 控制是否已关闭
check_acsctl_disabled() {
    local desc="$1"
    local output

    # 优先尝试 sudo，失败后回退到普通权限
    output=$(sudo lspci -vvv 2>/dev/null | grep -i acsctl || lspci -vvv 2>/dev/null | grep -i acsctl || true)
    if [[ -z "$output" ]]; then
        print_status "acsctl_disabled" "PASS" "$desc" "未检测到 ACSCtl 信息（设备可能不支持 ACS）"
        return 0
    fi

    if echo "$output" | grep -q '+'; then
        print_status "acsctl_disabled" "FAIL" "$desc" "检测到 ACS 控制未完全关闭，存在 '+' 标志"
        return 1
    else
        print_status "acsctl_disabled" "PASS" "$desc"
        return 0
    fi
}

# 检查 IOMMU 是否已关闭
check_iommu_disabled() {
    local desc="$1"

    if ls /sys/class/iommu 2>/dev/null | grep -qE '^dmar'; then
        print_status "iommu_disabled" "FAIL" "$desc" "检测到 IOMMU 已启用（/sys/class/iommu 下存在 dmar*）"
        return 1
    else
        print_status "iommu_disabled" "PASS" "$desc"
        return 0
    fi
}

# 检查服务是否处于 active 状态
check_service_active() {
    local service="$1"
    local desc="$2"

    if ! command -v systemctl >/dev/null 2>&1; then
        print_status "$service" "WARN" "$desc" "非 systemd 系统，无法检测服务状态"
        return 2
    fi

    local state
    state=$(systemctl is-active "$service" 2>/dev/null || echo inactive)
    if [[ "$state" == "active" ]]; then
        print_status "$service" "PASS" "$desc"
        return 0
    else
        print_status "$service" "FAIL" "$desc" "服务状态为 $state，请执行 systemctl start $service"
        return 1
    fi
}

# 检查 ulimit 配置是否为 unlimited
check_ulimit_unlimited() {
    local key="$1"
    local desc="$2"
    local output
    local value

    output=$(ulimit -a 2>/dev/null || true)
    value=$(echo "$output" | grep -i "$key" | awk '{print $NF}' | tr '[:upper:]' '[:lower:]' || true)

    if [[ "$value" == "unlimited" ]]; then
        print_status "ulimit_${key// /_}" "PASS" "$desc"
        return 0
    else
        print_status "ulimit_${key// /_}" "FAIL" "$desc" "当前值为 ${value:-unknown}，请在 limits.conf 中设置为 unlimited"
        return 1
    fi
}

# 从 nvcc --version 输出中提取 CUDA 版本
extract_cuda_version() {
    local output="$1"
    local version

    version=$(echo "$output" | grep -oE 'release [0-9]+\.[0-9]+' | head -1 | awk '{print $2}')
    if [[ -z "$version" ]]; then
        version=$(echo "$output" | grep -oE 'V[0-9]+\.[0-9]+' | head -1 | sed 's/V//')
    fi
    echo "$version"
}

# 从 apt list 输出中提取 NCCL 包对应的 CUDA 版本
extract_nccl_cuda_version() {
    local output="$1"
    local pkg="$2"
    local line

    line=$(echo "$output" | grep -E "^${pkg}/" | grep '\[installed\]' | head -1)
    # 格式示例: libnccl2/unknown,now 2.26.2-1+cuda12.8 amd64 [installed]
    echo "$line" | grep -oE 'cuda[0-9]+\.[0-9]+' | head -1 | sed 's/cuda//'
}

# 检查并比对版本号
check_versions() {
    local nvcc_version=""
    local libnccl2_version=""
    local libnccl_dev_version=""
    local nvcc_output
    local apt_output

    if [[ -x "/usr/local/cuda/bin/nvcc" ]]; then
        nvcc_output=$(/usr/local/cuda/bin/nvcc --version 2>/dev/null || true)
        nvcc_version=$(extract_cuda_version "$nvcc_output")
    fi

    if command -v apt >/dev/null 2>&1; then
        apt_output=$(apt list --installed 2>/dev/null | grep -E '^libnccl' || true)
        libnccl2_version=$(extract_nccl_cuda_version "$apt_output" "libnccl2")
        libnccl_dev_version=$(extract_nccl_cuda_version "$apt_output" "libnccl-dev")
    fi

    if [[ "$JSON_OUTPUT" == true ]]; then
        return
    fi

    if [[ "$SUMMARY_ONLY" == true ]]; then
        return
    fi

    if [[ "$QUIET_MODE" == true && -z "$nvcc_version" && -z "$libnccl2_version" && -z "$libnccl_dev_version" ]]; then
        return
    fi

    echo
    echo -e "${BLUE}NCCL / CUDA 版本比对:${NC}"
    printf "  %-20s %s\n" "nvcc CUDA 版本:" "${nvcc_version:-未检测到}"
    printf "  %-20s %s\n" "libnccl2 CUDA 版本:" "${libnccl2_version:-未检测到}"
    printf "  %-20s %s\n" "libnccl-dev CUDA 版本:" "${libnccl_dev_version:-未检测到}"

    if [[ -n "$nvcc_version" && -n "$libnccl2_version" && -n "$libnccl_dev_version" ]]; then
        if [[ "$nvcc_version" == "$libnccl2_version" && "$nvcc_version" == "$libnccl_dev_version" ]]; then
            echo -e "  ${GREEN}版本一致 ✓${NC}"
        else
            echo -e "  ${YELLOW}版本不一致，建议安装与 nvcc CUDA 版本匹配的 NCCL 包${NC}"
        fi
    fi
}

# 构建 JSON 输出
output_json() {
    local -n results_ref=$1
    local first=true

    echo "{"
    echo '  "commands": {'
    for key in "${!results_ref[@]}"; do
        if [[ "$first" == true ]]; then
            first=false
        else
            echo ","
        fi
        if [[ "${results_ref[$key]}" == "true" ]]; then
            printf '    "%s": true' "$key"
        else
            printf '    "%s": false' "$key"
        fi
    done
    echo
    echo '  }'
    echo "}"
}

# 主检查逻辑
main() {
    parse_arguments "$@"

    if [[ "$JSON_OUTPUT" == false ]]; then
        echo -e "${BLUE}GPU/NCCL/MPI 关键命令与库检测脚本 v${VERSION}${NC}"
        echo "============================================================="
    fi

    declare -A results
    local pass_count=0
    local fail_count=0
    local warn_count=0

    # 1. nvidia-smi
    if check_command "nvidia-smi" "NVIDIA GPU 驱动管理工具"; then
        results["nvidia-smi"]=true; ((pass_count++))
    else
        results["nvidia-smi"]=false; ((fail_count++))
    fi

    # 2. dcgmi
    if check_command "dcgmi" "NVIDIA 数据中心 GPU 管理器"; then
        results["dcgmi"]=true; ((pass_count++))
    else
        results["dcgmi"]=false; ((fail_count++))
    fi

    # 3. /usr/local/cuda/bin/nvcc
    if check_path_command "/usr/local/cuda/bin/nvcc" "CUDA 编译器"; then
        results["/usr/local/cuda/bin/nvcc"]=true; ((pass_count++))
    else
        results["/usr/local/cuda/bin/nvcc"]=false; ((fail_count++))
    fi

    # 4. ibstat
    if check_command "ibstat" "InfiniBand 状态查询工具"; then
        results["ibstat"]=true; ((pass_count++))
    else
        results["ibstat"]=false; ((fail_count++))
    fi

    # 5. mpirun
    if check_command "mpirun" "MPI 运行命令（多机测试必需）"; then
        results["mpirun"]=true; ((pass_count++))
    else
        results["mpirun"]=false; ((fail_count++))
    fi

    # 6. libnccl2
    local ret
    check_apt_package "libnccl2" "NCCL 运行时库"
    ret=$?
    if [[ $ret -eq 0 ]]; then
        results["libnccl2"]=true; ((pass_count++))
    elif [[ $ret -eq 2 ]]; then
        results["libnccl2"]=false; ((warn_count++))
    else
        results["libnccl2"]=false; ((fail_count++))
    fi

    # 7. libnccl-dev
    check_apt_package "libnccl-dev" "NCCL 开发库"
    ret=$?
    if [[ $ret -eq 0 ]]; then
        results["libnccl-dev"]=true; ((pass_count++))
    elif [[ $ret -eq 2 ]]; then
        results["libnccl-dev"]=false; ((warn_count++))
    else
        results["libnccl-dev"]=false; ((fail_count++))
    fi

    # 8. nvidia_peermem
    if check_module_loaded "nvidia_peermem" "nvidia_peermem 内核模块已加载"; then
        results["nvidia_peermem"]=true; ((pass_count++))
    else
        results["nvidia_peermem"]=false; ((fail_count++))
    fi

    # 9. nouveau 已卸载
    if check_module_unloaded "nouveau" "nouveau 开源驱动已卸载"; then
        results["nouveau_unloaded"]=true; ((pass_count++))
    else
        results["nouveau_unloaded"]=false; ((fail_count++))
    fi

    # 10. ACS 控制已关闭
    if check_acsctl_disabled "ACS 控制已关闭（P2P 优化）"; then
        results["acsctl_disabled"]=true; ((pass_count++))
    else
        results["acsctl_disabled"]=false; ((fail_count++))
    fi

    # 11. IOMMU 已关闭
    if check_iommu_disabled "IOMMU 已关闭（GPU 性能优化）"; then
        results["iommu_disabled"]=true; ((pass_count++))
    else
        results["iommu_disabled"]=false; ((fail_count++))
    fi

    # 12. nvidia-fabricmanager 服务已激活
    if check_service_active "nvidia-fabricmanager.service" "NVIDIA Fabric Manager 服务已激活"; then
        results["nvidia_fabricmanager_active"]=true; ((pass_count++))
    else
        results["nvidia_fabricmanager_active"]=false; ((fail_count++))
    fi

    # 13. ulimit max locked memory
    if check_ulimit_unlimited "max locked memory" "ulimit max locked memory 为 unlimited"; then
        results["ulimit_max_locked_memory"]=true; ((pass_count++))
    else
        results["ulimit_max_locked_memory"]=false; ((fail_count++))
    fi

    # 14. ulimit max memory size
    if check_ulimit_unlimited "max memory size" "ulimit max memory size 为 unlimited"; then
        results["ulimit_max_memory_size"]=true; ((pass_count++))
    else
        results["ulimit_max_memory_size"]=false; ((fail_count++))
    fi

    # 版本比对
    check_versions

    # 输出结果
    if [[ "$JSON_OUTPUT" == true ]]; then
        output_json results
        exit 0
    fi

    if [[ "$QUIET_MODE" == false ]]; then
        echo
        echo -e "${BLUE}检测结果摘要:${NC}"
        echo -e "  ${GREEN}通过: $pass_count${NC}"
        echo -e "  ${RED}失败: $fail_count${NC}"
        if [[ $warn_count -gt 0 ]]; then
            echo -e "  ${YELLOW}警告: $warn_count${NC}"
        fi

        if [[ $fail_count -eq 0 ]]; then
            echo -e "\n${GREEN}所有关键命令与库检测通过。${NC}"
        else
            echo -e "\n${YELLOW}存在检测失败的项，请根据上方提示进行修复。${NC}"
        fi
    fi

    # 静默模式下，有失败时返回非 0 退出码
    if [[ $fail_count -gt 0 ]]; then
        exit 1
    fi

    exit 0
}

main "$@"
