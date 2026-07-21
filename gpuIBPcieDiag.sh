#!/bin/bash
# ═══════════════════════════════════════════════════════════
# 智算节点健康检测脚本（v3）
# 检测范围：GPU 掉卡、GPU PCIe 链路、IB 端口状态、IB PCIe 链路
# 适用场景：智算集群日常巡检、故障排查
# ═══════════════════════════════════════════════════════════

set -uo pipefail

# ── 颜色定义 ──
RED=$'\033[0;31m'
GREEN=$'\033[0;32m'
YELLOW=$'\033[0;33m'
BLUE=$'\033[0;34m'
CYAN=$'\033[0;36m'
BOLD=$'\033[1m'
NC=$'\033[0m'

ISSUES=0

warn()   { echo -e "  ${RED}❌ $*${NC}"; ((ISSUES++)) || true; }
ok()     { echo -e "  ${GREEN}✅ $*${NC}"; }
notice() { echo -e "  ${YELLOW}⚠️  $*${NC}"; }
info()   { echo -e "  ${BLUE}ℹ️ $*${NC}"; }
tip()    { echo -e "  ${CYAN}💡 $*${NC}"; }
section(){ echo -e "\n${BOLD}${CYAN}▶ $*${NC}"; echo "  ----------------------------------------"; }

# ═══════════════════════════════════════════════════════════
# 通用函数
# ═══════════════════════════════════════════════════════════

gt_to_gen() {
    case "$1" in
        2)  echo "Gen1" ;; 5)  echo "Gen2" ;; 8)  echo "Gen3" ;;
        16) echo "Gen4" ;; 32) echo "Gen5" ;; 64) echo "Gen6" ;;
        *)  echo "Gen?" ;;
    esac
}

gen_to_speed() {
    case "$1" in
        1) echo 2 ;; 2) echo 5 ;; 3) echo 8 ;;
        4) echo 16 ;; 5) echo 32 ;; 6) echo 64 ;;
        *) echo 0 ;;
    esac
}

# 带宽估算: $1=GT/s数字  $2=lane宽度
estimate_bw() {
    local speed="$1" width="$2"
    case "$speed" in
        2)  echo $((width / 4)) ;;
        5)  echo $((width / 2)) ;;
        8)  echo $((width * 1)) ;;
        16) echo $((width * 2)) ;;
        32) echo $((width * 4)) ;;
        64) echo $((width * 8)) ;;
        *)  echo 0 ;;
    esac
}

# 从 lspci 输出中提取 PCIe Speed 和 Width（兼容 -oP 和 -oE）
extract_pcie_speed() {
    local line="$1"
    local val
    val=$(echo "$line" | grep -oP 'Speed \K[0-9]+(?=GT/s)' 2>/dev/null) || \
    val=$(echo "$line" | grep -oE 'Speed [0-9]+GT/s' | grep -oE '[0-9]+' | head -n1) || \
    val=""
    echo "$val"
}

extract_pcie_width() {
    local line="$1"
    local val
    val=$(echo "$line" | grep -oP 'Width x\K[0-9]+' 2>/dev/null) || \
    val=$(echo "$line" | grep -oE 'Width x[0-9]+' | grep -oE '[0-9]+' | head -n1) || \
    val=""
    echo "$val"
}

# ═══════════════════════════════════════════════════════════
# 阶段 1：GPU 掉卡检测
# ═══════════════════════════════════════════════════════════
check_gpu_presence() {
    section "GPU 掉卡检测（硬件枚举 vs 驱动可见）"

    # ── 硬件层：lspci 枚举 ──
    local hw_count=0
    if command -v lspci >/dev/null 2>&1; then
        hw_count=$(lspci 2>/dev/null | grep -iE '3D controller.*NVIDIA|VGA.*NVIDIA|Display.*NVIDIA|NVIDIA.*(3D|VGA|Display)' | wc -l || echo 0)
        hw_count=$(echo "$hw_count" | tr -d '[:space:]')
    fi

    # ── 驱动层：nvidia-smi ──
    local drv_count=0
    local drv_ok=false
    local smi_output=""
    if command -v nvidia-smi >/dev/null 2>&1; then
        smi_output=$(nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader 2>/dev/null) || true
        if [ -n "$smi_output" ]; then
            drv_count=$(echo "$smi_output" | wc -l | tr -d '[:space:]')
            drv_ok=true
        fi
    fi

    echo "  硬件可见 (lspci):      $hw_count 张"
    echo "  驱动可见 (nvidia-smi):  $drv_count 张"

    # ── 判定逻辑（优先以 nvidia-smi 为准）──
    if [ "$drv_ok" = true ] && [ "$drv_count" -gt 0 ]; then
        if [ "$hw_count" -gt 0 ] && [ "$drv_count" -lt "$hw_count" ]; then
            local lost=$((hw_count - drv_count))
            warn "疑似 $lost 张 GPU 掉卡（硬件 $hw_count - 驱动 $drv_count = $lost）"
            tip "排查: lspci | grep -i nvidia 对比 nvidia-smi -L"
            tip "内核日志: dmesg | grep -iE 'fallen off|Xid|nvidia' | tail -20"
        elif [ "$hw_count" -eq 0 ]; then
            notice "lspci 未匹配到 NVIDIA 设备（可能设备命名不同），但 nvidia-smi 可见 $drv_count 张 GPU"
            ok "以 nvidia-smi 为准：$drv_count 张 GPU 在线"
        else
            ok "所有 GPU 正常在线（$drv_count 张）"
        fi
    elif [ "$hw_count" -gt 0 ] && [ "$drv_count" -eq 0 ]; then
        warn "所有 GPU 掉卡！硬件可见 $hw_count 张，但 nvidia-smi 无法识别"
        tip "检查驱动: lsmod | grep nvidia"
        tip "内核日志: dmesg | grep -iE 'xid|nvidia|nvrm' | tail -20"
    elif [ "$hw_count" -eq 0 ] && [ "$drv_count" -eq 0 ]; then
        info "未检测到 NVIDIA GPU（硬件和驱动均不可见）"
    fi

    # ── Xid 错误历史 ──
    if command -v dmesg >/dev/null 2>&1; then
        local xid_lines
        xid_lines=$(dmesg 2>/dev/null | grep 'NVRM: Xid' || true)
        local xid_count=0
        if [ -n "$xid_lines" ]; then
            xid_count=$(echo "$xid_lines" | wc -l | tr -d '[:space:]')
        fi
        if [ "$xid_count" -gt 0 ]; then
            notice "dmesg 中发现 $xid_count 条 Xid 错误记录（GPU 异常历史）"
            echo "  最近 5 条："
            echo "$xid_lines" | tail -5 | sed 's/^/    /'
        fi
    fi
}

# ═══════════════════════════════════════════════════════════
# 阶段 2：GPU PCIe 链路检测
# 逐字段独立查询，避免 GPU 名称中逗号/空格导致 CSV 字段错位
# ═══════════════════════════════════════════════════════════
check_gpu_pcie() {
    section "GPU PCIe 链路状态"

    if ! command -v nvidia-smi >/dev/null 2>&1; then
        notice "nvidia-smi 不可用，跳过 GPU PCIe 检测"
        return
    fi

    # 修复：用 awk 取第一列，保留换行符让 for 循环正常迭代
    local gpu_indices
    gpu_indices=$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null | awk '{print $1}') || true
    if [ -z "$gpu_indices" ]; then
        info "无可用的 GPU 进行 PCIe 检测"
        return
    fi

    echo "  GPU | 型号                       | 设计链路    | 当前链路    | 带宽估算  | 状态"
    echo "  ----|----------------------------|-------------|-------------|-----------|----------"

    for gpu_id in $gpu_indices; do
        # 防御性清理：确保 gpu_id 是纯数字
        gpu_id=$(echo "$gpu_id" | tr -dc '0-9')
        [ -z "$gpu_id" ] && continue

        # 每个字段单独查询，彻底避免 CSV 拆分校位
        local name pci g_max w_max g_cur w_cur
        name=$(nvidia-smi --id="$gpu_id" --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1) || true
        pci=$(nvidia-smi --id="$gpu_id" --query-gpu=pci.bus_id --format=csv,noheader 2>/dev/null | head -n1) || true
        g_max=$(nvidia-smi --id="$gpu_id" --query-gpu=pcie.link.gen.max --format=csv,noheader,nounits 2>/dev/null | awk '{print $1}') || true
        w_max=$(nvidia-smi --id="$gpu_id" --query-gpu=pcie.link.width.max --format=csv,noheader,nounits 2>/dev/null | awk '{print $1}') || true
        g_cur=$(nvidia-smi --id="$gpu_id" --query-gpu=pcie.link.gen.current --format=csv,noheader,nounits 2>/dev/null | awk '{print $1}') || true
        w_cur=$(nvidia-smi --id="$gpu_id" --query-gpu=pcie.link.width.current --format=csv,noheader,nounits 2>/dev/null | awk '{print $1}') || true

        # 清理前后空白
        name=$(echo "$name" | xargs)
        pci=$(echo "$pci" | xargs)

        # 检查是否获取到有效数据
        if [ -z "$g_max" ] || [ -z "$g_cur" ]; then
            notice "GPU $gpu_id: 无法获取 PCIe 信息（可能设备异常）"
            continue
        fi

        # 缩短型号：去掉 "NVIDIA " 前缀，截取前 26 字符
        local short_name="${name#NVIDIA }"
        short_name="${short_name:0:26}"

        # 带宽估算
        local cap_bw cur_bw
        cap_bw=$(estimate_bw "$(gen_to_speed "${g_max:-0}")" "${w_max:-0}")
        cur_bw=$(estimate_bw "$(gen_to_speed "${g_cur:-0}")" "${w_cur:-0}")

        # 状态判定
        local status=""
        if [ "${g_max:-0}" != "${g_cur:-0}" ] && [ "${w_max:-0}" != "${w_cur:-0}" ]; then
            status="${RED}速度+宽度降级${NC}"
            ((ISSUES++)) || true
        elif [ "${g_max:-0}" != "${g_cur:-0}" ]; then
            status="${RED}速度降级${NC}"
            ((ISSUES++)) || true
        elif [ "${w_max:-0}" != "${w_cur:-0}" ]; then
            status="${RED}宽度降级${NC}"
            ((ISSUES++)) || true
        else
            status="${GREEN}正常${NC}"
        fi

        printf "  %-4s | %-26s | Gen%s x%-4s | Gen%s x%-4s | ~%3s GB/s | %b\n" \
            "$gpu_id" "$short_name" "${g_max:-?}" "${w_max:-?}" "${g_cur:-?}" "${w_cur:-?}" "$cur_bw" "$status"
    done
}

# ═══════════════════════════════════════════════════════════
# 阶段 3：IB 端口状态检测
# ═══════════════════════════════════════════════════════════
check_ib_port_status() {
    section "IB 端口链路状态（全部端口）"

    if ! command -v ibstat >/dev/null 2>&1; then
        notice "ibstat 不可用，跳过 IB 端口检测"
        tip "安装: apt install infiniband-diags 或 yum install infiniband-diags"
        return
    fi

    local found_any=false
    local CAs
    CAs=$(ibstat 2>/dev/null | grep "^CA '" | awk -F"'" '{print $2}') || true

    if [ -z "$CAs" ]; then
        info "未发现任何 InfiniBand HCA 设备"
        return
    fi

    echo "  CA 设备        | 端口   | 类型   | 物理链路 | 状态        | 速率   | 判定"
    echo "  ---------------|--------|--------|----------|-------------|--------|----------"

    for ca in $CAs; do
        found_any=true
        local ca_type port_count
        ca_type=$(ibstat "$ca" 2>/dev/null | grep "CA type:" | awk '{print $3}') || true
        port_count=$(ibstat "$ca" 2>/dev/null | grep "Number of ports:" | awk '{print $NF}') || true

        [ -z "$port_count" ] && port_count=1

        for port_num in $(seq 1 "$port_count"); do
            local port_out state phys_state rate
            port_out=$(ibstat "$ca" "$port_num" 2>/dev/null) || continue

            state=$(echo "$port_out" | grep -E "State:" | grep -v "Physical" | head -n1 | awk -F: '{gsub(/^[ \t]+|[ \t]+$/, "", $NF); print $NF}')
            phys_state=$(echo "$port_out" | grep "Physical state:" | head -n1 | awk -F: '{gsub(/^[ \t]+|[ \t]+$/, "", $NF); print $NF}')
            rate=$(echo "$port_out" | grep "Rate:" | head -n1 | awk '{print $2}')

            state=$(echo "$state" | tr -d '(),')
            phys_state=$(echo "$phys_state" | tr -d '(),')

            local verdict=""
            case "${state,,}" in
                active)
                    verdict="${GREEN}正常${NC}"
                    ;;
                initializing)
                    verdict="${YELLOW}初始化中${NC}"
                    ((ISSUES++)) || true
                    ;;
                down)
                    verdict="${RED}端口 Down${NC}"
                    ((ISSUES++)) || true
                    ;;
                *)
                    verdict="${YELLOW}未知: ${state:-N/A}${NC}"
                    ((ISSUES++)) || true
                    ;;
            esac

            printf "  %-14s | port %-2s | %-6s | %-8s | %-11s | %-5s  | %b\n" \
                "$ca" "$port_num" "$ca_type" "$phys_state" "$state" "${rate:-0}G" "$verdict"
        done
    done

    [ "$found_any" = false ] && info "未发现任何 InfiniBand HCA 设备"
}

# ═══════════════════════════════════════════════════════════
# 阶段 4：IB PCIe 链路检测（仅 ≥400G Active 端口）
# ═══════════════════════════════════════════════════════════
check_ib_pcie() {
    section "InfiniBand (≥400G Active) PCIe 链路"

    if ! command -v ibstat >/dev/null 2>&1; then
        notice "ibstat 不可用，跳过 IB PCIe 检测"
        return
    fi
    if ! command -v lspci >/dev/null 2>&1; then
        notice "lspci 不可用，跳过 IB PCIe 检测"
        return
    fi

    local found=false
    local checked_pci_addrs=""

    local CAs
    CAs=$(ibstat 2>/dev/null | grep "^CA '" | awk -F"'" '{print $2}') || true

    for ca in $CAs; do
        # 设备级信息：CA type（从设备级查询，非端口级）
        local ca_type
        ca_type=$(ibstat "$ca" 2>/dev/null | grep "CA type:" | awk '{print $3}') || true

        # 端口级信息：速率和状态（从端口 1 查询）
        local port_out rate port_state
        port_out=$(ibstat "$ca" 1 2>/dev/null) || continue
        rate=$(echo "$port_out" | grep "Rate:" | head -n1 | awk '{print $2}') || true
        port_state=$(echo "$port_out" | grep -E "State:" | grep -v "Physical" | head -n1 | awk -F: '{gsub(/^[ \t]+|[ \t]+$/, "", $NF); print $NF}')
        port_state=$(echo "$port_state" | tr -d '(),')

        # 仅检测 ≥400G
        if [ -z "$rate" ] || [ "$rate" -lt 400 ] 2>/dev/null; then
            continue
        fi

        # 仅检测 Active 端口
        if [ "${port_state,,}" != "active" ]; then
            notice "$ca ($ca_type, ${rate}G) — 端口未 Active (当前: $port_state)，跳过 PCIe 分析"
            continue
        fi

        # 获取 PCI 地址
        local sys_path="/sys/class/infiniband/$ca/device"
        if [ ! -e "$sys_path" ]; then
            notice "$ca — 无法定位 PCI 设备路径"
            continue
        fi

        local pci_addr
        pci_addr=$(basename "$(readlink -f "$sys_path")")

        # 去重：同一物理 HCA 的多端口共享 PCIe 链路
        if echo "$checked_pci_addrs" | grep -qw "$pci_addr"; then
            continue
        fi
        checked_pci_addrs="$checked_pci_addrs $pci_addr"

        found=true
        echo ""
        echo -e "  ${BOLD}IB: $ca ($ca_type, ${rate}G)${NC}"
        echo "  PCI 地址: $pci_addr"

        local lspci_out lnkcap lnksta
        lspci_out=$(lspci -vvv -s "$pci_addr" 2>/dev/null) || { notice "lspci 查询失败: $pci_addr"; continue; }
        lnkcap=$(echo "$lspci_out" | grep "LnkCap:" | head -n1)
        lnksta=$(echo "$lspci_out" | grep "LnkSta:" | head -n1)

        local cap_speed sta_speed cap_width sta_width
        cap_speed=$(extract_pcie_speed "$lnkcap")
        sta_speed=$(extract_pcie_speed "$lnksta")
        cap_width=$(extract_pcie_width "$lnkcap")
        sta_width=$(extract_pcie_width "$lnksta")

        echo "  PCIe Cap: Speed ${cap_speed:-?}GT/s ($(gt_to_gen "${cap_speed:-0}")) Width x${cap_width:-?}"
        echo "  PCIe Sta: Speed ${sta_speed:-?}GT/s ($(gt_to_gen "${sta_speed:-0}")) Width x${sta_width:-?}"

        # PCIe 速度对比
        if [ -n "${cap_speed:-}" ] && [ -n "${sta_speed:-}" ]; then
            if [ "$cap_speed" != "$sta_speed" ]; then
                warn "PCIe 速度降级: $(gt_to_gen "$sta_speed") ($sta_speed GT/s) < $(gt_to_gen "$cap_speed") ($cap_speed GT/s)"
            else
                ok "PCIe 速度正常: $(gt_to_gen "$sta_speed") ($sta_speed GT/s)"
            fi
        fi

        # PCIe 宽度对比
        if [ -n "${cap_width:-}" ] && [ -n "${sta_width:-}" ]; then
            if [ "$cap_width" != "$sta_width" ]; then
                warn "PCIe 宽度降级: x$sta_width < x$cap_width"
            else
                ok "PCIe 宽度正常: x$sta_width"
            fi
        fi

        # 带宽估算
        if [ -n "${sta_speed:-}" ] && [ -n "${sta_width:-}" ]; then
            local bw required_bw
            bw=$(estimate_bw "$sta_speed" "$sta_width")
            required_bw=$((rate / 8))

            if [ "$bw" -lt "$required_bw" ]; then
                warn "PCIe 带宽不足: ~${bw} GB/s < ${required_bw} GB/s (${rate}G 所需)"
                tip "当前: $(gt_to_gen "${sta_speed:-0}") x${sta_width}，建议: $(gt_to_gen "${cap_speed:-0}") x${cap_width}"
            else
                ok "PCIe 带宽充足: ~${bw} GB/s ≥ ${required_bw} GB/s"
            fi
        fi
    done

    [ "$found" = false ] && info "未发现速率 ≥400G 且 Active 的 InfiniBand 设备"
}

# ═══════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════
echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║       🖥️ 智算节点健康检测（GPU + IB + PCIe）            ║${NC}"
echo -e "${BOLD}║       $(date '+%Y-%m-%d %H:%M:%S')                                ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════════════════════╝${NC}"

echo ""
echo "  主机名: $(hostname 2>/dev/null || echo 'N/A')"
echo "  内核:   $(uname -r 2>/dev/null || echo 'N/A')"
echo "  CPU:    $(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | awk -F: '{print $2}' | xargs || echo 'N/A')"

# 阶段 1：GPU 掉卡检测
check_gpu_presence

# 阶段 2：GPU PCIe 链路
check_gpu_pcie

# 阶段 3：IB 端口状态（全量检测）
check_ib_port_status

# 阶段 4：IB PCIe 链路（仅 ≥400G Active 端口）
check_ib_pcie

# ── 汇总 ──
echo ""
echo "=================================================="
if [ "$ISSUES" -gt 0 ]; then
    echo -e "  ${RED}${BOLD}⚠️  检测完成，发现 $ISSUES 个问题，请逐一排查。${NC}"
else
    echo -e "  ${GREEN}${BOLD}✅ 检测完成，未发现异常。${NC}"
fi
echo "=================================================="
echo ""

exit "$ISSUES"
