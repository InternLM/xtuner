      

all_npu_id=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)

all_start_cpus=(13 27 53 67 93 107 133 147 173 187 213 227 253 267 293 307)   

# 将10进制CPU号转换为16进制mask码
cpu_to_mask() {
    local cpu=$1
    local group=$((cpu / 32))
    local bit=$((cpu % 32))
    local value=$((1 << bit))
    local mask=$(printf "%08x" $value)

    # 拼接成逗号分隔的掩码字符串
    for ((i=1; i<=group; i++)); do
        mask="${mask},00000000"
    done
    echo $mask
}

check_and_stop_irqbalance() {
    # 检查 irqbalance 是否存在
    if systemctl list-unit-files | grep -q irqbalance.service; then
        if systemctl is-active --quiet irqbalance; then
            echo "检测到 irqbalance 服务正在运行，正在关闭..."
            systemctl stop irqbalance
        else
            echo "irqbalance 服务已安装，但当前未运行"
        fi
    elif service --status-all 2>/dev/null | grep -q irqbalance; then
        if service irqbalance status >/dev/null 2>&1; then
            echo "检测到 irqbalance 服务正在运行，正在关闭..."
            service irqbalance stop
            echo "irqbalance 已关闭"
        else
            echo "irqbalance 服务已安装，但当前未运行"
        fi
    else
        echo "当前环境未检测到 irqbalance 服务"
    fi
}

bind_irq() {
    echo "==== 开始对所有 NPU 卡的中断绑核 ===="
    check_and_stop_irqbalance
    SQ_IRQ_LIST=($(cat /proc/interrupts | grep sq_send_trigger_irq | cut -d: -f1))
    for i in "${!all_npu_id[@]}"; do
        SQ_CPU=${all_start_cpus[$i]}
        CQ_CPU=$((SQ_CPU+1))

        if [[ "${#all_npu_id[@]}" -eq 8 ]]; then
            CARD=${all_npu_id[$i]}
            CHIP_ID=0
        else
            CARD=$((all_npu_id[$i] / 2))
            CHIP_ID=$((all_npu_id[$i] % 2))
        fi

        # 获取 PCI 地址
        PCI_ADDR=$(npu-smi info -t board -i $CARD -c $CHIP_ID| grep "PCIe Bus Info" | awk '{print $NF}' | tr '[:upper:]' '[:lower:]')
        if [ -z "$PCI_ADDR" ]; then
            echo "未找到 NPU 卡 $CARD 的 PCI 地址"
            continue
        fi

        # 获取该卡的 MSI 中断号列表
        NPU_IRQ_LIST=$(ls /sys/bus/pci/devices/$PCI_ADDR/msi_irqs/ | sort -n)
        for irq in "${SQ_IRQ_LIST[@]}"; do
            if echo "${NPU_IRQ_LIST[@]}" | grep -qw "$irq"; then
                SQ_IRQ=$irq
                CQ_IRQ=$((SQ_IRQ+1))
            fi
        done
        if [ -z "$SQ_IRQ" ]; then
            echo "未找到 NPU 卡 $CARD 的 SQ 中断"
            continue
        fi

        echo "NPU卡 ${all_npu_id[$i]} (PCI $PCI_ADDR): SQ IRQ=$SQ_IRQ → CPU$SQ_CPU, CQ IRQ=$CQ_IRQ → CPU$CQ_CPU"

        # 绑 SQ
        echo $(cpu_to_mask $SQ_CPU) > /proc/irq/$SQ_IRQ/smp_affinity
        # 绑 CQ
        echo $(cpu_to_mask $CQ_CPU) > /proc/irq/$CQ_IRQ/smp_affinity
    done
}

bind_irq