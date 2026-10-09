import logging
import os
import re
import shutil
import subprocess
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import psutil


CPU_MASK_BIT = 32
MAIN_PROCESS_RANGE = 5
ACL_THREAD_RANGE = 1
RELEASE_THREAD_RANGE = 1
ALLOWED_CPUS_PATH = "/proc/self/status"

DEFAULT_CPU_BIND_CONFIG = {
    "custom_bind": [
        {
            "process_name": "xtuner.v1.train.cli.sft",
            "cpu_list": [
                "15-22",
                "29-36",
                "55-62",
                "69-76",
                "95-102",
                "109-116",
                "135-142",
                "149-156",
                "175-182",
                "189-196",
                "215-222",
                "229-236",
                "255-262",
                "269-276",
                "295-302",
                "309-316",
            ],
            "bind_sub_process": True,
        },
        {
            "process_name": "acl_thread",
            "cpu_list": [
                "23",
                "37",
                "63",
                "77",
                "103",
                "117",
                "143",
                "157",
                "183",
                "197",
                "223",
                "237",
                "263",
                "277",
                "303",
                "317",
            ],
            "is_thread": True,
            "mem_bind": True,
        },
        {
            "process_name": "release_thread",
            "cpu_list": [
                "24",
                "38",
                "64",
                "78",
                "104",
                "118",
                "144",
                "158",
                "184",
                "198",
                "224",
                "238",
                "264",
                "278",
                "304",
                "318",
            ],
            "is_thread": True,
        },
        {
            "process_name": "hccp_connect",
            "cpu_list": [
                "25",
                "39",
                "65",
                "79",
                "105",
                "119",
                "145",
                "159",
                "185",
                "199",
                "225",
                "239",
                "265",
                "279",
                "305",
                "319",
            ],
            "is_thread": True,
        },
    ]
}

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s]:%(message)s")


def execute_command(cmd: List[str]) -> Tuple[str, int]:
    with subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as p:
        out, _ = p.communicate(timeout=1000)
    return out.decode(), p.returncode


def cpu_to_mask(cpus: List[int]) -> str:
    groups: Dict[int, int] = defaultdict(int)
    for cpu in cpus:
        group = cpu // CPU_MASK_BIT
        bit = cpu % CPU_MASK_BIT
        groups[group] |= 1 << bit

    max_group = max(groups.keys())
    mask_parts = []
    for group in reversed(range(max_group + 1)):
        mask_parts.append(f"{groups.get(group, 0):08x}")
    return ",".join(mask_parts)


def expand_cpu_list(cpu_str: str) -> List[int]:
    cpus: List[int] = []
    for part in cpu_str.split(","):
        if "-" in part:
            start, end = map(int, part.split("-"))
            cpus.extend(range(start, end + 1))
        else:
            cpus.append(int(part))
    return cpus


def get_npu_map_info() -> Dict[str, Dict[str, str]]:
    npu_map_info: Dict[str, Dict[str, str]] = {}
    npu_info, _ = execute_command(["npu-smi", "info", "-m"])
    npu_map = npu_info.strip().split("\n")[1:]
    for line in npu_map:
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        npu_id, chip_id, chip_logic_id = parts[:3]
        if chip_logic_id.isdigit():
            npu_map_info.setdefault(npu_id, {})[chip_id] = chip_logic_id
    logging.debug(f"build npu_map_info: {npu_map_info}")
    return npu_map_info


class DeviceInfo:
    def __init__(self):
        self.main_pid_list: List[List[int]] = []
        self.npu_map_info: Dict[str, Dict[str, str]] = get_npu_map_info()
        self.allowed_cpus: List[int] = self.parse_allowed_cpus()
        self.running_npu_list: List[int] = self.get_running_npus()
        self.npu_affinity: Dict[int, List[int]] = self.parse_topo_affinity()

    @staticmethod
    def parse_allowed_cpus() -> List[int]:
        if not os.path.exists(ALLOWED_CPUS_PATH):
            return []
        with open(ALLOWED_CPUS_PATH) as f:
            for line in f:
                if line.startswith("Cpus_allowed_list"):
                    allowed_cpu_list = expand_cpu_list(line.split()[1])
                    logging.debug(f"CPUs_allowed_list: {allowed_cpu_list}")
                    return allowed_cpu_list
        return []

    @staticmethod
    def parse_topo_affinity() -> Dict[int, List[int]]:
        chip_logic_id = 0
        affinity: Dict[int, List[int]] = {}
        affinity_message, _ = execute_command(["npu-smi", "info", "-t", "topo"])
        for line in affinity_message.splitlines():
            if line.startswith("NPU"):
                parts = line.split()
                last_part = parts[-1]
                if last_part != "Affinity":
                    affinity[chip_logic_id] = expand_cpu_list(last_part)
                chip_logic_id += 1
        logging.debug(f"build affinity map: {affinity}")
        return affinity

    def get_running_npus(self) -> List[int]:
        npu_message, _ = execute_command(["npu-smi", "info"])
        in_proc_section = False
        running_npu_set: set[int] = set()
        chip_pid_map: Dict[int, List[Tuple[int, int]]] = {}
        for line in npu_message.splitlines():
            line = line.strip()
            if line.startswith("| NPU") and "Process id" in line:
                in_proc_section = True
                continue
            if not in_proc_section or not line.startswith("| "):
                continue
            parts = [p.strip() for p in line.strip("|").split("|")]
            if len(parts) < 4 or not parts[1].isdigit():
                continue
            pid = int(parts[1])
            try:
                mem = int(parts[3])
            except ValueError:
                mem = 0
            npu_id, chip_id = parts[0].split()[:2]
            chip_logic_id_str = self.npu_map_info.get(npu_id, {}).get(chip_id)
            if chip_logic_id_str and chip_logic_id_str.isdigit():
                chip_logic_id = int(chip_logic_id_str)
                chip_pid_map.setdefault(chip_logic_id, []).append((pid, mem))
                running_npu_set.add(chip_logic_id)

        self.main_pid_list = []
        for npu in sorted(running_npu_set):
            pid_mem_list: List[Tuple[int, int]] = chip_pid_map.get(npu, [])
            if pid_mem_list:
                max_pid = max(pid_mem_list, key=lambda x: x[1])[0]
                self.main_pid_list.append([max_pid])
        logging.debug(f"identifying the running NPU card: {running_npu_set}")
        return sorted(running_npu_set)


class CpuAlloc:
    def __init__(self, device_info: Optional[DeviceInfo] = None):
        self.device_info: DeviceInfo = device_info or DeviceInfo()
        self.cpu_node: Dict[int, int] = {}
        self.numa_to_cpu_map: Dict[int, List[int]] = defaultdict(list)
        self.npu_cpu_pool: Dict[int, List[int]] = {}
        self.npu_cpu_pool_all: Dict[int, List[int]] = {}
        self.assign_main: Dict[int, List[int]] = {}
        self.assign_acl: Dict[int, List[int]] = {}
        self.assign_rel: Dict[int, List[int]] = {}

    @staticmethod
    def average_distribute(groups: Dict[str, List[int]], pool: Dict[int, List[int]]) -> Dict[int, List[int]]:
        result: Dict[int, List[int]] = {}
        for key, npu_list in groups.items():
            cpu_list = sorted(pool[npu_list[0]])
            cpu_num_per_npu = len(cpu_list) // len(npu_list)
            for i, npu in enumerate(npu_list):
                start_index = i * cpu_num_per_npu
                end_index = (i + 1) * cpu_num_per_npu if i < len(npu_list) - 1 else len(cpu_list)
                result[npu] = cpu_list[start_index:end_index]
        return result

    @staticmethod
    def get_acl_main_threads() -> List[int]:
        thread_message, _ = execute_command(["ps", "-Te"])
        pids: List[int] = []
        acl_threads_set = set()
        for line in thread_message.splitlines():
            if "acl_thread" in line:
                pid = line.split()[0]
                if pid not in acl_threads_set:
                    acl_threads_set.add(pid)
                    pids.append(int(pid))
        return pids

    def dev_alloc(self) -> tuple[List[int], List[str]]:
        dev_pid_list: List[int] = []
        dev_cpu_list: List[str] = []
        out, _ = execute_command(["ps", "aux"])
        for line in out.splitlines():
            m = re.search(r"dev(\d+)_sq_task", line)
            if not m:
                continue
            dev_id = int(m.group(1))
            pid = int(line.split()[1])
            cpus = self.npu_cpu_pool_all.get(dev_id, [])
            if cpus:
                core = cpus[2] if len(cpus) >= 3 else cpus[0]
                dev_pid_list.append(pid)
                dev_cpu_list.append(str(core))
        return dev_pid_list, dev_cpu_list

    def irq_alloc(self) -> tuple[List[int], List[str]]:
        sq_irqs = []
        irq_id_list: List[int] = []
        irq_cpu_list: List[str] = []
        with open("/proc/interrupts") as f:
            for line in f:
                if "sq_send_trigger_irq" in line:
                    irq = line.split(":")[0].strip()
                    sq_irqs.append(irq)

        for npu in sorted(self.npu_cpu_pool_all.keys()):
            cpus = self.npu_cpu_pool_all[npu]
            if len(cpus) < 2:
                continue

            info, _ = execute_command(["npu-smi", "info", "-t", "board", "-i", str(npu)])
            pci_addr = ""
            for line in info.splitlines():
                if "PCIe Bus Info" in line:
                    pci_addr = line.split()[-1].lower()
                    break
            if not pci_addr:
                raise RuntimeError(f"Can't find PCI address of NPU{npu} .")

            msi_irq_dir = f"/sys/bus/pci/devices/{pci_addr}/msi_irqs/"
            if not os.path.exists(msi_irq_dir):
                raise RuntimeError(f"Can't find MSI interrupt directory of NPU{npu} .")

            npu_irq_list = sorted(os.listdir(f"/sys/bus/pci/devices/{pci_addr}/msi_irqs/"), key=lambda x: int(x))
            for irq in sq_irqs:
                if irq in npu_irq_list:
                    irq_id_list.extend([int(irq), int(irq) + 1])
                    irq_cpu_list.extend([str(cpus[0]), str(cpus[1])])
                    break
        return irq_id_list, irq_cpu_list

    def build_cpu_node_map(self) -> None:
        cpu_numa_map, _ = execute_command(["lscpu", "-e=CPU,NODE"])
        for line in cpu_numa_map.splitlines():
            line = line.strip()
            if not line or not line[0].isdigit():
                continue
            cpu_str, node_str = line.split()
            cpu = int(cpu_str)
            node = int(node_str)
            self.cpu_node[cpu] = node
            self.numa_to_cpu_map[node].append(cpu)
        if not self.numa_to_cpu_map:
            raise RuntimeError("lscpu 输出错误，未检测到 NUMA 节点")

    def extend_numa(self, cpu_list: List[int]) -> List[int]:
        if not cpu_list:
            return []
        nodes = {self.cpu_node[c] for c in cpu_list}
        if len(nodes) != 1:
            return cpu_list
        node = list(nodes)[0]
        next_node = (node + 1) % len(self.numa_to_cpu_map)
        extended = cpu_list[:]
        for cpu in self.numa_to_cpu_map[next_node]:
            if cpu in self.device_info.allowed_cpus:
                extended.append(cpu)
        return sorted(set(extended))

    def handle_no_affinity(self) -> None:
        num_running_npu = len(self.device_info.running_npu_list)
        num_numa_node = len(self.numa_to_cpu_map)
        if num_numa_node == 0 or num_running_npu == 0:
            return
        npu_num_per_node = (num_running_npu // num_numa_node) + (1 if num_running_npu % num_numa_node else 0)
        index = 0
        for node in sorted(self.numa_to_cpu_map):
            cpus = [c for c in self.numa_to_cpu_map[node] if c in self.device_info.allowed_cpus]
            if not cpus:
                continue
            npu_num_this_node = min(npu_num_per_node, num_running_npu - index)
            total_cpu_num = len(cpus)
            base_cpu_num = total_cpu_num // npu_num_this_node
            extra_cpu_num = total_cpu_num % npu_num_this_node
            start_index = 0
            for i in range(npu_num_this_node):
                take_cpu_num = base_cpu_num + (1 if i < extra_cpu_num else 0)
                end_index = start_index + take_cpu_num
                select_cpus_list = cpus[start_index:end_index]
                if index < num_running_npu:
                    npu = self.device_info.running_npu_list[index]
                    self.npu_cpu_pool[npu] = select_cpus_list
                    index += 1
                start_index = end_index

    def build_cpu_pools_all(self) -> None:
        raw_pool: Dict[int, List[int]] = {}

        if self.device_info.npu_affinity:
            for npu, cpus in self.device_info.npu_affinity.items():
                filtered = [c for c in cpus if c in self.device_info.allowed_cpus]
                raw_pool[npu] = filtered
        else:
            self.handle_no_affinity()
            raw_pool = self.npu_cpu_pool.copy()

        groups: Dict[str, List[int]] = defaultdict(list)
        for npu, cpus in raw_pool.items():
            groups[str(cpus)].append(npu)

        final_pool: Dict[int, List[int]] = {}
        for key, npu_list in groups.items():
            if len(npu_list) == 1:
                final_pool[npu_list[0]] = raw_pool[npu_list[0]]
            else:
                final_pool.update(self.average_distribute({key: npu_list}, raw_pool))
        logging.debug(f"npu_cpu_pool_all: {final_pool}")
        self.npu_cpu_pool_all = final_pool

    def build_cpu_pools_running(self) -> None:
        self.build_cpu_node_map()
        raw_pool: Dict[int, List[int]] = {}

        if self.device_info.npu_affinity:
            for npu in self.device_info.running_npu_list:
                cpus = self.device_info.npu_affinity.get(npu, [])
                filtered = [c for c in cpus if c in self.device_info.allowed_cpus]
                raw_pool[npu] = filtered
        else:
            self.handle_no_affinity()
            for npu in self.device_info.running_npu_list:
                if npu in self.npu_cpu_pool:
                    raw_pool[npu] = self.npu_cpu_pool[npu]

        groups: Dict[str, List[int]] = defaultdict(list)
        for npu, cpus in raw_pool.items():
            groups[str(cpus)].append(npu)

        final_pool: Dict[int, List[int]] = {}
        for key, npu_list in groups.items():
            if len(npu_list) == 1:
                final_pool[npu_list[0]] = raw_pool[npu_list[0]]
            else:
                final_pool.update(self.average_distribute({key: npu_list}, raw_pool))
        logging.debug(f"npu_cpu_pool: {final_pool}")
        self.npu_cpu_pool = final_pool

    def allocate(self, main_range: int, acl_range: int, rel_range: int) -> None:
        for npu, pool in self.npu_cpu_pool.items():
            usable_pool = pool[3:]
            need = main_range + acl_range + rel_range
            if len(usable_pool) < need:
                raise RuntimeError(f"NPU{npu} 的 CPU 数量不足，默认方案需要至少 {need} 个")
            self.assign_main[npu] = usable_pool[:main_range]
            self.assign_acl[npu] = usable_pool[main_range : main_range + acl_range]
            self.assign_rel[npu] = usable_pool[main_range + acl_range : main_range + acl_range + rel_range]


class CustomBind:
    def __init__(
        self,
        process_name: str = "",
        cpu_list: Optional[List[str]] = None,
        bind_sub_process: bool = False,
        is_thread: bool = False,
        is_irq: bool = False,
        mem_bind: bool = False,
        pid: Optional[List[int]] = None,
        irq_id: Optional[List[int]] = None,
    ):
        self.process_name = process_name
        self.bind_sub_process = bind_sub_process
        self.is_thread = is_thread
        self.is_irq = is_irq
        self.mem_bind = mem_bind
        self.pid = pid or []
        self.irq_id = irq_id or []
        self.cpu_list = [expand_cpu_list(seg) for seg in (cpu_list or [])]

    @staticmethod
    def get_main_pid_from_docker(pid: int) -> int:
        pid_file = f"/proc/{pid}/status"
        if not os.path.exists(pid_file):
            return 0
        out, return_code = execute_command(["grep", "Ngid", pid_file])
        if return_code != 0:
            return 0
        parts = out.strip().split()
        if parts[-1] != "0":
            return int(parts[-1])
        else:
            return 0

    def get_real_main_pid_list(
        self, pid_list: List[Tuple[int, int]], main_pid_list: List[List[int]]
    ) -> List[List[int]]:
        real_main_pid_list: List[List[int]] = []
        for pid, ppid in pid_list:
            per_real_pid_list: List[int] = []
            for pids in main_pid_list:
                if pid in pids:
                    per_real_pid_list.append(pid)
                    continue
                elif ppid in pids:
                    per_real_pid_list.append(ppid)
                    continue
                real_pid = self.get_main_pid_from_docker(pid)
                if real_pid in pids:
                    per_real_pid_list.append(pid)
                    continue
                real_ppid = self.get_main_pid_from_docker(ppid)
                if real_ppid in pids:
                    per_real_pid_list.append(ppid)
            if per_real_pid_list:
                real_main_pid_list.append(per_real_pid_list)
        return real_main_pid_list

    def find_threads(self, current_pid) -> List[Tuple[int, int]]:
        if self.pid:
            pid_list = []
            for p in self.pid:
                try:
                    ppid = int(subprocess.check_output(["ps", "-o", "ppid=", "-p", str(p)], text=True).strip())
                except subprocess.CalledProcessError:
                    ppid = -1
                if p == current_pid or ppid == current_pid:
                    pid_list.append((p, ppid))
            return pid_list

        select_idx = 1 if self.is_thread else 0
        out, _ = execute_command(["ps", "-Te"]) if self.is_thread else execute_command(["ps", "-eo", "pid,ppid,cmd"])
        pid_list = []
        for line in out.splitlines():
            if self.process_name in line:
                parts = line.split()
                if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                    pid = int(parts[select_idx])
                    ppid = int(parts[1 - select_idx])
                    if pid == current_pid or ppid == current_pid:
                        pid_list.append((pid, ppid))
        if not pid_list:
            raise RuntimeError(f"未找到进程名称包含 {self.process_name} 的进程")
        return pid_list

    def irq_bind(self) -> None:
        if not shutil.which("systemctl"):
            logging.warning(
                "The systemctl command cannot be used in the current environment.If the irqbalance "
                "service is enabled, manually disable the irqbalance service.Otherwise, the "
                "interrupt-core binding cannot take effect."
            )
        else:
            out, return_code = execute_command(["systemctl", "list-unit-files"])
            if return_code == 0 and "irqbalance.service" in out:
                _, return_code = execute_command(["systemctl", "is-active", "--quiet", "irqbalance"])
                if return_code == 0:
                    logging.info("The irqbalance service is running and has been stopped.")
                    _, return_code = execute_command(["systemctl", "stop", "irqbalance"])
                    if return_code != 0:
                        logging.warning(
                            "The irqbalance service cannot be stopped.You need to manually stop it."
                            "Otherwise, the interrupt-core binding cannot take effect."
                        )

        for irq_id, target_cpu_list in zip(self.irq_id, self.cpu_list):
            affinity_file_path = f"/proc/irq/{irq_id}/smp_affinity"
            with open(affinity_file_path, "w") as f:
                f.write(cpu_to_mask(target_cpu_list))
                logging.info(f"Bind the interrupt of IRQ-{irq_id} to CPU{target_cpu_list}")

    def execute_bind(self, pid: int, cpu_list_str: str, process_type: str, cpu_node: Dict[int, int]) -> None:
        cmd = ["taskset", "-acp" if self.bind_sub_process else "-cp", cpu_list_str, str(pid)]
        logging.info(f"Bind the {self.process_name or 'target'} ({process_type}={pid}) to CPU{cpu_list_str}")
        _, return_code = execute_command(cmd)
        if return_code != 0:
            raise RuntimeError(f"Failed to execute the command: {' '.join(cmd)}")

    def bind(self, cpu_allocer: CpuAlloc, current_pid: int) -> None:
        process_type = "pid" if not self.is_thread else "tid"
        pid_list = self.find_threads(current_pid)
        if not pid_list:
            return
        real_main_pid_list = self.get_real_main_pid_list(pid_list, cpu_allocer.device_info.main_pid_list)
        cpu_index = -1
        for pid, ppid in pid_list:
            cpu_list_str = ""
            if len(self.cpu_list) == 1:
                cpu_list_str = ",".join(map(str, self.cpu_list[0]))
                self.execute_bind(pid, cpu_list_str, process_type, cpu_allocer.cpu_node)
                continue
            if len(pid_list) == len(self.cpu_list):
                cpu_index += 1
                cpu_list_str = ",".join(map(str, self.cpu_list[cpu_index]))
            for pids in real_main_pid_list:
                if pid in pids or ppid in pids:
                    cpu_list_str = ",".join(map(str, self.cpu_list[real_main_pid_list.index(pids)]))
            if not cpu_list_str:
                logging.error(
                    f"Failed to bind process {pid_list} to CPU {self.cpu_list}. Please ensure that "
                    f"the number of processes to be bound is the same as the number of cpu_list you have "
                    f"entered, or ensure that the Ngid field in the /proc/{pid}/status file is not 0. "
                    f"It is recommended that the script be executed on the host machine."
                )
                continue
            self.execute_bind(pid, cpu_list_str, process_type, cpu_allocer.cpu_node)
        logging.debug(f"pid_list: {pid_list}")
        logging.debug(f"real_main_pid_list: {real_main_pid_list}")


def export_bind_config(cpu_alloc: CpuAlloc, config: dict) -> Dict:
    main_cpu_list: List[str] = []
    acl_cpu_list: List[str] = []
    rel_cpu_list: List[str] = []
    config = {"custom_bind": []}
    cpu_alloc.build_cpu_pools_all()
    cpu_alloc.allocate(MAIN_PROCESS_RANGE, ACL_THREAD_RANGE, RELEASE_THREAD_RANGE)
    main_pid_list = cpu_alloc.get_acl_main_threads()
    dev_pid_list, dev_cpu_list = cpu_alloc.dev_alloc()
    irq_id_list, irq_cpu_list = cpu_alloc.irq_alloc()
    for npu in sorted(cpu_alloc.device_info.running_npu_list):
        main_cpu_list.append(",".join(map(str, cpu_alloc.assign_main[npu])))
        acl_cpu_list.append(",".join(map(str, cpu_alloc.assign_acl[npu])))
        rel_cpu_list.append(",".join(map(str, cpu_alloc.assign_rel[npu])))

    config["custom_bind"].append({"pid": main_pid_list, "cpu_list": main_cpu_list, "bind_sub_process": True})
    config["custom_bind"].append({"process_name": "acl_thread", "cpu_list": acl_cpu_list, "is_thread": True})
    config["custom_bind"].append({"process_name": "release_thread", "cpu_list": rel_cpu_list, "is_thread": True})
    config["custom_bind"].append({"pid": dev_pid_list, "cpu_list": dev_cpu_list})
    config["custom_bind"].append({"irq_id": irq_id_list, "cpu_list": irq_cpu_list, "is_irq": True})
    return config


def load_custom_bind(data: Dict) -> List[CustomBind]:
    binders = []
    for item in data.get("custom_bind", []):
        binders.append(
            CustomBind(
                process_name=item.get("process_name", ""),
                cpu_list=item["cpu_list"],
                bind_sub_process=item.get("bind_sub_process", False),
                is_thread=item.get("is_thread", False),
                is_irq=item.get("is_irq", False),
                mem_bind=item.get("mem_bind", False),
                pid=item.get("pid", []),
                irq_id=item.get("irq_id", []),
            )
        )
    return binders


def run(rank_id: int, config: dict | None = None) -> None:
    loop_count = 0
    input_data = config if config is not None else DEFAULT_CPU_BIND_CONFIG
    cpu_allocer = CpuAlloc(DeviceInfo())
    current_pid = psutil.Process().pid
    binder_list = load_custom_bind(input_data)
    for bind in binder_list:
        loop_count += 1
        logging.info(f"Start binding core round {loop_count}: {bind.__dict__}")
        try:
            if bind.is_irq:
                bind.irq_bind()
            else:
                if not bind.pid and not bind.process_name:
                    logging.error("No input bound object. One of 'pid, process_name, irq_id' are required.")
                    continue
                if len(bind.cpu_list) > 1:
                    bind.cpu_list = [bind.cpu_list[rank_id % 16]]
                bind.bind(cpu_allocer, current_pid)
        except Exception as e:
            logging.error(e)
        logging.info(f"===== Round {loop_count} of core binding has ended. =====")
