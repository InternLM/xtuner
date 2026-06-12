import unittest
import socket
import time
from typing import Dict

import httpx
import ray

from xtuner.v1.rl.utils import AutoCPUWorkers, CPUResourceManager, CPUResourcesConfig


GIB = 1024**3


@ray.remote(num_cpus=1)
class NaiveCPUWorker:
    def __init__(self, config: Dict, num_cpus = 1):
        self.config = config
        self.num_cpus = num_cpus
        if self.config["worker_type"] == "fake_receiver":
            from fastapi import FastAPI
            self.app = FastAPI()
            self.received_data = []
            self.addr = "127.0.0.1"
            self.port = self._set_free_port()
            print(f"Worker started at {self.addr}:{self.port}")
            self._setup_receiver()

    def _setup_receiver(self):
        """Set up the HTTP server for the receiver"""
        @self.app.post("/receive")
        async def receive_data(data: dict):
            self.received_data.append(data)
            return {"status": "success", "received": data}

    def _set_free_port(self):
        """Get an available port number"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port

    def start_server(self):
        """Start the HTTP server"""
        import threading
        import uvicorn

        def run_server():
            uvicorn.run(self.app, host=self.addr, port=self.port, log_level="critical")

        self.server = threading.Thread(target=run_server, daemon=True)
        self.server.start()
        time.sleep(1)  # Wait for the server to start
        return f"http://{self.addr}:{self.port}"

    def send_data(self, data: dict):
        """Send data to the target URL"""
        if self.config["worker_type"] != "fake_sender":
            raise ValueError("Only sender workers can send data")

        if not self.config["target_url"]:
            raise ValueError("target_url is not set")

        url = f"{self.config['target_url']}/receive"
        response = httpx.post(url, json=data, timeout=10.0)
        return response.json()

    def get_received_data(self):
        """Get the received data"""
        if self.config["worker_type"] != "fake_receiver":
            raise ValueError("Only receiver workers can get received data")
        return self.received_data


class TestAutoCPUWorkers(unittest.TestCase):
    """Test the functionality of AutoCPUWorkers"""

    @classmethod
    def setUpClass(cls):
        """Set up Ray environment before tests"""
        if not ray.is_initialized():
            # This test owns its Ray cluster; do not attach to an existing one on
            # a shared machine.
            ray.init(address="local", num_cpus=16, ignore_reinit_error=True, include_dashboard=False)

    @classmethod
    def tearDownClass(cls):
        """Shut down Ray environment after tests"""
        if ray.is_initialized():
            ray.shutdown()

    def test_create_cpu_workers(self):
        """Test creating CPU workers"""
        num_workers = 4
        cpu_config = CPUResourcesConfig(
            num_workers=num_workers,
            num_cpus_per_worker=1,
            cpu_memory_per_worker=1024**3,
        )

        # Create a receiver worker
        receiver_config = dict(worker_type="fake_receiver")

        workers_list, pg = AutoCPUWorkers.from_config(
            NaiveCPUWorker,
            receiver_config,
            cpu_config
        )

        self.assertEqual(len(workers_list), num_workers)
        self.assertIsNotNone(pg)

        # Cleanup
        ray.util.remove_placement_group(pg)

    def test_multiple_sends(self):
        """Test sending data multiple times"""
        # Create receiver worker
        receiver_config = dict(worker_type="fake_receiver")
        cpu_resources_config = CPUResourcesConfig(
            num_workers=2,
            num_cpus_per_worker=1,
            cpu_memory_per_worker=1024**3,
        )
        pg = AutoCPUWorkers.build_placement_group(cpu_resources_config)

        receiver_workers = AutoCPUWorkers.from_placement_group(
            NaiveCPUWorker,
            receiver_config,
            pg,
            num_workers=1,
            # default num_workers=-1 means the creation will consume all bundles
        )
        receiver_worker = receiver_workers[0]

        # Start the server for the receiver
        receiver_url = ray.get(receiver_worker.start_server.remote())

        # Create sender worker
        sender_config = dict(
            worker_type="fake_sender",
            target_url=receiver_url
        )
        sender_workers = AutoCPUWorkers.from_placement_group(
            NaiveCPUWorker,
            sender_config,
            pg,
            num_workers=1,
        )
        sender_worker = sender_workers[0]

        # Send multiple data packets
        test_messages = [
            {"id": 1, "message": "First message"},
            {"id": 2, "message": "Second message"},
            {"id": 3, "message": "Third message"}
        ]

        for msg in test_messages:
            response = ray.get(sender_worker.send_data.remote(msg))
            self.assertEqual(response["status"], "success")

        # Verify all data is received
        time.sleep(0.5)
        received_data = ray.get(receiver_worker.get_received_data.remote())
        self.assertEqual(len(received_data), 3)

        for i, msg in enumerate(test_messages):
            self.assertEqual(received_data[i], msg)

        # Cleanup
        ray.util.remove_placement_group(pg)


class TestCPUResourceManager(unittest.TestCase):
    def _install_resource_summary(
        self,
        manager: CPUResourceManager,
        *,
        external_cpus: float = 8,
        external_memory: int = 16 * GIB,
        max_node_external_cpus: float = 4,
    ):
        state = {
            "external_cpus": external_cpus,
            "external_memory": external_memory,
            "max_node_external_cpus": max_node_external_cpus,
        }

        def build_summary():
            registered_cpus = sum(
                pool.num_workers * pool.num_cpus_per_worker for pool in manager.pools.values()
            )
            registered_memory = sum(
                pool.num_workers * pool.cpu_memory_per_worker for pool in manager.pools.values()
            )
            return {
                "cluster_cpus": state["external_cpus"],
                "available_cpus": state["external_cpus"],
                "accelerator_cpus": 0.0,
                "external_capacity_cpus": state["external_cpus"],
                "ray_external_in_use_cpus": 0.0,
                "registered_external_cpus": registered_cpus,
                "remaining_after_registered_cpus": state["external_cpus"] - registered_cpus,
                "cluster_memory": state["external_memory"],
                "available_memory": state["external_memory"],
                "accelerator_memory": 0.0,
                "external_capacity_memory": state["external_memory"],
                "ray_external_in_use_memory": 0.0,
                "registered_external_memory": registered_memory,
                "remaining_after_registered_memory": state["external_memory"] - registered_memory,
                "max_node_external_cpus": state["max_node_external_cpus"],
            }

        manager._build_resource_summary = build_summary
        return state

    def test_register_success_and_duplicate_names(self):
        # register 会立即校验资源；同名 pool 再注册时应生成稳定的唯一名字。
        manager = CPUResourceManager()
        self._install_resource_summary(manager, external_cpus=8, external_memory=16 * GIB)
        config = CPUResourcesConfig(num_workers=1, num_cpus_per_worker=1, cpu_memory_per_worker=GIB)

        manager.register("judger", config)
        manager.register("judger", config)

        self.assertEqual(list(manager.pools), ["judger", "judger#2"])
        self.assertIs(manager.pools["judger"], config)
        self.assertIs(manager.pools["judger#2"], config)

    def test_register_rolls_back_when_total_cpu_is_insufficient(self):
        # 总 external CPU 不足时，失败的注册项不能残留在 manager.pools 中。
        manager = CPUResourceManager()
        self._install_resource_summary(manager, external_cpus=1, external_memory=16 * GIB)
        config = CPUResourcesConfig(num_workers=2, num_cpus_per_worker=1, cpu_memory_per_worker=GIB)

        with self.assertRaisesRegex(RuntimeError, "available_outside_accelerator_pg"):
            manager.register("judger", config)

        self.assertNotIn("judger", manager.pools)

    def test_register_rolls_back_when_memory_is_insufficient(self):
        # external memory 不足时，失败的注册项同样需要回滚。
        manager = CPUResourceManager()
        self._install_resource_summary(manager, external_cpus=8, external_memory=GIB)
        config = CPUResourcesConfig(num_workers=1, num_cpus_per_worker=1, cpu_memory_per_worker=2 * GIB)

        with self.assertRaisesRegex(RuntimeError, "memory requested"):
            manager.register("judger", config)

        self.assertNotIn("judger", manager.pools)

    def test_register_rolls_back_when_worker_cpu_exceeds_largest_node(self):
        # 单个 worker 请求的 CPU 不能超过任一节点可提供的 external CPU。
        manager = CPUResourceManager()
        self._install_resource_summary(
            manager,
            external_cpus=8,
            external_memory=16 * GIB,
            max_node_external_cpus=1,
        )
        config = CPUResourcesConfig(num_workers=1, num_cpus_per_worker=2, cpu_memory_per_worker=GIB)

        with self.assertRaisesRegex(RuntimeError, "largest node"):
            manager.register("judger", config)

        self.assertNotIn("judger", manager.pools)

    def test_validate_or_raise_checks_existing_pools_without_mutating_them(self):
        # validate_or_raise 会重新校验已有 pools；失败时不能清空已注册状态，便于调用方诊断。
        manager = CPUResourceManager()
        summary_state = self._install_resource_summary(manager, external_cpus=8, external_memory=16 * GIB)
        config = CPUResourcesConfig(num_workers=2, num_cpus_per_worker=1, cpu_memory_per_worker=GIB)
        manager.register("judger", config)

        summary_state["external_cpus"] = 1
        with self.assertRaisesRegex(RuntimeError, "available_outside_accelerator_pg"):
            manager.validate_or_raise()

        self.assertEqual(list(manager.pools), ["judger"])
        self.assertIs(manager.pools["judger"], config)


if __name__ == '__main__':
    unittest.main()
