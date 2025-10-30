import unittest
import socket
import time
from typing import Dict

import httpx
import ray

from xtuner.v1.ray.base import AutoCPUWorkers, BaseCPUWorker, CPUResourcesConfig


@ray.remote(num_cpus=1)
class NaiveCPUWorker(BaseCPUWorker):
    def __init__(self, config: Dict, num_cpus = 1):
        super().__init__(config, num_cpus)
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
            ray.init(num_cpus=16, ignore_reinit_error=True)
    
    @classmethod
    def tearDownClass(cls):
        """Shut down Ray environment after tests"""
        if ray.is_initialized():
            ray.shutdown()
    
    def test_create_cpu_workers(self):
        """Test creating CPU workers"""
        num_workers = 4
        cpu_config = CPUResourcesConfig.from_total(
            total_cpus=4,
            total_memory=4 * 1024**3,
            num_workers=num_workers,
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
        cpu_resources_config = CPUResourcesConfig.from_total(
            total_cpus=2,
            total_memory=2 * 1024**3,
            num_workers=2,
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


if __name__ == '__main__':
    unittest.main()
