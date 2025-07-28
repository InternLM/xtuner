import os
import unittest
import socket
import ray



from xtuner.v1.ray.utils import find_master_addr_and_port, get_accelerator_ids, get_ray_accelerator
import parametrize


class TestFindMasterAddrAndPort(unittest.TestCase):
    """测试 find_master_addr_and_port 函数"""
    
    @classmethod
    def setUpClass(cls):
        """测试前设置 Ray 环境"""
        # 如果 Ray 还没有初始化，则初始化它
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

    @classmethod
    def tearDownClass(cls):
        """测试后关闭 Ray 环境"""
        # 关闭 Ray（如果需要的话）
        if ray.is_initialized():
            ray.shutdown()
    
    def test_find_master_addr_and_port_actual(self):
        """测试 find_master_addr_and_port 函数在实际 Ray 环境中的行为"""
        # 实际调用远程函数
        addr, port = ray.get(find_master_addr_and_port.remote())
        
        # 验证返回的地址是有效的 IP 地址
        try:
            socket.inet_aton(addr)
            is_valid_ip = True
        except socket.error:
            is_valid_ip = False
        
        self.assertTrue(is_valid_ip, f"返回的地址 {addr} 不是有效的 IP 地址")
        
        # 验证端口是有效的端口号（介于 0 和 65535 之间）
        self.assertTrue(0 < port <= 65535, f"返回的端口 {port} 不是有效的端口号")
    
    def test_find_master_addr_and_port_port_availability(self):
        """测试 find_master_addr_and_port 返回的端口是可用的"""
        # 调用远程函数
        addr, port = ray.get(find_master_addr_and_port.remote())
        
        # 验证返回的端口确实是可用的
        # 创建一个新的套接字
        test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        try:
            # 尝试绑定到这个端口 - 这应该成功，因为 find_master_addr_and_port
            # 会在获取端口号后关闭套接字连接
            test_socket.bind(('', port))
            test_socket.listen(1)
            self.assertTrue(True, "端口可以被绑定，证明它是可用的")
        except OSError as e:
            self.fail(f"端口 {port} 不可用: {e}")
        finally:
            test_socket.close()


class TestGetAcceleratorIds(unittest.TestCase):
    """测试 get_accelerator_ids 函数"""
    
    @classmethod
    def setUpClass(cls):
        """测试前设置 Ray 环境并检测可用的加速器类型"""
        # 如果 Ray 还没有初始化，则初始化它
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        
    @classmethod
    def tearDownClass(cls):
        """测试后关闭 Ray 环境"""
        # 关闭 Ray（如果需要的话）
        if ray.is_initialized():
            ray.shutdown()

    @parametrize.parametrize("num_accelerators", [1, 2, 4])
    def test_get_accelerator_ids(self, num_accelerators: int):
        """测试获取 GPU ID 列表"""

        accelerator = get_ray_accelerator()
        if accelerator == "GPU":
            options = {"num_gpus": num_accelerators}
        elif accelerator == "NPU":
            options = {"resources": {"NPU": num_accelerators}}
        else:
            self.assertFalse(True, f"Unsupported accelerator type: {accelerator}")

        ids = ray.get(get_accelerator_ids.options(**options).remote(accelerator))
        self.assertIsInstance(ids, list, "GPU IDs 应该是列表类型")
        self.assertEqual(len(ids), num_accelerators, "GPU IDs 列表不应为空")



if __name__ == '__main__':
    unittest.main()
