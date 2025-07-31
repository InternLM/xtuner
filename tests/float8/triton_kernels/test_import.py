import unittest

import ray


class TestCpuActorImport(unittest.TestCase):
    """测试 xtuner.v1.float8.triton_kernels 在 CPU ray Actor上的导入"""
    
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
    
    def test_cpu_actor_triton_kernels_import(self):
        """测试 xtuner.v1.float8.triton_kernels 在 CPU ray Actor上的导入"""
        # 远程函数
        def _import_triton_kernels():
            from xtuner.v1.float8.triton_kernels import (
                per_block_dequant_gemm,
                per_block_quant_gemm,
                per_tile_quant,
            )
            return 0

        return_code = ray.get(
            ray.remote(_import_triton_kernels)
            .options(num_gpus=0, num_cpus=1)
            .remote(),
            timeout=30)
        self.assertTrue(return_code == 0, "导入 xtuner.v1.float8.triton_kernels 失败")

if __name__ == '__main__':
    unittest.main()
