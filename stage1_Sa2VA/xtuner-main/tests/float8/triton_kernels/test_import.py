import unittest

import ray


class TestCpuActorImport(unittest.TestCase):
    """Test importing xtuner.v1.float8.triton_kernels in a CPU Ray actor."""
    
    @classmethod
    def setUpClass(cls):
        """Set up the Ray environment before tests."""
        # Initialize Ray if it has not been initialized yet.
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

    @classmethod
    def tearDownClass(cls):
        """Shut down the Ray environment after tests."""
        # Shut down Ray if needed.
        if ray.is_initialized():
            ray.shutdown()
    
    def test_cpu_actor_triton_kernels_import(self):
        """Test importing xtuner.v1.float8.triton_kernels in a CPU Ray actor."""
        # Remote function.
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
