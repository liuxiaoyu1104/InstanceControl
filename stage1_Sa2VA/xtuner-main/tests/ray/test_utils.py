import os
import unittest
import socket
import ray



from xtuner.v1.ray.utils import find_master_addr_and_port, get_accelerator_ids, get_ray_accelerator
import parametrize


class TestFindMasterAddrAndPort(unittest.TestCase):
    """Test the find_master_addr_and_port function."""
    
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
    
    def test_find_master_addr_and_port_actual(self):
        """Test find_master_addr_and_port behavior in an actual Ray environment."""
        # Call the remote function.
        addr, port = ray.get(find_master_addr_and_port.remote())
        
        # Verify that the returned address is a valid IP address.
        try:
            socket.inet_aton(addr)
            is_valid_ip = True
        except socket.error:
            is_valid_ip = False
        
        self.assertTrue(is_valid_ip, f"返回的地址 {addr} 不是有效的 IP 地址")
        
        # Verify that the port is valid, between 0 and 65535.
        self.assertTrue(0 < port <= 65535, f"返回的端口 {port} 不是有效的端口号")
    
    def test_find_master_addr_and_port_port_availability(self):
        """Test that the port returned by find_master_addr_and_port is available."""
        # Call the remote function.
        addr, port = ray.get(find_master_addr_and_port.remote())
        
        # Verify that the returned port is actually available.
        # Create a new socket.
        test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        try:
            # Try binding to this port; this should succeed because find_master_addr_and_port
            # closes the socket connection after obtaining the port number.
            test_socket.bind(('', port))
            test_socket.listen(1)
            self.assertTrue(True, "端口可以被绑定，证明它是可用的")
        except OSError as e:
            self.fail(f"端口 {port} 不可用: {e}")
        finally:
            test_socket.close()


class TestGetAcceleratorIds(unittest.TestCase):
    """Test the get_accelerator_ids function."""
    
    @classmethod
    def setUpClass(cls):
        """Set up the Ray environment before tests and detect available accelerator types."""
        # Initialize Ray if it has not been initialized yet.
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        
    @classmethod
    def tearDownClass(cls):
        """Shut down the Ray environment after tests."""
        # Shut down Ray if needed.
        if ray.is_initialized():
            ray.shutdown()

    @parametrize.parametrize("num_accelerators", [1, 2, 4])
    def test_get_accelerator_ids(self, num_accelerators: int):
        """Test retrieving the GPU ID list."""

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
