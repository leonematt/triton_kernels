import pytest
import torch
import torch.nn.functional as F
import nexus
import os

VARIANTS = [
  {'BLOCK_SIZE': 128},
  {'BLOCK_SIZE': 256},
  {'BLOCK_SIZE': 512},
  {'BLOCK_SIZE': 1024},
]

@pytest.fixture(scope="module")
def runtime_and_device():
  rt = nexus.get_runtime("cuda")
  dev = rt.get_devices()[0]
  return rt, dev

def launch_silu(dev, variant, x):
  BLOCK_SIZE = variant['BLOCK_SIZE']
  n_elements = x.numel()
  output = torch.zeros_like(x)

  nb_x = dev.create_buffer(x)
  nb_output = dev.create_buffer(output)

  kernel_name = f"silu_kernel_BLOCK_SIZE_{BLOCK_SIZE}"
  lib_path = f"ptx_kernels/{kernel_name}.ptx"

  if not os.path.exists(lib_path):
    pytest.skip(f"Kernel file not found: {lib_path}")

  lib = dev.load_library(lib_path)
  kern = lib.get_kernel(kernel_name)

  sched = dev.create_schedule()
  cmd = sched.create_command(kern)

  cmd.set_arg(0, nb_x)
  cmd.set_arg(1, nb_output)
  cmd.set_arg(2, n_elements)
  cmd.set_arg(3, 0)

  grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
  cmd.finalize([grid_size, 1, 1], [128, 1, 1])
  sched.run()
  nb_output.copy(output)

  return output

@pytest.mark.parametrize("variant", VARIANTS, ids=lambda v: f"BLOCK_{v['BLOCK_SIZE']}")
def test_silu_basic(runtime_and_device, variant):
  _, dev = runtime_and_device
  x = torch.randn(4096, dtype=torch.float32)
  output = launch_silu(dev, variant, x)
  expected = F.silu(x)
  max_diff = (output - expected).abs().max().item()
  print(f"\nsilu basic BLOCK={variant['BLOCK_SIZE']}: max diff = {max_diff:.6e}")
  assert torch.allclose(output, expected, rtol=1e-5, atol=1e-5), f"Max diff: {max_diff:.6e}"
  print("✅ PASSED")

@pytest.mark.parametrize("variant", VARIANTS, ids=lambda v: f"BLOCK_{v['BLOCK_SIZE']}")
def test_silu_all_negative(runtime_and_device, variant):
  _, dev = runtime_and_device
  x = -torch.abs(torch.randn(2048, dtype=torch.float32))
  output = launch_silu(dev, variant, x)
  expected = F.silu(x)
  max_diff = (output - expected).abs().max().item()
  print(f"\nsilu all-negative BLOCK={variant['BLOCK_SIZE']}: max diff = {max_diff:.6e}")
  assert torch.allclose(output, expected, rtol=1e-5, atol=1e-5), f"Max diff: {max_diff:.6e}"
  print("✅ PASSED")

@pytest.mark.parametrize("variant", VARIANTS, ids=lambda v: f"BLOCK_{v['BLOCK_SIZE']}")
def test_silu_all_positive(runtime_and_device, variant):
  _, dev = runtime_and_device
  x = torch.abs(torch.randn(2048, dtype=torch.float32)) + 0.01
  output = launch_silu(dev, variant, x)
  expected = F.silu(x)
  max_diff = (output - expected).abs().max().item()
  print(f"\nsilu all-positive BLOCK={variant['BLOCK_SIZE']}: max diff = {max_diff:.6e}")
  assert torch.allclose(output, expected, rtol=1e-5, atol=1e-5), f"Max diff: {max_diff:.6e}"
  print("✅ PASSED")

@pytest.mark.parametrize("variant", VARIANTS, ids=lambda v: f"BLOCK_{v['BLOCK_SIZE']}")
def test_silu_zeros(runtime_and_device, variant):
  _, dev = runtime_and_device
  x = torch.zeros(1024, dtype=torch.float32)
  output = launch_silu(dev, variant, x)
  print(f"\nsilu zeros BLOCK={variant['BLOCK_SIZE']}")
  assert torch.all(output == 0.0), "silu of zeros should be zero"
  print("✅ PASSED")

@pytest.mark.parametrize("variant", VARIANTS, ids=lambda v: f"BLOCK_{v['BLOCK_SIZE']}")
def test_silu_odd_size(runtime_and_device, variant):
  _, dev = runtime_and_device
  x = torch.randn(1337, dtype=torch.float32)
  output = launch_silu(dev, variant, x)
  expected = F.silu(x)
  max_diff = (output - expected).abs().max().item()
  print(f"\nsilu odd-size (n=1337) BLOCK={variant['BLOCK_SIZE']}: max diff = {max_diff:.6e}")
  assert torch.allclose(output, expected, rtol=1e-5, atol=1e-5), f"Max diff: {max_diff:.6e}"
  print("✅ PASSED")

@pytest.mark.parametrize("variant", VARIANTS, ids=lambda v: f"BLOCK_{v['BLOCK_SIZE']}")
def test_silu_large(runtime_and_device, variant):
  _, dev = runtime_and_device
  x = torch.randn(1024 * 1024, dtype=torch.float32)
  output = launch_silu(dev, variant, x)
  expected = F.silu(x)
  max_diff = (output - expected).abs().max().item()
  print(f"\nsilu large (1M) BLOCK={variant['BLOCK_SIZE']}: max diff = {max_diff:.6e}")
  assert torch.allclose(output, expected, rtol=1e-5, atol=1e-5), f"Max diff: {max_diff:.6e}"
  print("✅ PASSED")

if __name__ == "__main__":
  pytest.main([__file__, "-v", "-s"])
