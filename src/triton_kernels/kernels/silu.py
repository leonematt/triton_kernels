import triton
import triton.language as tl

@triton.jit
def silu_kernel(
  input_ptr,
  output_ptr,
  n_elements,
  BLOCK_SIZE: tl.constexpr,
):
  pid = tl.program_id(axis=0)
  block_start = pid * BLOCK_SIZE
  offsets = block_start + tl.arange(0, BLOCK_SIZE)
  mask = offsets < n_elements

  x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
  
  # SiLU math: x * sigmoid(x)
  output = x * tl.sigmoid(x)
  tl.store(output_ptr + offsets, output, mask=mask)

VARIANTS = [
  {'BLOCK_SIZE': 128},
  {'BLOCK_SIZE': 256},
  {'BLOCK_SIZE': 512},
  {'BLOCK_SIZE': 1024},
]