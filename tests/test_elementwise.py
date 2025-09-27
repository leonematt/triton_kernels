import numpy
import sys
import torch
import nexus

# Create input tensors and result tensor
buf0 = torch.full((1024,), 3.0, dtype=torch.float32)  # First input: all 3's
buf1 = torch.ones(1024, dtype=torch.float32) 
res2 = torch.zeros(1024, dtype=torch.float32)

# Get runtime and device
rt = nexus.get_runtime("cuda")
dev = rt.get_devices()[0]

# Create device buffers
nb0 = dev.create_buffer(buf0)
nb1 = dev.create_buffer(buf1)
nb2 = dev.create_buffer(res2)

# Load the library and get the elementwise kernel
lib = dev.load_library_file("ptx_kernels/elementwise_add_kernel_BLOCK_SIZE1024.ptx")  # Adjust path as needed
kern = lib.get_kernel('elementwise_add_kernel_BLOCK_SIZE1024')

# Create schedule and command
sched = dev.create_schedule()
cmd = sched.create_command(kern)

# Set kernel arguments (5 parameters total)
cmd.set_arg(0, nb0)        # a_ptr - first input tensor
cmd.set_arg(1, nb1)        # b_ptr - second input tensor  
cmd.set_arg(2, nb2)        # output_ptr - output tensor
cmd.set_arg(3, 1024)       # n_elements - number of elements
cmd.set_arg(4, 1024)       # n_elements - number of elements
# Note: The 5th parameter might be for operation type or other metadata
# You may need to create a buffer for this or pass it differently
# cmd.set_arg(4, ???)      # This depends on what the 5th parameter expects

# Finalize with grid and block dimensions
# For 1024 elements with block size 128: need 8 blocks (1024/128 = 8)
cmd.finalize([8, 1, 1], [128, 1, 1])

# Run the kernel
sched.run()

# Copy result back to host
nb2.copy(res2)

print(res2)