#!/usr/bin/env python3

import torch
import triton
import triton.language as tl

@triton.jit
def elementwise_kernel(
    a_ptr,          # Pointer to first input tensor
    b_ptr,          # Pointer to second input tensor  
    output_ptr,     # Pointer to output tensor
    n_elements,     # Number of elements
    OP_NAME: tl.constexpr, # Operation name
    BLOCK_SIZE: tl.constexpr, # Block size (must be power of 2)
):
    """
    Single unified elementwise kernel supporting multiple operations.
    The OP_NAME and BLOCK_SIZE are compile-time constants, allowing Triton
    to create a specialized, optimized kernel for each combination.
    """
    # Get the program ID, which determines the block of data this instance processes
    pid = tl.program_id(axis=0)
    
    # Calculate the starting offset for this block
    block_start = pid * BLOCK_SIZE
    
    # Generate a vector of offsets for the elements this block will handle
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask to prevent out-of-bounds memory access
    mask = offsets < n_elements
    
    # Load the data from global memory
    # For division, use 1.0 as the default for `b` to avoid division by zero
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    if OP_NAME == 'DIV':
        b = tl.load(b_ptr + offsets, mask=mask, other=1.0)
    else:
        b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    
    # Perform the specified operation
    # This `if/elif` chain is resolved at compile time due to `tl.constexpr`
    if OP_NAME == 'ADD':
        output = a + b
    elif OP_NAME == 'SUB':
        output = a - b
    elif OP_NAME == 'MUL':
        output = a * b
    elif OP_NAME == 'DIV':
        output = a / b
    
    # Store the result back to global memory
    tl.store(output_ptr + offsets, output, mask=mask)

    # Test variants with different block sizes and operations
VARIANTS = [
        # Addition variants
        {'BLOCK_SIZE': 128, 'OP_NAME': 'ADD'},
        {'BLOCK_SIZE': 256, 'OP_NAME': 'ADD'},
        {'BLOCK_SIZE': 512, 'OP_NAME': 'ADD'},
        {'BLOCK_SIZE': 1024, 'OP_NAME': 'ADD'},
        
        # Subtraction variants
        {'BLOCK_SIZE': 128, 'OP_NAME': 'SUB'},
        {'BLOCK_SIZE': 256, 'OP_NAME': 'SUB'},
        {'BLOCK_SIZE': 512, 'OP_NAME': 'SUB'},
        {'BLOCK_SIZE': 1024, 'OP_NAME': 'SUB'},
        
        # Multiplication variants
        {'BLOCK_SIZE': 128, 'OP_NAME': 'MUL'},
        {'BLOCK_SIZE': 256, 'OP_NAME': 'MUL'},
        {'BLOCK_SIZE': 512, 'OP_NAME': 'MUL'},
        {'BLOCK_SIZE': 1024, 'OP_NAME': 'MUL'},
        
        # Division variants
        {'BLOCK_SIZE': 128, 'OP_NAME': 'DIV'},
        {'BLOCK_SIZE': 256, 'OP_NAME': 'DIV'},
        {'BLOCK_SIZE': 512, 'OP_NAME': 'DIV'},
        {'BLOCK_SIZE': 1024, 'OP_NAME': 'DIV'},
    ]
    
def run():
    tensor_size = 4096
    print(f"--- Running All Variants (Tensor Size: {tensor_size}) ---")

    # Loop through and test each variant
    for variant in VARIANTS:
        op_name = variant['OP_NAME']
        block_size = variant['BLOCK_SIZE']

        # Using end="" keeps the status on the same line
        print(f"Testing OP={op_name:<4}, BLOCK_SIZE={block_size:4}... ", end="")

        try:
            ## --- Data and Kernel Execution ---
            a = torch.randn(tensor_size, device='cuda', dtype=torch.float32)
            b = torch.randn(tensor_size, device='cuda', dtype=torch.float32)
            
            # For division, clamp b to avoid division by tiny numbers
            if op_name == 'DIV':
                b.clamp_(min=0.1)

            output = torch.empty_like(a)
            
            grid = (triton.cdiv(tensor_size, block_size),)
            
            elementwise_kernel[grid](
                a_ptr=a,
                b_ptr=b,
                output_ptr=output,
                n_elements=tensor_size,
                OP_NAME=op_name,
                BLOCK_SIZE=block_size
            )

            ## --- Verification ---
            if op_name == 'ADD': expected_output = a + b
            elif op_name == 'SUB': expected_output = a - b
            elif op_name == 'MUL': expected_output = a * b
            elif op_name == 'DIV': expected_output = a / b
            
            is_correct = torch.allclose(output, expected_output)
            status = "✅ PASSED" if is_correct else "❌ FAILED"
            print(status)

        except Exception as e:
            print(f"❌ ERROR: {e}")

    print("\n--- All variants tested. ---")
    
if __name__ == "__main__":
    run()
