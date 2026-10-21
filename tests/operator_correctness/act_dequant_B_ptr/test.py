from typing import Tuple
import torch
import triton
import triton.language as tl
import time 
import pytest
from OpenMixOpl.triton import (
    act_dequant_B_ptr,
    act_dequant_T_nostride_B_ptr,
    act_dequant_noT_stride_B_ptr
)

def ceil_div(a, b):
    return (a + b - 1) // b

def calc_diff(x, y):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim

# Test parameters for all dequantization tests
test_parameters = [
    (29100, 5120), (29100, 1536),
    (27906, 5120), (27906, 1536),
    (27908, 5120), (27908, 1536),
    (28205, 5120), (28205, 1536),
    (28411, 5120), (28411, 1536)
]

@pytest.mark.parametrize("M, K", test_parameters)
def test_act_dequant(M, K):
    block_size = 128
    dtype_x = torch.float8_e4m3fn
    dtype_s = torch.float32
    device = 'cuda'
    atol = 1e-3
    rtol = 1e-2

    # --- Input data preparation --- ---
    x = torch.randint(-128, 127, (M, K), device=device).to(dtype_x).contiguous()
    k_blocks = ceil_div(K, block_size)
    s = torch.randn((M, k_blocks), dtype=dtype_s, device=device) * 0.01 + 0.1 # s shape: [M, k_blocks]

    # --- Triton Kernel ---
    y = act_dequant_B_ptr(x, s, block_size)

    # --- Reference implementation ---
    assert K % block_size == 0
    x_reshape = torch.reshape(x,(M,k_blocks,block_size))
    s_useful = torch.reshape(s,(M,k_blocks,1))
    y_ref1 = x_reshape.to(torch.float32) * s_useful
    y_ref1 = torch.reshape(y_ref1,(M,K))
    y_ref = y_ref1

    # --- Assert correctness ---
    # Using calc_diff
    diff_y = calc_diff(y, y_ref)
    assert diff_y < 0.001, f'M={M}, K={K}, diff_y={diff_y:.5f}'
    # Using torch.allclose
    are_close = torch.allclose(y.float(), y_ref.float(), atol=atol, rtol=rtol)
    if not are_close:
         diff = torch.abs(y.float() - y_ref.float())
         print(f"Discrepancy found: M={M}, K={K} in test_act_dequant:")
         print(f"Max difference: {diff.max()}")
         max_diff_idx = torch.argmax(diff)
         max_diff_coords = (max_diff_idx // K, max_diff_idx % K)
         print(f"Max difference location (index): {max_diff_coords}")
         print(f"Triton output value: {y[max_diff_coords].float().item()}")
         print(f"Reference output value: {y_ref[max_diff_coords].float().item()}")
         k_block_idx_diff = max_diff_coords[1] // block_size
         print(f"Scale used at max difference: {s[max_diff_coords[0], k_block_idx_diff].item()}")
    assert are_close


if __name__ == '__main__':
    torch.manual_seed(0) 
    print(f"Running all tests in {__file__} using pytest...")
    pytest.main(['-v', __file__])