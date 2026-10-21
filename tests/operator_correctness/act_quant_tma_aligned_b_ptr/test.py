from typing import Tuple
import pytest
import torch
import triton
import triton.language as tl
import time 

from OpenMixOpl.triton import (
    act_quant_B_ptr,
    act_quant_vanilla,
    act_quant_TMA_aligned_B_ptr
)

# Global constants and configurations
PERF = True
def ceil_div(a, b):
    return (a + b - 1) // b

def calc_diff(x, y):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim

# PyTorch reference implementation for FP8 quantization
def per_token_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    block_size = 128

    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    s = torch.empty((m, (n + block_size - 1) // block_size), device=x.device, dtype=torch.float32)

    for block_idx in range(s.size(1)):
        start = block_idx * block_size
        end = min(start + block_size, n)
        block = x[:, start:end]

        block_amax = block.abs().float().amax(dim=1, keepdim=True).clamp(1e-4)
        scale = block_amax / 448.0
        quantized_block = (block * (448.0 / block_amax)).to(torch.float8_e4m3fn)

        y[:, start:end] = quantized_block
        s[:, block_idx] = scale.squeeze(1)
    return y, s

# Test parameters for test_act_quant
test_parameters_act_quant = [
    (64, 512), (64, 1536), (64, 7168), (64, 16384), (64, 2048),
    (128, 128), (128, 1536), (128, 7168), (128, 16384), (128, 2048),
    (4096, 512), (4096, 1536), (4096, 7168), (4096, 16384), (4096, 2048),
]
# Test parameters for test_act_quant_tmaalign
test_parameters_act_quant_tmaalign = [
    (65, 512),
    (64, 1536),
    (64, 7168),
    (64, 16384),
    (64, 2048),
    (128, 128),
    (128, 1536),
    (128, 7168),
    (128, 16384),
    (128, 2048),
    (4096, 512),
    (4096, 1536),
    (4096, 7168),
    (4096, 16384),
    (4097, 2048),
    (29100, 5120),
    (29100, 1536),
    (27906, 5120),
    (27906, 1536),
    (27908, 5120),
    (27908, 1536),
    (28205, 5120),
    (28205, 1536),
    (28411, 5120),
    (28411, 1536),
]


@pytest.mark.parametrize("M, K", test_parameters_act_quant_tmaalign)
def test_act_quant_tmaalign(M, K):
    block_size = 128 # block_size for kernel call

    # --- input data ---
    x = torch.randn(M, K, device='cuda', dtype=torch.bfloat16)
    # Non-contiguous input test (commented out as in original)
    # x = torch.randn(K, M, device='cuda', dtype=torch.bfloat16).transpose(0,1)

    # --- Triton Kernels ---
    y_orig, s_orig_tma = act_quant_vanilla(x, block_size=block_size) # Python TMA
    y_tma, s_tma = act_quant_TMA_aligned_B_ptr(x, block_size)        # Kernel TMA

    # --- reference implementation ---
    ref_tuple = per_token_cast_to_fp8(x)
    y_ref = ref_tuple[0]
    s_ref = ref_tuple[1]

    # --- Assert correctness ---
    # Validate y
    diff_y_tma_ref = calc_diff(y_tma, y_ref)
    assert diff_y_tma_ref < 0.001, f'M={M}, K={K} (y_tma vs y_ref), diff_y={diff_y_tma_ref:.5f}'
    diff_y_orig_tma = calc_diff(y_orig, y_tma)
    assert diff_y_orig_tma < 0.001, f'M={M}, K={K} (y_orig vs y_tma), diff_y={diff_y_orig_tma:.5f}'

    # Validate s (scales)
    s_tma_valid_data = s_tma[:M, :]
    s_tma_comparable = s_tma_valid_data.contiguous()
    s_orig_valid_data = s_orig_tma[:M, :]
    s_orig_comparable = s_orig_valid_data.contiguous()

    diff_s_tma_ref = calc_diff(s_tma_comparable, s_ref)
    assert diff_s_tma_ref < 0.001, f'M={M}, K={K} (s_tma vs s_ref), diff_s={diff_s_tma_ref:.5f}'
    diff_s_orig_tma = calc_diff(s_orig_comparable, s_tma_comparable)
    assert diff_s_orig_tma < 0.001, f'M={M}, K={K} (s_orig vs s_tma), diff_s={diff_s_orig_tma:.5f}'

    assert s_orig_tma.shape == s_tma.shape
    if s_tma.shape[1]!= 1: assert s_orig_tma.stride() == s_tma.stride()


if __name__ == '__main__':
    torch.manual_seed(0)
    print(f"Running all tests in {__file__} using pytest...")
    pytest.main(['-v', __file__])
