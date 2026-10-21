from typing import Tuple
import pytest
import torch
import triton
import triton.language as tl
import time

import OpenMixOpl
from  OpenMixOpl.triton import act_multi_cast_transpose_fp8
PERF =  True
def calc_diff(x, y):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim
# PyTorch reference implementation for FP8 quantization
def per_token_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    # assert x.is_contiguous()
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

@pytest.mark.parametrize("input_contiguous", [True, False])
@pytest.mark.parametrize(
    "M, K, m_splits",  
    [
        (1024, 768, [256, 512, 256]),   # Example: M=1024, K=768, split into [256, 512, 256]
        (512, 512, [128, 128, 128, 128]),  
        (2048, 1024, [512, 512, 512, 512]),
        (1024, 2048, [256, 768]), 
        (768, 768, [768]),  
        (512, 256, [100, 200, 212]), 
        (1536, 2048, [512, 512, 512]),
        (2080 + 1819 + 3106 + 2028 + 2121 + 413 + 1301 + 1344 + 1366 + 3675 + 1855 + 2048 + 2392 + 3511 + 1595 + 1047, 1280, [
            2080,
            1819,
            3106,
            2028,
            2121,
            413,
            1301,
            1344,
            1366,
            3675,
            1855,
            2048,
            2392,
            3511,
            1595,
            1047,
        ]),
        
    ],
)
def test_act_multi_cast_transpose_fp8(M, K, m_splits,input_contiguous):
    """Verify act quantization with different [M, K] shapes."""
    torch.manual_seed(41)
    block_size = 128

    # Generate input 
    if input_contiguous:
        x = torch.randn(M, K, device='cuda', dtype=torch.bfloat16)
    else:
        x = torch.randn(K, M, device='cuda', dtype=torch.bfloat16)
        x = x.transpose(0,1)

    # --- Triton Kernel execution ---
    y_list_fused, s_list_fused = act_multi_cast_transpose_fp8(x, m_splits, block_size)

     # --- Reference implementation ---
    x_splits = torch.split(x, m_splits, dim=0)
    y_list_ref = []
    s_list_ref = []

    for x_sub in x_splits:
        x_transpose = x_sub.transpose(0, 1)
        y_sub_ref, s_sub_ref = per_token_cast_to_fp8(x_transpose)
        y_list_ref.append(y_sub_ref)
        s_list_ref.append(s_sub_ref)
    
    # --- Assert correctness ---
    for i in range(len(y_list_fused)):
        y_fused = y_list_fused[i]
        y_ref = y_list_ref[i]
        s_fused = s_list_fused[i]
        s_ref = s_list_ref[i]

        # Check shapes
        assert y_fused.shape == y_ref.shape, \
            f"Split {i}, Contiguous: {input_contiguous}: y shapes mismatch: {y_fused.shape} vs {y_ref.shape}"
        assert s_fused.shape == s_ref.shape, \
            f"Split {i}, Contiguous: {input_contiguous}: s shapes mismatch: {s_fused.shape} vs {s_ref.shape}"

        # Check quantization and scales
        diff_y = calc_diff(y_fused, y_ref) 
        assert diff_y < 0.001, f'Split {i}, Contiguous: {input_contiguous}: diff_quant={diff_y:.5f}'
        diff_s = calc_diff(s_fused, s_ref)
        assert diff_s < 0.001, f'Split {i}, Contiguous: {input_contiguous}: diff_scale={diff_s:.5f}'

if __name__ == '__main__':
    torch.manual_seed(0) 
    print(f"Running all tests in {__file__} using pytest...")
    pytest.main(['-v', __file__])
