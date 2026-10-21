import torch
import triton
import triton.language as tl
from typing import Tuple
import pytest

import OpenMixOpl
from OpenMixOpl.triton import weight_multi_cast_transpose_fused
from OpenMixOpl.triton import weight_multi_cast_transpose_fused_2d_3d


def ceil_div(a, b):
    return (a + b - 1) // b

def calc_diff(x, y):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim

TEST_SHAPES_WEIGHT_MULTI_CAST = [
    (4, 4096, 7168),
    (4, 7168, 2048),
    (8, 4096, 7168),
    (8, 7168, 2048),
    (32, 4096, 7168),
    (32, 7168, 2048),
]

def per_block_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros((ceil_div(m, 128) * 128, ceil_div(n, 128) * 128), dtype=x.dtype, device=x.device)
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    x_scaled = (x_view * (448.0 / x_amax)).to(torch.float8_e4m3fn)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), (x_amax / 448.0).view(x_view.size(0), x_view.size(2))


@pytest.mark.parametrize("inp_shapes", TEST_SHAPES_WEIGHT_MULTI_CAST)
def test_weight_multi_cast_transpose_fused(inp_shapes):

    block_size = 128

    num, N, K = inp_shapes
    n_blocks = ceil_div(N, block_size)
    k_blocks = ceil_div(K, block_size)

    # Generate input
    x = []
    for i in range(num):
        tensor = torch.randn(N, K, device='cuda', dtype=torch.bfloat16)
        x.append(tensor)
    
    # Reference implementation
    y_ref = torch.empty((num, N, K), device='cuda', dtype=torch.float8_e4m3fn)
    s_ref = torch.empty((num, n_blocks, k_blocks), device='cuda', dtype=torch.float32)
    y_T_ref = torch.empty((num, K, N), device='cuda', dtype=torch.float8_e4m3fn)
    s_T_ref = torch.empty((num, k_blocks, n_blocks), device='cuda', dtype=torch.float32)

    for i in range(len(x)):
        y, s = per_block_cast_to_fp8(x[i])
        y_ref[i] = y
        s_ref[i] = s
        y_T_ref[i] = y.T
        s_T_ref[i] = s.T

    # Fused implementation
    y_out1, s_out1 = weight_multi_cast_transpose_fused(x, 1, block_size)

    y_out2, s_out2, y_T_out2, s_T_out2 = weight_multi_cast_transpose_fused(x, 2, block_size)

    y_T_out3, s_T_out3 = weight_multi_cast_transpose_fused(x, 3, block_size)


    # Verify
    assert calc_diff(y_out1, y_ref) < 0.001, f'The difference is too significant: {y_out1}, {y_ref}'
    torch.testing.assert_close(s_out1.to(torch.float32), s_ref.to(torch.float32), atol=1e-3, rtol=1e-3)

    assert calc_diff(y_out2, y_ref) < 0.001, f'The difference is too significant: {y_out2}, {y_ref}'
    torch.testing.assert_close(s_out2.to(torch.float32), s_ref.to(torch.float32), atol=1e-3, rtol=1e-3)
    assert calc_diff(y_T_out2, y_T_ref) < 0.001, f'The difference is too significant: {y_T_out2}, {y_T_ref}'
    torch.testing.assert_close(s_T_out2.to(torch.float32), s_T_ref.to(torch.float32), atol=1e-3, rtol=1e-3)

    assert calc_diff(y_T_out3, y_T_ref) < 0.001, f'The difference is too significant: {y_T_out3}, {y_T_ref}'
    torch.testing.assert_close(s_T_out3.to(torch.float32), s_T_ref.to(torch.float32), atol=1e-3, rtol=1e-3)


if __name__ == '__main__':
    torch.manual_seed(0) 
    print(f"Running all tests in {__file__} using pytest...")
    pytest.main(['-v', __file__])