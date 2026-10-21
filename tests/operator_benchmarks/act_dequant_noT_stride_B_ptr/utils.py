import torch
import triton

from typing import Tuple

import time

# M N K    for gemm
TEST_SHAPES_GEMM = [
    # from deepGEEM gemm test shape
    (64,   7168, 576),
    (64,   2112, 7168),
    (64,   24576,1536),
    (64,   32768,512),
    (64,   7168, 16384),
    (64,   4096, 7168),
    (64,   7168, 2048),
    (128,  7168, 576),
    (128,  2112, 7168),
    (128,  24576,1536),
    (128,  32768,512),
    (128,  7168, 16384),
    (128,  4096, 7168),
    (128,  7168, 2048),
    (4096, 7168, 576),
    (4096, 2112, 7168),
    (4096, 24576,1536),
    (4096, 32768,512),
    (4096, 7168, 16384),
    (4096, 4096, 7168),
    (4096, 7168, 2048),
]

# M N K    for wgrad gemm
TEST_SHAPES_WGRAD_GEMM = [
    # from deepGEEM gemm test shape
    ( 7168,  2112, 4096),
    ( 1536, 24576, 4096),
    (  512, 32768, 4096),
    (16384,  7168, 4096),
    ( 7168,  4096, 4096),
    ( 2048,  7168, 4096),
    ( 7168,  2112, 8192),
    ( 1536, 24576, 8192),
    (  512, 32768, 8192),
    (16384,  7168, 8192),
    ( 7168,  4096, 8192),
    ( 2048,  7168, 8192),
]

# for act_quant/dequant/transpose_quant
distinct_MK =[
        #from TE float8_blockwise_scaling test shape
        # full tile cases
        (128, 128),
        (256, 256),
        (256, 1024),
        (1024, 256),
        # Padding required cases
        (256, 272),
        (303, 300),
        (305, 256),
        # Some larger tiles.
        (2000, 2000),
        (2048, 2000),
        (2000, 1024),
        (2048, 1024),

        #from deepGEEM gemm test shape
        (4096, 576),
        (4096, 7168),
        (4096, 1536),
        (4096, 512),
        (4096, 16384),
        (4096, 7168),
        (4096, 2048),
        # from deepGEMM grouped_gemm test shape
        (32768, 7168),
        (32768, 2048),
        (8192 , 7168),
        (8192 , 2048),
        (1024 , 7168),
        (1024 , 2048),
        
        # # others
        # (18400, 5120),
        # (18400, 1536),
        # (23296, 5120),
        # (23296, 1536),
        # (29100, 5120),
        # (29100, 1536),
]

distinct_NK = set((N, K) for M, N, K in TEST_SHAPES_GEMM)  #for weight_quant...

TEST_SHAPES_WEIGHT_MULTI_CAST = [
    (4, 4096, 7168),
    (4, 7168, 2048),
    (8, 4096, 7168),
    (8, 7168, 2048),
    (32, 4096, 7168),
    (32, 7168, 2048),
]

def get_tma_aligned_size(x: int, element_size: int) -> int:
    tma_alignment_bytes = 16
    if tma_alignment_bytes % element_size != 0:
         raise ValueError(f"Element size {element_size} is not compatible with TMA alignment {tma_alignment_bytes}")
    alignment = tma_alignment_bytes // element_size
    return ceil_div(x, alignment) * alignment

def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def per_block_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros((ceil_div(m, 128) * 128, ceil_div(n, 128) * 128), dtype=x.dtype, device=x.device)
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    x_scaled = (x_view * (448.0 / x_amax)).to(torch.float8_e4m3fn)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), (x_amax / 448.0).view(x_view.size(0), x_view.size(2))

def per_token_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    # assert x.is_contiguous()
    m, n = x.shape
    block_size = 128

    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)  # quantized FP8 tensor
    s = torch.empty((m, (n + block_size - 1) // block_size), device=x.device, dtype=torch.float32)  # scale factor

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
    
def calc_diff(x, y):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim

def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"
def supports_tma():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9 and triton.__version__ >="3.2.0"

# return the average e2e time (in ms)
def average_elapsed_time_e2e(callable_func, warmup=10, repeat=100):
    # warmup
    for _ in range(warmup):
        callable_func()
    torch.cuda.synchronize()
    
    # repeat
    start_time = time.time()
    for _ in range(repeat):
        callable_func()
    torch.cuda.synchronize()
    end_time = time.time()
    
    return (end_time - start_time) / repeat * 1000.0


# return the average kernel time (in ms)
def average_elapsed_time_kernel(callable_func, quantiles=[0.5, 0.2, 0.8], repeat=100):
    ms, min_ms, max_ms = triton.testing.do_bench(callable_func, quantiles=quantiles, rep=100)
    return ms

