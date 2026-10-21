import torch

import OpenMixOpl
from OpenMixOpl.triton import weight_multi_cast_transpose_fused, weight_multi_cast_transpose_fused_2d_3d

from utils import TEST_SHAPES_WEIGHT_MULTI_CAST, ceil_div, average_elapsed_time_e2e, average_elapsed_time_kernel

from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockQuantizer
from transformer_engine.pytorch.constants import TE_DType
from benchmark_csv import BenchmarkCSVWriter

def TE_quant(x, flag, te_quantizer):
    num = len(x)
    N, K = x[0].shape
    block_size = 128

    n_blocks = ceil_div(N, block_size)
    k_blocks = ceil_div(K, block_size)
    te_fp8_out = te_quantizer.make_empty(
        (N, K),
        dtype=x[0].dtype,
        device=x[0].device,
        requires_grad=False
    )
    
    if flag == 1:
        te_fp8_y = torch.empty((num, N, K), device='cuda', dtype=torch.float8_e4m3fn)
        te_fp8_si = torch.empty((num, n_blocks, k_blocks), device='cuda', dtype=torch.float32)

        for i in range(num):
            te_fp8_out = te_quantizer.update_quantized(x[i], te_fp8_out)
            te_fp8_y[i] = te_fp8_out._rowwise_data.view(dtype=torch.float8_e4m3fn)[:N, :K]
            te_fp8_si[i] = te_fp8_out._rowwise_scale_inv[:n_blocks, :k_blocks]
        return te_fp8_y, te_fp8_si

    elif flag == 2:
        te_fp8_y = torch.empty((num, N, K), device='cuda', dtype=torch.float8_e4m3fn)
        te_fp8_si = torch.empty((num, n_blocks, k_blocks), device='cuda', dtype=torch.float32)
        te_fp8_y_T = torch.empty((num, K, N), device='cuda', dtype=torch.float8_e4m3fn)
        te_fp8_si_T = torch.empty((num, k_blocks, n_blocks), device='cuda', dtype=torch.float32)

        for i in range(num):
            te_fp8_out = te_quantizer.update_quantized(x[i], te_fp8_out)
            te_fp8_y[i] = te_fp8_out._rowwise_data.view(dtype=torch.float8_e4m3fn)[:N, :K]
            te_fp8_si[i] = te_fp8_out._rowwise_scale_inv[:n_blocks, :k_blocks]
            te_fp8_y_T[i] = te_fp8_out._columnwise_data.view(dtype=torch.float8_e4m3fn)[:K, :N]
            te_fp8_si_T[i] = te_fp8_out._columnwise_scale_inv[:k_blocks, :n_blocks]
        return te_fp8_y, te_fp8_si, te_fp8_y_T, te_fp8_si_T

    elif flag == 3:
        te_fp8_y_T = torch.empty((num, K, N), device='cuda', dtype=torch.float8_e4m3fn)
        te_fp8_si_T = torch.empty((num, k_blocks, n_blocks), device='cuda', dtype=torch.float32)

        for i in range(num):
            te_fp8_out = te_quantizer.update_quantized(x[i], te_fp8_out)
            te_fp8_y_T[i] = te_fp8_out._columnwise_data.view(dtype=torch.float8_e4m3fn)[:K, :N]
            te_fp8_si_T[i] = te_fp8_out._columnwise_scale_inv[:k_blocks, :n_blocks]
        return te_fp8_y_T, te_fp8_si_T


def benchmark_weight_multi_cast_transpose_fused(flag):
    block_size = 128
    
    shapes = TEST_SHAPES_WEIGHT_MULTI_CAST
    csv_writer = BenchmarkCSVWriter.for_script(
        "perf_data",
        fieldnames=[
            'num', 'N', 'K', 'triton_kernel_ms', 'te_kernel_ms', 'kernel_speedup',
        ],
    )

    # e2e perf param
    WARMUP = 20
    REPEAT = 100

    # kernel perf param
    quantiles = [0.5, 0.2, 0.8]
    repeat = 100

    if flag == 1:
        rowwize = True
        columnwise = False
    elif flag == 2:
        rowwize = True
        columnwise = True
    elif flag == 3:
        rowwize = True
        columnwise = True

    # te_quantizer
    te_quantizer = Float8BlockQuantizer(
        fp8_dtype=TE_DType[torch.float8_e4m3fn],
        rowwise=rowwize,        # return quant
        columnwise=columnwise,  # return quant.T
        amax_epsilon=0,
        force_pow_2_scales=False,
        block_scaling_dim=2,    # per-block
    )
    te_quantizer.internal = True

    print("benchmark_weight_multi_cast_transpose_fused_flag_" + str(flag))
    print('=' * 180)
    print(f"| {'num':^6} | {'N':^8} | {'K':^8} | {'triton kernel time(ms)':^25} | {'te kernel time(ms)':^25} | {'kernel speed up':^15} |")
    print('=' * 180)

    # cast
    for num, N, K in shapes:
        
        # Generate input
        x = []
        for _ in range(num):
            x.append(torch.randn(N, K, device='cuda', dtype=torch.bfloat16))

        # Performance
        # baseline te
        func_call_te = lambda: TE_quant(x, flag, te_quantizer)
        
        ms_te_e2e = average_elapsed_time_e2e(func_call_te, warmup=WARMUP, repeat=REPEAT)
        ms_te_kernel = average_elapsed_time_kernel(func_call_te, quantiles=quantiles, repeat=repeat)
        
        # fused triton
        func_call_triton = lambda: weight_multi_cast_transpose_fused(x, flag, block_size)

        ms_triton_e2e = average_elapsed_time_e2e(func_call_triton, warmup=WARMUP, repeat=REPEAT)        
        ms_triton_kernel = average_elapsed_time_kernel(func_call_triton, quantiles=quantiles, repeat=repeat)

        csv_writer.add_row(
            num=num,
            N=N,
            K=K,
            triton_kernel_ms=ms_triton_kernel,
            te_kernel_ms=ms_te_kernel,
            kernel_speedup=ms_te_kernel / ms_triton_kernel,
        )

        # Print
        print(f"| {num:^6} | {N:^8} | {K:^8} | {ms_triton_kernel:^25.6} | {ms_te_kernel:^25.6} | {ms_te_kernel / ms_triton_kernel:^15.6} |")

    print('=' * 180)

    csv_path = csv_writer.save()
    print(f'CSV results saved to: {csv_path}')


if __name__ == "__main__":

    benchmark_weight_multi_cast_transpose_fused(flag=1)
