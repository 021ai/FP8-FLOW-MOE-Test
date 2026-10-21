import torch
import time
import triton 

# OpenMixOpl imports
from OpenMixOpl.triton import act_dequant_noT_stride_B_ptr
from utils import (
    distinct_MK,
    per_token_cast_to_fp8,
    average_elapsed_time_e2e,
    average_elapsed_time_kernel,
    ceil_div
)

# Transformer Engine imports
import transformer_engine
import transformer_engine_torch as tex
from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockQuantizer, Float8BlockwiseQTensor
from benchmark_csv import BenchmarkCSVWriter

_fp8_block_quantizer_qtensor_instance = None
def TE_Fp8BlockQuantize_ReturnQTensor(tensor: torch.Tensor) -> Float8BlockwiseQTensor:
    global _fp8_block_quantizer_qtensor_instance
    if _fp8_block_quantizer_qtensor_instance is None:
        with torch.no_grad():
            _fp8_block_quantizer_qtensor_instance = Float8BlockQuantizer(
                fp8_dtype=tex.DType.kFloat8E4M3,
                rowwise=True,
                columnwise=False,
                amax_epsilon=0.0,
                force_pow_2_scales=True,
                block_scaling_dim=1,
            )
    with torch.no_grad():
        quantized_tensor = _fp8_block_quantizer_qtensor_instance(tensor.contiguous())
    return quantized_tensor

def test_perf_act_dequant_noT_stride_B_ptr():
    block_size = 128
    triton_fp8type = torch.float8_e4m3fn

    shapes = distinct_MK
    WARMUP = 20
    REPEAT_E2E = 100
    output_orig_dtype = torch.bfloat16

    csv_writer = BenchmarkCSVWriter.for_script(
        "perf_data",
        fieldnames=[
            'M', 'K', 'triton_kernel_ms', 'te_kernel_ms', 'kernel_speedup',
        ],
    )

    DO_BENCH_QUANTILES = [0.5, 0.2, 0.8]
    DO_BENCH_REPEAT = 100

    col_width_shape = 8
    col_width_time = 22
    col_width_speedup = 15

    print("\nPerformance Test: Block Dequantization (act_dequant_noT_stride_B_ptr vs TE Block Dequant)")
    header = (f"| {'M':^{col_width_shape}} | {'K':^{col_width_shape}} | "
              f"{'Triton Krn (ms)':^{col_width_time}} | {'TE BQ Krn (ms)':^{col_width_time}} | {'Krn S(Tri/BQ)':^{col_width_speedup}} |")
    separator = '-' * len(header)
    print(separator)
    print(header)
    print(separator)

    if _fp8_block_quantizer_qtensor_instance is None:
        print("Pre-warming TE BlockQuantizer instance...")
        if torch.cuda.is_available():
            dummy_tensor = torch.randn(16, 128, device='cuda', dtype=torch.bfloat16)
            TE_Fp8BlockQuantize_ReturnQTensor(dummy_tensor)
            print("TE BlockQuantizer instance warmed up.")
            torch.cuda.synchronize()
        else:
            print("CUDA not available, TE quantizer pre-warming skipped.")

    for M, K in shapes: 
        if not torch.cuda.is_available():
            print(f"| {M:^{col_width_shape}} | {K:^{col_width_shape}} | {'SKIPPING (CUDA not available)':^{col_width_time*4 + col_width_speedup*2 + 9}} |")
            continue
        try:
            x_original_bf16 = torch.randn(M, K, device='cuda', dtype=torch.bfloat16)
            x_fp8_for_dequant, _ = per_token_cast_to_fp8(x_original_bf16.clone())
            x_fp8_for_dequant = x_fp8_for_dequant.contiguous()
            k_blocks = ceil_div(K, block_size)
            M_aligned = ceil_div(M, 4) * 4
            s_row_major_temp = torch.rand((k_blocks, M_aligned), dtype=torch.float32 , device='cuda') * 0.01 + 0.1
            s_row_major_temp = s_row_major_temp.contiguous()
            x_scales_input_triton = s_row_major_temp.T
            te_quantized_tensor_obj = TE_Fp8BlockQuantize_ReturnQTensor(x_original_bf16.clone())
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"| {M:^{col_width_shape}} | {K:^{col_width_shape}} | "
                      f"{'OOM (Input Gen)':^{col_width_time*4 + col_width_speedup*2 + 9}} |")
                continue
            else:
                raise e

        ms_triton_bq_e2e, ms_te_bq_e2e = "N/A", "N/A"
        ms_triton_kernel, ms_te_bq_kernel = "N/A", "N/A"

        try:
            func_call_triton = lambda: act_dequant_noT_stride_B_ptr(
                x_fp8_for_dequant, x_scales_input_triton, 
                block_size=block_size, fp8type=triton_fp8type
            )
            ms_triton_bq_e2e = average_elapsed_time_e2e(func_call_triton, warmup=WARMUP, repeat=REPEAT_E2E)
            try:
                ms_triton_kernel = average_elapsed_time_kernel(func_call_triton, quantiles=DO_BENCH_QUANTILES, repeat=DO_BENCH_REPEAT)
            except Exception as e_bench_triton:
                print(f"Warning: do_bench for Triton (noT_stride) failed for shape ({M},{K}): {e_bench_triton}")
                ms_triton_kernel = "Err_Bench"

            func_call_te_bq = lambda: te_quantized_tensor_obj.dequantize(dtype=output_orig_dtype)
            ms_te_bq_e2e = average_elapsed_time_e2e(func_call_te_bq, warmup=WARMUP, repeat=REPEAT_E2E)
            try:
                ms_te_bq_kernel = average_elapsed_time_kernel(func_call_te_bq, quantiles=DO_BENCH_QUANTILES, repeat=DO_BENCH_REPEAT)
            except Exception as e_bench_te:
                print(f"Warning: do_bench for TE BQ (dequant) failed for shape ({M},{K}): {e_bench_te}")
                ms_te_bq_kernel = "Err_Bench"
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                oom_marker = "OOM"
                if isinstance(ms_triton_bq_e2e, str): ms_triton_bq_e2e = oom_marker
                if isinstance(ms_te_bq_e2e, str): ms_te_bq_e2e = oom_marker
                if isinstance(ms_triton_kernel, str) and ms_triton_kernel == "N/A": ms_triton_kernel = oom_marker
                if isinstance(ms_te_bq_kernel, str) and ms_te_bq_kernel == "N/A": ms_te_bq_kernel = oom_marker
                print(f"| {M:^{col_width_shape}} | {K:^{col_width_shape}} | "
                      f"{str(ms_triton_bq_e2e):^{col_width_time}} | {str(ms_te_bq_e2e):^{col_width_time}} | {'N/A':^{col_width_speedup}} | "
                      f"{str(ms_triton_kernel):^{col_width_time}} | {str(ms_te_bq_kernel):^{col_width_time}} | {'N/A':^{col_width_speedup}} |")
                del x_original_bf16, x_fp8_for_dequant, s_row_major_temp, x_scales_input_triton, te_quantized_tensor_obj
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                continue
            else:
                raise e
        
        speedup_e2e_triton_vs_te_bq = (ms_te_bq_e2e / ms_triton_bq_e2e
                                 if isinstance(ms_te_bq_e2e, float) and ms_te_bq_e2e > 0 and \
                                    isinstance(ms_triton_bq_e2e, float) and ms_triton_bq_e2e > 0
                                 else float('nan'))
        
        speedup_kernel_triton_vs_te_bq = (ms_te_bq_kernel / ms_triton_kernel
                                    if isinstance(ms_te_bq_kernel, float) and ms_te_bq_kernel > 0 and \
                                       isinstance(ms_triton_kernel, float) and ms_triton_kernel > 0
                                    else float('nan'))

        csv_writer.add_row(
            M=M,
            K=K,
            triton_kernel_ms=ms_triton_kernel,
            te_kernel_ms=ms_te_bq_kernel,
            kernel_speedup=speedup_kernel_triton_vs_te_bq,
        )

        def format_val(val, width, prec=6):
            if isinstance(val, float): return f"{val:^{width}.{prec}f}"
            return f"{str(val):^{width}}"

        print(f"| {M:^{col_width_shape}} | {K:^{col_width_shape}} | "
              f"{format_val(ms_triton_kernel, col_width_time)} | "
              f"{format_val(ms_te_bq_kernel, col_width_time)} | "
              f"{format_val(speedup_kernel_triton_vs_te_bq, col_width_speedup, prec=2)}x |")

        del x_original_bf16, x_fp8_for_dequant, s_row_major_temp, x_scales_input_triton, te_quantized_tensor_obj
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(separator)
    csv_path = csv_writer.save()
    print(f'CSV results saved to: {csv_path}')

if __name__ == "__main__":
    torch.manual_seed(0)
    test_perf_act_dequant_noT_stride_B_ptr()