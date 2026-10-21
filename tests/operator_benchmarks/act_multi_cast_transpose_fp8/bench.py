import torch
import time
import triton 
from typing import List, Tuple

# OpenMixOpl import
from OpenMixOpl.triton import act_multi_cast_transpose_fp8
from utils import (
    average_elapsed_time_e2e,
    average_elapsed_time_kernel 
)

# Transformer Engine imports
import transformer_engine
import transformer_engine_torch as tex
from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockQuantizer
from benchmark_csv import BenchmarkCSVWriter

# --- TE Quantizer ---
_fp8_block_quantizer_instance_for_multi = None
def TE_Fp8BlockQuantize_for_multi(tensor_to_quant: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    global _fp8_block_quantizer_instance_for_multi
    if _fp8_block_quantizer_instance_for_multi is None:
        with torch.no_grad():
            _fp8_block_quantizer_instance_for_multi = Float8BlockQuantizer(
                fp8_dtype=tex.DType.kFloat8E4M3,
                rowwise=True,
                columnwise=False,
                amax_epsilon=0.0,
                force_pow_2_scales=True,
                block_scaling_dim=1,
            )
    with torch.no_grad():
        quantized_obj = _fp8_block_quantizer_instance_for_multi(tensor_to_quant.contiguous())
    return quantized_obj._rowwise_data, quantized_obj._rowwise_scale_inv

def te_ref_multi_cast_transpose_fp8(x: torch.Tensor, m_splits: List[int]
                                   ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    x_subs = torch.split(x, m_splits, dim=0)
    y_list_te_final = []
    s_list_te_final = []
    for x_sub_i in x_subs:
        x_sub_i_T = x_sub_i.transpose(0, 1).contiguous()
        y_sub_te_i, s_inv_sub_te_i = TE_Fp8BlockQuantize_for_multi(x_sub_i_T)
        y_list_te_final.append(y_sub_te_i)
        s_list_te_final.append(s_inv_sub_te_i)
    return y_list_te_final, s_list_te_final

def test_perf_act_multi_cast_transpose_fp8():
    block_size_param = 128
    shapes_with_splits = [
        (512, 512, [128, 128, 128, 128]),
        (2048, 1024, [512, 512, 512, 512]),
        (4096, 1280, [1024, 1024, 1024, 1024]),
        (8192, 1024, [512, 512, 1024, 1024, 1024, 1024, 1536, 1536]),
        (31701, 1280, [2080, 1819, 3106, 2028, 2121, 413, 1301, 1344, 1366, 3675, 1855, 2048, 2392, 3511, 1595, 1047]),
        # deepgemm
        (4 * 8192, 7168, [8192] * 4),   
        (4 * 8192, 2048, [8192] * 4),    
        (8 * 4096, 7168, [4096] * 8),    
        (8 * 4096, 2048, [4096] * 8),    
        (32 * 256, 7168, [256] * 32),   
        (32 * 256, 2048, [256] * 32),  
    ]

    WARMUP = 20
    REPEAT_E2E = 100

    csv_writer = BenchmarkCSVWriter.for_script(
        "perf_data",
        fieldnames=[
            'M', 'K', 'm_splits',
            'triton_kernel_ms', 'te_kernel_ms', 'kernel_speedup',
        ],
    )

    DO_BENCH_QUANTILES = [0.5, 0.2, 0.8]
    DO_BENCH_REPEAT = 100

    col_width_shape = 24
    col_width_time = 22 
    col_width_speedup = 15 

    print("\nPerformance Test: Activation Multi-Split Cast & Transpose to FP8 (Triton vs TE Ref)")
    header = (f"| {'M, K, Splits':^{col_width_shape}} | "
              f"{'Triton Krn (ms)':^{col_width_time}} | {'TE Ref Krn (ms)':^{col_width_time}} | {'Krn S(Tri/TE)':^{col_width_speedup}} |")
    separator = '-' * len(header)
    print(separator)
    print(header)
    print(separator)

    if _fp8_block_quantizer_instance_for_multi is None:
        print("Pre-warming TE BlockQuantizer for multi-split test...")
        if torch.cuda.is_available():
            dummy_tensor_for_te = torch.randn(128, 128, device='cuda', dtype=torch.bfloat16)
            TE_Fp8BlockQuantize_for_multi(dummy_tensor_for_te)
            print("TE BlockQuantizer instance warmed up.")
            torch.cuda.synchronize()
        else:
            print("CUDA not available, TE quantizer pre-warming skipped.")

    for M, K, m_splits in shapes_with_splits:
        shape_str = f"{M}, {K}, {len(m_splits)}"
        if not torch.cuda.is_available():
            print(f"| {shape_str:^{col_width_shape}} | {'SKIPPING (CUDA not available)':^{col_width_time*4 + col_width_speedup*2 + 9}} |")
            continue
        try:
            x = torch.randn(M, K, device='cuda', dtype=torch.bfloat16)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"| {shape_str:^{col_width_shape}} | "
                      f"{'OOM (Input Gen)':^{col_width_time*4 + col_width_speedup*2 + 9}} |")
                continue
            else:
                raise e

        ms_triton_e2e, ms_te_ref_e2e = "N/A", "N/A"
        ms_triton_kernel, ms_te_ref_kernel = "N/A", "N/A"

        try:
            func_call_triton = lambda: act_multi_cast_transpose_fp8(
                x, m_splits, block_size=block_size_param
            )
            ms_triton_e2e = average_elapsed_time_e2e(func_call_triton, warmup=WARMUP, repeat=REPEAT_E2E)
            try:
                ms_triton_kernel = average_elapsed_time_kernel(func_call_triton, quantiles=DO_BENCH_QUANTILES, repeat=DO_BENCH_REPEAT)
            except Exception as e_bench_triton:
                print(f"Warning: do_bench for Triton (multi-split) failed for shape ({shape_str}): {e_bench_triton}")
                ms_triton_kernel = "Err_Bench"


            func_call_te_ref = lambda: te_ref_multi_cast_transpose_fp8(
                x.clone(), m_splits
            )
            ms_te_ref_e2e = average_elapsed_time_e2e(func_call_te_ref, warmup=WARMUP, repeat=REPEAT_E2E)
            try:
                ms_te_ref_kernel = average_elapsed_time_kernel(func_call_te_ref, quantiles=DO_BENCH_QUANTILES, repeat=DO_BENCH_REPEAT)
            except Exception as e_bench_te:
                print(f"Warning: do_bench for TE Ref (multi-split) failed for shape ({shape_str}): {e_bench_te}")
                ms_te_ref_kernel = "Err_Bench"
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                oom_marker = "OOM"
                if isinstance(ms_triton_e2e, str): ms_triton_e2e = oom_marker
                if isinstance(ms_te_ref_e2e, str): ms_te_ref_e2e = oom_marker
                if isinstance(ms_triton_kernel, str) and ms_triton_kernel == "N/A": ms_triton_kernel = oom_marker
                if isinstance(ms_te_ref_kernel, str) and ms_te_ref_kernel == "N/A": ms_te_ref_kernel = oom_marker
                print(f"| {shape_str:^{col_width_shape}} | "
                      f"{str(ms_triton_e2e):^{col_width_time}} | {str(ms_te_ref_e2e):^{col_width_time}} | {'N/A':^{col_width_speedup}} | "
                      f"{str(ms_triton_kernel):^{col_width_time}} | {str(ms_te_ref_kernel):^{col_width_time}} | {'N/A':^{col_width_speedup}} |")
                del x
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                continue
            else:
                raise e

        speedup_e2e_triton_vs_te = (ms_te_ref_e2e / ms_triton_e2e
                                 if isinstance(ms_te_ref_e2e, float) and ms_te_ref_e2e > 0 and \
                                    isinstance(ms_triton_e2e, float) and ms_triton_e2e > 0
                                 else float('nan'))
        
        speedup_kernel_triton_vs_te = (ms_te_ref_kernel / ms_triton_kernel
                                    if isinstance(ms_te_ref_kernel, float) and ms_te_ref_kernel > 0 and \
                                       isinstance(ms_triton_kernel, float) and ms_triton_kernel > 0
                                    else float('nan'))

        csv_writer.add_row(
            M=M,
            K=K,
            m_splits=m_splits,
            triton_kernel_ms=ms_triton_kernel,
            te_kernel_ms=ms_te_ref_kernel,
            kernel_speedup=speedup_kernel_triton_vs_te,
        )

        def format_val(val, width, prec=6):
            if isinstance(val, float): return f"{val:^{width}.{prec}f}"
            return f"{str(val):^{width}}"

        print(f"| {shape_str:^{col_width_shape}} | "
              f"{format_val(ms_triton_kernel, col_width_time)} | "
              f"{format_val(ms_te_ref_kernel, col_width_time)} | "
              f"{format_val(speedup_kernel_triton_vs_te, col_width_speedup, prec=2)}x |")

        del x
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(separator)
    csv_path = csv_writer.save()
    print(f'CSV results saved to: {csv_path}')

if __name__ == "__main__":
    torch.manual_seed(0)
    test_perf_act_multi_cast_transpose_fp8()