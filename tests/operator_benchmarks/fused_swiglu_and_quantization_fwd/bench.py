import torch
import csv
import os
import transformer_engine
import transformer_engine_torch as tex
from megatron.core.fusions.fused_weighted_swiglu_quant import fused_weighted_swiglu_quant
from megatron.core.fp8_utils import fp8_quantize
from megatron.core.fusions.fused_bias_swiglu import weighted_swiglu
from megatron.core.enums import Fp8Recipe

seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

def perf_test_cuda_kernel(cuda_kernel_fn):
    if torch.cuda.is_available():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        # Warmup
        for _ in range(20):
            cuda_kernel_fn()
        torch.cuda.synchronize()
        
        start_event.record()
        for _ in range(100):
            cuda_kernel_fn()
        end_event.record()
        torch.cuda.synchronize()

        elapsed_time_ms = start_event.elapsed_time(end_event)
        return elapsed_time_ms / 100
    else:
        raise RuntimeError("CUDA is not available.")

def _benchmark_fwd_swiglu_quant(
    te_dtype, num_tokens, num_expert, hidden_size, topK, num_out_tokens, MOE_INTERMEDIATE_SIZE
):
    if num_out_tokens is None:
        num_out_tokens = num_tokens * topK

    dtype = torch.bfloat16 if te_dtype == tex.DType.kBFloat16 else torch.float16

    print(f"Fwd Benchmark | token:{num_tokens} hidden_size:{hidden_size} expert:{num_expert} topK:{topK} MOE_INTERMEDIATE_SIZE:{MOE_INTERMEDIATE_SIZE}")

    input = torch.rand((num_out_tokens, 2*MOE_INTERMEDIATE_SIZE), dtype=dtype).cuda()
    weights = torch.rand((num_out_tokens, 1), dtype=dtype).cuda()
    fp8_recipe = Fp8Recipe.blockwise

    def weight_swiglu_quant():
        weighted_swiglu_result = weighted_swiglu(input, weights)
        fp8_quantize(fp8_recipe, weighted_swiglu_result)

    def fusion_weight_swiglu_quant():
        fused_weighted_swiglu_quant(input, weights)

    t_naive = perf_test_cuda_kernel(weight_swiglu_quant)
    t_fused = perf_test_cuda_kernel(fusion_weight_swiglu_quant)

    torch.cuda.empty_cache()
    speedup = t_naive / t_fused
    
    print(f"  -> fwd: naive: {t_naive:.3f} ms,  fused: {t_fused:.3f} ms, speedup: {speedup:.2f}x")
    
    return {
        "fwd_naive_ms": round(t_naive, 3),
        "fwd_fused_ms": round(t_fused, 3),
        "fwd_speedup": round(speedup, 2),
    }

if __name__ == "__main__":
    te_dtype = tex.DType.kBFloat16
    configs = [
        (4096, 1280, 64,  7, 1024),   # 2B
        (4096, 2048, 64,  6, 1408),   # 16B
        (4096, 5120, 160, 6, 1536),   # 200B
        (4096, 7168, 256, 8, 2048),   # 600B
    ]

    header = ["config", "fwd_naive_ms", "fwd_fused_ms", "fwd_speedup"]
    rows = []

    print("Starting Forward SwiGLU+Quant Benchmark...\n")
    for num_tokens, hidden_size, num_expert, topK, moe_intermediate_size in configs:
        config_str = f"({num_tokens}, {hidden_size}, {num_expert}, {topK}, {moe_intermediate_size})"
        result = _benchmark_fwd_swiglu_quant(
            te_dtype, num_tokens, num_expert, hidden_size, topK, None, moe_intermediate_size
        )
        rows.append([
            config_str,
            result["fwd_naive_ms"],
            result["fwd_fused_ms"],
            result["fwd_speedup"],
        ])

    output_csv = os.path.join(os.path.dirname(__file__), "perf_data.csv")
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"\n[SUCCESS] Forward performance results saved to {output_csv}")