import torch
import pytest
from typing import Dict, List

import transformer_engine
import transformer_engine_torch as tex
from megatron.core.fusions.fused_weighted_swiglu_quant import fused_weighted_swiglu_quant_back
from megatron.core.fp8_utils import fp8_quantize
from megatron.core.fusions.fused_bias_swiglu import weighted_swiglu_back
from megatron.core.enums import Fp8Recipe
from OpenMixOpl.triton import act_quant_B_ptr

seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# TE tensor dtypes
_te_dtypes: List[tex.DType] = [tex.DType.kBFloat16]

def calc_diff(x, y):
    """全局统计误差"""
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    if denominator == 0: return torch.tensor(0.0, device=x.device)
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim

def dtype_tols(te_dtype: tex.DType) -> Dict[str, float]:
    if te_dtype == tex.DType.kFloat32: return dict(rtol=1.0e-6, atol=1.0e-6)
    if te_dtype == tex.DType.kFloat16: return dict(rtol=3.0e-3, atol=1.0e-5)
    if te_dtype == tex.DType.kBFloat16: return dict(rtol=2.0e-2, atol=1.0e-5)
    if te_dtype in (tex.DType.kFloat8E5M2, tex.DType.kFloat8E4M3): return dict(rtol=2.0e-1, atol=1.0e-1)
    raise ValueError(f"Unsupported dtype ({te_dtype})")

def assert_tensor_close_with_diff(name, actual, expected, tols=None):
    """双重判定：全局误差 + 逐元素误差，并自带 Debug 溯源"""
    actual_f32, expected_f32 = actual.float(), expected.float()
    
    diff_val = calc_diff(actual_f32, expected_f32).item()
    assert diff_val < 0.001, f"[{name}] Global calc_diff failed: {diff_val:.5f} (Threshold: 0.001)"
    
    if tols != None:
        rtol, atol = tols['rtol'], tols['atol']
        are_close = torch.allclose(actual_f32, expected_f32, atol=atol, rtol=rtol)
        
        if not are_close:
            diff = torch.abs(actual_f32 - expected_f32)
            max_diff_val = diff.max().item()
            max_idx = torch.argmax(diff)
            if actual_f32.dim() == 2:
                coords = (max_idx // actual_f32.size(1), max_idx % actual_f32.size(1))
            else:
                coords = (max_idx.item(),)
                
            debug_msg = f"\n[{name}] Element-wise allclose failed!\n"
            debug_msg += f"  Max diff: {max_diff_val:.6f} at coords {coords}\n"
            debug_msg += f"  Actual: {actual_f32[coords].item()}, Expected: {expected_f32[coords].item()}\n"
            assert False, debug_msg


def _run_bwd_test(te_dtype, num_tokens, num_expert, hidden_size, topK, num_out_tokens, MOE_INTERMEDIATE_SIZE):
    if topK > num_expert:
        pytest.skip("topK should be smaller than the number of experts.")
    if num_out_tokens is None:
        num_out_tokens = num_tokens * topK

    print(f"\n[Correctness Bwd SwiGLU+Quant] token:{num_tokens} hidden_size:{hidden_size} topK:{topK} MOE_INTERMEDIATE:{MOE_INTERMEDIATE_SIZE}")

    if te_dtype == tex.DType.kFloat32: dtype = torch.float32
    elif te_dtype == tex.DType.kFloat16: dtype = torch.float16
    elif te_dtype == tex.DType.kBFloat16: dtype = torch.bfloat16
    else: pytest.skip("Invalid dtype.")

    input = torch.rand((num_out_tokens, 2 * MOE_INTERMEDIATE_SIZE), dtype=dtype).cuda()
    weights = torch.rand((num_out_tokens, 1), dtype=dtype).cuda()
    grad_output = torch.rand((num_out_tokens, MOE_INTERMEDIATE_SIZE), dtype=dtype).cuda()
    fp8_recipe = Fp8Recipe.blockwise

    # --- 1. Naive Pipeline ---
    ref_tmp, ref_wgrad = weighted_swiglu_back(grad_output, input, weights)
    ref_quantized, ref_scale = act_quant_B_ptr(ref_tmp)

    # --- 2. Fused Pipeline ---
    fused_quantized, fused_scale, fused_wgrad = fused_weighted_swiglu_quant_back(grad_output, input, weights)

    # --- 3. 双重准则断言验证 ---
    # a) 量化后的 FP8 梯度
    fp8_tols = dict(rtol=2.0e-1, atol=1.0e-1)
    assert_tensor_close_with_diff("Bwd Quantized FP8 Output", fused_quantized, ref_quantized, fp8_tols)
    
    # b) FP8 Scale
    if ref_scale is not None and fused_scale is not None:
        scale_tols = None
        assert_tensor_close_with_diff("Bwd FP8 Scale", fused_scale, ref_scale, scale_tols)
        
    # c) 权重梯度 (wgrad) - 通常为 BF16
    bf16_tols = dtype_tols(tex.DType.kBFloat16)
    assert_tensor_close_with_diff("Bwd Weight Gradient", fused_wgrad, ref_wgrad, bf16_tols)
        
    print("  -> SwiGLU Bwd correctness PASSED (Dual criteria).")


@pytest.mark.parametrize("te_dtype", _te_dtypes)
@pytest.mark.parametrize("num_out_tokens", [None])
@pytest.mark.parametrize(
    "num_tokens, hidden_size, num_expert, topK, MOE_INTERMEDIATE_SIZE",
    [
        (4096, 1280, 64,  7, 1024),
        (4096, 2048, 64,  6, 1408),
        (4096, 5120, 160, 6, 1536),
        (4096, 7168, 256, 8, 2048),
    ],
)
def test_swiglu_quant_bwd_correctness(te_dtype, num_tokens, num_expert, hidden_size, topK, num_out_tokens, MOE_INTERMEDIATE_SIZE):
    _run_bwd_test(te_dtype, num_tokens, num_expert, hidden_size, topK, num_out_tokens, MOE_INTERMEDIATE_SIZE)

if __name__ == "__main__":
    print("Running Backward correctness tests...")
    pytest.main(['-v', __file__])