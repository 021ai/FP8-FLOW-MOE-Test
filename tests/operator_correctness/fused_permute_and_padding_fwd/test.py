import torch
import pytest
from typing import Dict

from transformer_engine.pytorch import (
    moe_permute_with_probs as te_permute_with_probs,
    moe_permute_and_pad_with_probs as te_permute_and_pad_with_probs,
)
from transformer_engine.pytorch.utils import is_bf16_compatible
import transformer_engine_torch as tex
from transformer_engine.pytorch import Fp8Padding

seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

def calc_diff(x, y):
    """全局统计误差 (Cosine Similarity 变体): 1 - 2*(x·y)/(x²+y²)"""
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

def assert_tensor_close_with_diff(name, actual, expected, tols, expert_offsets=None):
    """双重判定：全局误差 + 逐元素误差，并自带 Debug 溯源"""
    actual_f32, expected_f32 = actual.float(), expected.float()
    
    # 1. 全局统计误差
    diff_val = calc_diff(actual_f32, expected_f32).item()
    assert diff_val < 0.001, f"[{name}] Global calc_diff failed: {diff_val:.5f} (Threshold: 0.001)"
    
    # 2. 逐元素容差
    rtol, atol = tols['rtol'], tols['atol']
    are_close = torch.allclose(actual_f32, expected_f32, atol=atol, rtol=rtol)
    
    if not are_close:
        diff = torch.abs(actual_f32 - expected_f32)
        max_diff_val = diff.max().item()
        max_idx = torch.argmax(diff)
        coords = (max_idx // actual_f32.size(1), max_idx % actual_f32.size(1))
        
        # Debug 溯源
        debug_msg = f"\n[{name}] Element-wise allclose failed!\n"
        debug_msg += f"  Max diff: {max_diff_val:.6f} at coords {coords}\n"
        debug_msg += f"  Actual: {actual_f32[coords].item()}, Expected: {expected_f32[coords].item()}\n"
        
        # 如果提供了 expert_offsets，定位是哪个 expert 出了问题
        if expert_offsets is not None:
            expert_id = torch.searchsorted(expert_offsets, torch.tensor(coords[0], device=expert_offsets.device)).item()
            debug_msg += f"  This element belongs to Expert ID: {expert_id}\n"
            
        assert False, debug_msg


def _run_correctness_permute_pad(te_dtype, num_tokens, num_expert, hidden_size, topK, num_out_tokens, align_size=16):
    if topK > num_expert: pytest.skip("topK should be smaller than num_expert.")
    if num_out_tokens is None: num_out_tokens = num_tokens * topK

    print(f"\n[Correctness Permute+Pad] token:{num_tokens} hidden_size:{hidden_size} expert:{num_expert} topK:{topK}")
    
    dtype = torch.bfloat16 if te_dtype == tex.DType.kBFloat16 else torch.float16

    _tmp_tensor = torch.zeros((num_tokens * num_expert,))
    _tmp_tensor[: int(num_out_tokens)] = 1.0
    routing_map = torch.reshape(_tmp_tensor[torch.randperm(num_tokens * num_expert)], (num_tokens, num_expert)).bool().cuda()

    probs = torch.rand(num_tokens, num_expert).cuda() * routing_map
    probs = (probs / probs.sum(dim=1, keepdim=True)).to(dtype)
    probs.requires_grad_(True)

    tokens_per_expert = routing_map.sum(dim=0).cpu()
    target_tokens_per_expert = (torch.ceil(tokens_per_expert / align_size) * align_size).long()
    num_permute_pad_out_tokens = target_tokens_per_expert.sum().item()
    
    # 记录每个 Expert 在最终输出中的起始 Offset，用于 Debug
    expert_offsets = torch.cumsum(torch.cat([torch.tensor([0]), target_tokens_per_expert])[:-1], dim=0).cuda()
    
    permute_pad_fwd_input = torch.rand((num_tokens, hidden_size), dtype=dtype).cuda()
    permute_pad_fwd_input.requires_grad_(True)
    permute_pad_bwd_input = torch.rand((num_permute_pad_out_tokens, hidden_size), dtype=dtype).cuda()

    # --- 1. Naive Pipeline ---
    permuted_output, permuted_probs, row_id_map = te_permute_with_probs(
        permute_pad_fwd_input, probs, routing_map, num_out_tokens=num_out_tokens,
    )
    tokens_per_expert_list = tokens_per_expert.tolist()
    fp8_padding = Fp8Padding(num_expert, align_size)
    permuted_paded_output, _ = fp8_padding(permuted_output, tokens_per_expert_list)
    permuted_paded_probs, _ = fp8_padding(permuted_probs.unsqueeze(-1), tokens_per_expert_list)
    permuted_paded_output.backward(permute_pad_bwd_input, retain_graph=True)

    # --- 2. Fused Pipeline ---
    fusion_permute_and_pad_fwd_input = permute_pad_fwd_input.detach()
    fusion_permute_and_pad_fwd_input.requires_grad_(True)
    fusion_probs = probs.detach()
    fusion_probs.requires_grad_(True)

    (
        fusion_permuted_padded_output,
        fusion_permuted_padded_probs,
        fusion_row_id_map,
        pad_offsets,
        target_tokens_per_expert,
    ) = te_permute_and_pad_with_probs(
        fusion_permute_and_pad_fwd_input, fusion_probs, routing_map, tokens_per_expert, align_size,
    )
    fusion_permuted_padded_probs = fusion_permuted_padded_probs.unsqueeze(-1)
    fusion_permuted_padded_output.backward(permute_pad_bwd_input.detach(), retain_graph=True)

    # --- 3. 双重准则断言验证 ---
    tols = dtype_tols(te_dtype)

    assert_tensor_close_with_diff(
        "Fwd Output (Permuted+Padded)", fusion_permuted_padded_output, permuted_paded_output, tols, expert_offsets
    )
    assert_tensor_close_with_diff(
        "Fwd Probs", fusion_permuted_padded_probs, permuted_paded_probs, tols, expert_offsets
    )
    assert_tensor_close_with_diff(
        "Bwd Grad Input", fusion_permute_and_pad_fwd_input.grad, permute_pad_fwd_input.grad, tols
    )
    
    print("  -> Permute and Pad correctness PASSED (Dual criteria).")


_te_dtypes = [tex.DType.kBFloat16] if is_bf16_compatible() else []

@pytest.mark.parametrize("te_dtype", _te_dtypes)
@pytest.mark.parametrize("num_out_tokens", [None])
@pytest.mark.parametrize(
    "num_tokens, num_expert, hidden_size, topK",
    [
        (4096, 64, 1280, 7), (4096, 64, 2048, 6), (4096, 160, 5120, 6),
        (4096, 256, 7168, 8), (4096, 384, 8192, 8), (4096, 512, 9216, 8),
    ],
)
def test_permute_and_pad_correctness(te_dtype, num_tokens, num_expert, hidden_size, topK, num_out_tokens):
    _run_correctness_permute_pad(te_dtype, num_tokens, num_expert, hidden_size, topK, num_out_tokens)

if __name__ == "__main__":
    pytest.main(['-v', __file__])