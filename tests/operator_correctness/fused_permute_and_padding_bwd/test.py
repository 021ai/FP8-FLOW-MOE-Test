import torch
import pytest
from typing import Dict

from transformer_engine.pytorch import (
    moe_permute_with_probs as te_permute_with_probs,
    moe_permute_and_pad_with_probs as te_permute_and_pad_with_probs,
    moe_unpermute as te_unpermute,
)
from transformer_engine.pytorch.utils import is_bf16_compatible
import transformer_engine_torch as tex
from transformer_engine.pytorch import Fp8Unpadding

seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

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

def assert_tensor_close_with_diff(name, actual, expected, tols):
    """双重判定：全局误差 + 逐元素误差"""
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
        
        debug_msg = f"\n[{name}] Element-wise allclose failed!\n"
        debug_msg += f"  Max diff: {max_diff_val:.6f} at coords {coords} (Original Token ID: {coords[0]})\n"
        debug_msg += f"  Actual: {actual_f32[coords].item()}, Expected: {expected_f32[coords].item()}\n"
        assert False, debug_msg


def _run_correctness_unpermute_unpad(te_dtype, num_tokens, num_expert, hidden_size, topK, num_out_tokens, align_size=16):
    if topK > num_expert: pytest.skip("topK should be smaller than num_expert.")
    if num_out_tokens is None: num_out_tokens = num_tokens * topK

    print(f"\n[Correctness Unpermute+Unpad] token:{num_tokens} hidden_size:{hidden_size} expert:{num_expert} topK:{topK}")
    
    dtype = torch.bfloat16 if te_dtype == tex.DType.kBFloat16 else torch.float16

    _tmp_tensor = torch.zeros((num_tokens * num_expert,))
    _tmp_tensor[: int(num_out_tokens)] = 1.0
    routing_map = torch.reshape(_tmp_tensor[torch.randperm(num_tokens * num_expert)], (num_tokens, num_expert)).bool().cuda()

    probs = torch.rand(num_tokens, num_expert).cuda() * routing_map
    probs = (probs / probs.sum(dim=1, keepdim=True)).to(dtype)

    tokens_per_expert = routing_map.sum(dim=0).cpu()
    target_tokens_per_expert = (torch.ceil(tokens_per_expert / align_size) * align_size).long()
    num_permute_pad_out_tokens = target_tokens_per_expert.sum().item()
    
    permute_pad_fwd_input = torch.rand((num_tokens, hidden_size), dtype=dtype).cuda()
    restore_shape = permute_pad_fwd_input.shape
    unpermute_unpad_bwd_input = torch.rand((num_tokens, hidden_size), dtype=dtype).cuda()

    # 准备 Fwd 数据（获取 row_id_map 和 pad_offsets）
    _, _, fusion_row_id_map, fusion_pad_offsets, _ = te_permute_and_pad_with_probs(
        permute_pad_fwd_input, probs, routing_map, tokens_per_expert, align_size,
    )
    
    unpermute_unpad_fwd_input_naive = torch.rand((num_permute_pad_out_tokens, hidden_size), dtype=dtype).cuda()
    unpermute_unpad_fwd_input_naive.requires_grad_(True)
    fusion_unpermute_unpad_fwd_input = unpermute_unpad_fwd_input_naive.detach()
    fusion_unpermute_unpad_fwd_input.requires_grad_(True)

    # --- 1. Naive Pipeline (Unpad -> Unpermute) ---
    fp8_unpadding = Fp8Unpadding(num_expert, align_size)
    tokens_per_expert_list = tokens_per_expert.tolist()
    
    unpaded_output = fp8_unpadding(unpermute_unpad_fwd_input_naive, tokens_per_expert_list)
    unpermuted_unpaded_output = te_unpermute(unpaded_output, fusion_row_id_map, restore_shape=restore_shape)
    unpermuted_unpaded_output.backward(unpermute_unpad_bwd_input, retain_graph=True)

    # --- 2. Fused Pipeline (Unpermute with pad_offsets) ---
    fusion_unpermuted_unpaded_output = te_unpermute(
        fusion_unpermute_unpad_fwd_input, fusion_row_id_map, restore_shape=restore_shape, pad_offsets=fusion_pad_offsets,
    )
    fusion_unpermuted_unpaded_output.backward(unpermute_unpad_bwd_input.detach(), retain_graph=True)

    # --- 3. 双重准则断言验证 ---
    tols = dtype_tols(te_dtype)

    assert_tensor_close_with_diff(
        "Fwd Output (Unpermuted+Unpadded)", fusion_unpermuted_unpaded_output, unpermuted_unpaded_output, tols
    )
    assert_tensor_close_with_diff(
        "Bwd Grad Input", fusion_unpermute_unpad_fwd_input.grad, unpermute_unpad_fwd_input_naive.grad, tols
    )
    
    print("  -> Unpermute and Unpad correctness PASSED (Dual criteria).")


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
def test_unpermute_and_unpad_correctness(te_dtype, num_tokens, num_expert, hidden_size, topK, num_out_tokens):
    _run_correctness_unpermute_unpad(te_dtype, num_tokens, num_expert, hidden_size, topK, num_out_tokens)

if __name__ == "__main__":
    pytest.main(['-v', __file__])