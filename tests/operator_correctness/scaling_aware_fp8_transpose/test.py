import random
import math
import torch
import pytest
from typing import List

import transformer_engine
import transformer_engine_torch as tex
from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockQuantizer

# 尝试导入 Triton 的被测算子实现 (请确保路径正确)
# 假设被测算子实现在 benchmark 所在的文件中，或者您将其单独抽出
try:
    from transformer_engine.pytorch.triton.blockwise_scaling_aware_fp8_transpose import blockwise_scaling_aware_fp8_transpose
except ImportError:
    # 仅作占位提示，请替换为实际的被测算子模块导入路径
    raise ImportError("Please import the `blockwise_scaling_aware_fp8_transpose` correctly.")

seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# TE tensor dtypes
_te_dtypes: List[tex.DType] = [tex.DType.kBFloat16]

# ==============================================================================
# 1. 双重判定准则与 Debug 溯源机制
# ==============================================================================
def calc_diff(x, y):
    """全局统计误差 (Cosine Similarity 变体): 1 - 2*(x·y)/(x²+y²)"""
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    if denominator == 0: return torch.tensor(0.0, device=x.device)
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim

def assert_tensor_close_with_diff(name, actual, expected, rtol, atol, is_bitwise=False):
    """
    双重判定：全局误差 + 逐元素误差
    is_bitwise=True 用于 uint8 原始 FP8 数据，要求 100% 比特级一致
    """
    actual_f32 = actual.float()
    expected_f32 = expected.float()
    
    # 1. 全局统计误差
    # 如果是比特级比对，误差理论上必须是 0，否则说明转码错误
    diff_threshold = 1e-7 if is_bitwise else 0.001
    diff_val = calc_diff(actual_f32, expected_f32).item()
    assert diff_val < diff_threshold, f"[{name}] Global calc_diff failed: {diff_val:.5f} (Threshold: {diff_threshold})"
    
    # 2. 逐元素容差
    if is_bitwise:
        are_close = torch.equal(actual, expected)
    else:
        are_close = torch.allclose(actual_f32, expected_f32, atol=atol, rtol=rtol)
    
    if not are_close:
        diff = torch.abs(actual_f32 - expected_f32)
        max_diff_val = diff.max().item()
        max_idx = torch.argmax(diff)
        
        # 兼容 1D/2D/3D 坐标转换
        if actual_f32.dim() == 2:
            coords = (max_idx // actual_f32.size(1), max_idx % actual_f32.size(1))
        else:
            coords = (max_idx.item(),)
            
        debug_msg = f"\n[{name}] Element-wise check failed!\n"
        debug_msg += f"  Max diff: {max_diff_val} at coords {coords}\n"
        debug_msg += f"  Actual: {actual[coords].item()}, Expected: {expected[coords].item()}\n"
        assert False, debug_msg


# ==============================================================================
# 2. 纯 PyTorch Naive 参考实现 (逐字节精准还原 Triton 逻辑)
# ==============================================================================
def naive_scaling_aware_transpose(rowwise_data, rowwise_scale_inv, m_splits, block_size=128):
    device = rowwise_data.device
    cols = rowwise_data.shape[1]
    rsi_cols = rowwise_scale_inv.shape[1]
    
    rowwise_data_list = list(torch.split(rowwise_data, m_splits, dim=0))
    rowwise_scale_inv_list = list(torch.split(rowwise_scale_inv, m_splits, dim=0))
    
    rowwise_scale_inv_t_list = []
    columnwise_data_list = []
    columnwise_scale_inv_list = []
    
    for i, (data, si) in enumerate(zip(rowwise_data_list, rowwise_scale_inv_list)):
        m = data.shape[0]
        nbrows = (m + block_size - 1) // block_size
        nbcols = (cols + block_size - 1) // block_size
        
        # a) 处理 rowwise_scale_inv_t (转置)
        si_t = torch.zeros((nbcols, m), dtype=torch.float32, device=device)
        si_t[:rsi_cols, :] = si.T
        rowwise_scale_inv_t_list.append(si_t)
        
        # b) 处理 columnwise_data 和 scale
        csi = torch.zeros((nbrows, cols), dtype=torch.float32, device=device)
        col_data_out = torch.zeros((cols, m), dtype=torch.uint8, device=device)
        
        for b in range(nbrows):
            row_start = b * block_size
            row_end = min(row_start + block_size, m)
            
            data_block = data[row_start:row_end, :]          # [block_size, cols]
            si_block = si[row_start:row_end, :]              # [block_size, rsi_cols]
            
            # target_si = max(si, axis=0)
            target_si = si_block.max(dim=0)[0]               # [rsi_cols]
            
            # 将 target_si 展平/复制以匹配 cols
            target_si_full = torch.zeros(cols, dtype=torch.float32, device=device)
            for c_idx in range(nbcols):
                c_start = c_idx * block_size
                c_end = min(c_start + block_size, cols)
                target_si_full[c_start:c_end] = target_si[c_idx]
            csi[b, :] = target_si_full
            
            # FP8 解码
            sign = (data_block >> 7) & 1                       # [block_size, cols]
            exp = (data_block >> 3) & 0xF                      # [block_size, cols]
            mant = data_block & 0x7                            # [block_size, cols]
            
            # 提取 float32 scale 的指数位
            bits_target = target_si_full.view(torch.int32).unsqueeze(0) # [1, cols]
            bits_si = si_block.view(torch.int32)                        # [block_size, rsi_cols]
            
            bits_si_full = torch.zeros((row_end - row_start, cols), dtype=torch.int32, device=device)
            for c_idx in range(nbcols):
                c_start = c_idx * block_size
                c_end = min(c_start + block_size, cols)
                bits_si_full[:, c_start:c_end] = bits_si[:, c_idx:c_idx+1]
            
            exp_t = ((bits_target & 0x7F800000) >> 23) - 127
            exp_s = ((bits_si_full & 0x7F800000) >> 23) - 127
            
            # 计算 k (指数差)
            k = exp_t - exp_s
            
            # exp_new = exp - k
            exp_new = exp.to(torch.int32) - k
            exp_new = torch.where(exp_new < 1, 0, exp_new)
            
            # 重新编码
            new_data = (sign << 7) | (exp_new << 3) | mant
            new_data = torch.where(exp == 0, torch.tensor(0, dtype=torch.uint8, device=device), new_data.to(torch.uint8))
            
            col_data_out[:, row_start:row_end] = new_data.T
            
        columnwise_data_list.append(col_data_out)
        columnwise_scale_inv_list.append(csi)
        
    return rowwise_data_list, rowwise_scale_inv_t_list, columnwise_data_list, columnwise_scale_inv_list

# ==============================================================================
# 3. 辅助生成输入数据的函数
# ==============================================================================
def make_m_splits(total_rows, group_m, base, jitter_max=512, seed=42):
    random.seed(seed)
    splits = []
    for i in range(group_m):
        jitter = random.randint(-jitter_max // 2, jitter_max // 2)
        val = max(1, base + jitter)
        splits.append(val)
    cur_sum = sum(splits)
    diff = total_rows - cur_sum
    i = 0
    while diff != 0:
        idx = i % group_m
        if diff > 0:
            splits[idx] += 1
            diff -= 1
        else:
            if splits[idx] > 1:
                splits[idx] -= 1
                diff += 1
        i += 1
    return splits

def get_quantizer(all_gather_usage):
    return Float8BlockQuantizer(
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=True,
        amax_epsilon=0.0,
        force_pow_2_scales=True,
        block_scaling_dim=1,
        all_gather_usage=all_gather_usage,
    )

# ==============================================================================
# 4. 核心测试逻辑
# ==============================================================================
def _run_correctness_test(te_dtype, num_tokens, num_expert, hidden_size, topK, ep, dp):
    if te_dtype == tex.DType.kBFloat16: dtype = torch.bfloat16
    elif te_dtype == tex.DType.kFloat16: dtype = torch.float16
    elif te_dtype == tex.DType.kFloat32: dtype = torch.float32
    else: pytest.skip("Invalid dtype.")

    print(f"\n[Correctness] token:{num_tokens} hidden_size:{hidden_size} expert:{num_expert} topK:{topK} ep:{ep} dp:{dp}")

    local_num_expert = num_expert // ep
    total_tokens = num_tokens * topK
    local_total_tokens = (total_tokens // ep) * dp
    tokens_per_expert_base = local_total_tokens // local_num_expert

    inp = torch.rand((local_total_tokens, hidden_size), dtype=dtype).cuda()
    m_splits = make_m_splits(
        local_total_tokens, local_num_expert,
        base=tokens_per_expert_base, jitter_max=512,
        seed=100 + local_num_expert,
    )

    # 生成初始 FP8 输入数据
    inp_fp8 = get_quantizer(all_gather_usage=True)(inp)
    inp_fp8._data_format = tex.Float8BlockScaleTensorFormat.GEMM_READY
    
    # 获取原始底层 Data 和 Scale
    raw_data = inp_fp8._rowwise_data.view(torch.uint8) if inp_fp8._rowwise_data.dtype != torch.uint8 else inp_fp8._rowwise_data
    raw_scale_inv = inp_fp8._rowwise_scale_inv

    # ---------------------------------------------------------
    # 执行被测算子 (Triton)
    # ---------------------------------------------------------
    triton_out = blockwise_scaling_aware_fp8_transpose(raw_data, raw_scale_inv, m_splits, block_size=128)

    # ---------------------------------------------------------
    # 执行参考基准 (Naive PyTorch)
    # ---------------------------------------------------------
    naive_out = naive_scaling_aware_transpose(raw_data, raw_scale_inv, m_splits, block_size=128)

    # ---------------------------------------------------------
    # 结果比对
    # ---------------------------------------------------------
    # 返回值结构: (rowwise_data_list, rowwise_scale_inv_t_list, columnwise_data_list, columnwise_scale_inv_list)
    assert len(triton_out) == len(naive_out) == 4, "Mismatch in output tuple length"
    assert len(triton_out[0]) == len(naive_out[0]) == local_num_expert, "Mismatch in number of experts"

    # Scale 的容差 (FP32, 需严格一致)
    scale_rtol, scale_atol = 1.0e-5, 1.0e-6

    for i in range(local_num_expert):
        # 1. 验证 rowwise_data_list (原始切分视图)
        assert_tensor_close_with_diff(
            f"Rowwise Data [{i}]", triton_out[0][i], naive_out[0][i], rtol=0.0, atol=0.0, is_bitwise=True
        )
        
        # 2. 验证 rowwise_scale_inv_t_list (转置后的 Scale)
        assert_tensor_close_with_diff(
            f"Rowwise Scale Inv T [{i}]", triton_out[1][i], naive_out[1][i], rtol=scale_rtol, atol=scale_atol
        )
        
        # 3. 验证 columnwise_data_list (核心输出：转码后的 FP8 数据)
        assert_tensor_close_with_diff(
            f"Columnwise Data [{i}]", triton_out[2][i], naive_out[2][i], rtol=0.0, atol=0.0, is_bitwise=True
        )
        
        # 4. 验证 columnwise_scale_inv_list (块级列向 Scale)
        assert_tensor_close_with_diff(
            f"Columnwise Scale Inv [{i}]", triton_out[3][i], naive_out[3][i], rtol=scale_rtol, atol=scale_atol
        )

    print(f"  -> scaling_aware_fp8_transpose correctness PASSED (Dual criteria).")


# ==============================================================================
# 5. 测试用例入口
# ==============================================================================
@pytest.mark.parametrize("te_dtype", _te_dtypes)
@pytest.mark.parametrize(
    "num_tokens, num_expert, hidden_size, topK, ep, dp",
    [
        (4096, 64,  1280, 7, 4,  4),
        (4096, 64,  2048, 6, 8,  4),
        (4096, 160, 5120, 6, 8,  4),
        (4096, 256, 7168, 8, 32, 6),
        (4096, 384, 8192, 8, 64, 6),
        (4096, 512, 9216, 8, 64, 6),
    ],
)
def test_scaling_aware_fp8_transpose_correctness(te_dtype, num_tokens, num_expert, hidden_size, topK, ep, dp):
    _run_correctness_test(te_dtype, num_tokens, num_expert, hidden_size, topK, ep, dp)

if __name__ == "__main__":
    print("Running correctness tests for scaling_aware_fp8_transpose...")
    pytest.main(['-v', __file__])