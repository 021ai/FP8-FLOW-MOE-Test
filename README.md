# 高性能算子交付测试说明

## 一、交付概览

1. **交付范围：** 本次共交付高精度、高性能算子共 13 个，包含 10 个 FP8 低精度算子和 3 个融合算子，涵盖核心计算、MoE 路由分发、激活函数及量化加速等底层核心场景。
2. **测试环境基准配置：**
   * **硬件平台：** NVIDIA H100-SXM5-80GB，1 GPU。
   * **软件栈依赖：** Ubuntu 24.04.2 LTS, PyTorch 2.7, CUDA 12.9, TransformerEngine-FP8FlowMoe, Megatron-LM-FP8FlowMoe。

---

## 二、算子清单

| 序号 | 算子名称 (Op Name) | 核心功能简述 |
| :--- | :--- | :--- |
| 1 | `act_dequant_B_ptr` | 根据内存排布为行优先的缩放因子，将 bf16 格式的张量反向化为 fp8 格式 |
| 2 | `act_dequant_noT_stride_B_ptr` | 根据内存排布为列优先的缩放因子，将 bf16 格式的张量反向化为 fp8 格式 |
| 3 | `act_dequant_T_nostride_B_ptr` | 根据内存排布为行优先且经过转置的缩放因子，将 bf16 格式的张量反向化为 fp8 格式 |
| 4 | `act_quant_B_ptr` | 对输入的 bf16 格式张量进行分组量化，得到 fp8 格式张量 |
| 5 | `act_quant_tma_aligned_b_ptr` | 对输入的 bf16 格式张量进行分组量化，得到 fp8 格式张量，且确保缩放因子与 TMA 对齐 |
| 6 | `weight_multi_cast_transpose_fused` | 执行逐块量化操作，将多个二维张量堆叠得到的三维张量，量化为 FP8 格式的三维张量 |
| 7 | `weight_multi_cast_transpose_fused_2d_3d` | 执行逐块量化操作，将多个二维张量拼接得到的二维张量，量化为 FP8 格式的三维张量 |
| 8 | `fused_swiglu_and_quantization_fwd` | 激活函数和量化操作融合前向算子 |
| 9 | `fused_swiglu_and_quantization_bwd` | 激活函数和量化操作融合反向算子 |
| 10 | `fused_permute_and_padding_fwd` | Token 重排和元素补齐操作融合前向算子 |
| 11 | `fused_permute_and_padding_bwd` | Token 重排和元素补齐操作融合反向算子 |
| 12 | `scaling_aware_fp8_transpose` | 反量化、转置和量化操作融合算子 |
| 13 | `act_multi_cast_transpose_fp8` | 根据 `m_splits` 对输入张量进行逻辑分割，在每个子块上执行转置和 FP8 量化 |

---

## 三、验收测试维度与指标定义

### 1. 正确性验证
为保障算子的数学精度与工程可用性，本测试采用全局相似度误差和逐元素容差两个指标作为正确性判定的依据。由于不同的算子存在功能和原理上的差异，13 个算子的正确性判定标准有所区别，具体可参考各算子的测试脚本。两个指标的含义分别如下：

#### 1.1 指标 1：全局相似度误差
* 该指标反映了算子输出与基准参考输出之间的全局向量夹角/余弦相似度偏差。
* **计算公式：** $\text{diff} = 1 - \frac{2 \sum (x \cdot y)}{\sum (x^2 + y^2)}$
* **判定阈值：** $\text{diff} < 0.001$，即全局误差容忍度小于 $0.1\%$。
* **判定意义：** 该指标能够有效卡住整体的宏观精度，防止因缩放因子应用错误（如整体少乘或多乘一个 Scale）导致的系统性精度崩塌，符合大模型对 FP8 反量化的极高精度要求。

#### 1.2 指标 2：逐元素双阈值容差
* 该指标使用 PyTorch 原生 `torch.allclose` 函数进行逐元素比对，采用绝对误差 (`atol`) 与相对误差 (`rtol`) 混合判定。
* **判定意义：** 满足公式 $|y - y_{\text{ref}}| \le \text{atol} + \text{rtol} \times |y_{\text{ref}}|$。该指标用于捕捉极值点（Outliers）、离群点或因边界对齐（Padding）导致的小范围溢出/截断错误。

### 2. 性能验证
为全面评估算子性能，测试时参考以下两个维度的指标：

#### 2.1 Kernel 纯计算耗时 (Kernel Latency)
* **定义：** 剥离 Python 开销和显存分配开销，纯计算核心在 GPU 上的执行时间。
* **测量方法：** 使用基于 GPU Event 的 `do_bench` 工具，重复执行 100 次。记录并提取中位数（Median, $50\%$）、高水位线（$80\%$）和低水位线（$20\%$）的分位数，以消除系统噪声。

#### 2.2 加速比要求 (Speedup Ratio)
* **定义：** 被测算子相对于基准（TE）的耗时缩减倍数，即 Kernel 纯计算加速比。
* **计算公式：** $\text{Speedup} = \frac{\text{Time}_{\text{TE}}}{\text{Time}_{\text{Triton}}}$。数值大于 1 表示被测算子性能优于 TE 基准。

---

## 四、测试执行流程

### 1. 环境准备
所有依赖包均安装完毕，直接进入测试目录：
```bash
cd $TEST_PATH/
```

### 2. 实验执行
测试目录结构如下：
```text
tests/
├── operator_benchmarks/          # 算子性能测试，包含13个算子各自的测试目录
│   ├── act_dequant_B_ptr/
│   ├── act_dequant_noT_stride_B_ptr/
│   ├── ……
│   └── run_all_benchmarks.py          # 性能一键测试脚本
└── operator_correctness/          # 算子正确性测试
    ├── act_dequant_B_ptr/
    ├── act_dequant_noT_stride_B_ptr/
    ├── ……
    └── run_all_tests.py          # 正确性一键测试脚本
```

所有算子支持一键测试和各算子独立测试两种模式，测试命令分别如下：

* **一键测试**
  ```bash
  # 正确性测试
  cd tests/operator_correctness
  python run_all_tests.py

  # 性能测试
  cd tests/operator_benchmarks
  python run_all_benchmarks.py
  ```

* **各算子独立测试**
  ```bash
  # 正确性测试
  cd tests/operator_correctness
  cd act_dequant_B_ptr/      # 进入对应算子目录
  python bench.py | tee result.out

  # 性能测试
  cd tests/operator_benchmarks
  cd act_dequant_B_ptr/      # 进入对应算子目录
  python test.py
  ```

### 3. 结果收集
* **正确性验证：** 一键测试的结果会保存在 `$TEST_PATH/tests/operator_correctness/test_results_YYYYMMDD_HHMMSS/`，例如：`$TEST_PATH/tests/operator_correctness/test_results_20261021_095046`。若采用各算子独立测试模式，测试结果会在终端输出，也可在算子目录下的 `result.out` 文件中查看。
* **性能验证：** 一键测试的结果会保存在 `$TEST_PATH/tests/operator_benchmarks/bench_results_YYYYMMDD_HHMMSS/`，例如：`$TEST_PATH/tests/operator_benchmarks/bench_results_20261021_095356`。若采用各算子独立测试模式，测试结果会在终端输出，也可在算子目录下的 `perf_data.csv` 文件中查看。