import os
import sys
import subprocess
import time
from datetime import datetime
import shutil

import pathlib

# 在 main() 函数开头加入清理逻辑
def clean_pycache(root_dir):
    """清理所有 __pycache__ 目录，防止缓存干扰第三方验收"""
    count = 0
    for p in pathlib.Path(root_dir).rglob("__pycache__"):
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
            count += 1
    if count > 0:
        print(f"🧹 Cleaned up {count} __pycache__ directories.")

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # ---> 启动时自动清理 <---
    clean_pycache(current_dir)
    
    # 1. 获取项目根目录 (向上退两级)
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    
    env = os.environ.copy()
    env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")
    env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7" # 按需修改

    # 2. 创建归档目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(current_dir, f"bench_results_{timestamp}")
    os.makedirs(result_dir, exist_ok=True)

    print("==========================================================")
    print("Starting FP8-FLOW-MOE Performance Benchmarks...")
    print(f"Results will be archived to: {result_dir}")
    print("==========================================================")

    # 3. 收集所有 bench.py 脚本路径
    bench_scripts = []
    for root, dirs, files in os.walk(current_dir):
        if "bench.py" in files:
            # 排除根目录下可能存在的 benchmark_csv.py 等
            script_path = os.path.join(root, "bench.py")
            bench_scripts.append(script_path)

    bench_scripts.sort() # 排序，保证每次执行顺序一致
    total_scripts = len(bench_scripts)
    print(f"Found {total_scripts} benchmark scripts to execute.\n")

    failed_scripts = []

    # 4. 顺序执行每一个 bench.py
    for idx, script_path in enumerate(bench_scripts, 1):
        op_name = os.path.basename(os.path.dirname(script_path))
        print(f"----------------------------------------------------------")
        print(f"[{idx}/{total_scripts}] Running benchmark for: {op_name}")
        print(f"Script: {script_path}")
        print(f"----------------------------------------------------------")

        start_time = time.time()
        
        # 切换工作目录到算子目录，确保生成的 perf_data.csv 直接落在对应目录下
        work_dir = os.path.dirname(script_path)
        
        try:
            # 实时打印输出
            process = subprocess.Popen(
                [sys.executable, "bench.py"],
                cwd=work_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            
            for line in process.stdout:
                print(line, end="")
            
            process.wait()
            
            if process.returncode != 0:
                print(f"❌ Error: Benchmark for {op_name} failed with return code {process.returncode}")
                failed_scripts.append(op_name)
            else:
                elapsed = time.time() - start_time
                print(f"✅ Finished {op_name} in {elapsed:.2f}s")
                
                # 5. 收集产物 (将生成的 csv 复制到归档目录)
                # 您的脚本可能生成 perf_data.csv 或 bench_results.csv
                for out_file in ["perf_data.csv", "bench_results.csv"]:
                    src_file = os.path.join(work_dir, out_file)
                    if os.path.exists(src_file):
                        dst_file = os.path.join(result_dir, f"{op_name}_{out_file}")
                        shutil.copy2(src_file, dst_file)
                        print(f"   -> Saved {out_file} to archive.")

        except Exception as e:
            print(f"❌ Exception running {op_name}: {e}")
            failed_scripts.append(op_name)
            
        # 清理显存，防止下一个算子 OOM
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
            
        time.sleep(2) # 稍微停顿，让 GPU 散热/冷却，保证下一个性能测试准确

    # 6. 汇总报告
    print("\n==========================================================")
    print("Benchmark Execution Summary:")
    print(f"Total Scripts: {total_scripts}")
    print(f"Passed: {total_scripts - len(failed_scripts)}")
    print(f"Failed: {len(failed_scripts)}")
    
    if failed_scripts:
        print("\nFailed Operators:")
        for f in failed_scripts:
            print(f" - {f}")
        sys.exit(1)
    else:
        print(f"\n🎉 ALL BENCHMARKS PASSED! Check results in: {result_dir}")
        sys.exit(0)

if __name__ == "__main__":
    sys.exit(main())