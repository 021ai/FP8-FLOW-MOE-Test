import os
import sys
import time
import subprocess
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

def is_package_installed(package_name):
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False

def main():
    # 1. 获取当前脚本所在目录 (即 tests/operator_correctness)
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # ---> 启动时自动清理 <---
    clean_pycache(current_dir)
    
    # 2. 获取项目根目录 (向上退两级，即 FP8-FLOW-MOE-Test)
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    
    env = os.environ.copy()
    env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")
    env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7" # 按需修改

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(current_dir, f"test_results_{timestamp}")
    os.makedirs(result_dir, exist_ok=True)

    print("==========================================================")
    print("Starting FP8-FLOW-MOE Correctness Tests...")
    print(f"Script Location: {current_dir}")
    print(f"Project Root: {project_root}")
    print(f"Results will be saved to: {result_dir}")
    print("==========================================================")

    test_target = "." 
    
    cmd = [
        sys.executable, "-m", "pytest",
        test_target,
        "--import-mode=importlib",
        "-o", "python_files=test.py", 
        "-v", "-s", "-rf",
        f"--junitxml={os.path.join(result_dir, 'junit_results.xml')}",
    ]

    if is_package_installed("pytest_html"):
        print("✨ pytest-html detected. HTML report will be generated.")
        cmd.append(f"--html={os.path.join(result_dir, 'test_report.html')}")
        cmd.append("--self-contained-html")
    else:
        print("⚠️ pytest-html NOT installed. Skipping HTML report generation.")

    start_time = time.time()
    try:
        process = subprocess.Popen(
            cmd, env=env, cwd=current_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        
        log_file_path = os.path.join(result_dir, "console_log.txt")
        with open(log_file_path, "w", encoding="utf-8") as f:
            for line in process.stdout:
                print(line, end="") 
                f.write(line)       
                
        process.wait()
        return_code = process.returncode
    except Exception as e:
        print(f"Error running tests: {e}")
        return_code = 1

    elapsed_time = time.time() - start_time

    print("\n==========================================================")
    if return_code == 0:
        print(f"✅ ALL TESTS PASSED! (耗时: {elapsed_time:.2f} 秒)")
        if os.path.exists(os.path.join(result_dir, 'test_report.html')):
            print(f"Please check HTML report at: {os.path.join(result_dir, 'test_report.html')}")
        print(f"Check Log at: {log_file_path}")
    else:
        print(f"❌ TESTS FAILED! (耗时: {elapsed_time:.2f} 秒)")
        if os.path.exists(os.path.join(result_dir, 'test_report.html')):
            print(f"Please check HTML report at: {os.path.join(result_dir, 'test_report.html')}")
        print(f"Check Log at: {log_file_path}")
    print("==========================================================\n")
    
    return return_code

if __name__ == "__main__":
    sys.exit(main())