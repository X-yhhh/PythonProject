# test.py
import sys

print("✨ 这是 test.py 文件")
print(f"当前 Python 版本: {sys.version}")
print(f"Python 可执行文件路径: {sys.executable}")

# 测试是否能使用标准库
import json
data = {"name": "Alice", "age": 30}
print("JSON 序列化测试:", json.dumps(data, indent=2))

# 测试第三方库（可选）
try:
    import numpy as np
    print(f"NumPy 版本: {np.__version__}")
    arr = np.array([1, 2, 3])
    print("NumPy 数组示例:", arr)
except ImportError:
    print("⚠️ NumPy 未安装。如需安装，请在 Terminal 中运行：pip install numpy")