import os
import sys
# --- Force PATH for diagnosis ---
if sys.platform == 'win32':
    os.environ['PATH'] = r'C:\Windows\System32' + os.pathsep + os.environ['PATH']

import numba
import cupy as cp

print(f"CUDA available: {hasattr(numba, 'cuda')}")

@numba.cuda.jit
def simple_kernel(array):
    i = numba.cuda.grid(1)
    if i < array.size:
        array[i] *= 2

try:
    x = cp.arange(10)
    simple_kernel[1, 10](x)
    print(f"Simple kernel ran successfully. Result: {x}")
    print("✅ This confirms Numba and CuPy can work in this environment.")
except Exception as e:
    print(f"❌ Kernel execution failed: {e}")