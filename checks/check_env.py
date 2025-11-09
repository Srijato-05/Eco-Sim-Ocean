import os
import sys
import numba

print("--- Python Executable ---")
print(sys.executable)
print("-" * 30)

print("--- Numba CUDA Check ---")
print(f"hasattr(numba, 'cuda'): {hasattr(numba, 'cuda')}")
print("-" * 30)

print("--- Details of Key Environment Variables ---")
for key in ['PATH', 'CUDA_HOME', 'CUDA_PATH']:
    value = os.environ.get(key, '*** NOT SET ***')
    print(f"\n## {key}:")
    if key == 'PATH':
        for p in value.split(os.pathsep):
            print(f"  - {p}")
    else:
        print(f"  {value}")