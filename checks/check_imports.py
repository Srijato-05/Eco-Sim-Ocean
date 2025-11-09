import sys
import os
import numba

print("--- Python Path (sys.path) ---")
# sys.path shows where Python looks for modules. The first entry is key.
for i, p in enumerate(sys.path):
    print(f"{i}: {p}")
print("-" * 30)

print("--- Numba Module Info ---")
try:
    numba_path = numba.__file__
    print(f"Numba version: {numba.__version__}")
    print(f"Numba is being imported from: {numba_path}")

    is_correct_loc = 'site-packages' in numba_path
    print(f"Is this the correct Conda location? {is_correct_loc}")
    print("-" * 30)

    print("--- CUDA Check ---")
    has_cuda = hasattr(numba, 'cuda')
    print(f"Does the imported numba have '.cuda'? {has_cuda}")
    if not has_cuda:
        print("\n!!! CRITICAL: 'numba.cuda' was not found.")
        print("Searching for modules loaded from your local project directory...")
        print("Any file listed below could be the cause of the conflict.")
        print("-" * 30)

        # This part finds modules that Python imported from your project folder
        # instead of from the standard library or Conda environment.
        project_dir = os.getcwd()
        found_conflict = False
        for name, module in sorted(sys.modules.items()):
            if hasattr(module, '__file__') and module.__file__:
                if project_dir in module.__file__:
                    print(f"  -> CONFLICT? Module '{name}' loaded from: {module.__file__}")
                    found_conflict = True

        if not found_conflict:
            print("No obvious conflicting modules found loaded from the project directory.")

except Exception as e:
    print(f"An error occurred: {e}")