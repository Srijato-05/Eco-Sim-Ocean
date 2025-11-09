print("--- Starting Numba import test ---")
try:
    import numba

    print(f"Successfully imported numba. Version: {numba.__version__}")
    print(f"Does numba have 'cuda' attribute? {hasattr(numba, 'cuda')}")

    # If it has the cuda attribute, let's inspect it
    if hasattr(numba, 'cuda'):
        print("'numba.cuda' exists. Trying to use it...")
        from numba import cuda

        @cuda.jit
        def simple_kernel():
            pass

        print("SUCCESS: Successfully defined a Numba CUDA kernel.")
    else:
        print("FAILURE: numba module has no 'cuda' attribute.")

except Exception as e:
    print(f"An exception occurred: {e}")

print("--- Test finished ---")