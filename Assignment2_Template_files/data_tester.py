import numpy as np
import pandas as pd
import torch as t
import time

from data_loader import (
    load_dataset, load_dataset_np, normalize_array, normalize_array_np,
    load_dataset_pd, split_xy, split_training_test
)
from vector_product import (
    dot_product, find_largest_dot_product_py, dot_product_np, find_largest_dot_product_np,
    mat_mul_np, mat_mul_t, dot_product_t
)


if __name__ == "__main__":
    # Test data loading
    print('Testing data loading...')
    try:
        data_list = load_dataset('GasProperties.csv')
        print(f'Loaded {len(data_list)} rows (list of lists)')
    except Exception as e:
        print('load_dataset failed:', e)
    try:
        data_np = load_dataset_np('GasProperties.csv')
        print(f'Loaded {data_np.shape[0]} rows (NumPy array)')
    except Exception as e:
        print('load_dataset_np failed:', e)
    try:
        data_pd = load_dataset_pd('GasProperties.csv')
        print(f'Loaded {data_pd.shape[0]} rows (Pandas DataFrame)')
    except Exception as e:
        print('load_dataset_pd failed:', e)

    # Test normalization
    print('\nTesting normalization with timing...')
    try:
        start = time.perf_counter()
        nrows = normalize_array(data_list)
        elapsed = time.perf_counter() - start
        print(f'normalize_array processed {nrows} rows in {elapsed:.4f} seconds')
    except Exception as e:
        print('normalize_array failed:', e)

    try:
        start = time.perf_counter()
        nrows_np = normalize_array_np(data_np)
        elapsed = time.perf_counter() - start
        print(f'normalize_array_np processed {nrows_np} rows in {elapsed:.4f} seconds')
    except Exception as e:
        print('normalize_array_np failed:', e)    

    # Test splitting
    print('\nTesting split_xy and split_training_test...')
    try:
        X, Y = split_xy(data_pd)
        print(f'split_xy: X shape {X.shape}, Y shape {Y.shape}')
        X_train, Y_train, X_test, Y_test = split_training_test(X, Y)
        print(f'split_training_test: X_train {X_train.shape}, X_test {X_test.shape}')
    except Exception as e:
        print('Splitting failed:', e)

    # Test vector and matrix operations
    print('\nTesting vector and matrix operations...')
    try:
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        print('dot_product:', dot_product(a, b))
        print('find_largest_dot_product_py:', find_largest_dot_product_py([a, b], b))
        a_np = np.array(a)
        b_np = np.array(b)
        print('dot_product_np:', dot_product_np(a_np, b_np))
        print('find_largest_dot_product_np:', find_largest_dot_product_np(np.array([a, b]), b_np))
        A = np.random.randn(3, 3)
        B = np.random.randn(3, 3)
        print('mat_mul_np:', mat_mul_np(A, B))
        A_t = t.tensor(A, dtype=t.float64)
        B_t = t.tensor(B, dtype=t.float64)
        print('mat_mul_t:', mat_mul_t(A_t, B_t))
        print('dot_product_t:', dot_product_t(A_t[0], B_t[0]))
    except Exception as e:
        print('Vector/matrix operation failed:', e)

    # Matrix multiplication: 32-bit vs 64-bit precision
    '''print('\nMatrix multiplication: 32-bit vs 64-bit precision')
    try:
        # Use X from split_xy if available, else generate random
        try:
            X_data = X
        except NameError:
            X_data = np.random.randn(1000, 1000)
       
        X32 = t.tensor(X_data, dtype=t.float32)
        X64 = t.tensor(X_data, dtype=t.float64)
        # 32-bit
        start = time.time()
        result32 = t.matmul(X32, X32)
        time32 = time.time() - start
        # 64-bit
        start = time.time()
        result64 = t.matmul(X64, X64)
        time64 = time.time() - start
        print(f"32-bit time: {time32:.6f} s, 64-bit time: {time64:.6f} s")
        if abs(time32 - time64) > 1e-3:
            print("64-bit (float64) operations are usually slower than 32-bit (float32) because they require more memory and computational resources. Most consumer hardware is optimized for 32-bit operations, so 64-bit math can be significantly slower.")
        else:
            print("Computation times are similar; this may be due to small matrix size or hardware with strong double-precision support.")
    except Exception as e:
        print('Matrix multiplication precision comparison failed:', e)'''

    # Matrix multiplication: 32-bit vs 64-bit precision
    print('\nMatrix multiplication: 32-bit vs 64-bit precision')
    try:
        # Use X from split_xy if available, else fallback to a reasonably-sized random square matrix
        try:
            X_data = X
        except NameError:
            X_data = np.random.randn(1000, 1000)

        # Choose a square size for benchmarking. Prefer a slice from X_data when possible.
        preferred_size = 256
        k = min(preferred_size, X_data.shape[0], X_data.shape[1])
        if k < 2:
            # Data is too narrow; fall back to a small random matrix
            A = np.random.randn(4, 4)
        else:
            # If X_data is large enough in both dims, take a top-left k x k slice; otherwise use a random k x k
            if X_data.shape[0] >= k and X_data.shape[1] >= k:
                A = X_data[:k, :k].astype(float)
            else:
                A = np.random.randn(k, k)

        X32 = t.tensor(A, dtype=t.float32)
        X64 = t.tensor(A, dtype=t.float64)

        # 32-bit
        start = time.time()
        result32 = t.matmul(X32, X32)
        time32 = time.time() - start
        # 64-bit
        start = time.time()
        result64 = t.matmul(X64, X64)
        time64 = time.time() - start
        print(f"32-bit time: {time32:.6f} s, 64-bit time: {time64:.6f} s")
        if abs(time32 - time64) > 1e-3:
            print("64-bit (float64) operations are usually slower than 32-bit (float32) because they require more memory and computational resources. Most consumer hardware is optimized for 32-bit operations, so 64-bit math can be significantly slower.")
        else:
            print("Computation times are similar; this may be due to small matrix size or hardware with strong double-precision support.")
    except Exception as e:
        print('Matrix multiplication precision comparison failed:', e)
    
