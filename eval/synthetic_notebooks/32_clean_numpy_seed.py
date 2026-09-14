# 32_clean_numpy_seed.py
import numpy as np

np.random.seed(123)
arr = np.random.rand(50, 5)
print(arr.mean())
