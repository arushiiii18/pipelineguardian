# 35_missing_random_state_kmeans.py
from sklearn.cluster import KMeans
import numpy as np

np.random.seed(42)
X = np.random.rand(100, 4)
km = KMeans(n_clusters=4)
km.fit(X)
