"""Fetch and inspect remaining sklearn fetch_* datasets."""
import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.datasets import (
    fetch_olivetti_faces,
    fetch_20newsgroups_vectorized,
    fetch_lfw_people,
    fetch_lfw_pairs,
    fetch_rcv1,
    fetch_species_distributions,
)

print("=== Olivetti Faces ===")
d = fetch_olivetti_faces()
print(f"  data: {d.data.shape}, target: {d.target.shape}, classes: {len(np.unique(d.target))}")
print(f"  dtype: {d.data.dtype}, sparse: {scipy.sparse.issparse(d.data)}")

print()
print("=== 20 Newsgroups Vectorized ===")
d2 = fetch_20newsgroups_vectorized(subset="all")
print(f"  data: {d2.data.shape}, target: {d2.target.shape}, classes: {len(np.unique(d2.target))}")
print(f"  sparse: {scipy.sparse.issparse(d2.data)}, nnz: {d2.data.nnz}")
density = d2.data.nnz / (d2.data.shape[0] * d2.data.shape[1])
print(f"  density: {density:.6f}")

print()
print("=== LFW People ===")
d3 = fetch_lfw_people(min_faces_per_person=70)
print(f"  data: {d3.data.shape}, target: {d3.target.shape}, classes: {len(np.unique(d3.target))}")
print(f"  target_names: {d3.target_names}")

print()
print("=== LFW Pairs ===")
d4 = fetch_lfw_pairs()
print(f"  pairs: {d4.pairs.shape}, target: {d4.target.shape}, classes: {np.unique(d4.target)}")

print()
print("=== RCV1 ===")
try:
    d5 = fetch_rcv1()
    print(f"  data: {d5.data.shape}, target: {d5.target.shape}")
    print(f"  data sparse: {scipy.sparse.issparse(d5.data)}, target sparse: {scipy.sparse.issparse(d5.target)}")
except Exception as e:
    print(f"  Error: {e}")

print()
print("=== Species Distributions ===")
d6 = fetch_species_distributions()
print(f"  type: {type(d6)}")
print(f"  keys: {list(d6.keys())}")
print(f"  coverages shape: {d6.coverages.shape}")
train = d6.train
print(f"  train: {type(train)}, len={len(train)}")
