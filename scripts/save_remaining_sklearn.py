"""Save remaining viable sklearn fetch_* datasets as CSVs."""
import os

import numpy as np
import pandas as pd
from sklearn.datasets import (
    fetch_20newsgroups_vectorized,
    fetch_lfw_pairs,
    fetch_lfw_people,
    fetch_olivetti_faces,
)

# 1. Olivetti Faces: 400 x 4096, 40-class classification
print("=== Saving Olivetti Faces ===")
d = fetch_olivetti_faces()
cols = [f"pixel_{i}" for i in range(d.data.shape[1])]
df = pd.DataFrame(d.data, columns=cols)
df["target"] = d.target
out = "datasets/sklearn/Olivetti_Faces"
os.makedirs(out, exist_ok=True)
df.to_csv(f"{out}/olivetti_faces.csv", index=False)
print(f"  Saved: {df.shape} -> {out}/olivetti_faces.csv")

# 2. LFW People (min_faces_per_person=70): 1288 x 2914, 7-class
print("\n=== Saving LFW People ===")
d3 = fetch_lfw_people(min_faces_per_person=70)
cols3 = [f"pixel_{i}" for i in range(d3.data.shape[1])]
df3 = pd.DataFrame(d3.data, columns=cols3)
df3["target"] = d3.target
out3 = "datasets/sklearn/LFW_People"
os.makedirs(out3, exist_ok=True)
df3.to_csv(f"{out3}/lfw_people.csv", index=False)
print(f"  Saved: {df3.shape} -> {out3}/lfw_people.csv")
print(f"  Classes: {d3.target_names.tolist()}")

# 3. LFW Pairs: flatten each pair → 5828 features, binary classification
print("\n=== Saving LFW Pairs ===")
d4 = fetch_lfw_pairs()
# pairs shape: (2200, 2, 62, 47)
n = d4.pairs.shape[0]
flat = d4.pairs.reshape(n, -1)  # (2200, 5828)
cols4 = [f"pixel_{i}" for i in range(flat.shape[1])]
df4 = pd.DataFrame(flat, columns=cols4)
df4["target"] = d4.target
out4 = "datasets/sklearn/LFW_Pairs"
os.makedirs(out4, exist_ok=True)
df4.to_csv(f"{out4}/lfw_pairs.csv", index=False)
print(f"  Saved: {df4.shape} -> {out4}/lfw_pairs.csv")

# 4. 20 Newsgroups Vectorized: reduce to top 500 features by total TF-IDF
print("\n=== Saving 20 Newsgroups Vectorized ===")
d2 = fetch_20newsgroups_vectorized(subset="all")
# Select top 500 features by sum of TF-IDF values
col_sums = np.array(d2.data.sum(axis=0)).flatten()
top_500_idx = np.argsort(col_sums)[-500:]
reduced = d2.data[:, top_500_idx].toarray()  # dense (18846, 500)
feature_names = np.array(d2.feature_names) if hasattr(d2, "feature_names") else None
if feature_names is not None:
    cols2 = [feature_names[i] for i in top_500_idx]
else:
    cols2 = [f"tfidf_{i}" for i in range(500)]
df2 = pd.DataFrame(reduced, columns=cols2)
df2["target"] = d2.target
out2 = "datasets/sklearn/20_Newsgroups"
os.makedirs(out2, exist_ok=True)
df2.to_csv(f"{out2}/20newsgroups.csv", index=False)
print(f"  Saved: {df2.shape} -> {out2}/20newsgroups.csv")
print(f"  Classes: {len(np.unique(d2.target))}")

print("\nDone!")
