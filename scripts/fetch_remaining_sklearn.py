"""Fetch and inspect remaining sklearn fetch_* datasets.

The six ``fetch_*`` calls are independent network downloads, so we kick them
off in parallel via a thread executor and only print the summaries once every
fetch has completed.
"""
import asyncio

import numpy as np
import scipy.sparse
from sklearn.datasets import (
    fetch_20newsgroups_vectorized,
    fetch_lfw_pairs,
    fetch_lfw_people,
    fetch_olivetti_faces,
    fetch_rcv1,
    fetch_species_distributions,
)

from app.concurrency import gather_with_concurrency


async def _fetch_all() -> dict[str, object]:
    coros = {
        "olivetti": asyncio.to_thread(fetch_olivetti_faces),
        "20news": asyncio.to_thread(fetch_20newsgroups_vectorized, subset="all"),
        "lfw_people": asyncio.to_thread(fetch_lfw_people, min_faces_per_person=70),
        "lfw_pairs": asyncio.to_thread(fetch_lfw_pairs),
        "rcv1": asyncio.to_thread(fetch_rcv1),
        "species": asyncio.to_thread(fetch_species_distributions),
    }
    results = await gather_with_concurrency(
        list(coros.values()), limit=6, return_exceptions=True
    )
    return dict(zip(coros.keys(), results))


def main() -> None:
    fetched = asyncio.run(_fetch_all())

    print("=== Olivetti Faces ===")
    d = fetched["olivetti"]
    print(f"  data: {d.data.shape}, target: {d.target.shape}, classes: {len(np.unique(d.target))}")
    print(f"  dtype: {d.data.dtype}, sparse: {scipy.sparse.issparse(d.data)}")

    print()
    print("=== 20 Newsgroups Vectorized ===")
    d2 = fetched["20news"]
    print(f"  data: {d2.data.shape}, target: {d2.target.shape}, classes: {len(np.unique(d2.target))}")
    print(f"  sparse: {scipy.sparse.issparse(d2.data)}, nnz: {d2.data.nnz}")
    density = d2.data.nnz / (d2.data.shape[0] * d2.data.shape[1])
    print(f"  density: {density:.6f}")

    print()
    print("=== LFW People ===")
    d3 = fetched["lfw_people"]
    print(f"  data: {d3.data.shape}, target: {d3.target.shape}, classes: {len(np.unique(d3.target))}")
    print(f"  target_names: {d3.target_names}")

    print()
    print("=== LFW Pairs ===")
    d4 = fetched["lfw_pairs"]
    print(f"  pairs: {d4.pairs.shape}, target: {d4.target.shape}, classes: {np.unique(d4.target)}")

    print()
    print("=== RCV1 ===")
    d5 = fetched["rcv1"]
    if isinstance(d5, BaseException):
        print(f"  Error: {d5}")
    else:
        print(f"  data: {d5.data.shape}, target: {d5.target.shape}")
        print(f"  data sparse: {scipy.sparse.issparse(d5.data)}, target sparse: {scipy.sparse.issparse(d5.target)}")

    print()
    print("=== Species Distributions ===")
    d6 = fetched["species"]
    print(f"  type: {type(d6)}")
    print(f"  keys: {list(d6.keys())}")
    print(f"  coverages shape: {d6.coverages.shape}")
    train = d6.train
    print(f"  train: {type(train)}, len={len(train)}")


if __name__ == "__main__":
    main()
