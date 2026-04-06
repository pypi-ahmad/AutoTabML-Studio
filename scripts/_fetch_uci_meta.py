"""Temporary script to fetch UCI dataset metadata for the 200-dataset batch."""
import sys

from ucimlrepo import fetch_ucirepo

NEW_IDS = [
    # Batch-1 IDs we can re-frame with alternative targets
    14, 15, 17, 20, 33, 43, 46, 53, 95, 109, 151, 225,
    # Additional IDs that might exist on the server
    4, 5, 6, 7, 11, 21, 24, 25, 34, 35, 36, 37, 41, 48, 49,
    51, 55, 56, 57, 61, 64, 65, 66, 67, 68, 71, 72, 77, 79,
    84, 85, 86, 93, 97, 98, 99, 100, 102, 103, 104, 106, 108,
    112, 113, 114, 115, 118, 119, 120, 121, 123, 124, 125,
]

for uid in NEW_IDS:
    try:
        ds = fetch_ucirepo(id=uid)
        name = ds.metadata.name
        targets = list(ds.data.targets.columns) if ds.data.targets is not None else []
        n_rows = len(ds.data.features) if ds.data.features is not None else 0
        n_cols = ds.data.features.shape[1] if ds.data.features is not None else 0
        # Determine task type
        task = "auto"
        task_raw = getattr(ds.metadata, "task", "") or ""
        if "classif" in task_raw.lower():
            task = "classification"
        elif "regress" in task_raw.lower():
            task = "regression"
        tgt = targets[0] if targets else "unknown"
        print(f'    ({uid}, "{name}", "{tgt}", "{task}"),  # {n_rows} rows, {n_cols} cols')
    except Exception as e:
        print(f"    # SKIP {uid}: {e}", file=sys.stderr)
