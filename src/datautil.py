# Dataset loading that doesn't download a whole 10k-image split for a smoke.

from __future__ import annotations


def load_rows(
    dataset_id: str,
    split: str,
    sample_size: int | None = None,
    config: str | None = None,
):
    from datasets import load_dataset

    args = (dataset_id, config) if config else (dataset_id,)
    if sample_size:
        ds = load_dataset(*args, split=split, streaming=True)
        return list(ds.take(sample_size))
    return list(load_dataset(*args, split=split))
