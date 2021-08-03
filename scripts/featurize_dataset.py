import os
from functools import partial
from typing import Callable

from datasets import load_dataset, Dataset

from src.utils import pmap, smiles2graph, smiles2graph_v2

_CACHE_DIR = "/raid/data/cache/"


def make_dataset(name: str, fn: Callable, dataset: Dataset) -> None:
    """
    Use batched dataset.map with batch_size > dataset size and pmap to most
    efficiently process transforms. Keep transformed dataset in memory (don't auto
    cache) and manually save to given location.
    :param name: unique name of dataset
    :param fn: function to run on dataset examples to create new data
    :param dataset: Dataset to featurize
    """
    path = os.path.join(_CACHE_DIR, name)
    if not os.path.exists(path):
        print(f"Saving new dataset to {path}")
        dataset.map(
            fn, batched=True, batch_size=5_000_000, keep_in_memory=True
        ).save_to_disk(path)
        print(f"{name} dataset created!")
    else:
        print(f"{name} already exists.")


# %%
if __name__ == "__main__":
    # Process once, cache, memmapped loading
    pcqm4m = load_dataset("src/pcqm4m_dataset.py", cache_dir=_CACHE_DIR)

    make_dataset(
        "smiles2graph_pos_enc",
        partial(
            smiles2graph_v2,
            add_junction_tree=False,
            n_virtual_nodes=0,
            add_pos_enc=True,
        ),
        pcqm4m,
    )