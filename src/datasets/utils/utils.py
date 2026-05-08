# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

try:
    from src.utils.cluster import dataset_paths
except ImportError:
    dataset_paths = None

from src.utils.logging import get_logger

import numpy as np
import torch

logger = get_logger("Datasets utils")

from torch.utils.data import default_collate


def default_collate_with_patient_ids(batch):
    """
    Uses PyTorch's native default_collate for standard elements,
    but manually extracts and passes through patient_ids as a list.
    """
    # 1. Extract just the patient IDs into a Python list
    patient_ids = [item[3] for item in batch]

    # 2. Create a new batch containing ONLY the elements PyTorch knows how to collate
    # (clips, label, clip_indices)
    batch_without_pids = [(item[0], item[1], item[2]) for item in batch]

    # 3. Let PyTorch's battle-tested default_collate handle the tensors exactly as it did before
    collated_clips, collated_labels, collated_indices = default_collate(batch_without_pids)

    # 4. Return the fully collated tensors plus our raw list of patient IDs
    return (collated_clips, collated_labels, collated_indices, patient_ids)


def get_dataset_paths(datasets: list[str]):
    if dataset_paths is None:
        raise ImportError("dataset_paths function not available from src.utils.cluster")
    paths = []
    for d in datasets:
        try:
            path = dataset_paths().get(d)
        except Exception:
            raise Exception(f"Unknown dataset: {d}")
        paths.append(path)
    logger.info(f"Datapaths {paths}")
    return paths
