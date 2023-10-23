import os
import random
import torch
import numpy as np


def get_data_loader_from_dataset(dataset_path, batch_size=256, shuffle=True):
    dataset = np.load(dataset_path)
    data = dataset["data"]
    labels = dataset["labels"]
    assert (
        isinstance(data, np.ndarray)
        and isinstance(labels, np.ndarray)
        and len(data) == len(labels)
    )
    assert data.dtype == np.float32  # and label.dtype == np.longlong
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(data), torch.from_numpy(labels)
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle
    )
    return dataloader
