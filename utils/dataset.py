import os
import random
import torch
import numpy as np

from loguru import logger


def get_data_loader_from_dataset(dataset_path, train=True, batch_size=256, shuffle=True):
    # local dataset
    if dataset_path.startswith("/home"):
        data = []
        labels = []
        for cur_file in os.listdir(dataset_path):
            cur_dataset = np.load(os.path.join(dataset_path, cur_file))
            cur_data = cur_dataset["data"]
            cur_labels = cur_dataset["labels"]
            data.append(cur_data)
            labels.append(cur_labels)
        data = np.concatenate(data, axis=0)
        labels = np.concatenate(labels, axis=0)
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

    elif dataset_path.startswith("/CIFAR10"):
        import torchvision
        import torchvision.transforms as transforms

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
                ),  # mean, std for 3 channels
            ]
        )
        dataloader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10(
                root="./data", train=train, download=True, transform=transform
            ),
            batch_size=batch_size,
            shuffle=shuffle,
        )
    return dataloader
