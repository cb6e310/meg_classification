import os
import random
import torch
import numpy as np

from loguru import logger

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from .augmentations import DataTransform


class SiameseDataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, samples, labels):
        super(Dataset, self).__init__()
        X_train = samples
        y_train = labels

        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)

        if (
            X_train.shape.index(min(X_train.shape)) != 1
        ):  # make sure the Channels in second dim
            X_train = X_train.permute(0, 2, 1)

        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train)
            self.y_data = torch.from_numpy(y_train).long()
        else:
            self.x_data = X_train
            self.y_data = y_train

        # expand last dim for channel
        if len(self.x_data.shape) < 4:
            self.x_data = self.x_data.unsqueeze(3)
        print("dataset info: classes, shape")
        print(
            self.y_data.unique().shape[0],
            self.x_data.shape,
        )

        self.aug1, self.aug2 = DataTransform(self.x_data)

    def __getitem__(self, index):
        return (
            self.x_data[index],
            self.y_data[index],
            self.aug1[index],
            self.aug2[index],
            index,
        )

    def __len__(self):
        return self.x_data.shape[0]


def data_generator_all(data_path, configs, training_mode):
    train_dataset = torch.load(os.path.join(data_path, "train.pt"))
    valid_dataset = torch.load(os.path.join(data_path, "val.pt"))
    test_dataset = torch.load(os.path.join(data_path, "test.pt"))

    train_dataset = Load_Dataset(train_dataset, configs, training_mode)
    valid_dataset = Load_Dataset(valid_dataset, configs, training_mode)
    test_dataset = Load_Dataset(test_dataset, configs, training_mode)

    # print(len(train_dataset)) # HAR: 7352 , wisdm: 2617
    # print(len(valid_dataset)) # HAR: 1471, wisdm: 655
    # print(len(test_dataset))  # HAR: 2947, wisdm: 819

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=configs.batch_size,
        shuffle=True,
        drop_last=configs.drop_last,
        num_workers=0,
    )

    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=configs.batch_size,
        shuffle=False,
        drop_last=configs.drop_last,
        num_workers=0,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=configs.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    return train_loader, valid_loader, test_loader


def data_generator(data_path, configs, training_mode, batch_size=128, drop_last=True):
    train_dataset = torch.load(os.path.join(data_path, "train.pt"))
    test_dataset = torch.load(os.path.join(data_path, "test.pt"))

    train_dataset = Load_Dataset(train_dataset, configs, training_mode)
    test_dataset = Load_Dataset(test_dataset, configs, training_mode)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
        num_workers=0,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    return train_loader, test_loader


def get_data_loader_from_dataset(
    dataset_path, train=True, batch_size=256, shuffle=True, siamese=False
):
    # local dataset
    if dataset_path.startswith("/home"):
        data = []
        labels = []
        for cur_file in os.listdir(dataset_path):
            # HLTS decoding format
            if cur_file.endswith("pt"):
                cur_dataset = torch.load(os.path.join(dataset_path, cur_file))
                cur_data = cur_dataset["samples"]
                cur_labels = cur_dataset["labels"]

            # Fan decoding format
            elif cur_file.endswith("npz"):
                cur_dataset = np.load(os.path.join(dataset_path, cur_file))
                cur_data = cur_dataset["data"]
                cur_labels = cur_dataset["labels"]
            data.append(cur_data)
            labels.append(cur_labels)
        data = np.concatenate(data, axis=0)
        labels = np.concatenate(labels, axis=0)
        print("data info:", type(data), type(labels), data.shape, labels.shape)
        data.astype(np.float32)
        labels.astype(np.longlong)
        assert (
            isinstance(data, np.ndarray)
            and isinstance(labels, np.ndarray)
            and len(data) == len(labels)
        )
        assert data.dtype == np.float32 or torch.float32
          # and label.dtype == np.longlong

        if siamese:
            dataset = SiameseDataset(data, labels)
        else:
            dataset = torch.utils.data.TensorDataset(
                torch.from_numpy(data), torch.from_numpy(labels)
            )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle
        )

    # elif dataset_path.startswith("/CIFAR10"):
    #     import torchvision
    #     import torchvision.transforms as transforms

    #     transform = transforms.Compose(
    #         [
    #             transforms.ToTensor(),
    #             transforms.Normalize(
    #                 (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    #             ),  # mean, std for 3 channels
    #         ]
    #     )
    #     dataloader = torch.utils.data.DataLoader(
    #         torchvision.datasets.CIFAR10(
    #             root="./data", train=train, download=True, transform=transform
    #         ),
    #         batch_size=batch_size,
    #         shuffle=shuffle,
    #     )
    return dataloader
