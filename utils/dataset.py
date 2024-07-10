import os
import random
import torch
import numpy as np

from loguru import logger

from torch.utils.data import DataLoader
from torch.utils.data import Dataset, TensorDataset

from utils.helpers import (
    centerize_vary_length_series,
    split_with_nan,
)

from sklearn.preprocessing import MinMaxScaler, StandardScaler


class SimpleDataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, samples, labels):
        super(Dataset, self).__init__()
        X_train = samples
        y_train = labels

        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)

        # if (
        #     X_train.shape.index(min(X_train.shape)) != 1
        # ):  # make sure the Channels in second dim
        #     X_train = X_train.permute(0, 2, 1)

        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train)
            self.y_data = torch.from_numpy(y_train).long()
        else:
            self.x_data = X_train
            self.y_data = y_train

        # expand last dim for channel
        if len(self.x_data.shape) < 4:
            self.x_data = self.x_data.unsqueeze(3)
        logger.info("dataset info: classes, shape")
        logger.info("{}, {}".format(self.y_data.unique(), self.x_data.shape))
        logger.info(
            "data statistics: max {}, min {}, mean {}, std {}".format(
                self.x_data.max(),
                self.x_data.min(),
                self.x_data.mean(),
                self.x_data.std(),
            )
        )
        scaler = StandardScaler()
        self.x_norm = (
            scaler.fit_transform(self.x_data.reshape(-1, self.x_data.shape[-1]))
            .reshape(self.x_data.shape)
        )
        logger.info("dataset scaler mean and std: {}, {}".format(scaler.mean_, scaler.scale_))
        pass
        # self.aug1, self.aug2 = DataTransform(self.x_data)
        # logger.info("SiameseDataset: Augmentation done")

    def __getitem__(self, index):
        return (
            self.x_norm[index],
            self.y_data[index],
            index,
        )

    def __len__(self):
        return self.x_data.shape[0]


def TS2Vec_Train_Dataset(train_data, cfg):
    max_train_length = cfg.MODEL.ARGS.MAX_TRAIN_LENGTH
    # swap timestamp and channel
    train_data = train_data.transpose(0, 2, 1)
    assert train_data.ndim == 3

    if max_train_length is not None:
        sections = train_data.shape[1] // max_train_length
        if sections >= 2:
            train_data = np.concatenate(
                split_with_nan(train_data, sections, axis=1), axis=0
            )

    temporal_missing = np.isnan(train_data).all(axis=-1).any(axis=0)
    if temporal_missing[0] or temporal_missing[-1]:
        train_data = centerize_vary_length_series(train_data)

    train_data = train_data[~np.isnan(train_data).all(axis=2).all(axis=1)]

    train_dataset = TensorDataset(torch.from_numpy(train_data).to(torch.float))
    return train_dataset


def get_data_loader_from_dataset(
    dataset_path,
    cfg,
    train=True,
    batch_size=256,
    shuffle=True,
    siamese=False,
):
    # local dataset
    if dataset_path.startswith("/home"):
        data = []
        labels = []
        if os.path.isdir(dataset_path):
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

                elif  "sleepedf20" in cfg.DATASET.TYPE  and cur_file.endswith("npy"):
                    cur_dataset = np.load(os.path.join(dataset_path, cur_file))
                    cur_data = cur_dataset[:, :-1]
                    cur_data = np.expand_dims(cur_data, axis=1)
                    cur_labels = cur_dataset[:, -1]

                else:
                    continue
                data.append(cur_data)
                labels.append(cur_labels)
        elif "semi" in dataset_path:
            dataset_path = np.load(dataset_path)
            x_train_labeled = dataset_path['data_labeled']
            y_train_labeled = dataset_path['labels_labeled']
            data.append(x_train_labeled)
            labels.append(y_train_labeled)

        data = np.concatenate(data, axis=0)
        labels = np.concatenate(labels, axis=0)
        logger.info("data info:", data.shape, labels.shape)
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
            dataset = SimpleDataset(data, labels)
        elif cfg.MODEL.TYPE == "TS2Vec" or "TSEncoder":
            dataset = TS2Vec_Train_Dataset(data, cfg)
        elif cfg.MODEL.TYPE == "Consistency":
            pass
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
