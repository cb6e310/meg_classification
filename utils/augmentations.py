import numpy as np

import torch
from scipy.interpolate import CubicSpline
import random

from utils.augclass import *
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torchvision.transforms import Compose

from loguru import logger


class AutoAUG(Module):
    def __init__(self, cfg) -> None:
        super(AutoAUG, self).__init__()
        self.cfg = cfg
        self.all_augs = [
            crop(resize=cfg.DATASET.POINTS),
            timeshift(),
            jitter(),
            scaling(),
            window_warp(),
        ]

        self.normal_augs_wo_spec = [
            crop(resize=cfg.DATASET.POINTS),
            timeshift(),
            window_warp(),
        ]

        self.sensitive_base_augs = [
            crop(resize=cfg.DATASET.POINTS),
        ]

    @staticmethod
    def random_jitter(x, max_sigma=0.5):
        output = jitter(sigma=max_sigma, random_sigma=True)(x)
        return output

    @staticmethod
    def random_timereverse(x):
        output = TimeReverse()(x)
        return output

    @staticmethod
    def random_signflip(x):
        output = SignFlip()(x)
        return output

    @staticmethod
    def random_ftsurrogate(x):
        output = FTSurrogate()(x)
        return output

    @staticmethod
    def random_timeshift(x):
        output = TimeShift(max_shift=0.4)(x)
        return output

    @staticmethod
    def random_scaling(x):
        output = Scaling()(x)
        return output

    @staticmethod
    def random_frequencyshift(x, sfreq):
        sfreq
        output = FrequencyShift(sfreq=100)(x)
        return output

    def forward(self, x, step=None):
        # x shape: (batch, seq_len, channels)
        x = x.transpose(1, 2)

        if self.training and step is None and self.cfg.MODEL.TYPE == "current":
            raise ValueError("step is required during training")
        if self.training and self.cfg.MODEL.TYPE != "CurrentCLR":
            transform = Compose(self.all_augs)
            aug1 = transform(x)
            aug2 = transform(x)
            aug1 = aug1.transpose(1, 2)
            aug2 = aug2.transpose(1, 2)
            return aug1, aug2

        if step == "clr":
            # transform = Compose(self.normal_augs_wo_spec)
            base_aug = Compose(self.sensitive_base_augs)
            x1 = base_aug(x)
            x2 = base_aug(x)
            aug1, _ = self.random_signflip(x1)
            aug2, _ = self.random_signflip(x2)
            aug1 = aug1.transpose(1, 2)
            aug2 = aug2.transpose(1, 2)
            return aug1, aug2

        elif step == "rec":
            aug1 = self.random_jitter(x, max_sigma=1)[0]
            # aug1=x
            aug2 = x
            aug1 = aug1.transpose(1, 2)
            aug2 = aug2.transpose(1, 2)

            return aug1, aug2

        elif step == "cls":
            # 1/3 for jitter and 1/3 for cutout and 1/3 for no spec
            transform = Compose(self.normal_augs_wo_spec)
            spec_transform_jitter = Compose([jitter()])
            spec_transform_cutout = Compose([cutout()])

            batch_size = x.size(0)
            labels = torch.zeros(batch_size, dtype=torch.long)
            indices = torch.randperm(batch_size)

            third_batch = batch_size // 3

            noise_indices = indices[:third_batch]
            x[noise_indices] = spec_transform_jitter(x[noise_indices])
            labels[noise_indices] = 1

            cutout_indices = indices[third_batch : 2 * third_batch]
            x[cutout_indices] = spec_transform_cutout(x[cutout_indices])
            labels[cutout_indices] = 2

            aug1 = transform(x)
            aug1 = aug1.transpose(1, 2)
            return aug1, labels

        elif step == "pred":
            # transform = Compose(self.normal_augs_wo_spec)
            # x = transform(x)
            spec_x, labels = self.random_signflip(x)
            spec_x = spec_x.transpose(1, 2)
            return spec_x, labels

        else:
            raise ValueError("step should be one of 'clr', 'rec', 'cls', 'pred'")
