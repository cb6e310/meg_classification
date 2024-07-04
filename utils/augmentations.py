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
            # window_warp(),
            # Normalize(),
            # scaling(max_sigma=0.2),
        ]

        self.normal_augs_wo_spec = [
            crop(resize=cfg.DATASET.POINTS),
            TimeReverse(random=False),
            TimeShift(random=False),
            SignFlip(random=False),
        ]

        self.sensitive_base_augs = Compose([
            crop(resize=cfg.DATASET.POINTS),
        ])

        self.Normalize = Compose([Normalize()])

    def random_jitter(
        self,
        x,
    ):
        mean = self.cfg.DATASET.MEAN
        std = self.cfg.DATASET.STD
        x_raw = x * std + mean
        max_sigma = std * 0.2
        output, labels = jitter(sigma=max_sigma, random_sigma=True)(x_raw)
        output = (output - mean) / std
        return output, labels

    def random_scaling(self, x, max_sigma=0.5):
        mean = self.cfg.DATASET.MEAN
        std = self.cfg.DATASET.STD
        x_raw = x * std + mean
        output, labels = scaling(max_sigma=max_sigma)(x_raw)
        output = (output - mean) / std
        return output, labels

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
    def random_frequencyshift(x, sfreq=100):

        output = FrequencyShift(sfreq=sfreq)(x)
        return output

    @staticmethod
    def random_drift(x, max_drift_points=20):
        output = Drift(max_drift=0.4, max_drift_points=max_drift_points)(x)
        return output

    def forward(self, x, step=None):
        # in aug, it will change x shape to (batch, seq_len, channels)
        x = x.transpose(1, 2)

        if self.training and step is None and self.cfg.MODEL.TYPE == "current":
            raise ValueError("step is required during training")
        # if (
        #     self.training
        #     and self.cfg.MODEL.TYPE != "CurrentCLR"
        #     and self.cfg.MODEL.TYPE != "CurrentSimCLR"
        # ):
        #     transform = Compose(self.normal_augs_wo_spec)
        #     aug1 = transform(x)
        #     aug2 = transform(x)
        #     aug1 = aug1.transpose(1, 2)
        #     aug2 = aug2.transpose(1, 2)
        #     return aug1, aug2
        if step == "equimod_clr":
            aug1, _= self.random_jitter(x)
            aug2, _= self.random_jitter(x)
            aug1= self.sensitive_base_augs(aug1)
            aug2= self.sensitive_base_augs(aug2)
            aug1 = aug1.transpose(1, 2)
            aug2 = aug2.transpose(1, 2)
            return aug1, aug2

        if step == "clr":
            base_aug = Compose(self.normal_augs_wo_spec)

            # base_aug = Compose(self.sensitive_base_augs)
            x1 = base_aug(x)
            x2 = base_aug(x)
            aug1 = x1
            aug2 = x2
            # aug1, _ = self.random_jitter(x1, )
            # aug1 = self.Normalize(aug1)
            # aug2 = self.Normalize(aug2)
            # aug2, _ = self.random_jitter(x2, max_sigma=0.2)

            # aug1, _ = self.random_scaling(x1)
            # aug2, _ = self.random_scaling(x2)
            aug1 = aug1.transpose(1, 2)
            aug2 = aug2.transpose(1, 2)
            return aug1, aug2

        elif step == "rec":
            aug1, _ = self.random_jitter(
                x,
            )
            # aug1=x
            aug2, _ = self.random_jitter(
                x,
            )
            # aug1 = self.Normalize(aug1)
            # aug2 = self.Normalize(aug2)
            aug1 = aug1.transpose(1, 2)
            aug2 = aug2.transpose(1, 2)

            return aug1, aug2

        elif step == "pred":
            # transform = Compose(self.normal_augs_wo_spec)
            # x = transform(x)
            spec_x, labels = self.random_jitter(x)
            # spec_x = self.Normalize(spec_x)
            spec_x = spec_x.transpose(1, 2)
            return spec_x, labels

        elif step == "equimod_pred":
            spec_x, labels = self.random_jitter(x)
            spec_x = self.sensitive_base_augs(spec_x)
            # spec_x = self.Normalize(spec_x)
            spec_x = spec_x.transpose(1, 2)
            return spec_x, labels
        else:
            # linear eval
            # x = self.Normalize(x)
            x = x.transpose(1, 2)
            return x
