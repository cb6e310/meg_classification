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
        output = TimeReverse(random=True)(x)
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
            spec_x, labels = self.random_timereverse(x)
            spec_x = self.sensitive_base_augs(spec_x)
            # spec_x = self.Normalize(spec_x)
            spec_x = spec_x.transpose(1, 2)
            return spec_x, labels
        else:
            # linear eval
            # x = self.Normalize(x)
            x = x.transpose(1, 2)
            return x

class InfoTSAUG(Module):
    def __init__(
        self,cfg, aug_p1=0.2, aug_p2=0.0, used_augs=None, device=None, dtype=None
    ) -> None:
        super(InfoTSAUG, self).__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        all_augs = [
            jitter(),
            TimeReverse(random=False),
            TimeShift(random=False),
            SignFlip(random=False),
        ]

        if used_augs is not None:
            self.augs = []
            for i in range(len(used_augs)):
                if used_augs[i]:
                    self.augs.append(all_augs[i])
        else:
            self.augs = all_augs
        self.weight = Parameter(torch.empty((2, len(self.augs)), **factory_kwargs))
        self.reset_parameters()
        self.aug_p1 = aug_p1
        self.aug_p2 = aug_p2

    def get_sampling(self, temperature=1.0, bias=0.0):
        if self.training:
            bias = bias + 0.0001  # If bias is 0, we run into problems
            eps = (bias - (1 - bias)) * torch.rand(self.weight.size()) + (1 - bias)
            gate_inputs = torch.log(eps) - torch.log(1 - eps)
            gate_inputs = gate_inputs.cuda()
            gate_inputs = (gate_inputs + self.weight) / temperature
            # para = torch.sigmoid(gate_inputs)
            para = torch.softmax(gate_inputs, -1)
            return para
        else:
            return torch.softmax(self.weight, -1)

    def reset_parameters(self) -> None:
        # init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        torch.nn.init.normal_(self.weight, mean=0.0, std=0.01)

    def forward(self, x):
        if self.aug_p1 == 0 and self.aug_p2 == 0:
            return x.clone(), x.clone()
        para = self.get_sampling()

        if random.random() > self.aug_p1 and self.training:
            aug1 = x.clone()
        else:
            xs1_list = []
            for aug in self.augs:
                xs1_list.append(aug(x))
            xs1 = torch.stack(xs1_list, 0)
            # logger.debug(xs1.shape)
            xs1_flattern = torch.reshape(
                xs1, (xs1.shape[0], xs1.shape[1] * xs1.shape[2] * xs1.shape[3])
            )
            # logger.debug(xs1_flattern.shape)
            # logger.debug(para[0])
            # logger.debug("--")
            aug1 = torch.reshape(torch.unsqueeze(para[0], -1) * xs1_flattern, xs1.shape)
            aug1 = torch.sum(aug1, 0)

        aug2 = x.clone()

        return aug1, aug2
