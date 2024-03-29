import numpy as np

import torch
from scipy.interpolate import CubicSpline
import random

from utils.augclass import *
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter

from loguru import logger


class AutoAUG(Module):
    def __init__(
        self, aug_p1=0.2, aug_p2=0.0, used_augs=None, device=None, dtype=None
    ) -> None:
        super(AutoAUG, self).__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        all_augs = [
            subsequence(),
            cutout(),
            jitter(),
            scaling(),
            time_warp(),
            window_slice(),
            window_warp(),
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
            logger.debug("yep")
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
