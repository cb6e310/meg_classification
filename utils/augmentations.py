import numpy as np
import torch
from scipy.interpolate import CubicSpline
import random 

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

    def forward(self, xt):
        x, t = xt
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



def DataTransform(sample):
    aug_1 = scaling(sample, sigma=1.1)
    aug_2 = jitter(permutation(sample, max_segments=5, seg_mode="random"), sigma=0.8)

    # You can choose any augmentation transformations or combinations listed as follows:
    # x_jitter = jitter(sample, sigma=0.8) #sigma: [0.01, 0.5]
    # x_scaling = scaling(sample, sigma=1.1) #sigma: [0.1, 2.]
    # x_permutation = permutation(sample, max_segments=5, seg_mode="random") #max_segments: [3., 6.]
    # x_rotation = rotation(sample)
    # x_magnitude_warp = magnitude_warp(sample, sigma=0.2, knot=4) #sigma: [0.1, 2.], knot: [3, 5]
    # x_time_warp = time_warp(sample, sigma=0.2, knot=4) #sigma: [0.01, 0.5], knot: [3, 5]
    # x_window_slice = window_slice(sample, reduce_ratio=0.9) #reduce_ratio: [0.95, 0.6]

    return aug_1, aug_2


def jitter(x, sigma=0.8):
    x_new = x + np.random.normal(loc=0.0, scale=sigma, size=x.shape)
    return x_new


def scaling(x, sigma=0.1):
    x=x.squeeze()
    scaler = np.random.normal(loc=1.0, scale=sigma, size=(1, x.shape[1]))
    noise = np.matmul(np.ones((x.shape[2], 1)), scaler)
    x = x.permute(0, 2, 1)
    x_new = x.clone()
    # for i in range(0, x.shape[0]):
    #     x_new[i] = x[i] * noise
    x_new = x * noise
    return x_new.permute(0, 2, 1).unsqueeze(3)


def permutation(x, max_segments=5, seg_mode="random"):
    orig_steps = np.arange(x.shape[2])
    # print(0, orig_steps)
    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))
    x_new = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(
                    x.shape[2] - 2, num_segs[i] - 1, replace=False
                )
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            # print(1, num_segs, splits)
            random.shuffle(splits)
            # print(2, splits)
            warp = np.concatenate(splits).ravel()
            x_new[i] = pat[0, warp]
        else:
            x_new[i] = pat
    return torch.from_numpy(x_new)


def rotation(x):
    flip = np.random.choice([-1, 1], size=(x.shape[0], x.shape[2]))
    rotate_axis = np.arange(x.shape[2])
    np.random.shuffle(rotate_axis)
    x_new = torch.from_numpy(flip)[:, np.newaxis, :] * x[:, :, rotate_axis]
    return x_new


def magnitude_warp(x, sigma=0.2, knot=4):
    orig_steps = np.arange(x.shape[1])
    random_warps = np.random.normal(
        loc=1.0, scale=sigma, size=(x.shape[0], knot + 2, x.shape[2])
    )
    warp_steps = (
        np.ones((x.shape[2], 1)) * (np.linspace(0, x.shape[1] - 1.0, num=knot + 2))
    ).T
    x_new = np.zeros_like(x)
    for i, pat in enumerate(x):
        li = []
        for dim in range(x.shape[2]):
            li.append(
                CubicSpline(warp_steps[:, dim], random_warps[i, :, dim])(orig_steps)
            )
        warper = np.array(li).T
        x_new[i] = pat * warper
    return torch.from_numpy(x_new)


def time_warp(x, sigma=0.2, knot=4):
    orig_steps = np.arange(x.shape[1])
    random_warps = np.random.normal(
        loc=1.0, scale=sigma, size=(x.shape[0], knot + 2, x.shape[2])
    )
    warp_steps = (
        np.ones((x.shape[2], 1)) * (np.linspace(0, x.shape[1] - 1.0, num=knot + 2))
    ).T
    x_new = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            time_warp = CubicSpline(
                warp_steps[:, dim], warp_steps[:, dim] * random_warps[i, :, dim]
            )(orig_steps)
            scale = (x.shape[1] - 1) / time_warp[-1]
            x_new[i, :, dim] = np.interp(
                orig_steps, np.clip(scale * time_warp, 0, x.shape[1] - 1), pat[:, dim]
            ).T
    return torch.from_numpy(x_new)


def window_slice(x, reduce_ratio=0.9):
    target_len = np.ceil(reduce_ratio * x.shape[1]).astype(int)
    if target_len >= x.shape[1]:
        return x
    starts = np.random.randint(
        low=0, high=x.shape[1] - target_len, size=(x.shape[0])
    ).astype(int)
    ends = (target_len + starts).astype(int)
    x_new = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            x_new[i, :, dim] = np.interp(
                np.linspace(0, target_len, num=x.shape[1]),
                np.arange(target_len),
                pat[starts[i] : ends[i], dim],
            ).T
    return x_new
