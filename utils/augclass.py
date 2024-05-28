import numpy as np
import tsaug
import torch
import time
from torch.nn.functional import interpolate


def totensor(x):
    return torch.from_numpy(x).type(torch.FloatTensor).cuda()


class cutout:
    def __init__(self, perc=0.1) -> None:
        self.perc = perc

    def __call__(self, ts):
        seq_len = ts.shape[1]
        new_ts = ts.clone()
        win_len = int(self.perc * seq_len)
        start = np.random.randint(0, seq_len - win_len - 1)
        end = start + win_len
        start = max(0, start)
        end = min(end, seq_len)
        new_ts[:, start:end, :] = 0.0
        return new_ts


class jitter:
    def __init__(self, sigma=0.3, random_sigma=False) -> None:
        self.sigma = sigma
        self.random_sigma = random_sigma

    def __call__(self, x):
        if self.random_sigma:
            # Random sigma for each sample in the batch
            sigmas_labels = np.random.uniform(
                0, self.sigma, size=(x.shape[0], 1, 1)
            )  # (batch_size, 1, 1)
            sigmas_labels = torch.tensor(
                sigmas_labels, dtype=torch.float32, device=x.device
            ).squeeze(-1)
            sigmas = torch.tensor(
                sigmas_labels, dtype=torch.float32, device=x.device
            ).expand(-1, x.shape[1], x.shape[2])

            # Generate noise
            noise = torch.randn_like(x)

            # Apply noise scaled by sigmas
            pertubated = x + noise * sigmas
            return pertubated, sigmas_labels

        else:
            return x + torch.normal(
                mean=0.0, std=self.sigma, size=x.shape, device=x.device
            )


class scaling:
    def __init__(self, sigma=0.5) -> None:
        self.sigma = sigma

    def __call__(self, x):
        factor = torch.normal(
            mean=1.0, std=self.sigma, size=(x.shape[0], x.shape[2])
        ).cuda()
        res = torch.multiply(x, torch.unsqueeze(factor, 1))
        return res


class timeshift:
    def __init__(self, shift_max=10):
        self.shift_max = shift_max

    def __call__(self, x):
        shift = np.random.randint(-self.shift_max, self.shift_max)
        return torch.roll(x, shifts=shift, dims=1)


class time_warp:
    def __init__(self, n_speed_change=100, max_speed_ratio=10) -> None:
        self.transform = tsaug.TimeWarp(
            n_speed_change=n_speed_change, max_speed_ratio=max_speed_ratio
        )

    def __call__(self, x_torch):
        x = x_torch.cpu().detach().numpy()
        x_tran = self.transform.augment(x)
        return totensor(x_tran.astype(np.float32))


class magnitude_warp:

    def __init__(self, n_speed_change: int = 100, max_speed_ratio=10) -> None:
        self.transform = tsaug.TimeWarp(
            n_speed_change=n_speed_change, max_speed_ratio=max_speed_ratio
        )

    def __call__(self, x_torch):
        x = x_torch.cpu().detach().numpy()
        # x_t = np.transpose(x, (0, 2, 1))
        x_tran = self.transform.augment(x)
        return totensor(x_tran.astype(np.float32))


class window_slice:
    def __init__(self, reduce_ratio=0.5, diff_len=True) -> None:
        self.reduce_ratio = reduce_ratio
        self.diff_len = diff_len

    def __call__(self, x):

        # begin = time.time()
        # x = torch.transpose(x, 2, 1)

        target_len = np.ceil(self.reduce_ratio * x.shape[2]).astype(int)
        if target_len >= x.shape[2]:
            return x
        if self.diff_len:
            starts = np.random.randint(
                low=0, high=x.shape[2] - target_len, size=(x.shape[0])
            ).astype(int)
            ends = (target_len + starts).astype(int)
            croped_x = torch.stack(
                [x[i, :, starts[i] : ends[i]] for i in range(x.shape[0])], 0
            )

        else:
            start = np.random.randint(low=0, high=x.shape[2] - target_len)
            end = target_len + start
            croped_x = x[:, :, start:end]

        ret = interpolate(croped_x, x.shape[2], mode="linear", align_corners=False)
        ret = torch.transpose(ret, 2, 1)
        # end = time.time()
        # old_window_slice()(x)
        # end2 = time.time()
        # print(end-begin,end2-end)
        return ret


class window_warp:
    def __init__(self, window_ratio=0.3, scales=[0.5, 2.0]) -> None:
        self.window_ratio = window_ratio
        self.scales = scales

    def __call__(self, x):
        # https://halshs.archives-ouvertes.fr/halshs-01357973/document
        x = x.cpu().detach().numpy()
        warp_scales = np.random.choice(self.scales, x.shape[0])
        warp_size = np.ceil(self.window_ratio * x.shape[1]).astype(int)
        window_steps = np.arange(warp_size)

        window_starts = np.random.randint(
            low=1, high=x.shape[1] - warp_size - 1, size=(x.shape[0])
        ).astype(int)
        window_ends = (window_starts + warp_size).astype(int)

        ret = np.zeros_like(x)
        for i, pat in enumerate(x):
            for dim in range(x.shape[2]):
                start_seg = pat[: window_starts[i], dim]
                window_seg = np.interp(
                    np.linspace(0, warp_size - 1, num=int(warp_size * warp_scales[i])),
                    window_steps,
                    pat[window_starts[i] : window_ends[i], dim],
                )
                end_seg = pat[window_ends[i] :, dim]
                warped = np.concatenate((start_seg, window_seg, end_seg))
                ret[i, :, dim] = np.interp(
                    np.arange(x.shape[1]),
                    np.linspace(0, x.shape[1] - 1.0, num=warped.size),
                    warped,
                ).T

        return torch.from_numpy(ret).type(torch.FloatTensor).cuda()
        # begin = time.time()
        # B, T, D = x_torch.size()
        # x = x_torch
        # # https://halshs.archives-ouvertes.fr/halshs-01357973/document
        # warp_scales = np.random.choice(self.scales, B)
        # warp_size = np.ceil(self.window_ratio * T).astype(int)
        # window_steps = np.arange(warp_size)

        # window_starts = np.random.randint(low=1, high=T - warp_size - 1, size=(B)).astype(
        #     int
        # )
        # window_ends = (window_starts + warp_size).astype(int)

        # rets = []

        # for i in range(x.shape[0]):
        #     window_seg = torch.unsqueeze(x[i, window_starts[i] : window_ends[i], :], 0)
        #     window_seg_inter = interpolate(
        #         window_seg,
        #         int(warp_size * warp_scales[i]),
        #         mode="linear",
        #         align_corners=False,
        #     )[0]
        #     start_seg = x[i, : window_starts[i], :]
        #     end_seg = x[i, window_ends[i] :, :]
        #     ret_i = torch.cat([start_seg, window_seg_inter, end_seg], -1)
        #     ret_i_inter = interpolate(
        #         torch.unsqueeze(ret_i, 0), T, mode="linear", align_corners=False
        #     )
        #     rets.append(ret_i_inter)

        # ret = torch.cat(rets, 0)
        # # end = time.time()
        # # old_window_warp()(x_torch)
        # # end2 = time.time()
        # # print(end-begin,end2-end)
        # return ret


class subsequence:
    def __init__(self) -> None:
        pass

    def __call__(self, x):
        ts = x
        seq_len = ts.shape[1]
        ts_l = x.size(1)
        crop_l = np.random.randint(low=2, high=ts_l + 1)
        new_ts = ts.clone()
        start = np.random.randint(ts_l - crop_l + 1)
        end = start + crop_l
        start = max(0, start)
        end = min(end, seq_len)
        new_ts[:, :start, :] = 0.0
        new_ts[:, end:, :] = 0.0
        return new_ts
