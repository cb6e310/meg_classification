import os
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


import numpy as np

from collections import OrderedDict
import getpass
from utils.helpers import (
    AverageMeter,
    accuracy,
    adjust_learning_rate,
    log_msg,
    timing_start,
    timing_end,
)

from utils.config import (
    show_cfg,
    log_msg,
    save_cfg,
)
from models.losses import (
    hierarchical_contrastive_loss,
)

from loguru import logger

from utils.validate import validate


def torch_pad_nan(arr, left=0, right=0, dim=0):
    if left > 0:
        padshape = list(arr.shape)
        padshape[dim] = left
        arr = torch.cat((torch.full(padshape, np.nan), arr), dim=dim)
    if right > 0:
        padshape = list(arr.shape)
        padshape[dim] = right
        arr = torch.cat((arr, torch.full(padshape, np.nan)), dim=dim)
    return arr


def pad_nan_to_target(array, target_length, axis=0, both_side=False):
    assert array.dtype in [np.float16, np.float32, np.float64]
    pad_size = target_length - array.shape[axis]
    if pad_size <= 0:
        return array
    npad = [(0, 0)] * array.ndim
    if both_side:
        npad[axis] = (pad_size // 2, pad_size - pad_size // 2)
    else:
        npad[axis] = (0, pad_size)
    return np.pad(array, pad_width=npad, mode="constant", constant_values=np.nan)


def split_with_nan(x, sections, axis=0):
    assert x.dtype in [np.float16, np.float32, np.float64]
    arrs = np.array_split(x, sections, axis=axis)
    target_length = arrs[0].shape[axis]
    for i in range(len(arrs)):
        arrs[i] = pad_nan_to_target(arrs[i], target_length, axis=axis)
    return arrs


def take_per_row(A, indx, num_elem):
    all_indx = indx[:, None] + np.arange(num_elem)
    return A[torch.arange(all_indx.shape[0])[:, None], all_indx]


def centerize_vary_length_series(x):
    prefix_zeros = np.argmax(~np.isnan(x).all(axis=-1), axis=1)
    suffix_zeros = np.argmax(~np.isnan(x[:, ::-1]).all(axis=-1), axis=1)
    offset = (prefix_zeros + suffix_zeros) // 2 - prefix_zeros
    rows, column_indices = np.ogrid[: x.shape[0], : x.shape[1]]
    offset[offset < 0] += x.shape[1]
    column_indices = column_indices - offset[:, np.newaxis]
    return x[rows, column_indices]


def data_dropout(arr, p):
    B, T = arr.shape[0], arr.shape[1]
    mask = np.full(B * T, False, dtype=np.bool)
    ele_sel = np.random.choice(B * T, size=int(B * T * p), replace=False)
    mask[ele_sel] = True
    res = arr.copy()
    res[mask.reshape(B, T)] = np.nan
    return res


class TS2VecTrainer:
    def __init__(self, experiment_name, model, criterion, train_loader, val_loader, cfg):
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.model = model
        self.optimizer = self.init_optimizer(cfg)
        self.best_acc = -1
        self.best_epoch = 0
        self.resume_epoch = -1

        username = getpass.getuser()
        # init loggers
        cur_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())
        experiment_name = experiment_name + "_" + cur_time
        self.log_path = os.path.join(cfg.LOG.PREFIX, experiment_name)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        # self.tf_writer = SummaryWriter(os.path.join(self.log_path, "train.events"))

        if not cfg.EXPERIMENT.RESUME:
            save_cfg(self.cfg, os.path.join(self.log_path, "config.yaml"))
            os.mkdir(os.path.join(self.log_path, "checkpoints"))

        # Choose here if you want to start training from a previous snapshot (None for new training)
        if cfg.EXPERIMENT.RESUME:
            if cfg.EXPERIMENT.CHECKPOINT != "":
                chkp_path = os.path.join(
                    "results", cfg.EXPERIMENT.CHECKPOINT, "checkpoints"
                )
                chkps = [f for f in os.listdir(chkp_path) if f[:4] == "chkp"]

                # Find which snapshot to restore
                if chkp_idx is None:
                    chosen_chkp = "current_chkp.tar"
                else:
                    chosen_chkp = np.sort(chkps)[cfg.EXPERIMENT.CHKP_IDX]
                chosen_chkp = os.path.join(
                    "results", cfg.EXPERIMENT.CHECKPOINT, "checkpoints", chosen_chkp
                )

                print(log_msg("Loading model from {}".format(chosen_chkp), "INFO"))
                print(
                    log_msg("Current epoch: {}".format(cfg.EXPERIMENT.CHKP_IDX), "INFO")
                )

                self.chkp = torch.load(chosen_chkp)
                self.model.load_state_dict(self.chkp["model_state_dict"])
                self.optimizer.load_state_dict(self.chkp["optimizer"])
                self.best_acc = self.chkp["best_acc"]
                self.best_epoch = self.chkp["epoch"]
                self.resume_epoch = self.chkp["epoch"]

                if self.chkp["dataset"] != cfg.DATASET.TYPE:
                    print(
                        log_msg(
                            "ERROR: dataset in checkpoint is different from current dataset",
                            "ERROR",
                        )
                    )
                    exit()
                if self.chkp["model"] != cfg.MODEL.TYPE:
                    print(
                        log_msg(
                            "ERROR: model in checkpoint {} is different from current model {}".format,
                            "ERROR",
                        )
                    )
                    exit()
            else:
                chosen_chkp = None
                print(
                    "Resume is True but no checkpoint path is given. Start from scratch."
                )
        else:
            chosen_chkp = None
            print("Start from scratch.")

        self.device = "cuda"
        self.lr = cfg.SOLVER.LR
        self.batch_size = cfg.SOLVER.BATCH_SIZE
        self.max_train_length = cfg.MODEL.ARGS.MAX_TRAIN_LENGTH
        self.temporal_unit = cfg.MODEL.ARGS.TEMPORAL_UNIT

        self.net = torch.optim.swa_utils.AveragedModel(self.model)
        self.net.update_parameters(self.model)

        self.n_epochs = 0
        self.n_iters = 0

    def init_optimizer(self, cfg):
        if cfg.SOLVER.TYPE == "SGD":
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=cfg.SOLVER.LR,
                momentum=cfg.SOLVER.MOMENTUM,
                weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            )
        elif cfg.SOLVER.TYPE == "Adam":
            optimizer = optim.Adam(
                self.model.module.get_learnable_parameters(),
                lr=cfg.SOLVER.LR,
                weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            )
        else:
            raise NotImplementedError(cfg.SOLVER.TYPE)
        return optimizer

    def train(self, n_iters=None):
        """Training the TS2Vec model.

        Args:
            train_data (numpy.ndarray): The training data. It should have a shape of (n_instance, n_timestamps, n_features). All missing data should be set to NaN.
            n_epochs (Union[int, NoneType]): The number of epochs. When this reaches, the training stops.
            n_iters (Union[int, NoneType]): The number of iterations. When this reaches, the training stops. If both n_epochs and n_iters are not specified, a default setting would be used that sets n_iters to 200 for a dataset with size <= 100000, 600 otherwise.
            verbose (bool): Whether to print the training loss after each epoch.

        Returns:
            loss_log: a list containing the training losses on each epoch.
        """

        loss_log = []
        n_epochs = self.cfg.SOLVER.EPOCHS

        
        while True:
            if n_epochs is not None and self.n_epochs >= n_epochs:
                break

            cum_loss = 0
            n_epoch_iters = 0

            interrupted = False
            for batch in self.train_loader:
                if n_iters is not None and self.n_iters >= n_iters:
                    interrupted = True
                    break

                x = batch[0]
                if (
                    self.max_train_length is not None
                    and x.size(1) > self.max_train_length
                ):
                    window_offset = np.random.randint(
                        x.size(1) - self.max_train_length + 1
                    )
                    x = x[:, window_offset : window_offset + self.max_train_length]
                x = x.to(self.device)

                ts_l = x.size(1)
                crop_l = np.random.randint(
                    low=2 ** (self.temporal_unit + 1), high=ts_l + 1
                )
                crop_left = np.random.randint(ts_l - crop_l + 1)
                crop_right = crop_left + crop_l
                crop_eleft = np.random.randint(crop_left + 1)
                crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
                crop_offset = np.random.randint(
                    low=-crop_eleft, high=ts_l - crop_eright + 1, size=x.size(0)
                )

                self.optimizer.zero_grad()
                out1 = self.model(
                    take_per_row(x, crop_offset + crop_eleft, crop_right - crop_eleft)
                )
                out1 = out1[:, -crop_l:]

                out2 = self.model(
                    take_per_row(x, crop_offset + crop_left, crop_eright - crop_left)
                )
                out2 = out2[:, :crop_l]
                logger.debug(f"out1: {out1.shape}, out2: {out2.shape}")
                loss = hierarchical_contrastive_loss(
                    out1, out2, temporal_unit=self.temporal_unit
                )

                loss.backward()
                self.optimizer.step()
                self.net.update_parameters(self.model)

                cum_loss += loss.item()
                n_epoch_iters += 1

                self.n_iters += 1
            if interrupted:
                break

            cum_loss /= n_epoch_iters
            loss_log.append(cum_loss)
            logger.info(f"Epoch #{self.n_epochs}: loss={cum_loss}")
            self.n_epochs += 1

            # validate
            if self.cfg.EXPERIMENT.TASK == "pretext":
                test_acc, test_loss = 0, 0

            # saving checkpoint
            # saving occasional checkpoints

            state = {
                "epoch": self.n_epochs,
                "model_state_dict": self.model.state_dict(),
                "model": self.cfg.MODEL.TYPE,
                "optimizer": self.optimizer.state_dict(),
                "best_acc": self.best_acc,
                "dataset": self.cfg.DATASET.TYPE,
            }

            if (self.n_epochs + 1) % self.cfg.EXPERIMENT.CHECKPOINT_GAP == 0:
                logger.info("Saving checkpoint to {}".format(self.log_path))
                chkp_path = os.path.join(
                    self.log_path,
                    "checkpoints",
                    "epoch_{}_chkp.tar".format(self.n_epochs),
                )
                torch.save(state, chkp_path)

        return loss_log

    def _eval_with_pooling(self, x, mask=None, slicing=None, encoding_window=None):
        out = self.net(x.to(self.device, non_blocking=True), mask)
        if encoding_window == "full_series":
            if slicing is not None:
                out = out[:, slicing]
            out = F.max_pool1d(
                out.transpose(1, 2),
                kernel_size=out.size(1),
            ).transpose(1, 2)

        elif isinstance(encoding_window, int):
            out = F.max_pool1d(
                out.transpose(1, 2),
                kernel_size=encoding_window,
                stride=1,
                padding=encoding_window // 2,
            ).transpose(1, 2)
            if encoding_window % 2 == 0:
                out = out[:, :-1]
            if slicing is not None:
                out = out[:, slicing]

        elif encoding_window == "multiscale":
            p = 0
            reprs = []
            while (1 << p) + 1 < out.size(1):
                t_out = F.max_pool1d(
                    out.transpose(1, 2),
                    kernel_size=(1 << (p + 1)) + 1,
                    stride=1,
                    padding=1 << p,
                ).transpose(1, 2)
                if slicing is not None:
                    t_out = t_out[:, slicing]
                reprs.append(t_out)
                p += 1
            out = torch.cat(reprs, dim=-1)

        else:
            if slicing is not None:
                out = out[:, slicing]

        return out.cpu()

    def encode(
        self,
        data,
        mask=None,
        encoding_window=None,
        causal=False,
        sliding_length=None,
        sliding_padding=0,
        batch_size=None,
    ):
        """Compute representations using the model.

        Args:
            data (numpy.ndarray): This should have a shape of (n_instance, n_timestamps, n_features). All missing data should be set to NaN.
            mask (str): The mask used by encoder can be specified with this parameter. This can be set to 'binomial', 'continuous', 'all_true', 'all_false' or 'mask_last'.
            encoding_window (Union[str, int]): When this param is specified, the computed representation would the max pooling over this window. This can be set to 'full_series', 'multiscale' or an integer specifying the pooling kernel size.
            causal (bool): When this param is set to True, the future informations would not be encoded into representation of each timestamp.
            sliding_length (Union[int, NoneType]): The length of sliding window. When this param is specified, a sliding inference would be applied on the time series.
            sliding_padding (int): This param specifies the contextual data length used for inference every sliding windows.
            batch_size (Union[int, NoneType]): The batch size used for inference. If not specified, this would be the same batch size as training.

        Returns:
            repr: The representations for data.
        """
        assert self.net is not None, "please train or load a net first"
        assert data.ndim == 3
        if batch_size is None:
            batch_size = self.batch_size
        n_samples, ts_l, _ = data.shape

        org_training = self.net.training
        self.net.eval()

        with torch.no_grad():
            output = []
            for batch in self.train_loader:
                x = batch[0]
                if sliding_length is not None:
                    reprs = []
                    if n_samples < batch_size:
                        calc_buffer = []
                        calc_buffer_l = 0
                    for i in range(0, ts_l, sliding_length):
                        l = i - sliding_padding
                        r = i + sliding_length + (sliding_padding if not causal else 0)
                        x_sliding = torch_pad_nan(
                            x[:, max(l, 0) : min(r, ts_l)],
                            left=-l if l < 0 else 0,
                            right=r - ts_l if r > ts_l else 0,
                            dim=1,
                        )
                        if n_samples < batch_size:
                            if calc_buffer_l + n_samples > batch_size:
                                out = self._eval_with_pooling(
                                    torch.cat(calc_buffer, dim=0),
                                    mask,
                                    slicing=slice(
                                        sliding_padding, sliding_padding + sliding_length
                                    ),
                                    encoding_window=encoding_window,
                                )
                                reprs += torch.split(out, n_samples)
                                calc_buffer = []
                                calc_buffer_l = 0
                            calc_buffer.append(x_sliding)
                            calc_buffer_l += n_samples
                        else:
                            out = self._eval_with_pooling(
                                x_sliding,
                                mask,
                                slicing=slice(
                                    sliding_padding, sliding_padding + sliding_length
                                ),
                                encoding_window=encoding_window,
                            )
                            reprs.append(out)

                    if n_samples < batch_size:
                        if calc_buffer_l > 0:
                            out = self._eval_with_pooling(
                                torch.cat(calc_buffer, dim=0),
                                mask,
                                slicing=slice(
                                    sliding_padding, sliding_padding + sliding_length
                                ),
                                encoding_window=encoding_window,
                            )
                            reprs += torch.split(out, n_samples)
                            calc_buffer = []
                            calc_buffer_l = 0

                    out = torch.cat(reprs, dim=1)
                    if encoding_window == "full_series":
                        out = F.max_pool1d(
                            out.transpose(1, 2).contiguous(),
                            kernel_size=out.size(1),
                        ).squeeze(1)
                else:
                    out = self._eval_with_pooling(
                        x, mask, encoding_window=encoding_window
                    )
                    if encoding_window == "full_series":
                        out = out.squeeze(1)

                output.append(out)

            output = torch.cat(output, dim=0)

        self.net.train(org_training)
        return output.numpy()
