import os
import time
from tqdm import tqdm
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import getpass

from utils.helpers import (
    AverageMeter,
    accuracy,
    adjust_learning_rate,
    log_msg,
)

from utils.config import (
    show_cfg,
    log_msg,
    save_cfg,
)

from models.losses import (
    OrthLoss,
)

from loguru import logger
from models.losses import hierarchical_contrastive_loss
from utils.validate import validate


class Ts2vecTrainer:
    def __init__(
        self, experiment_name, model, criterion, train_loader, val_loader, aug, cfg
    ):
        self.current_ckpt_path = ""
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.clr_criterion = criterion
        # self.pred_criterion = torch.nn.BCEWithLogitsLoss().cuda()
        self.temporal_unit = 0.5
        self.model = model
        self.aug = aug

        # check model name is BYOL

        self.optimizer = self.init_optimizer(cfg)
        self.scheduler = self.init_scheduler(cfg)
        self.best_acc = -1
        self.best_epoch = 0
        self.resume_epoch = -1

        self.current_epoch = 0

        username = getpass.getuser()
        # init loggers
        if not cfg.EXPERIMENT.DEBUG:

            if not cfg.EXPERIMENT.RESUME:
                # cur_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())
                # experiment_name = experiment_name + "_" + cur_time
                self.log_path = os.path.join(cfg.LOG.PREFIX, experiment_name)
                if not os.path.exists(self.log_path):
                    os.makedirs(self.log_path)
                save_cfg(self.cfg, os.path.join(self.log_path, "config.yaml"))
                os.makedirs(os.path.join(self.log_path, "checkpoints"),exist_ok=True)
                chosen_chkp = None
                logger.info("Start from scratch.")

            # Choose here if you want to start training from a previous snapshot (None for new training)
            else:
                self.log_path = os.path.join(cfg.LOG.PREFIX, cfg.EXPERIMENT.CHECKPOINT)
                if cfg.EXPERIMENT.CHECKPOINT != "":
                    chkp_path = os.path.join(
                        cfg.LOG.PREFIX, cfg.EXPERIMENT.CHECKPOINT, "checkpoints"
                    )
                    chkps = [f for f in os.listdir(chkp_path) if f[:4] == "chkp"]

                    # Find which snapshot to restore
                    if cfg.EXPERIMENT.CHKP_IDX is None:
                        chosen_chkp = "current_chkp.tar"
                    else:
                        chosen_chkp = "epoch_{}_0_chkp.tar".format(
                            cfg.EXPERIMENT.CHKP_IDX
                        )
                    chosen_chkp = os.path.join(
                        cfg.LOG.PREFIX,
                        cfg.EXPERIMENT.CHECKPOINT,
                        "checkpoints",
                        chosen_chkp,
                    )

                    print(log_msg("Loading model from {}".format(chosen_chkp), "INFO"))
                    print(
                        log_msg(
                            "Current epoch: {}".format(cfg.EXPERIMENT.CHKP_IDX), "INFO"
                        )
                    )

                    self.chkp = torch.load(chosen_chkp)
                    self.model.load_state_dict(self.chkp["model_state_dict"])
                    self.optimizer.load_state_dict(self.chkp["optimizer"])
                    self.scheduler.load_state_dict(self.chkp["scheduler"])
                    # self.best_acc = self.chkp["best_acc"]
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
            self.writer = SummaryWriter(os.path.join(self.log_path, "writer"))
            logger.info("Log path: {}".format(self.log_path))

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

    def init_scheduler(self, cfg):
        # check optimizer
        assert self.optimizer is not None

        if cfg.SOLVER.SCHEDULER.TYPE == "step":
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=cfg.SOLVER.SCHEDULER.STEP_SIZE,
                gamma=cfg.SOLVER.SCHEDULER.GAMMA,
            )
        elif cfg.SOLVER.SCHEDULER.TYPE == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=cfg.SOLVER.EPOCHS
            )
        elif cfg.SOLVER.SCHEDULER.TYPE == "ExponentialLR":
            scheduler = optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=cfg.SOLVER.SCHEDULER.GAMMA
            )
        else:
            raise NotImplementedError(cfg.SOLVER.SCHEDULER.TYPE)
        return scheduler

    def log(self, epoch, it, log_dict):
        # import wandb

        # # wandb.log({"current lr": lr})
        # wandb.log(log_dict)
        # resolve log_dict
        log_dict = {k: v.avg for k, v in log_dict.items()}
        self.writer.add_scalars(
            "losses", log_dict, global_step=epoch * len(self.train_loader) + it
        )

    def log_worklog(self, epoch, log_dict):
        with open(os.path.join(self.log_path, "worklog.txt"), "a") as writer:
            lines = ["epoch: {}\t".format(epoch)]
            for k, v in log_dict.items():
                lines.append("{}: {:.4f}\t".format(k, v))
            lines.append(os.linesep)
            writer.writelines(lines)

    def train(self, repetition_id=0):
        self.current_epoch = 0

        if self.resume_epoch != -1:
            self.current_epoch = self.resume_epoch + 1

        while self.current_epoch < self.cfg.SOLVER.EPOCHS:
            self.train_epoch(self.current_epoch, repetition_id=repetition_id)
            self.current_epoch += 1
        print(
            log_msg(
                "repetition_id:{} Best accuracy:{} Epoch:{}".format(
                    repetition_id, self.best_acc, self.best_epoch
                ),
                "EVAL",
            )
        )

        with open(os.path.join(self.log_path, "worklog.txt"), "a") as writer:
            writer.write(
                "repetition_id:{}\tbest_acc:{:.4f}\tepoch:{}".format(
                    repetition_id, float(self.best_acc), self.best_epoch
                )
            )
            writer.write(os.linesep + "-" * 25 + os.linesep)
        return self.best_acc, self.current_ckpt_path

    def train_epoch(self, epoch, repetition_id=0):
        lr = self.cfg.SOLVER.LR
        train_meters = {
            "training_time": AverageMeter(),
            # "data_time": AverageMeter(),
            "loss_clr": AverageMeter(),
            "loss_pred": AverageMeter(),
            "top1": AverageMeter(),
        }
        num_iter = len(self.train_loader)
        pbar = tqdm(range(num_iter))

        # train loops
        self.model.train()
        for idx, data in enumerate(self.train_loader):
            msg, imgs = self.after_warmup_iter(data, epoch, train_meters, idx)
            pbar.set_description(log_msg(msg, "TRAIN"))
            pbar.update()
            if not self.cfg.EXPERIMENT.DEBUG:
                self.log(epoch, idx, train_meters)

        # update lr
        # get current lr
        lr = self.optimizer.param_groups[0]["lr"]
        if lr > self.cfg.SOLVER.SCHEDULER.MIN_LR:
            self.scheduler.step()

        pbar.close()

        # validate
        if self.cfg.EXPERIMENT.TASK == "pretext":
            test_acc, test_loss = 0, 0
        else:
            test_acc, test_loss = validate(self.val_loader, self.model)

        # log
        log_dict = OrderedDict(
            {
                "loss_clr": train_meters["loss_clr"].avg,
                "loss_pred": train_meters["loss_pred"].avg,
            }
        )
        if not self.cfg.EXPERIMENT.DEBUG:
            self.log_worklog(epoch, log_dict)
            # saving checkpoint
            # saving occasional checkpoints

            state = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "model": self.cfg.MODEL.TYPE,
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                # "best_acc": self.best_acc,
                # "loss": train_meters["losses"].avg,
                "dataset": self.cfg.DATASET.TYPE,
            }

            if (epoch + 1) % self.cfg.EXPERIMENT.CHECKPOINT_GAP == 0:
                logger.info("Saving checkpoint to {}".format(self.log_path))
                chkp_path = os.path.join(
                    self.log_path,
                    "checkpoints",
                    "epoch_{}_{}_chkp.tar".format(epoch, repetition_id),
                )
                torch.save(state, chkp_path)
                self.current_ckpt_path = chkp_path

        # save best checkpoint with loss or accuracy
        if self.cfg.EXPERIMENT.TASK != "pretext":
            if test_acc > self.best_acc:
                chkp_path = os.path.join(
                    self.log_path,
                    "checkpoints",
                    "epoch_best_{}_chkp.tar".format(repetition_id),
                )
                torch.save(state, chkp_path)

    def after_warmup_iter(self, data, epoch, train_meters, data_itx: int = 0):
        train_start_time = time.time()
        x, _, _ = data
        batch_size = x.size(0)
        loss_clr = torch.tensor(0, dtype=torch.float32).cuda()
        loss_pred = torch.tensor(0, dtype=torch.float32).cuda()
        process_imgs = torch.tensor(0, dtype=torch.float32).cuda()

        loss_clr = self.clr_step(x)

        train_meters["training_time"].update(time.time() - train_start_time)
        train_meters["loss_clr"].update(
            loss_clr.cpu().detach().numpy().mean(), batch_size
        )

        msg = "Epoch:{}|Time(train):{:.2f}|clr:{:.4f}|lr:{:.6f}".format(
            epoch,
            # train_meters["data_time"].avg,
            train_meters["training_time"].avg,
            train_meters["loss_clr"].avg,
            self.optimizer.param_groups[0]["lr"],
        )

        return msg, process_imgs

    def clr_step(self, x):
        # clr step
        self.optimizer.zero_grad()
        x = x.float().cuda()
        x = x.transpose(1, 2)
        ts_l = x.size(1)
        crop_l = np.random.randint(low=2 ** (self.temporal_unit + 1), high=ts_l + 1)
        crop_left = np.random.randint(ts_l - crop_l + 1)
        crop_right = crop_left + crop_l
        crop_eleft = np.random.randint(crop_left + 1)
        crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
        crop_offset = np.random.randint(
            low=-crop_eleft, high=ts_l - crop_eright + 1, size=x.size(0)
        )

        out1, out2 = self.model(
            self.take_per_row(
                x, crop_offset + crop_eleft, crop_right - crop_eleft
            ).transpose(1, 2),
            self.take_per_row(
                x, crop_offset + crop_left, crop_eright - crop_left
            ).transpose(1, 2),
        )
        # out1 = out1.squeeze(-1)
        # out2 = out2.squeeze(-1)

        # b, t1, c1 = out1.shape
        # b, t2, c2 = out2.shape
        # t_min = min(t1, t2)
        # c_min = min(c1, c2)
        # out1 = out1[:, :t_min, :c_min]
        # out2 = out2[:, :t_min, :c_min]
        out1 = out1[:, -crop_l:]
        out2 = out2[:, :crop_l]
        b, t1, c1 = out1.shape
        b, t2, c2 = out2.shape
        c_min = min(c1, c2)
        out1 = out1[:, :, :c_min]
        out2 = out2[:, :, :c_min]

        loss_clr = self.clr_criterion(out1, out2)

        loss_clr = loss_clr.mean()

        # backward
        self.optimizer.zero_grad()
        loss_clr.backward()
        self.optimizer.step()

        return loss_clr

    @staticmethod
    def take_per_row(A, indx, num_elem):
        all_indx = indx[:, None] + np.arange(num_elem)
        return A[torch.arange(all_indx.shape[0])[:, None], all_indx]
