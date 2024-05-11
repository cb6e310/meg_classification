import os
import time
from tqdm import tqdm
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader

import numpy as np

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

from loguru import logger

from utils.validate import validate


class BYOLTrainer:
    def __init__(
        self, experiment_name, model, criterion, train_loader, val_loader, aug, cfg
    ):
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.model = model
        self.aug = aug
        self.tau_base = cfg.MODEL.ARGS.TAU_BASE
        self.warmup_epochs = cfg.MODEL.ARGS.WARMUP_EPOCHS

        # check model name is BYOL

        self.optimizer = self.init_optimizer(cfg)
        self.scheduler = self.init_scheduler(cfg)
        self.best_acc = -1
        self.best_epoch = 0
        self.resume_epoch = -1

        self.current_epoch = 0

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
                if cfg.EXPERIMENT.CHKP_IDX is None:
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
                self.scheduler.load_state_dict(self.chkp["scheduler"])
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

    def log(self, epoch, log_dict):
        if not self.cfg.EXPERIMENT.DEBUG:
            import wandb

            # wandb.log({"current lr": lr})
            wandb.log(log_dict)
        if log_dict["test_acc"] > self.best_acc:
            self.best_acc = log_dict["test_acc"]
            self.best_epoch = epoch
            if not self.cfg.EXPERIMENT.DEBUG:
                wandb.run.summary["best_acc"] = self.best_acc
        # worklog.txt
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
        return self.best_acc

    def train_epoch(self, epoch, repetition_id=0):
        lr = self.cfg.SOLVER.LR
        train_meters = {
            "training_time": AverageMeter(),
            # "data_time": AverageMeter(),
            "losses": AverageMeter(),
            "loss_rec": AverageMeter(),
            "loss_orthogonal": AverageMeter(),
            "loss_clr": AverageMeter(),
            "loss_cls": AverageMeter(),
            "top1": AverageMeter(),
        }
        num_iter = len(self.train_loader)
        pbar = tqdm(range(num_iter))

        # train loops
        self.model.train()
        if epoch < self.warmup_epochs:
            for idx, data in enumerate(self.train_loader):
                msg = self.before_warmup_iter(data, epoch, train_meters, idx)
                pbar.set_description(log_msg(msg, "TRAIN"))
                pbar.update()
        else:
            for idx, data in enumerate(self.train_loader):
                msg = self.after_warmup_iter(data, epoch, train_meters, idx)
                pbar.set_description(log_msg(msg, "TRAIN"))
                pbar.update()

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
                "train_acc": train_meters["top1"].avg,
                "train_loss": train_meters["losses"].avg,
                "test_acc": test_acc,
                "test_loss": test_loss,
            }
        )
        self.log(epoch, log_dict)
        # saving checkpoint
        # saving occasional checkpoints

        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "model": self.cfg.MODEL.TYPE,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "best_acc": self.best_acc,
            "loss": train_meters["losses"].avg,
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

        loss_total_rec, loss_rec_spec, loss_rec_normal, loss_orthogonal = self.rec_step(x)

        loss_clr = self.clr_step(x)

        loss_cls = self.cls_step(x)

        train_meters["training_time"].update(time.time() - train_start_time)
        train_meters["loss_total_rec"].update(
            loss_total_rec.cpu().detach().numpy().mean(), batch_size
        )
        train_meters["loss_rec_spec"].update(
            loss_rec_spec.cpu().detach().numpy().mean(), batch_size
        )
        train_meters["loss_rec_normal"].update(
            loss_rec_normal.cpu().detach().numpy().mean(), batch_size
        )
        train_meters["loss_orthogonal"].update(
            loss_orthogonal.cpu().detach().numpy().mean(), batch_size
        )
        train_meters["loss_clr"].update(
            loss_clr.cpu().detach().numpy().mean(), batch_size
        )
        train_meters["loss_cls"].update(
            loss_cls.cpu().detach().numpy().mean(), batch_size
        )

        msg = "Epoch:{}|Time(train):{:.2f}|Loss:{:.7f}|loss_rec:{:.7f}|loss_orthogonal:{:.7f}|loss_clr:{:.7f}|loss_cls:{:.7f}|lr:{:.6f}".format(
            epoch,
            # train_meters["data_time"].avg,
            train_meters["training_time"].avg,
            train_meters["loss_total_rec"].avg,
            train_meters["loss_rec_spec"].avg,
            train_meters["loss_rec_normal"].avg,
            train_meters["loss_orthogonal"].avg,
            train_meters["loss_clr"].avg,
            train_meters["loss_cls"].avg,
            self.optimizer.param_groups[0]["lr"],
        )

        return msg

    def before_warmup_iter(self, data, epoch, train_meters, data_itx: int = 0):
        train_start_time = time.time()
        x, _, _ = data
        batch_size = x.size(0)

        loss_total, loss_rec_spec, loss_rec_normal, loss_orthogonal = self.rec_step(x)

        train_meters["training_time"].update(time.time() - train_start_time)
        train_meters["losses"].update(
            loss_total.cpu().detach().numpy().mean(), batch_size
        )
        train_meters["loss_rec"].update(
            (loss_rec_spec + loss_rec_normal).cpu().detach().numpy().mean(), batch_size
        )
        train_meters["loss_orthogonal"].update(
            loss_orthogonal.cpu().detach().numpy().mean(), batch_size
        )

        msg = "Epoch:{}|Time(train):{:.2f}|Loss:{:.7f}|loss_rec:{:.7f}|loss_orthogonal:{:.7f}|lr:{:.6f}".format(
            epoch,
            # train_meters["data_time"].avg,
            train_meters["training_time"].avg,
            train_meters["losses"].avg,
            train_meters["loss_rec"].avg,
            train_meters["loss_orthogonal"].avg,
            self.optimizer.param_groups[0]["lr"],
        )
        return msg

    def rec_step(self, x):

        # rec step
        self.optimizer.zero_grad()
        x = x.float().cuda()
        x = torch.squeeze(x, -1)
        aug_spec, aug_normal = self.aug(x, step="rec")
        aug_spec = aug_spec.unsqueeze(-1)
        aug_normal = aug_normal.unsqueeze(-1)

        # forward
        (
            rec_spec_batch_one,
            rec_spec_batch_two,
            rec_normal_batch_one,
            rec_normal_batch_two,
            normal_inv_representation,
            spec_inv_representation,
            normal_cs_representation,
            spec_cs_representation,
        ) = self.model(
            step="rec", rec_batch_view_spec=aug_spec, rec_batch_view_normal=aug_normal
        )

        loss_rec_spec = self.rec_criterion(
            rec_spec_batch_one, aug_spec
        ) + self.rec_criterion(rec_spec_batch_two, aug_spec)

        loss_rec_normal = self.rec_criterion(
            rec_normal_batch_one, aug_normal
        ) + self.rec_criterion(rec_normal_batch_two, aug_normal)

        loss_orthogonal = self.orthogonal_criterion(
            normal_inv_representation, normal_cs_representation
        ) + self.orthogonal_criterion(spec_inv_representation, spec_cs_representation)

        loss_total = (
            self.cfg.MODEL.ARGS.REC_WEIGHT * (loss_rec_spec + loss_rec_normal)
            + loss_orthogonal
        )

        # backward
        self.optimizer.zero_grad()
        loss_total.backward()
        self.optimizer.step()

        return loss_total, loss_rec_spec, loss_rec_normal, loss_orthogonal

    def clr_step(self, x):
        # clr step
        self.optimizer.zero_grad()
        x = x.float().cuda()
        x = torch.squeeze(x, -1)
        aug_1, aug_2 = self.aug(x, step="clr")
        aug_1 = aug_1.unsqueeze(-1)
        aug_2 = aug_2.unsqueeze(-1)

        # forward
        (
            clr_online_pred_one,
            clr_online_pred_two,
            clr_target_proj_one,
            clr_target_proj_two,
        ) = self.model(step="clr", clr_batch_view_one=aug_1, clr_batch_view_two=aug_2)

        loss_clr = self.clr_criterion(
            clr_online_pred_one, clr_target_proj_two
        ) + self.clr_criterion(clr_online_pred_two, clr_target_proj_one)

        loss_clr = loss_clr.mean()

        # backward
        self.optimizer.zero_grad()
        loss_clr.backward()
        self.optimizer.step()

        return loss_clr

    def cls_step(self, x):
        # cls step
        self.optimizer.zero_grad()
        x = x.float().cuda()
        x = torch.squeeze(x, -1)
        aug, labels = self.aug(x, step="cls")
        aug = aug.unsqueeze(-1)

        # forward
        (
            cls_online_pred,
            cls_target_proj,
        ) = self.model(step="cls", cls_batch_view=aug, labels=labels)

        loss_cls = self.cls_criterion(cls_online_pred, labels)

        loss_cls = loss_cls.mean()

        # backward
        self.optimizer.zero_grad()
        loss_cls.backward()
        self.optimizer.step()

        return loss_cls