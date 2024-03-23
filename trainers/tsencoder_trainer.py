import os
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import OrderedDict
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


class TSEncoderTrainer:
    def __init__(self, experiment_name, model, criterion, train_loader, val_loader, augmentation, cfg):
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.model = model
        self.aug = augmentation
        self.optimizer = self.init_optimizer(cfg)
        self.scheduler = self.init_scheduler(cfg)
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
                self.model.parameters(),
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

    def __l1_regularization__(self, l1_penalty=3e-4):
        regularization_loss = 0
        for param in self.distiller.module.get_learnable_parameters():
            regularization_loss += torch.sum(abs(param))  # torch.norm(param, p=1)
        return l1_penalty * regularization_loss

    def train(self, repetition_id=0):
        epoch = 0
        if self.resume_epoch != -1:
            epoch = self.resume_epoch + 1
        # if self.cfg.EXPERIMENT.RESUME and self.cfg.EXPERIMENT.CHECKPOINT != "":
        #     epoch = state["epoch"] + 1
        #     self.distiller.load_state_dict(state["model"])
        #     self.optimizer.load_state_dict(state["optimizer"])
        #     self.best_acc = state["best_acc"]
        while epoch < self.cfg.SOLVER.EPOCHS:
            self.train_epoch(epoch, repetition_id=repetition_id)
            epoch += 1
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
            "top1": AverageMeter(),
        }
        num_iter = len(self.train_loader)
        pbar = tqdm(range(num_iter))

        # train loops
        self.model.train()
        for idx, data in enumerate(self.train_loader):
            msg = self.train_iter(data, epoch, train_meters, idx)
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

    def train_iter(self, data, epoch, train_meters, data_itx: int = 0):
        self.optimizer.zero_grad()
        train_start_time = time.time()

        # train_meters["data_time"].update(time.time() - train_start_time)
        x = data[0]
        x = x.float().cuda()
        if (
            self.cfg.MODEL.ARGS.MAX_TRAIN_LENGTH is not None
            and x.size(1) > self.cfg.MODEL.ARGS.MAX_TRAIN_LENGTH
        ):
            window_offset = np.random.randint(
                x.size(1) - self.cfg.MODEL.ARGS.MAX_TRAIN_LENGTH + 1
            )
            x = x[:, window_offset : window_offset + self.cfg.MODEL.ARGS.MAX_TRAIN_LENGTH]
        
        batch_size = x.size(0)

        aug_1, aug_2 = self.aug(x)

        # forward
        z_1 = self.model(aug_1)
        z_2 = self.model(aug_2)
        loss = self.criterion(z_1, z_2)
        loss = loss.mean()


        # backward
        loss.backward()
        self.optimizer.step()

        train_meters["training_time"].update(time.time() - train_start_time)
        train_meters["losses"].update(loss.cpu().detach().numpy().mean(), batch_size)
        # print info
        msg = "Epoch:{}|Time(train):{:.2f}|Loss:{:.2f}|lr:{:.6f}".format(
            epoch,
            # train_meters["data_time"].avg,
            train_meters["training_time"].avg,
            train_meters["losses"].avg,
            self.optimizer.param_groups[0]["lr"],
        )
        return msg
