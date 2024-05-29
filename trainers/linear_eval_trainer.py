import os
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np


import os
import time
from tqdm import tqdm
import torch.optim as optim
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


class LinearEvalTrainer:
    def __init__(
        self, experiment_name, model, classifier, criterion, train_loader, val_loader, cfg
    ):
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.model = model
        self.classifier = classifier
        self.optimizer = self.init_optimizer(cfg)
        self.scheduler = self.init_scheduler(cfg)
        self.best_acc = -1
        self.best_epoch = 0
        self.resume_epoch = -1

        # creating features from pretext model
        self.train_feat_loader, self.val_feat_loader = self.get_features(
            self.model, train_loader, val_loader
        )

        # username = getpass.getuser()
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
                self.classifier.parameters(),
                lr=cfg.SOLVER.LR,
                momentum=cfg.SOLVER.MOMENTUM,
                weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            )
        elif cfg.SOLVER.TYPE == "Adam":
            optimizer = optim.Adam(
                self.classifier.module.get_learnable_parameters(),
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
            # import wandb

            # # wandb.log({"current lr": lr})
            # wandb.log(log_dict)
            pass
        if log_dict["test_acc"] > self.best_acc:
            self.best_acc = log_dict["test_acc"]
            self.best_epoch = epoch
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

    def _inference(self, loader):
        feature_vector = []
        labels_vector = []
        for step, data in enumerate(loader):
            x, y, _ = data

            x = x.cuda(non_blocking=True)
            x = x.float()
            # get encoding
            # if self.cfg.MODEL.TYPE == "BYOL":
            #     x = torch.unsqueeze(x, -1)
            with torch.no_grad():
                if self.cfg.MODEL.ARGS.BACKBONE == "eegconvnet":
                    h = self.model(x, x, return_embedding=True, return_projection=False)[0][1]
            h = h.detach()

            feature_vector.extend(h.cpu().detach().numpy())
            labels_vector.extend(y.numpy())

            if step % 20 == 0:
                print(f"Step [{step}/{len(loader)}]\t Computing features...")

        feature_vector = np.array(feature_vector)
        labels_vector = np.array(labels_vector)
        print("Features shape {}".format(feature_vector.shape))
        return feature_vector, labels_vector

    def get_features(self, encoder, train_loader, test_loader):
        batch_size = self.cfg.SOLVER.BATCH_SIZE
        train_X, train_y = self._inference(train_loader)
        test_X, test_y = self._inference(test_loader)
        train = torch.utils.data.TensorDataset(
            torch.from_numpy(train_X), torch.from_numpy(train_y)
        )

        train_loader = torch.utils.data.DataLoader(
            train, batch_size=batch_size, shuffle=False
        )

        test = torch.utils.data.TensorDataset(
            torch.from_numpy(test_X), torch.from_numpy(test_y)
        )

        test_loader = torch.utils.data.DataLoader(
            test, batch_size=batch_size, shuffle=False
        )
        return train_loader, test_loader

    def train(self, repetition_id=0):
        epoch = 0
        if self.resume_epoch != -1:
            epoch = self.resume_epoch + 1

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
        num_iter = len(self.train_feat_loader)
        pbar = tqdm(range(num_iter))

        # train loops
        self.classifier.train()
        for idx, data in enumerate(self.train_feat_loader):
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
            test_acc, test_loss = self.validate()

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
        data, target = data
        data = data.float()
        data = data.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        batch_size = data.size(0)

        # forward
        preds = self.classifier(data)
        # logger.debug(self.classifier.training)
        loss = self.criterion(preds, target)
        loss = loss.mean()

        # backward
        loss.backward()
        self.optimizer.step()

        train_meters["training_time"].update(time.time() - train_start_time)
        train_meters["losses"].update(loss.cpu().detach().numpy().mean(), batch_size)
        acc1, _ = accuracy(preds, target, topk=(1, 2))
        train_meters["top1"].update(acc1[0], batch_size)
        msg = "Epoch:{}|Time(train):{:.2f}|Loss:{:.6f}|Top-1:{:.6f}|lr:{:.6f}".format(
            epoch,
            train_meters["training_time"].avg,
            train_meters["losses"].avg,
            train_meters["top1"].avg,
            self.optimizer.param_groups[0]["lr"],
        )
        return msg

    def validate(self):
        batch_time, losses, top1 = [AverageMeter() for _ in range(3)]
        num_iter = len(self.val_feat_loader)
        pbar = tqdm(range(num_iter))

        self.classifier.eval()
        start_time = time.time()
        for idx, (data, target) in enumerate(self.val_feat_loader):
            data = data.float()
            data = data.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            preds = self.classifier(data)
            loss = self.criterion(preds, target)
            loss = loss.mean()
            acc1, _ = accuracy(preds, target, topk=(1, 2))
            batch_size = data.size(0)
            losses.update(loss.cpu().detach().numpy().mean(), batch_size)
            top1.update(acc1[0], batch_size)

            # measure elapsed time
            batch_time.update(time.time() - start_time)
            start_time = time.time()
            msg = "Loss:{loss:.4f}| Top-1:{top1:.4f}".format(
                loss=losses.avg, top1=top1.avg
            )
            pbar.set_description(log_msg(msg, "EVAL"))
            pbar.update()
        pbar.close()
        self.classifier.train()
        return top1.avg, losses.avg
