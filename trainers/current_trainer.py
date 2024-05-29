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

from utils.validate import validate


class CurrentTrainer:
    def __init__(
        self, experiment_name, model, criterion, train_loader, val_loader, aug, cfg
    ):
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.clr_criterion = criterion
        self.rec_criterion = torch.nn.L1Loss().cuda()
        self.orthogonal_criterion = OrthLoss().cuda()
        self.cls_criterion = torch.nn.CrossEntropyLoss().cuda()
        self.pred_criterion = torch.nn.MSELoss().cuda()

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
        if not cfg.EXPERIMENT.DEBUG:

            if not cfg.EXPERIMENT.RESUME:
                cur_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())
                experiment_name = experiment_name + "_" + cur_time
                self.log_path = os.path.join(cfg.LOG.PREFIX, experiment_name)
                if not os.path.exists(self.log_path):
                    os.makedirs(self.log_path)
                save_cfg(self.cfg, os.path.join(self.log_path, "config.yaml"))
                os.mkdir(os.path.join(self.log_path, "checkpoints"))
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

    def log_imgs(self, epoch, it, imgs):
        img_list = [
            "x",
            "aug_spec",
            "aug_normal",
            "rec_spec_batch_one",
            "rec_spec_batch_two",
            "rec_normal_batch_one",
            "rec_normal_batch_two",
            # "rec_representation",
        ]
        imgs = imgs.squeeze()[:, ::50].cpu().detach().numpy()
        fig, axes = plt.subplots(7, 1, figsize=(10, 15), sharex=True)
        for i in range(7):
            axes[i].plot(imgs[i], color="black")
            axes[i].set_title(img_list[i])
            axes[i].grid(True)
        plt.tight_layout()

        # buf = BytesIO()
        # plt.savefig(buf, format="png")
        # plt.close(fig)
        # buf.seek(0)
        # image = Image.open(buf).convert("L")
        # image_np = np.array(image)
        self.writer.add_figure(
            "process_imgs",
            plt.gcf(),
            global_step=epoch * len(self.train_loader) + it,
        )

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
            "loss_total_rec": AverageMeter(),
            "loss_rec_spec": AverageMeter(),
            "loss_rec_normal": AverageMeter(),
            "loss_orthogonal": AverageMeter(),
            "loss_clr": AverageMeter(),
            "loss_cls": AverageMeter(),
            "loss_pred": AverageMeter(),
            "top1": AverageMeter(),
        }
        num_iter = len(self.train_loader)
        pbar = tqdm(range(num_iter))

        # train loops
        self.model.train()
        if epoch < self.warmup_epochs:
            for idx, data in enumerate(self.train_loader):
                msg, imgs = self.before_warmup_iter(data, epoch, train_meters, idx)
                pbar.set_description(log_msg(msg, "TRAIN"))
                pbar.update()
                if not self.cfg.EXPERIMENT.DEBUG:
                    self.log(epoch, idx, train_meters)
                    if self.cfg.EXPERIMENT.LOG_IMAGES:
                        self.log_imgs(epoch, idx, imgs)
        else:
            for idx, data in enumerate(self.train_loader):
                msg, imgs = self.after_warmup_iter(data, epoch, train_meters, idx)
                pbar.set_description(log_msg(msg, "TRAIN"))
                pbar.update()
                if not self.cfg.EXPERIMENT.DEBUG:
                    self.log(epoch, idx, train_meters)
                    if self.cfg.EXPERIMENT.LOG_IMAGES:
                        self.log_imgs(epoch, idx, imgs)

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
                "loss_total_rec": train_meters["loss_total_rec"].avg,
                "loss_rec_spec": train_meters["loss_rec_spec"].avg,
                "loss_rec_normal": train_meters["loss_rec_normal"].avg,
                "loss_orthogonal": train_meters["loss_orthogonal"].avg,
                "loss_clr": train_meters["loss_clr"].avg,
                "loss_cls": train_meters["loss_cls"].avg,
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
        loss_cls = torch.tensor(0, dtype=torch.float32).cuda()
        loss_rec_spec = torch.tensor(0, dtype=torch.float32).cuda()
        loss_rec_normal = torch.tensor(0, dtype=torch.float32).cuda()
        loss_total_rec = torch.tensor(0, dtype=torch.float32).cuda()
        loss_orthogonal = torch.tensor(0, dtype=torch.float32).cuda()
        loss_pred = torch.tensor(0, dtype=torch.float32).cuda()
        process_imgs = torch.tensor(0, dtype=torch.float32).cuda()
        _, loss_rec_spec, loss_rec_normal, loss_orthogonal, process_imgs = self.rec_step(
            x
        )

        loss_total_rec = loss_rec_spec + loss_rec_normal

        loss_clr = self.clr_step(x)

        # loss_pred = self.pred_step(x)

        # loss_cls = self.cls_step(x)

        loss_total = loss_clr + loss_cls + loss_total_rec

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
        train_meters["loss_pred"].update(
            loss_pred.cpu().detach().numpy().mean(), batch_size
        )

        msg = "Epoch:{}|Time(train):{:.2f}|rec_total:{:.4f}|rec_spec:{:.4f}|rec_normal:{:.4f}|orthogonal:{:.4f}|clr:{:.4f}|cls:{:.4f}|pred:{:.4f}|lr:{:.6f}".format(
            epoch,
            # train_meters["data_time"].avg,
            train_meters["training_time"].avg,
            train_meters["loss_total_rec"].avg,
            train_meters["loss_rec_spec"].avg,
            train_meters["loss_rec_normal"].avg,
            train_meters["loss_orthogonal"].avg,
            train_meters["loss_clr"].avg,
            train_meters["loss_cls"].avg,
            train_meters["loss_pred"].avg,
            self.optimizer.param_groups[0]["lr"],
        )

        return msg, process_imgs

    def before_warmup_iter(self, data, epoch, train_meters, data_itx: int = 0):
        train_start_time = time.time()
        x, _, _ = data
        batch_size = x.size(0)

        loss_total, loss_rec_spec, loss_rec_normal, loss_orthogonal, process_imgs = (
            self.rec_step(x)
        )

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
        loss_total = torch.tensor(0, dtype=torch.float32).cuda()
        loss_rec_spec = torch.tensor(0, dtype=torch.float32).cuda()
        loss_rec_normal = torch.tensor(0, dtype=torch.float32).cuda()
        loss_orthogonal = torch.tensor(0, dtype=torch.float32).cuda()

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
            # rec_representation,
            normal_inv_representation,
            spec_inv_representation,
            normal_acs_representation,
            spec_acs_representation,
        ) = self.model(
            step="rec", rec_batch_view_spec=aug_spec, rec_batch_view_normal=aug_normal
        )

        loss_rec_spec = self.rec_criterion(rec_spec_batch_one, aug_spec)
        +self.rec_criterion(rec_spec_batch_two, aug_spec)

        loss_rec_normal = self.rec_criterion(rec_normal_batch_one, aug_normal)
        +self.rec_criterion(rec_normal_batch_two, aug_normal)

        loss_orthogonal = self.orthogonal_criterion(
            normal_inv_representation, normal_acs_representation
        )
        +self.orthogonal_criterion(spec_inv_representation, spec_acs_representation)

        loss_total = (
            self.cfg.MODEL.ARGS.REC_WEIGHT * (loss_rec_normal + loss_rec_spec)
            + loss_orthogonal
        )

        # backward
        self.optimizer.zero_grad()
        loss_total.backward()
        self.optimizer.step()

        process_imgs = torch.cat(
            (
                x.unsqueeze(-1)[0],
                aug_spec[0],
                aug_normal[0],
                rec_spec_batch_one[0],
                rec_spec_batch_two[0],
                rec_normal_batch_one[0],
                rec_normal_batch_two[0],
                # rec_representation[0],
            ),
            dim=0,
        ).unsqueeze(1)

        return loss_total, loss_rec_spec, loss_rec_normal, loss_orthogonal, process_imgs

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
        ) = self.model(step="clr", clr_batch_view_1=aug_1, clr_batch_view_2=aug_2)

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
        aug = aug.unsqueeze(-1).cuda()
        labels = labels.cuda()

        # forward
        cls_online_pred = self.model(step="cls", cls_batch_view=aug)

        loss_cls = self.cls_criterion(cls_online_pred, labels)

        loss_cls = loss_cls.mean()
        loss_cls = loss_cls * self.cfg.MODEL.ARGS.CLS_WEIGHT

        # backward
        self.optimizer.zero_grad()
        loss_cls.backward()
        self.optimizer.step()

        return loss_cls

    def pred_step(self, x):
        # pred step
        self.optimizer.zero_grad()
        x = x.float().cuda()
        x = torch.squeeze(x, -1)
        aug, labels = self.aug(x, step="pred")
        aug = aug.unsqueeze(-1).cuda()
        labels = labels.cuda()

        # forward
        pred_online_pred = self.model(step="pred", pred_batch_view=aug)

        loss_pred = self.pred_criterion(pred_online_pred, labels)

        loss_pred = loss_pred.mean()
        loss_pred = loss_pred * self.cfg.MODEL.ARGS.PRED_WEIGHT

        # backward
        self.optimizer.zero_grad()
        loss_pred.backward()
        self.optimizer.step()

        return loss_pred
