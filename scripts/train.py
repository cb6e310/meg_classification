import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from statistics import mean, pstdev

from utils.config import (
    show_cfg,
    save_cfg,
    CFG as cfg,
)
from utils.helpers import (
    log_msg,
    setup_benchmark,
)
from utils.dataset import get_data_loader_from_dataset

from trainers import trainer_dict
from models import model_dict

from loguru import logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser("training for knowledge distillation.")
    parser.add_argument("--cfg", type=str, default="")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    ###############
    # Previous chkp
    ###############

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.EXPERIMENT.GPU_IDS
    experiment_name = cfg.EXPERIMENT.NAME
    if experiment_name == "":
        experiment_name = cfg.EXPERIMENT.TAG
    tags = cfg.EXPERIMENT.TAG.split(",")
    if args.opts:
        addtional_tags = ["{}:{}".format(k, v) for k, v in zip(opts[::2], opts[1::2])]
        tags += addtional_tags
        experiment_name += ",".join(addtional_tags)
    experiment_name = os.path.join(cfg.EXPERIMENT.PROJECT, experiment_name)
    if not cfg.EXPERIMENT.DEBUG:
        try:
            import wandb

            wandb.init(project=cfg.EXPERIMENT.PROJECT, name=experiment_name, tags=tags)
        except:
            print(log_msg("Failed to use WANDB", "INFO"))

    # cfg & loggers
    show_cfg(cfg)
    best_acc_l = []
    for repetition_id in range(cfg.EXPERIMENT.REPETITION_NUM):
        # set the random number seed
        setup_benchmark(cfg.EXPERIMENT.SEED + repetition_id)
        # init dataloader & models
        train_loader = get_data_loader_from_dataset(
            cfg.DATASET.ROOT
            + "/{}".format(cfg.DATASET.TYPE)
            + "/train",
            train=True,
            batch_size=cfg.SOLVER.BATCH_SIZE,
        )
        val_loader = get_data_loader_from_dataset(
            cfg.DATASET.ROOT
            + "/{}".format(cfg.DATASET.TYPE)
            + "/test",
            train=False,
            batch_size=cfg.DATASET.TEST.BATCH_SIZE,
        )

        model = model_dict[cfg.MODEL.TYPE][0](cfg)
        model = torch.nn.DataParallel(model.cuda())

        # train
        trainer = trainer_dict[cfg.SOLVER.TRAINER](
            experiment_name, model, train_loader, val_loader, cfg
        )
        best_acc = trainer.train(repetition_id=repetition_id)
        best_acc_l.append(float(best_acc))

    print(
        log_msg(
            "best_acc(mean±std)\t{:.2f} ± {:.2f}\t{}".format(
                mean(best_acc_l), pstdev(best_acc_l), best_acc_l
            ),
            "INFO",
        )
    )

    with open(os.path.join(trainer.log_path, "worklog.txt"), "a") as writer:
        writer.write("CONFIG:\n{}".format(cfg.dump()))
        writer.write(os.linesep + "-" * 25 + os.linesep)
        writer.write(
            "best_acc(mean±std)\t{:.2f} ± {:.2f}\t{}".format(
                mean(best_acc_l), pstdev(best_acc_l), best_acc_l
            )
        )
        writer.write(os.linesep + "-" * 25 + os.linesep)
    ###############
    # Prepare DataLoader
    ###############
