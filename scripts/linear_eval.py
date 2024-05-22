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
from models import model_dict, criterion_dict

from loguru import logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser("training for linear eval.")
    parser.add_argument("--cfg", type=str, default="")
    parser.add_argument("--opts", nargs="+", default=[])
    args = parser.parse_args()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    ###############
    # Previous chkp
    ###############
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.EXPERIMENT.GPU_IDS
    # print available GPUs
    logger.info("Available GPUs: {}".format(torch.cuda.device_count()))

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
        # try:
        #     import wandb

        #     wandb.init(project=cfg.EXPERIMENT.PROJECT, name=experiment_name, tags=tags)
        # except:
        #     print(log_msg("Failed to use WANDB", "INFO"))
        pass

    # cfg & loggers
    show_cfg(cfg)
    best_acc_l = []
    for repetition_id in range(cfg.EXPERIMENT.REPETITION_NUM):
        # set the random number seed
        setup_benchmark(cfg.EXPERIMENT.SEED + repetition_id)
        # init dataloader & models
        train_loader = get_data_loader_from_dataset(
            cfg.DATASET.ROOT + "/{}".format(cfg.DATASET.TYPE) + "/train",
            cfg,
            train=True,
            batch_size=cfg.SOLVER.BATCH_SIZE,
            siamese=cfg.MODEL.ARGS.SIAMESE,
        )
        val_loader = get_data_loader_from_dataset(
            cfg.DATASET.ROOT + "/{}".format(cfg.DATASET.TYPE) + "/test",
            cfg,
            train=False,
            batch_size=cfg.SOLVER.BATCH_SIZE,
            siamese=cfg.MODEL.ARGS.SIAMESE,
        )

        encoder = model_dict[cfg.MODEL.TYPE][0](cfg).cuda()
        # if cfg.EXPERIMENT.WORLD_SIZE > 1:
        #     encoder = torch.nn.DataParallel(encoder)

        pretrained_dict = torch.load(cfg.MODEL.ARGS.PRETRAINED_PATH)

        encoder.load_state_dict(pretrained_dict["model_state_dict"])

        print(
            "Loaded pretrained model from {}".format(cfg.MODEL.ARGS.PRETRAINED_PATH),
            "pretrained epoch: {}".format(pretrained_dict["epoch"]),
        )

        encoder.eval()

        classifier = model_dict[cfg.MODEL.ARGS.CLASSIFIER][0](cfg).cuda()

        criterion = criterion_dict[cfg.MODEL.CRITERION.TYPE](cfg).cuda()

        # train
        trainer = trainer_dict["linear_eval"](
            experiment_name, encoder, classifier, criterion, train_loader, val_loader, cfg
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
