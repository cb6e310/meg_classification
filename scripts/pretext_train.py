import os
import time
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
from utils.augmentations import AutoAUG, InfoTSAUG

from trainers import trainer_dict
from models import model_dict, criterion_dict

from loguru import logger

def train(cfg):
    if cfg.EXPERIMENT.CHECKPOINT_GAP<cfg.SOLVER.EPOCHS:
        raise "no enough epochs for ckpt"

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

    # cfg & loggers
    show_cfg(cfg)
    best_acc_l = []
    ckpts = []
    cur_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())
    experiment_name_w_time = experiment_name + "_" + cur_time
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

        if cfg.SOLVER.TRAINER == "InfoTS":
            aug = InfoTSAUG(cfg).cuda()
        else:
            aug = AutoAUG(cfg).cuda()

        model = model_dict[cfg.MODEL.TYPE][0](cfg).cuda()
        logger.info("model's device: {}".format(next(model.parameters()).device))

        # model = nn.DataParallel(model)

        criterion = criterion_dict[cfg.MODEL.CRITERION.TYPE](cfg).cuda()

        # train
        trainer = trainer_dict[cfg.SOLVER.TRAINER](
            experiment_name_w_time, model, criterion, train_loader, val_loader, aug, cfg
        )
        best_acc, ckpt_path = trainer.train(repetition_id=repetition_id)
        best_acc_l.append(float(best_acc))
        ckpts.append(ckpt_path)

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
    return ckpts, trainer.log_path

def linear_eval(cfg, ckpts=None, log_path=None):
    train_loader = get_data_loader_from_dataset(
        cfg.DATASET.ROOT + "/{}".format(cfg.DATASET.TYPE) + "/train",
        cfg,
        train=True,
        batch_size=cfg.EVAL_LINEAR.BATCH_SIZE,
        siamese=cfg.MODEL.ARGS.SIAMESE,
    )
    val_loader = get_data_loader_from_dataset(
        cfg.DATASET.ROOT + "/{}".format(cfg.DATASET.TYPE) + "/test",
        cfg,
        train=False,
        batch_size=cfg.EVAL_LINEAR.BATCH_SIZE,
        siamese=cfg.MODEL.ARGS.SIAMESE,
    )

    if cfg.SOLVER.TRAINER == "InfoTS":
        aug = InfoTSAUG(cfg).cuda()
    else:
        aug = AutoAUG(cfg).cuda()

    model = model_dict[cfg.MODEL.TYPE][0](cfg).cuda()
    logger.info("model's device: {}".format(next(model.parameters()).device))
    best_acc_l = []
    knn_acc_l = []
    if ckpts == None:
        # eval only mode, retrieve from early record
        log_path = cfg.EXPERIMENT.PRETRAINED_PATH
        max_epoch = -1
        ckpts = []

        for filename in os.listdir(os.path.join(log_path, "checkpoints")):
            if "eval" not in filename:
                parts = filename.split('_')
                if len(parts) > 1 and parts[1].isdigit():
                    number = int(parts[1])
                    if number > max_epoch:
                        max_epoch = number
        for filename in os.listdir()
        ckpts = [os.path.join(log_path,"checkpoints",filename)]
        elif number == max_epoch:
            ckpts.append(os.path.join(log_path,"checkpoints", filename))
    for ckpt in ckpts:
        pretrained_dict = torch.load(ckpt)

        model.load_state_dict(pretrained_dict["model_state_dict"])

        print(
            "Loaded pretrained model from {}".format(ckpt),
            "pretrained epoch: {}".format(pretrained_dict["epoch"]),
        )

        model.eval()

        classifier = model_dict[cfg.EVAL_LINEAR.CLASSIFIER][0](cfg).cuda()

        criterion = criterion_dict[cfg.EVAL_LINEAR.CRITERION](cfg).cuda()

        # train
        trainer = trainer_dict["linear_eval"](
            log_path,
            model,
            classifier,
            criterion,
            train_loader,
            val_loader,
            cfg,
        )
        best_acc, knn_acc = trainer.train()
        best_acc_l.append(float(best_acc))
        knn_acc_l.append(float(knn_acc))

    print(
        log_msg(
            "best_acc(mean±std)\t{:.2f} ± {:.2f}\t{}".format(
                mean(best_acc_l), pstdev(best_acc_l), best_acc_l
            ),
            "INFO",
        )
    )
    logger.info("save at {}".format(log_path))

    with open(os.path.join(trainer.log_path, "worklog_linear.txt"), "a") as writer:
        writer.write("CONFIG:\n{}".format(cfg.dump()))
        writer.write(os.linesep + "-" * 25 + os.linesep)
        
        writer.write(
            "best_acc(mean±std)\t{:.2f} ± {:.2f}\t{}\n".format(
                mean(best_acc_l), pstdev(best_acc_l), best_acc_l
            )
        )
        
        writer.write(
            "knn_acc(mean±std)\t{:.2f} ± {:.2f}\t{}\n".format(
                mean(knn_acc_l), pstdev(knn_acc_l), knn_acc_l
            )
        )

        writer.write(os.linesep + "-" * 25 + os.linesep)

if __name__ == "__main__":

    parser = argparse.ArgumentParser("")
    parser.add_argument("--cfg", type=str, default="")
    parser.add_argument("--opts", nargs="+", default=[])
    args = parser.parse_args()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    if cfg.EXPERIMENT.EVAL_ONLY == True:
        if cfg.EXPERIMENT.EVAL_LINEAR == True:
            logger.info("eval only, linear eval")
            linear_eval(cfg)
    else:
        ckpts = train(cfg)
        if cfg.EXPERIMENT.EVAL_NEXT == True:
            if cfg.EXPERIMENT.EVAL_LINEAR == True:
                linear_eval(cfg, ckpts)