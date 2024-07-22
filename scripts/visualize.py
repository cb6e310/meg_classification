import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
from statistics import mean, pstdev
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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


class VisualizeTrainer:
    def __init__(
        self,
        experiment_name,
        model,
        train_loader,
        val_loader,
        cfg,
    ):
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        # (main, inv, acs)
        self.feature_test = self.get_features(self.model, val_loader)

        self.log_path = os.path.join(cfg.LOG.PREFIX, experiment_name)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

    def _inference(self, loader):
        features_main = []
        features_1 = []
        features_2 = []
        y_list = []
        for step, data in enumerate(loader):
            x, y, _ = data
            x = x.cuda(non_blocking=True)
            x = x.float()
            # get encoding
            with torch.no_grad():
                main_features, feature_1, feature_2 = self.model(
                    step="visualize", clr_batch_view_1=x
                )

            features_main.extend(main_features.cpu().detach().numpy())
            features_1.extend(feature_1.cpu().detach().numpy())
            features_2.extend(feature_2.cpu().detach().numpy())
            y_list.extend(y.numpy())

            if step % 20 == 0:
                print(f"Step [{step}/{len(loader)}]\t Computing features...")

        main_features = np.array(features_main)
        features_1 = np.array(features_1)
        features_2 = np.array(features_2)
        y = np.array(y_list)
        print("Features shape {}".format(main_features.shape))
        return main_features, features_1, features_2, y

    def get_features(self, encoder, test_loader):
        batch_size = self.cfg.SOLVER.BATCH_SIZE
        features_main_test, features_1_test, features_2_test, y = self._inference(
            test_loader,
        )
        return (
            features_main_test,
            features_1_test,
            features_2_test,
            y
        )

    def get_tsne_disentangle(self):
        tsne = TSNE(n_components=2, random_state=0)
        _, feature_1_test, feature_2_test, y = self.feature_test
        # scaler = MinMaxScaler(feature_range=(-1, 1))

        # feature_1_test= scaler.fit_transform(feature_1_test)
        # feature_2_test= scaler.fit_transform(feature_2_test)
        feature = np.concatenate((feature_1_test,feature_2_test))
        
        
        logger.debug("calculating tsne")
        
        
        tsne = TSNE(n_components=2, random_state=42)
        tsne_output=tsne.fit_transform(feature)
        tsnescaler = MinMaxScaler((-1,1))
        tsne_output=tsnescaler.fit_transform(tsne_output)
        
        label_1 = np.ones(feature_1_test.shape[0])
        label_2 = np.zeros(feature_2_test.shape[0])
        targets = np.concatenate((label_1, label_2))
        
        df = pd.DataFrame(tsne_output, columns=['x', 'y'])
        df['targets'] = targets
        
        # sns.set(style="whitegrid")
        plt.figure(figsize=(10, 10))
        
        scatter = sns.scatterplot(
            x='x', y='y',
            hue='targets',
            palette=['red', 'blue'],
            data=df,
            marker='o',
            s=100,
            alpha=0.6,
            edgecolor="w",
            linewidth=0.5
        )
        
        plt.legend(title='Targets', loc='upper right', fontsize='large', title_fontsize='13', markerscale=1.2)
        plt.title('t-SNE visualization of Disentangled Features', fontsize=16)
        plt.xlabel('t-SNE feature 1', fontsize=14)
        plt.ylabel('t-SNE feature 2', fontsize=14)
        
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        
        plt.savefig(os.path.join(self.log_path, 'tsne.png'), bbox_inches='tight', dpi=300)
        plt.show()
        
        print('Done!')

    def get_tsne_class(self, ):
        outputs, feature_1_test, feature_2_test, targets = self.feature_test

        print('generating t-SNE plot...')
        # tsne_output = bh_sne(outputs)
        tsne = TSNE(random_state=0)
        tsne_output = tsne.fit_transform(outputs)

        df = pd.DataFrame(tsne_output, columns=['x', 'y'])
        df['targets'] = targets

        plt.rcParams['figure.figsize'] = 10, 10
        sns.scatterplot(
            x='x', y='y',
            hue='targets',
            palette=['red', 'blue'],
            data=df,
            marker='o',
            legend="full",
            alpha=0.5
        )

        plt.xticks([])
        plt.yticks([])
        plt.xlabel('')
        plt.ylabel('')

        plt.savefig(os.path.join(self.log_path,'tsne_classes.png'), bbox_inches='tight')
        print('done!')


def visualize(cfg, ckpts=None, log_path=None):
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

    log_path = cfg.EXPERIMENT.PRETRAINED_PATH
    max_epoch = -1
    ckpts = []

    for filename in os.listdir(os.path.join(log_path, "checkpoints")):
        if "eval" not in filename:
            parts = filename.split("_")
            if len(parts) > 1 and parts[1].isdigit():
                number = int(parts[1])
                if number > max_epoch:
                    max_epoch = number
                    ckpts = [os.path.join(log_path, "checkpoints", filename)]
                elif number == max_epoch:
                    ckpts.append(os.path.join(log_path, "checkpoints", filename))

    ckpt = ckpts[0]
    pretrained_dict = torch.load(ckpt)
    model = model_dict[cfg.MODEL.TYPE][0](cfg).cuda()

    model.load_state_dict(pretrained_dict["model_state_dict"])

    print(
        "Loaded pretrained model from {}".format(ckpt),
        "pretrained epoch: {}".format(pretrained_dict["epoch"]),
    )

    trainer = VisualizeTrainer(
        log_path,
        model,
        train_loader,
        val_loader,
        cfg,
    )
    trainer.get_tsne_disentangle()


if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.EXPERIMENT.GPU_IDS
    # print available GPUs
    logger.info("Available GPUs: {}".format(torch.cuda.device_count()))
    parser = argparse.ArgumentParser("")
    parser.add_argument("--cfg", type=str, default="")
    parser.add_argument("--opts", nargs="+", default=[])
    args = parser.parse_args()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    logger.info("eval only, visualization")
    visualize(cfg)
