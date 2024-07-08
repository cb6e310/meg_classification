from utils.helpers import AverageMeter, accuracy, log_msg
import time
from tqdm import tqdm
import torch
import torch.nn as nn

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def validate(val_loader, model):
    batch_time, losses, top1 = [AverageMeter() for _ in range(3)]
    num_iter = len(val_loader)
    pbar = tqdm(range(num_iter))

    model.eval()
    with torch.no_grad():
        start_time = time.time()
        for idx, (data, target) in enumerate(val_loader):
            data = data.float()
            data = data.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            output, loss = model(data, target)
            loss = loss.mean()
            acc1, _ = accuracy(output, target, topk=(1, 2))
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
    return top1.avg, losses.avg

def KNN_validate(train_loader, val_loader):
    # train_loader, val_loader: feature, labels from dataset through encoder
    # KNN
    knn = KNeighborsClassifier(n_neighbors=1)
    train_feature, train_label = train_loader.dataset.tensors
    train_feature = train_feature.cpu().numpy()
    train_feature = train_feature.squeeze()
    train_label = train_label.cpu().numpy()
    knn.fit(train_feature, train_label)

    val_feature, val_label = val_loader.dataset.tensors
    val_feature = val_feature.cpu().numpy()
    val_feature=val_feature.squeeze()
    val_label = val_label.cpu().numpy()
    pred = knn.predict(val_feature)
    acc = accuracy_score(val_label, pred)
    return acc