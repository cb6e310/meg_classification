import os
import random
import torch
import numpy as np
import logging


logger = logging.getLogger(__name__)
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(format=FORMAT)
logger.setLevel(logging.DEBUG)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # 尽可能提高确定性
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)


class AverageMeter(object):

    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def predict(model, data, num_classes=2, batch_size=1024, eval=False):
    model.cuda()
    data = torch.from_numpy(data)
    data_split = torch.split(data, batch_size, dim=0)
    output = torch.zeros(len(data), num_classes).cuda()  # 预测的置信度和置信度最大的标签编号
    start = 0
    if eval:
        model.eval()
        with torch.no_grad():
            for batch_data in data_split:
                batch_data = batch_data.cuda()
                batch_data = batch_data.float()
                output[start : start + len(batch_data)] = model(batch_data)
                start += len(batch_data)
    else:
        model.eval()
        for batch_data in data_split:
            batch_data = batch_data.cuda()
            batch_data = batch_data.float()
            output[start : start + len(batch_data)] = model(batch_data)
            start += len(batch_data)
    model.train()
    return output


def individual_predict(model, individual_data, eval=True):
    pred = predict(model, np.expand_dims(individual_data, 0), eval=eval)
    return pred[0]


def adjust_learning_rate(epoch, cfg, optimizer):
    steps = np.sum(epoch > np.asarray(cfg.SOLVER.LR_DECAY_STAGES))
    if steps > 0:
        new_lr = cfg.SOLVER.LR * (cfg.SOLVER.LR_DECAY_RATE ** steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr
        return new_lr
    return cfg.SOLVER.LR


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def log_msg(msg, mode="INFO"):
    color_map = {"INFO": 36, "TRAIN": 32, "EVAL": 31}
    msg = "\033[{}m[{}] {}\033[0m".format(color_map[mode], mode, msg)
    return msg