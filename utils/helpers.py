import os
import random
import torch
import numpy as np
from time import time
from loguru import logger

def check_model_device(model):
    device = None
    for param in model.parameters():
        if device is None:
            device = param.device
        elif param.device != device:
            return False  # Found a parameter on a different device
    return True

_timing_start_dict = {}
def timing_start(name: str):
    global _timing_start_dict
    _timing_start_dict[name] = time()


def timing_end(name: str):
    global _timing_start_dict
    if name not in _timing_start_dict:
        raise ValueError(f"Block {name} not started")
    logger.debug(f"Block {name} time cost: {time() - _timing_start_dict[name]}")


def torch_pad_nan(arr, left=0, right=0, dim=0):
    if left > 0:
        padshape = list(arr.shape)
        padshape[dim] = left
        arr = torch.cat((torch.full(padshape, np.nan), arr), dim=dim)
    if right > 0:
        padshape = list(arr.shape)
        padshape[dim] = right
        arr = torch.cat((arr, torch.full(padshape, np.nan)), dim=dim)
    return arr


def pad_nan_to_target(array, target_length, axis=0, both_side=False):
    assert array.dtype in [np.float16, np.float32, np.float64]
    pad_size = target_length - array.shape[axis]
    if pad_size <= 0:
        return array
    npad = [(0, 0)] * array.ndim
    if both_side:
        npad[axis] = (pad_size // 2, pad_size - pad_size // 2)
    else:
        npad[axis] = (0, pad_size)
    return np.pad(array, pad_width=npad, mode="constant", constant_values=np.nan)


def split_with_nan(x, sections, axis=0):
    assert x.dtype in [np.float16, np.float32, np.float64]
    arrs = np.array_split(x, sections, axis=axis)
    target_length = arrs[0].shape[axis]
    for i in range(len(arrs)):
        arrs[i] = pad_nan_to_target(arrs[i], target_length, axis=axis)
    return arrs


def take_per_row(A, indx, num_elem):
    all_indx = indx[:, None] + np.arange(num_elem)
    return A[torch.arange(all_indx.shape[0])[:, None], all_indx]


def centerize_vary_length_series(x):
    prefix_zeros = np.argmax(~np.isnan(x).all(axis=-1), axis=1)
    suffix_zeros = np.argmax(~np.isnan(x[:, ::-1]).all(axis=-1), axis=1)
    offset = (prefix_zeros + suffix_zeros) // 2 - prefix_zeros
    rows, column_indices = np.ogrid[: x.shape[0], : x.shape[1]]
    offset[offset < 0] += x.shape[1]
    column_indices = column_indices - offset[:, np.newaxis]
    return x[rows, column_indices]

def setup_benchmark(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # 尽可能提高确定性
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False # True will cause dilated conv much slower and fatty
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