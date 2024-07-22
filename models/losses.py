import torch
import torch.nn as nn
import torch.distributed as dist
from .blocks import *

# import some common builtin loss
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from loguru import logger


# for ts2vec
def hierarchical_contrastive_loss(z1, z2, alpha=0.5, temporal_unit=0):
    def instance_contrastive_loss(z1, z2):
        B, T = z1.size(0), z1.size(1)
        if B == 1:
            return z1.new_tensor(0.0)
        z = torch.cat([z1, z2], dim=0)  # 2B x T x C
        z = z.transpose(0, 1)  # T x 2B x C
        sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B
        logits = torch.tril(sim, diagonal=-1)[:, :, :-1]  # T x 2B x (2B-1)
        logits += torch.triu(sim, diagonal=1)[:, :, 1:]
        logits = -F.log_softmax(logits, dim=-1)

        i = torch.arange(B, device=z1.device)
        loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
        return loss

    def temporal_contrastive_loss(z1, z2):
        B, T = z1.size(0), z1.size(1)
        if T == 1:
            return z1.new_tensor(0.0)
        z = torch.cat([z1, z2], dim=1)  # B x 2T x C
        sim = torch.matmul(z, z.transpose(1, 2))  # B x 2T x 2T
        logits = torch.tril(sim, diagonal=-1)[:, :, :-1]  # B x 2T x (2T-1)
        logits += torch.triu(sim, diagonal=1)[:, :, 1:]
        logits = -F.log_softmax(logits, dim=-1)

        t = torch.arange(T, device=z1.device)
        loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2
        return loss

    loss = torch.tensor(0.0, device=z1.device)
    d = 0
    while z1.size(1) > 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2)
        if d >= temporal_unit:
            if 1 - alpha != 0:
                loss += (1 - alpha) * temporal_contrastive_loss(z1, z2)
        d += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)
    if z1.size(1) == 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2)
        d += 1
    return loss / d


class CrossEntropyLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        # logger.info(f"Logits shape: {logits.shape}, Labels shape: {labels.shape}")
        return self.criterion(logits, labels)


class RegressionLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()

    def forward(self, x, y):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        return 2 - 2 * (x * y).sum(dim=-1)


class NT_Xent(nn.Module):
    def __init__(self, cfg):
        super(NT_Xent, self).__init__()
        # self.batch_size = batch_size
        # self.temperature = temperature
        # self.world_size = worNoneld_size
        self.batch_size = cfg.SOLVER.BATCH_SIZE
        self.temperature = cfg.MODEL.CRITERION.ARGS.TEMPERATURE

        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        z_i = z_i.squeeze()
        z_j = z_j.squeeze()
        mask = self.mask_correlated_samples(z_i.shape[0])
        N = 2 * z_i.shape[0]
        # logger.debug(z_i.shape)
        z = torch.cat((z_i, z_j), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, int(N / 2))
        sim_j_i = torch.diag(sim, int(-N / 2))

        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)

        loss = self.criterion(logits, labels)
        loss /= N
        return loss


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, cfg, contrast_mode="all"):
        super(SupConLoss, self).__init__()
        self.temperature = cfg.MODEL.ARGS.TEMPERATURE
        self.contrast_mode = contrast_mode
        self.base_temperature = cfg.MODEL.ARGS.TEMPERATURE

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = torch.device("cuda") if features.is_cuda else torch.device("cpu")

        # print(features.shape)
        if len(features.shape) < 3:
            raise ValueError(
                "`features` needs to be [bsz, n_views, ...],"
                "at least 3 dimensions are required"
            )
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError("Unknown mode: {}".format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class InfoNCE(nn.Module):
    def __init__(self, cfg, normalize=False):
        super().__init__()
        self.tau = cfg.MODEL.ARGS.TEMPERATURE
        self.normalize = normalize

    def forward(self, xi, xj):
        # print(xi.shape)
        x = torch.cat((xi, xj), dim=0)

        is_cuda = x.is_cuda
        sim_mat = torch.mm(x, x.T)
        if self.normalize:
            sim_mat_denom = torch.mm(
                torch.norm(x, dim=1).unsqueeze(1), torch.norm(x, dim=1).unsqueeze(1).T
            )
            sim_mat = sim_mat / sim_mat_denom.clamp(min=1e-16)

        sim_mat = torch.exp(sim_mat / self.tau)

        # no diag because it's not diffrentiable -> sum - exp(1 / tau)
        # diag_ind = torch.eye(xi.size(0) * 2).bool()
        # diag_ind = diag_ind.cuda() if use_cuda else diag_ind

        # sim_mat = sim_mat.masked_fill_(diag_ind, 0)

        # top
        if self.normalize:
            sim_mat_denom = torch.norm(xi, dim=1) * torch.norm(xj, dim=1)
            sim_match = torch.exp(torch.sum(xi * xj, dim=-1) / sim_mat_denom / self.tau)
        else:
            sim_match = torch.exp(torch.sum(xi * xj, dim=-1) / self.tau)

        sim_match = torch.cat((sim_match, sim_match), dim=0)

        norm_sum = torch.exp(torch.ones(x.size(0)) / self.tau)
        norm_sum = norm_sum.cuda() if is_cuda else norm_sum
        loss = torch.mean(-torch.log(sim_match / (torch.sum(sim_mat, dim=-1) - norm_sum)))
        return loss

# class OrthLoss(nn.Module):

#     def __init__(self):
#         super(OrthLoss, self).__init__()
#         self.criterion = nn.CosineSimilarity(dim=1)
#     def forward(self, input1, input2):
#         # Zero mean
#         # input1 = torch.flatten(input1, start_dim=1)
#         # input2 = torch.flatten(input2, start_dim=1)
#         input1 = F.adaptive_avg_pool2d(input1, (1, 1)).squeeze()
#         input2 = F.adaptive_avg_pool2d(input2, (1, 1)).squeeze()
#         input1 = torch.squeeze(input1)
#         input2 = torch.squeeze(input2)
#         diff_loss = 1 - self.criterion(input1, input2).pow(2).mean()
#         # input1_mean = torch.mean(input1, dim=0, keepdims=True)
#         # input2_mean = torch.mean(input2, dim=0, keepdims=True)
#         # input1 = input1 - input1_mean
#         # input2 = input2 - input2_mean

#         # input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
#         # input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)

#         # input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
#         # input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

#         # diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))
#         # # diff_loss = torch.mean((input1_l2 * input2_l2).sum(dim=1).pow(2))

#         return diff_loss
class OrthLoss(nn.Module):

    def __init__(self):
        super(OrthLoss, self).__init__()
        self.criterion = nn.CosineSimilarity(dim=1)
        self.margin = 1

    def forward(self, input1, input2):
        input1 = F.adaptive_avg_pool2d(input1, (1, 1)).squeeze()
        input2 = F.adaptive_avg_pool2d(input2, (1, 1)).squeeze()
        input1 = torch.squeeze(input1)
        input2 = torch.squeeze(input2)
        input1 = F.normalize(input1, p=2, dim=1)
        input2 = F.normalize(input2, p=2, dim=1)
        # distance = F.pairwise_distance(input1, input2)
        distance = torch.cdist(input1, input2)
        contrastive_loss = torch.mean(torch.clamp(self.margin - distance, min=0.0))

        return contrastive_loss
