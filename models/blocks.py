import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from .helpers import *

from functools import wraps


def MaybeSyncBatchnorm(is_distributed=None):
    is_distributed = default(
        is_distributed, dist.is_initialized() and dist.get_world_size() > 1
    )
    return nn.SyncBatchNorm if is_distributed else nn.BatchNorm1d


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, x):
        print(x.shape)
        return x


# define torch.Tensor.view in torch.nn
class TensorView(nn.Module):
    def __init__(self):
        super(TensorView, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class Unsqueeze(nn.Module):
    def __init__(self, dim):
        super(Unsqueeze, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.unsqueeze(x, self.dim)


class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0, self.dim1 = dim0, dim1

    def forward(self, x):
        return torch.transpose(x, self.dim0, self.dim1)


class VARConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7):
        super(VARConv, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
        )

    def forward(self, x):
        # x: (batch_size, channels, seq_len)
        return self.conv(x)


class LFConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7):
        super(LFConv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, kernel_size),
            padding=(0, kernel_size - 1 // 2),
            groups=in_channels,
        )

    def forward(self, x):
        # x: (batch_size, channels, seq_len)
        x = x.unsqueeze(-2)
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(
                inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False
            ),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.BatchNorm2d(outchannel),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    inchannel, outchannel, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(outchannel),
            )

    def forward(self, x):
        out = self.left(x)
        out = out + self.shortcut(x)
        out = F.relu(out)

        return out


class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7):
        super(TemporalConv, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
        )


class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out
