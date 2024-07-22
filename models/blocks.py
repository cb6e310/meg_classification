import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from .helpers import *

from functools import wraps

from utils.helpers import timing_start, timing_end

from loguru import logger


def default(val, def_val):
    return def_val if val is None else val
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
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
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


# ts2vec
class SamePadConv(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, dilation=1, stride=1, groups=1
    ):
        super().__init__()
        self.receptive_field = (kernel_size - 1) * dilation + 1
        padding = self.receptive_field // 2
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            stride=stride,
            groups=groups,
        )
        self.remove = 1 if self.receptive_field % 2 == 0 else 0

    def forward(self, x):
        out = self.conv(x)
        if self.remove > 0:
            out = out[:, :, : -self.remove]
        return out


# ts2vec
class ConvBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, dilation, stride=1, final=False
    ):
        super().__init__()
        self.conv1 = SamePadConv(
            in_channels, out_channels, kernel_size, stride=stride, dilation=dilation
        )
        self.conv2 = SamePadConv(
            out_channels, out_channels, kernel_size, stride=1, dilation=dilation
        )
        if stride == 1:
            self.projector = (
                nn.Conv1d(in_channels, out_channels, 1)
                if in_channels != out_channels or final
                else None
            )
        else:
            self.projector = nn.Conv1d(in_channels, out_channels, 1, stride=stride)

    def forward(self, x):
        residual = x if self.projector is None else self.projector(x)
        x = F.gelu(x)
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        return x + residual


# ts2vec
class DilatedConvEncoder(nn.Module):
    def __init__(self, in_channels, channels, kernel_size, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            *[
                ConvBlock(
                    channels[i - 1] if i > 0 else in_channels,
                    channels[i],
                    kernel_size=kernel_size,
                    dilation=2**i,
                    stride=stride,
                    final=(i == len(channels) - 1),
                )
                for i in range(len(channels))
            ]
        )

    def forward(self, x):
        return self.net(x)


class UpSampling1DNet(nn.Module):
    def __init__(self, scale_factor):
        super(UpSampling1DNet, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return F.interpolate(
            x, scale_factor=self.scale_factor, mode="linear", align_corners=False
        )


class ConvDecoder(nn.Module):
    def __init__(self, input_channels, filter_size, channels=1, length=3000):
        super(ConvDecoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.Conv1d(input_channels, 256, kernel_size=filter_size, padding="same"),
            nn.ReLU(),
            UpSampling1DNet(scale_factor=2),
            nn.Conv1d(256, 128, kernel_size=filter_size, padding="same"),
            nn.ReLU(),
            UpSampling1DNet(scale_factor=2),
            nn.Conv1d(128, 64, kernel_size=filter_size, padding="same"),
            nn.ReLU(),
            UpSampling1DNet(scale_factor=2),
            nn.Conv1d(64, 32, kernel_size=filter_size, padding="same"),
            nn.ReLU(),
            UpSampling1DNet(scale_factor=2),
            nn.Conv1d(32, channels, kernel_size=filter_size, padding="same"),
        )

        self.fusion = nn.Conv1d(256, 256, kernel_size=1)

        self.length = length

    def forward(self, x1, x2):
        # encoded = F.normalize(x1 + x2, dim=2)
        encoded = x1+x2
        encoded = torch.squeeze(encoded)
        decoded = self.decoder(encoded)
        decoded = F.interpolate(
            decoded, size=self.length, mode="linear", align_corners=False
        )
        return decoded
