import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from .helpers import *

from functools import wraps

from utils.helpers import timing_start, timing_end

from loguru import logger


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


class ConvGenerator(nn.Module):
    def __init__(self, input_dim, output_channels, output_length):
        super(ConvGenerator, self).__init__()

        self.fc = nn.Linear(input_dim * 2, 1024)

        self.conv1 = nn.Conv1d(1024, 512, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(512, 256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(256, output_channels, kernel_size=3, padding=1)

        self.norm1 = nn.BatchNorm1d(512)
        self.norm2 = nn.BatchNorm1d(256)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.output_length = output_length

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        x = F.normalize(x, dim=1)

        x = self.fc(x)

        x = x.unsqueeze(2)

        x = self.relu(self.norm1(self.conv1(x)))
        x = self.relu(self.norm2(self.conv2(x)))
        x = self.tanh(self.conv3(x))

        x = nn.functional.interpolate(
            x, size=self.output_length, mode="linear", align_corners=True
        )

        return x.squeeze(2)

class TimeSeriesAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_length):
        super(TimeSeriesAutoencoder, self).__init__()

        self.decoder_fc = nn.Linear(input_dim, hidden_dims)  
        self.decoder_conv1 = nn.ConvTranspose1d(hidden_dims, hidden_dims // 2, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.decoder_conv2 = nn.ConvTranspose1d(hidden_dims // 2, output_length, kernel_size=5, stride=2, padding=2, output_padding=1)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh() 

    def forward(self, x1, x2):
        encoded_features = torch.cat((x1, x2), dim=1)
        x = F.normalize(encoded_features, dim=1)
        x = self.relu(self.decoder_fc(encoded_features))
        x = x.unsqueeze(2)  
        x = self.relu(self.decoder_conv1(x))
        x = self.decoder_conv2(x)
        return self.tanh(x).squeeze() 

class Generator(nn.Module):
    def __init__(self, input_dim, output_channels, output_length):
        super(Generator, self).__init__()
        self.fc = nn.Linear(input_dim*2, 512)

        # Up-sampling and Conv layers, adjusted to match the desired output dimensions
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1)
        self.final_conv = nn.Conv2d(8, output_channels, kernel_size=3, stride=1, padding=1)

        # Batch normalization layers
        self.norm1 = nn.BatchNorm2d(256)
        self.norm2 = nn.BatchNorm2d(128)
        self.norm3 = nn.BatchNorm2d(64)
        self.norm4 = nn.BatchNorm2d(32)
        self.norm5 = nn.BatchNorm2d(16)
        self.norm6 = nn.BatchNorm2d(8)

        # Activation functions
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.output_length = output_length

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        x = F.normalize(x, dim=1)
        x = self.fc(x)
        x = x.view(-1, 512, 1, 1)  # Reshape to match the dimensions for convolutions
        x = self.up(x)
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.up(x)
        x = self.relu(self.norm2(self.conv2(x)))
        x = self.up(x)
        x = self.relu(self.norm3(self.conv3(x)))
        x = self.up(x)
        x = self.relu(self.norm4(self.conv4(x)))
        x = self.up(x)
        x = self.relu(self.norm5(self.conv5(x)))
        x = self.up(x)
        x = self.relu(self.norm6(self.conv6(x)))
        x = self.final_conv(x)
        x = F.interpolate(x, size=(self.output_length, 1), mode='nearest')  # Adjust the size to [batchsize, 1, 3000, 1]
        return self.tanh(x)


# class Generator(nn.Module):
#     def __init__(self, input_dim, output_channels, output_length):
#         super(Generator, self).__init__()
#         self.fc = nn.Linear(input_dim*2, 512)

#         # Up-sampling and Conv layers, adjusted to match the desired output dimensions
#         self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         self.conv1 = nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1)
#         self.conv3 = nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1)
#         self.conv5 = nn.Conv2d(32, 8, kernel_size=3, stride=1, padding=1)
#         self.final_conv = nn.Conv2d(8, output_channels, kernel_size=3, stride=1, padding=1)

#         # Batch normalization layers
#         self.norm2 = nn.BatchNorm2d(128)
#         self.norm4 = nn.BatchNorm2d(32)
#         self.norm6 = nn.BatchNorm2d(8)

#         # Activation functions
#         self.relu = nn.ReLU()
#         self.tanh = nn.Tanh()
#         self.output_length = output_length

#     def forward(self, x1, x2):
#         x = torch.cat((x1, x2), dim=1)
#         x = F.normalize(x, dim=1)
#         x = self.fc(x)
#         x = x.view(-1, 512, 1, 1)  # Reshape to match the dimensions for convolutions

#         x = self.up(x)
#         x = self.relu(self.norm2(self.conv1(x)))
#         x = self.up(x)
#         x = self.relu(self.norm4(self.conv3(x)))
#         x = self.up(x)
#         x = self.relu(self.norm6(self.conv5(x)))
#         x = self.final_conv(x)

#         x = F.interpolate(x, size=(self.output_length, 1), mode='nearest')  # Adjust the size to [batchsize, 1, 3000, 1]
#         return self.tanh(x)

class GeneratorSimple(nn.Module):
    def __init__(self):
        super(GeneratorSimple, self).__init__()
        self.fc = nn.Linear(512, 256)  # 减少输出特征数量

        # 简化上采样和卷积层
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.final_conv = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

        # 减少批量归一化层
        self.norm1 = nn.BatchNorm2d(128)
        self.norm2 = nn.BatchNorm2d(64)

        # 激活函数
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 256, 1, 1)  # 适应卷积的维度需求
        x = self.up(x)
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.up(x)
        x = self.relu(self.norm2(self.conv2(x)))
        x = self.up(x)  # 进行额外的上采样以达到目标长度
        x = self.final_conv(x)
        x = F.interpolate(x, size=(3000, 1), mode='nearest')  # 调整大小至[batchsize, 1, 3000, 1]
        return self.tanh(x)

# class ConvDecoder(nn.Module):
#     def __init__(self, input_dim, output_channels, output_length):
#         super(ConvDecoder, self).__init__()
#         self.output_length = output_length

#         self.fc = nn.Linear(input_dim * 2, input_dim)

#         self.conv1 = nn.Conv1d(
#             in_channels=input_dim, out_channels=input_dim * 2, kernel_size=3, padding=1
#         )
#         self.conv2 = nn.Conv1d(
#             in_channels=input_dim * 2,
#             out_channels=output_channels,
#             kernel_size=3,
#             padding=1,
#         )

#         self.adjust_length = nn.Conv1d(
#             in_channels=output_channels, out_channels=output_channels, kernel_size=1
#         )

#     def forward(self, x1, x2):
#         # x = torch.cat((x1, x2), dim=1)
#         x = F.normalize(torch.cat((x1 , x2), dim=1), dim=1)

#         x = self.fc(x)

#         x = x.unsqueeze(2)

#         x = torch.relu(self.conv1(x))
#         x = torch.relu(self.conv2(x))

#         if x.shape[2] < self.output_length:
#             x = nn.functional.interpolate(
#                 x, size=self.output_length, mode="linear", align_corners=True
#             )
#         x = self.adjust_length(x)

#         return x
