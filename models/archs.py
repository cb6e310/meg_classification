from models.blocks import *
from collections import OrderedDict

import torchvision.models as models

from loguru import logger


class BaseNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def forward(self, x):
        raise NotImplementedError("Subclass must implement forward method")

    def compute_loss(self, output, target):
        criterion = nn.CrossEntropyLoss()
        return criterion(output, target)

    def get_learnable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


class VARCNN(BaseNet):
    def __init__(self, cfg):
        super().__init__(cfg)
        meg_channels = cfg.DATASET.CHANNELS
        points_length = cfg.DATASET.POINTS
        num_classes = cfg.DATASET.NUM_CLASSES

        max_pool = cfg.MODEL.ARGS.MAX_POOL
        sources_channels = cfg.MODEL.ARGS.SOURCE_CHANNELS

        Conv = VARConv

        self.net = nn.Sequential(
            OrderedDict(
                [
                    ("transpose0", Transpose(1, 2)),
                    ("Spatial", nn.Linear(meg_channels, sources_channels)),
                    ("transpose1", Transpose(1, 2)),
                    (
                        "Temporal_LF",
                        Conv(
                            in_channels=sources_channels,
                            out_channels=sources_channels,
                            kernel_size=7,
                        ),
                    ),
                    ("unsqueeze", Unsqueeze(-3)),
                    ("active", nn.ReLU()),
                    ("pool", nn.MaxPool2d((1, max_pool), (1, max_pool))),
                    ("view", TensorView()),
                    ("dropout", nn.Dropout(p=0.5)),
                    (
                        "linear",
                        nn.Linear(sources_channels * int(points_length / 2), num_classes),
                    ),
                ]
            )
        )

    def forward(self, x, target=None):
        preds = self.net(x)
        loss = self.compute_loss(preds, target)
        return preds, loss


class LFCNN(BaseNet):
    def __init__(self, cfg):
        super().__init__()
        meg_channels = cfg.DATASET.CHANNELS
        points_length = cfg.DATASET.POINTS
        num_classes = cfg.DATASET.NUM_CLASSES

        max_pool = cfg.MODEL.ARGS.MAX_POOL
        sources_channels = cfg.MODEL.ARGS.SOURCE_CHANNELS

        batch_size = cfg.SOLVER.BATCH_SIZE

        Conv = LFConv

        self.net = nn.Sequential(
            OrderedDict(
                [
                    ("transpose1", Transpose(1, 2)),
                    ("Spatial", nn.Linear(meg_channels, sources_channels)),
                    ("transpose1", Transpose(1, 2)),
                    ("unsqueeze", Unsqueeze(-2)),
                    (
                        "Temporal_LF",
                        Conv(
                            in_channels=sources_channels,
                            out_channels=sources_channels,
                            kernel_size=7,
                        ),
                    ),
                    ("active", nn.ReLU()),
                    ("transpose2", Transpose(1, 2)),
                    ("pool", nn.MaxPool2d((1, max_pool), (1, max_pool))),
                    ("view", TensorView(batch_size, -1)),
                    ("dropout", nn.Dropout(p=0.5)),
                    (
                        "linear",
                        nn.Linear(sources_channels * int(points_length / 2), num_classes),
                    ),
                ]
            )
        )

    def forward(self, x):
        return self.net(x)


class ResNet18(BaseNet):
    def __init__(self, cfg):
        super().__init__(cfg)
        meg_channels = cfg.DATASET.CHANNELS
        points_length = cfg.DATASET.POINTS
        num_classes = cfg.DATASET.NUM_CLASSES

        # sources_channels = cfg.MODEL.ARGS.SOURCE_CHANNELS

        batch_size = cfg.SOLVER.BATCH_SIZE

        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512 * 6 * 3, cfg.DATASET.NUM_CLASSES)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x, target=None):
        x = x.unsqueeze(1)
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        loss = self.compute_loss(out, target)

        return out, loss


class TimeNet(BaseNet):
    def __init__(self, cfg):
        super().__init__(cfg)
        meg_channels = cfg.DATASET.CHANNELS
        points_length = cfg.DATASET.POINTS
        num_classes = cfg.DATASET.NUM_CLASSES

        max_pool = cfg.MODEL.ARGS.MAX_POOL
        channel_1 = cfg.MODEL.ARGS.CHANNEL_1
        channel_2 = cfg.MODEL.ARGS.CHANNEL_2
        kernel_size_1 = cfg.MODEL.ARGS.KERNEL_SIZE_1
        kernel_size_2 = cfg.MODEL.ARGS.KERNEL_SIZE_2
        batch_norm_flag = cfg.MODEL.ARGS.BATCH_NORM_FLAG

        # sources_channels = cfg.MODEL.ARGS.SOURCE_CHANNELS

        batch_size = cfg.SOLVER.BATCH_SIZE

        self.net = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1d-1",
                        nn.Conv1d(
                            meg_channels,
                            channel_1,
                            kernel_size=kernel_size_1,
                            stride=1,
                            padding=kernel_size_1 // 2,
                            bias=False,
                        ),
                    ),
                    (
                        "bn1",
                        nn.BatchNorm1d(channel_1) if batch_norm_flag else nn.Identity(),
                    ),
                    ("relu1", nn.ReLU(inplace=True)),
                    (
                        "conv1d-2",
                        nn.Conv1d(
                            channel_1,
                            channel_2,
                            kernel_size=kernel_size_2,
                            stride=1,
                            padding=kernel_size_2 // 2,
                            bias=False,
                        ),
                    ),
                    (
                        "bn2",
                        nn.BatchNorm1d(channel_2) if batch_norm_flag else nn.Identity(),
                    ),
                    ("relu2", nn.ReLU(inplace=True)),
                    ("maxpool", nn.MaxPool1d(max_pool)),
                    ("dropout", nn.Dropout(0.5)),
                    ("flatten", TensorView()),
                    (
                        "linear",
                        nn.Linear(channel_2 * int(points_length / 2), num_classes),
                    ),
                ]
            )
        )

    def forward(self, x, target=None):
        out = self.net(x)
        loss = self.compute_loss(out, target)
        return out, loss
