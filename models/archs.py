from models.blocks import *
from models.losses import *
from collections import OrderedDict

import torchvision.models as models

from loguru import logger

backbone_dict = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
}

class BaseNet(nn.Module):
    def __init__(self, cfg):
        super(BaseNet, self).__init__()
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
                        "Temporal_VAR",
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
        dropout = cfg.MODEL.ARGS.DROP_OUT
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
                    ("dropout", nn.Dropout(p=dropout)),
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


class SimCLR(BaseNet):
    def __init__(self, cfg):
        super(SimCLR, self).__init__(cfg)
        input_channels = cfg.DATASET.CHANNELS
        num_classes = cfg.DATASET.NUM_CLASSES

        backbone_name = cfg.MODEL.ARGS.BACKBONE
        projection_dim = cfg.MODEL.ARGS.PROJECTION_DIM

        self.backbone = backbone_dict[backbone_name](pretrained=False)
        self.backbone.conv1 = nn.Conv2d(
            input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        print(self.backbone)
        self.projection_head = nn.Sequential(
            nn.Linear(self.backbone.fc.in_features, self.backbone.fc.in_features),
            nn.ReLU(),
            nn.Linear(self.backbone.fc.in_features, projection_dim),
        )
        self.backbone.fc = nn.Identity()

        self.criterion = InfoNCE(cfg)
        # self.criterion = contrastive_loss(cfg)

        # self.fc = nn.Linear(projection_dim, num_classes)

    def forward(self, x_i, x_j):
        h_i = self.backbone(x_i)
        h_j = self.backbone(x_j)

        z_i = F.normalize(self.projection_head(h_i), dim=-1)
        z_j = F.normalize(self.projection_head(h_j), dim=-1)
        # logger.debug(z_i.shape)

        # loss = self.compute_loss(z_i, z_j)

        return h_i, h_j, z_i, z_j 

class SimSiam(BaseNet):
    def __init__(self, cfg):
        super(SimSiam, self).__init__(cfg)
        input_channels = cfg.DATASET.CHANNELS
        num_classes = cfg.DATASET.NUM_CLASSES

        backbone_name = cfg.MODEL.ARGS.BACKBONE

