from models.blocks import *
from models.losses import *
from models.helpers import *
from collections import OrderedDict
import copy

import torchvision.models as models
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from torchvision import transforms as T

from loguru import logger

from utils.helpers import timing_start, timing_end


def create_VARCNNBackbone(cfg):
    return VARCNNBackbone(cfg)


backbone_dict = {
    "resnet18": [models.resnet18, 512],
    "resnet34": [models.resnet34, 512],
    "resnet50": [models.resnet50, 2048],
    "varcnn": [create_VARCNNBackbone, 1800],
}


class TSEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.input_dims = cfg.DATASET.CHANNELS
        self.output_dims = cfg.MODEL.ARGS.PROJECTION_DIM
        self.hidden_dims = cfg.MODEL.ARGS.HIDDEN_SIZE
        self.mask_mode = cfg.MODEL.ARGS.MASK_MODE
        self.depth = cfg.MODEL.ARGS.DEPTH
        self.input_fc = nn.Linear(self.input_dims, self.hidden_dims)
        self.feature_extractor = DilatedConvEncoder(
            self.hidden_dims,
            [self.hidden_dims] * self.depth + [self.output_dims],
            kernel_size=3,
        )
        self.repr_dropout = nn.Dropout(p=0.1)

    def forward(
        self, x, mask=None, return_embedding=True, return_projection=False
    ):  # x: B x T x input_dims
        x = x.transpose(1, 2).squeeze()
        nan_mask = ~x.isnan().any(axis=-1)
        x[~nan_mask] = 0
        x = self.input_fc(x)  # B x T x Ch

        # generate & apply mask
        if mask is None:
            if self.training:
                mask = self.mask_mode
            else:
                mask = "all_true"

        if mask == "binomial":
            mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == "continuous":
            mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == "all_true":
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
        elif mask == "all_false":
            mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
        elif mask == "mask_last":
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
            mask[:, -1] = False

        mask &= nan_mask
        x[~mask] = 0

        # conv encoder
        x = x.transpose(1, 2)  # B x Ch x T
        x = self.feature_extractor(x)
        if self.repr_dropout is not None:
            x = self.repr_dropout(x)  # B x Co x T
        x = x.transpose(1, 2)  # B x T x Co

        x = F.max_pool1d(x.transpose(1, 2).contiguous(), kernel_size=x.size(1)).transpose(
            1, 2
        )
        x = x.squeeze()
        # normalize feature
        x = F.normalize(x, dim=-1)

        return x


# MLP class for projector and predictor
def MLP(dim, projection_size, hidden_size=4096, sync_batchnorm=None):
    return nn.Sequential(
        nn.Linear(dim, hidden_size),
        MaybeSyncBatchnorm(sync_batchnorm)(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size),
    )


def SimSiamMLP(dim, projection_size, hidden_size=4096, sync_batchnorm=None):
    return nn.Sequential(
        nn.Linear(dim, hidden_size, bias=False),
        MaybeSyncBatchnorm(sync_batchnorm)(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, hidden_size, bias=False),
        MaybeSyncBatchnorm(sync_batchnorm)(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size, bias=False),
        MaybeSyncBatchnorm(sync_batchnorm)(projection_size, affine=False),
    )


# a wrapper class for the base neural network
# will manage the interception of the hidden layer output
# and pipe it into the projecter and predictor nets
class NetWrapper(nn.Module):
    def __init__(
        self,
        net,
        projection_size,
        projection_hidden_size,
        layer=-2,
        use_simsiam_mlp=False,
        sync_batchnorm=None,
    ):
        super().__init__()
        self.net = net
        self.layer = layer

        self.projector = None
        self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size

        self.use_simsiam_mlp = use_simsiam_mlp
        self.sync_batchnorm = sync_batchnorm

        self.hidden = {}
        self.hook_registered = False

    def _find_layer(self):
        if type(self.layer) == str:
            modules = dict([*self.net.named_modules()])
            return modules.get(self.layer, None)
        elif type(self.layer) == int:
            children = [*self.net.children()]
            return children[self.layer]
        return None

    def _hook(self, _, input, output):
        device = input[0].device
        self.hidden[device] = flatten(output)

    def _register_hook(self):
        layer = self._find_layer()
        assert layer is not None, f"hidden layer ({self.layer}) not found"
        handle = layer.register_forward_hook(self._hook)
        self.hook_registered = True

    @singleton("projector")
    def _get_projector(self, hidden):
        _, dim = hidden.shape
        create_mlp_fn = MLP if not self.use_simsiam_mlp else SimSiamMLP
        projector = create_mlp_fn(
            dim,
            self.projection_size,
            self.projection_hidden_size,
            sync_batchnorm=self.sync_batchnorm,
        )
        return projector.to(hidden)

    def get_representation(self, x):
        if self.layer == -1:
            return self.net(x)

        if not self.hook_registered:
            self._register_hook()

        self.hidden.clear()
        _ = self.net(x)
        hidden = self.hidden[x.device]
        self.hidden.clear()

        assert hidden is not None, f"hidden layer {self.layer} never emitted an output"
        return hidden

    def forward(self, x, return_projection=True):
        representation = self.get_representation(x)

        if not return_projection:
            return representation

        projector = self._get_projector(representation)
        projection = projector(representation)
        return projection, representation


# main class
class BYOL(nn.Module):
    def __init__(
        self,
        cfg,
        # net,
        # feature_size,
        # projection_size=256,
        # projection_hidden_size=4096,
        # moving_average_decay=0.99,
        # use_momentum=True,
        # sync_batchnorm=None,
    ):
        super().__init__()
        self.net = backbone_dict[cfg.MODEL.ARGS.BACKBONE][0](pretrained=False)
        feature_size = cfg.DATASET.POINTS
        channels = cfg.DATASET.CHANNELS
        hidden_layer = -2
        projection_size = cfg.MODEL.ARGS.PROJECTION_DIM
        projection_hidden_size = cfg.MODEL.ARGS.PROJECTION_HIDDEN_SIZE
        moving_average_decay = cfg.MODEL.ARGS.TAU_BASE
        use_momentum = cfg.MODEL.ARGS.USE_MOMENTUM
        sync_batchnorm = None

        self.net.conv1 = nn.Conv2d(
            channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )

        self.online_encoder = NetWrapper(
            self.net,
            projection_size,
            projection_hidden_size,
            layer=hidden_layer,
            use_simsiam_mlp=not use_momentum,
            sync_batchnorm=sync_batchnorm,
        )

        self.use_momentum = use_momentum
        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average_decay)

        self.online_predictor = MLP(
            projection_size, projection_size, projection_hidden_size
        )

        # get device of network and make wrapper same device
        device = get_module_device(self.net)
        self.to(device)

        # send a mock image tensor to instantiate singleton parameters
        self.forward(
            torch.randn(1, channels, feature_size, 1, device=device),
            torch.randn(1, channels, feature_size, 1, device=device),
        )

    @singleton("target_encoder")
    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.online_encoder)
        set_requires_grad(target_encoder, False)
        return target_encoder

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        assert (
            self.use_momentum
        ), "you do not need to update the moving average, since you have turned off momentum for the target encoder"
        assert self.target_encoder is not None, "target encoder has not been created yet"
        update_moving_average(
            self.target_ema_updater, self.target_encoder, self.online_encoder
        )

    def forward(
        self, batch_view_1, batch_view_2, return_embedding=False, return_projection=True
    ):
        # assert not (
        #     self.training and batch_view_1.shape[0] == 1
        # ), "you must have greater than 1 sample when training, due to the batchnorm in the projection layer"

        if return_embedding:
            return self.online_encoder(batch_view_1, return_projection=return_projection)

        views = torch.cat((batch_view_1, batch_view_2), dim=0)

        online_projections, _ = self.online_encoder(views)
        online_predictions = self.online_predictor(online_projections)

        online_pred_one, online_pred_two = online_predictions.chunk(2, dim=0)

        with torch.no_grad():
            target_encoder = (
                self._get_target_encoder() if self.use_momentum else self.online_encoder
            )

            target_projections, _ = target_encoder(views)
            target_projections = target_projections.detach()

            target_proj_one, target_proj_two = target_projections.chunk(2, dim=0)

        return (
            online_pred_one,
            online_pred_two,
            target_proj_one,
            target_proj_two,
        )


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


class VARCNNBackbone(BaseNet):
    def __init__(self, cfg):
        super().__init__(cfg)
        meg_channels = cfg.DATASET.CHANNELS
        points_length = cfg.DATASET.POINTS
        num_classes = cfg.DATASET.NUM_CLASSES

        sources_channels = 36

        Conv = VARConv

        # refact nn.Sequential
        self.transpose0 = Transpose(1, 2)
        self.Spatial = nn.Linear(meg_channels, sources_channels)
        self.transpose1 = Transpose(1, 2)
        self.Temporal_VAR = Conv(
            in_channels=sources_channels,
            out_channels=sources_channels,
            kernel_size=7,
        )
        self.unsqueeze = Unsqueeze(-3)
        self.active = nn.ReLU()
        self.pool = nn.MaxPool2d((1, 2), (1, 2))
        self.view = TensorView()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x, target=None):
        x = self.transpose0(x)
        x = x.squeeze()
        x = self.Spatial(x)
        x = self.transpose1(x)
        x = self.Temporal_VAR(x)
        x = self.unsqueeze(x)
        x = self.active(x)
        x = self.pool(x)
        x = self.view(x)
        x = self.dropout(x)
        return x


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

        if "resnet" in backbone_name:
            self.backbone = backbone_dict[backbone_name][0](pretrained=False)
            self.backbone.conv1 = nn.Conv2d(
                input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.projection_head = nn.Sequential(
                nn.Linear(self.backbone.fc.in_features, self.backbone.fc.in_features),
                nn.ReLU(),
                nn.Linear(self.backbone.fc.in_features, projection_dim),
            )
            self.backbone.fc = nn.Identity()
        elif "varcnn" in backbone_name:
            self.backbone = backbone_dict[backbone_name][0](cfg)
            self.projection_head = nn.Sequential(
                nn.Linear(backbone_dict[backbone_name][1], projection_dim),
            )
        print(self.backbone)

        # self.criterion = contrastive_loss(cfg)

        # self.fc = nn.Linear(projection_dim, num_classes)

    def forward(self, x_i, x_j, return_embedding=False, return_projection=True):
        h_i = self.backbone(x_i)
        h_j = self.backbone(x_j)
        z_i = F.normalize(self.projection_head(h_i), dim=-1)
        z_j = F.normalize(self.projection_head(h_j), dim=-1)
        # logger.debug(z_i.shape)

        # loss = self.compute_loss(z_i, z_j)
        if return_embedding:
            return h_i
        return h_i, h_j, z_i, z_j


class SimSiam(BaseNet):
    def __init__(self, cfg):
        super(SimSiam, self).__init__(cfg)
        input_channels = cfg.DATASET.CHANNELS
        num_classes = cfg.DATASET.NUM_CLASSES

        backbone_name = cfg.MODEL.ARGS.BACKBONE


# class BYOL(BaseNet):
#     def __init__(self, cfg):
#         super().__init__(cfg)
#         input_channels = cfg.DATASET.CHANNELS
#         num_classes = cfg.DATASET.NUM_CLASSES

#         backbone_name = cfg.MODEL.ARGS.BACKBONE
#         projection_hidden_size = cfg.MODEL.ARGS.PROJECTION_HIDDEN_SIZE
#         projection_dim = cfg.MODEL.ARGS.PROJECTION_DIM

#         self.target_network = backbone_dict[backbone_name][0](pretrained=False)
#         self.online_network = backbone_dict[backbone_name][0](pretrained=False)

#         self.target_network.conv1 = nn.Conv2d(
#             input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
#         )
#         self.online_network.conv1 = nn.Conv2d(
#             input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
#         )

#         projection_input_dim = self.online_network.fc.in_features

#         self.projection_head = nn.Sequential(
#             nn.Linear(projection_input_dim, projection_hidden_size),
#             MaybeSyncBatchnorm()(projection_hidden_size),
#             nn.ReLU(),
#             nn.Linear(projection_hidden_size, projection_dim),
#         )

#         self.online_predictor = nn.Sequential(
#             nn.Linear(projection_input_dim)
#         )

#         self.target_network.fc = nn.Identity()
#         self.online_network.fc = nn.Identity()

#     def forward(self, batch_view_1, batch_view_2):
#         # compute query feature
#         predictions_from_view_1 = self.projection_head(self.online_network(batch_view_1))
#         predictions_from_view_2 = self.projection_head(self.online_network(batch_view_2))

#         # compute key features
#         with torch.no_grad():
#             target_predictions_from_view_1 = self.target_network(batch_view_1)
#             target_predictions_from_view_2 = self.target_network(batch_view_2)

#         return (
#             predictions_from_view_1,
#             predictions_from_view_2,
#             target_predictions_from_view_1,
#             target_predictions_from_view_2,
#         )


class LinearClassifier(nn.Module):
    def __init__(self, cfg):
        super(LinearClassifier, self).__init__()
        n_features = cfg.MODEL.ARGS.N_FEATURES
        n_classes = cfg.DATASET.NUM_CLASSES
        self.model = nn.Linear(n_features, n_classes)

    def forward(self, x):
        return self.model(x)
