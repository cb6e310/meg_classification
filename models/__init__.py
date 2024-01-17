import os

from models.archs import *

from models.losses import *

model_dict = {
    "LFCNN": (LFCNN, ""),
    "VARCNN": (VARCNN, ""),
    "TimeNet": (TimeNet, ""),
    "SimCLR": (SimCLR, ""),
    "LinearClassifier": (LinearClassifier, ""),
    "BYOL": (BYOL, ""),
    "TS2Vec": (TSEncoder, ""),
}

criterion_dict = {
    "SupConLoss": SupConLoss,
    "NT_Xent": NT_Xent,
    "CE": CrossEntropyLoss,
    "InfoNCE": InfoNCE,
    "RegressionLoss": RegressionLoss,
}
