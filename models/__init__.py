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
}

criterion_dict = {
    "SupConLoss": SupConLoss,
    "NT_Xent": NT_Xent,
    "CE": CrossEntropyLoss,
    "InfoNCE": InfoNCE,
}
