import os

from models.archs import *

from models.losses import *

model_dict = {
    "VARCNN": (VARCNN, ""),
    "SimCLR": (SimCLR, ""),
    "LinearClassifier": (LinearClassifier, ""),
    "BYOL": (BYOL, ""),
    "SimSiam": (BYOL, ""),
    "TSEncoder": (TSEncoder, ""),
    "CurrentCLR": (CurrentCLR, ""),
    "CurrentSimCLR": (CurrentSimCLR, ""),
    "Equimod": (Equimod, ""),
    "ts2vec": (Ts2vec,"")
}

criterion_dict = {
    "SupConLoss": SupConLoss,
    "NT_Xent": NT_Xent,
    "CE": CrossEntropyLoss,
    "InfoNCE": InfoNCE,
    "RegressionLoss": RegressionLoss,
}
