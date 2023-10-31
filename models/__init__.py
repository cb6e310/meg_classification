import os

from models.archs import *

model_dict = {
    "LFCNN": (
        LFCNN,
        ""
    ),
    "VARCNN": (
        VARCNN,
        ""
    ),
    "ResNet18": (
        ResNet18,
        ""
    ),
    "TimeNet": (
        TimeNet,
        ""
    ),
}
