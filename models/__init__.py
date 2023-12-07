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
    "TimeNet": (
        TimeNet,
        ""
    ),
    "SimCLR": (
        SimCLR,
        ""
    ),
}
