import torch
from easydict import EasyDict as edict

config = {
    "gpu_mode": True,

    "iters": 50,  # 500
    "lr": 0.2,  # 0.01
    "max_break_count": 15,  # 15
    "break_threshold_ratio": 0.000005,

    "samples": 6000,  # 6000
    "motion_type": "Sim3",
    "rotation_format": "euler",

    "m": 9,
    "k0": -9,  # -8
    "depth": 3,
    "width": 128,
    "act_fn": "relu",

    "w_reg": 0.0,
    "w_ldmk": 0.1,
    "w_cd": 0.1,
    "trunc": 100
}