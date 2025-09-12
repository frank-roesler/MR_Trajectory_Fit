import math
import os
from datetime import datetime

export_path = os.path.join("results", datetime.now().strftime("%Y-%m-%d_%H-%M"))

params = {
    "train_steps": 10000,
    "lr": 1e-3,
    "grad_loss_weight": 1e-3,
    "slew_loss_weight": 1e-4,
    "zero_filling": True,
    "timesteps": 101,
    "duration": 0.5,  # ms
    "grad_max": 135,
    "slew_rate": 240,
    "gamma": 42.575575,  # Gyromagnetic ratio in MHz/T
    "img_size": 512,  # pixels
    "FoV": 224,  # mm
    "res": 50,
    "n_petals": 80,
    "model_size": 11,  # number of Fourier coefficients
    "loss_function": "combined",
}

kmax_traj = params["res"] / (2 * params["FoV"])  # 1/m
kmax_img = params["img_size"] / (2 * params["FoV"])  # 1/m
dt = params["duration"] / (params["timesteps"] - 1)
final_FT_scaling = 4 / params["img_size"] ** 2
