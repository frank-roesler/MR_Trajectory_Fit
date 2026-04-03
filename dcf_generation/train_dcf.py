import sys

sys.path.append("/Users/frankrosler/Desktop/PhD/Python/MIRTorch/")

import torch
from params import *
import matplotlib.pyplot as plt
from time import time
import numpy as np
from itertools import cycle
from dcf_utils import (
    TrajectoryDCFDataset,
    get_rosette_batch,
    get_dcf_batch,
    get_petal_batch_from_rosette,
    get_dcf_petal_batch_from_dcf,
    plot_loss,
    plot_final_examples,
    train_step,
)
from torch.utils.data import DataLoader
from glob import glob
from models import UNet1D
import os
from datetime import datetime

n_epochs = 10
batch_size = 64
learning_rate = 5e-4
n_steps = 5000  # total steps. After n_epochs * len(train_data) / batch_size steps, start computing dcfs if not already doing so.
compute_dcfs = False
data_dir = "dcf_generation/train_data/"


train_data = glob(os.path.join(data_dir, "*"))
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps")
print("Device:", device)

# dcfnet = DCFNet(input_size=2 * (params["timesteps"] - 1), output_size=params["timesteps"] - 1, n_hidden=8, n_features=256).to(device)
# dcfnet = FCN1D(channels=[2, 128, 256, 512, 256, 128, 1], kernel_size=21).to(device)
dcfnet = UNet1D(in_channels=2, out_channels=1, features=[16, 32, 64, 128, 256]).to(device)

dataset = TrajectoryDCFDataset(data_dir)
if len(dataset) > 0:
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    dataloader_cycle = cycle(dataloader)
optimizer = torch.optim.Adam(dcfnet.parameters(), lr=learning_rate)

losses = []
best_loss = float("inf")
t0 = time()
for step in range(n_steps):
    if step >= len(train_data):
        compute_dcfs = True
    if compute_dcfs:
        with torch.no_grad():
            rosette_batch = get_rosette_batch(batch_size, device=device)
            dcf_batch = get_dcf_batch(rosette_batch, device=device)
            petal_batch = get_petal_batch_from_rosette(rosette_batch)
            dcf_petal_batch = get_dcf_petal_batch_from_dcf(dcf_batch)
            for i in range(batch_size):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                torch.save((petal_batch[i], dcf_petal_batch[i]), os.path.join(data_dir, f"{timestamp}_{i}.pt"))
    else:
        petal_batch, dcf_petal_batch = next(dataloader_cycle)
    if step >= n_epochs * len(train_data) / batch_size:
        compute_dcfs = True
    loss, dcf_pred_batch = train_step(dcfnet, optimizer, petal_batch, dcf_petal_batch, device=device)

    if loss < 0.99 * best_loss:
        torch.save(dcfnet.state_dict(), os.path.join(f"trained_models/dcfnet_general_{dcfnet.name}.pt"))
        best_loss = loss

    losses.append(loss)
    if compute_dcfs == True or step % 10 == 0:
        plot_loss(losses, step, t0)

print("FINAL LOSS", np.mean(losses[-100:]).item())
plot_loss(losses, step, t0, block=True)
plot_final_examples(dcf_batch, dcf_pred_batch)
