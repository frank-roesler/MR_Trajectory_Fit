import sys
import os

sys.path.append(os.getcwd())

import torch
from params import *

# import matplotlib.pyplot as plt
from time import time
import numpy as np
from dcf_utils import (
    TrajectoryDCFDataset,
    get_rosette_batch,
    plot_loss,
    plot_final_examples,
    calculate_pipe_E,
    plot_E_colormap,
)
from torch.utils.data import DataLoader
from models import UNet1D, FourierCurve
import os
import torchkbnufft as tkbn
import matplotlib.pyplot as plt

output_dir = "trained_models/"

n_epochs = 10000
batch_size = 16
learning_rate = 5e-4

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps")
print("Device:", device)

dcfnet = UNet1D(in_channels=2, out_channels=1, features=[16, 32, 64, 128, 256], kernel_size=5).to(device)

optimizer = torch.optim.Adam(dcfnet.parameters(), lr=learning_rate)
model = FourierCurve(
    tmin=0,
    tmax=params["duration"],
    n_petals=params["n_petals"],
    initial_max=kmax_traj,
    n_coeffs=params["model_size"],
    coeff_lvl=0.5,
    angle_lvl=0.05,
).to(device)

# --- Dataloaders ---
base_data_dir = "dcf_generation/train_data/"
train_dataset = TrajectoryDCFDataset(os.path.join(base_data_dir, "train"))
val_dataset = TrajectoryDCFDataset(os.path.join(base_data_dir, "val"))
test_dataset = TrajectoryDCFDataset(os.path.join(base_data_dir, "test"))

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# --- Training Loop ---
best_val_loss = float("inf")
t0 = time()
train_losses_history = []
val_losses_history = []

im_shape = (params["img_size"], params["img_size"])
fwd_op = tkbn.KbNufft(im_size=im_shape).to(device)
adj_op = tkbn.KbNufftAdjoint(im_size=im_shape).to(device)

losses = []
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
for epoch in range(n_epochs):
    # Training Phase
    dcfnet.train()
    rosette_batch = get_rosette_batch(model, batch_size, device=device)
    rosette_batch = rosette_batch.permute(0, 2, 1).contiguous()
    dcf_pred_batch = dcfnet(rosette_batch)

    grid_density = adj_op(dcf_pred_batch + 0j, rosette_batch, norm="ortho")
    E_complex = fwd_op(grid_density, rosette_batch, norm="ortho")
    loss = torch.mean((E_complex.real - 1) ** 2 + E_complex.imag**2)

    losses.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        t0 = plot_loss(ax, losses, losses, epoch, n_epochs, t0, dcf_pred_batch[-1])


# --- Evaluation on Test Set ---
print("\n--- Final Test Set Evaluation ---")
dcfnet.eval()
test_l1_losses, test_l2_losses = [], []

with torch.no_grad():
    for test_rosette_batch, test_dcf_batch in test_dataloader:
        ktraj = test_rosette_batch.to(device).permute(0, 2, 1).contiguous()
        test_dcf_batch = test_dcf_batch.to(device)

        dcf_pred_batch = dcfnet(ktraj).squeeze(1)

        l1loss = torch.mean((test_dcf_batch - dcf_pred_batch).abs())
        l2loss = torch.mean((test_dcf_batch - dcf_pred_batch) ** 2)

        test_l1_losses.append(l1loss.item())
        test_l2_losses.append(l2loss.item())


print(f"Final Test L1 Loss: {np.mean(test_l1_losses):.6f}")
print(f"Final Test L2 Loss: {np.mean(test_l2_losses):.6f}")

# True vs Pred DCF
plot_final_examples(test_dcf_batch, dcf_pred_batch)

# E Colormap
print("Computing E = AA^HW for the final test batch...")

with torch.no_grad():
    # Calculate raw E
    E_raw = calculate_pipe_E(dcf_pred_batch, ktraj, params["img_size"], device)

    # Normalize E so the average density is exactly 1.0
    E_normalized = E_raw / (E_raw.mean(dim=1, keepdim=True) + 1e-8)

    # Plot the colormap of the rosette trajectory
    plot_E_colormap(E_normalized, ktraj)

with torch.no_grad():
    # Calculate raw E using the TRUE DCF
    E_raw_true = calculate_pipe_E(test_dcf_batch, ktraj, params["img_size"], device)

    # Normalize
    E_normalized_true = E_raw_true / (E_raw_true.mean(dim=1, keepdim=True) + 1e-8)

    # Plot the colormap of the true rosette density
    print("Plotting TRUE DCF Effective Density...")
    plot_E_colormap(E_normalized_true, ktraj)
