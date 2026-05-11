import sys
import os

sys.path.append(os.getcwd())

import torch
from params import *

# import matplotlib.pyplot as plt
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
    calculate_pipe_E,
    plot_E_colormap,
)
from torch.utils.data import DataLoader
from models import UNet1D, FourierCurve
import os
from datetime import datetime

output_dir = "trained_models/"

n_epochs = 100
batch_size = 64
learning_rate = 5e-4

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps")
print("Device:", device)

# dcfnet = DCFNet(input_size=2 * (params["timesteps"] - 1), output_size=params["timesteps"] - 1, n_hidden=8, n_features=256).to(device)
# dcfnet = FCN1D(channels=[2, 128, 256, 512, 256, 128, 1], kernel_size=21).to(device)
dcfnet = UNet1D(in_channels=2, out_channels=1, features=[16, 32, 64, 128, 256], kernel_size=5).to(device)
# dcfnet = ResNet1D().to(device)

optimizer = torch.optim.AdamW(dcfnet.parameters(), lr=learning_rate)
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

for epoch in range(n_epochs):
    # Training Phase
    dcfnet.train()
    epoch_train_losses = []
    
    for rosette_batch, dcf_batch in train_dataloader:
        loss, dcf_pred_batch = train_step(dcfnet, optimizer, rosette_batch, dcf_batch, device=device)
        epoch_train_losses.append(loss)
    
    avg_train_loss = np.mean(epoch_train_losses)
    train_losses_history.append(avg_train_loss)

    # Validation Phase
    dcfnet.eval()
    epoch_val_losses = []
    
    with torch.no_grad():
        for val_rosette_batch, val_dcf_batch in val_dataloader:
            ktraj = val_rosette_batch.to(device).permute(0, 2, 1).contiguous()
            val_dcf_batch = val_dcf_batch.to(device)
            
            dcf_pred_batch = dcfnet(ktraj).squeeze(1)
            
            val_loss = torch.mean((val_dcf_batch - dcf_pred_batch) ** 2)
            epoch_val_losses.append(val_loss.item())

    avg_val_loss = np.mean(epoch_val_losses)
    val_losses_history.append(avg_val_loss)

    # Plot the losses
    t0 = plot_loss(train_losses_history, val_losses_history, epoch, n_epochs, t0)

    # Save Best Model
    if avg_val_loss < best_val_loss:
        torch.save((dcfnet.state_dict(), dcfnet.kernel_size, dcfnet.features), 
                   os.path.join(output_dir, f"dcfnet_{params['img_size']}_{dcfnet.name}_best.pt"))
        best_val_loss = avg_val_loss

# --- Evaluation on Test Set ---
print("\n--- Final Test Set Evaluation ---")
dcfnet.eval()
test_l1_losses, test_l2_losses = [], []

last_ktraj = None
last_dcf_pred = None

with torch.no_grad():
    for test_rosette_batch, test_dcf_batch in test_dataloader:
        ktraj = test_rosette_batch.to(device).permute(0, 2, 1).contiguous()
        test_dcf_batch = test_dcf_batch.to(device)
        
        dcf_pred_batch = dcfnet(ktraj).squeeze(1)
        
        l1loss = torch.mean((test_dcf_batch - dcf_pred_batch).abs())
        l2loss = torch.mean((test_dcf_batch - dcf_pred_batch) ** 2)
        
        test_l1_losses.append(l1loss.item())
        test_l2_losses.append(l2loss.item())
        
        last_ktraj = ktraj
        last_dcf_pred = dcf_pred_batch

print(f"Final Test L1 Loss: {np.mean(test_l1_losses):.6f}")
print(f"Final Test L2 Loss: {np.mean(test_l2_losses):.6f}")

# True vs Pred DCF
plot_final_examples(test_dcf_batch, last_dcf_pred)

# E Colormap
print("Computing E = AA^HW for the final test batch...")

with torch.no_grad():
    # Calculate raw E
    E_raw = calculate_pipe_E(last_dcf_pred, last_ktraj, params["img_size"], device)
    
    # Normalize E so the average density is exactly 1.0
    E_normalized = E_raw / (E_raw.mean(dim=1, keepdim=True) + 1e-8)
    
    # Plot the colormap of the rosette trajectory
    plot_E_colormap(E_normalized, last_ktraj)

with torch.no_grad():
    # Calculate raw E using the TRUE DCF
    E_raw_true = calculate_pipe_E(test_dcf_batch, last_ktraj, params["img_size"], device)
    
    # Normalize
    E_normalized_true = E_raw_true / (E_raw_true.mean(dim=1, keepdim=True) + 1e-8)
    
    # Plot the colormap of the true rosette density
    print("Plotting TRUE DCF Effective Density...")
    plot_E_colormap(E_normalized_true, last_ktraj)