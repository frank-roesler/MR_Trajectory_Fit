import torch
from models import DCFNet, FourierCurve, Ellipse, FCN1D, UNet1D
from params import *
from utils import make_rosette, get_rotation_matrix
from torchkbnufft import calc_density_compensation_function
import matplotlib.pyplot as plt
from time import time
from glob import glob
import numpy as np
import os

total_steps = 1000
batch_size = 64
compute_dcfs = False
train_data_folder = "train_data"
train_data = glob(train_data_folder + "/*")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps")
print("Device:", device)

rotation_matrix = get_rotation_matrix(params["n_petals"], device=device).detach()
t = torch.linspace(0, params["duration"], steps=params["timesteps"]).unsqueeze(1).to(device)  # (timesteps, 1)
ft = 2 * torch.pi / params["duration"] * t[:-1, :]

# dcfnet = DCFNet(input_size=2 * (params["timesteps"] - 1), output_size=params["timesteps"] - 1, n_hidden=8, n_features=256).to(device)
# dcfnet = FCN1D(channels=[2, 128, 256, 512, 256, 128, 1], kernel_size=21).to(device)
dcfnet = UNet1D(in_channels=2, out_channels=1, features=[16, 32, 64, 128, 256]).to(device)

optimizer = torch.optim.Adam(dcfnet.parameters(), lr=1e-3)

losses = []
best_loss = float("inf")
for step in range(1, total_steps + 1):
    if step >= len(train_data):
        compute_dcfs = True
    t0 = time()
    if compute_dcfs:
        with torch.no_grad():
            traj_batch = []
            dcf_batch = []
            for b in range(batch_size):
                print(f"Computing DCFs for batch {b + 1} out of {batch_size}")
                model = FourierCurve(tmin=0, tmax=params["duration"], initial_max=kmax_traj, n_coeffs=params["model_size"], coeff_lvl=1e-1).to(device)
                traj = model(t)
                # ------------------------------------------------------
                # For training pure Ellipse model:
                # a = 1 + 0.2 * torch.randn(1)
                # b = 1 + 0.2 * torch.randn(1)
                # traj = torch.cat([0.5 * a * kmax_traj * (torch.cos(ft) - 1), 0.5 * b * kmax_traj * torch.sin(ft)], dim=-1)
                # plt.plot(traj[:, 0], traj[:, 1], linewidth=0.7, marker=".", markersize=3)
                # plt.show()
                # -------------------------------------------------------
                rosette, _, _ = make_rosette(traj, rotation_matrix, params["n_petals"], kmax_img, dt, zero_filling=params["zero_filling"])
                rosette_dcf = rosette.squeeze().permute(1, 0) / kmax_img * torch.pi
                dcf = calc_density_compensation_function(rosette_dcf, (params["img_size"], params["img_size"])).abs().squeeze()
                traj = torch.cat([traj[:, 0], traj[:, 1]], dim=0)
                traj_batch.append(traj / kmax_img * torch.pi)
                dcf_batch.append(dcf[:-2])
            traj_batch = torch.stack(traj_batch, dim=0)
            dcf_batch = torch.stack(dcf_batch, dim=0)
            os.makedirs(train_data_folder, exist_ok=True)
            torch.save((traj_batch, dcf_batch), f"{train_data_folder}/{torch.randint(0,int(1e+7),(1,)).item()}.pt")
    else:
        traj_batch, dcf_batch = torch.load(train_data[step])
    traj_batch = traj_batch.to(device).reshape(batch_size, 2, 100)
    dcf_batch = dcf_batch.to(device)
    dcf_pred_batch = dcfnet(traj_batch).squeeze()
    loss = torch.mean((dcf_batch - dcf_pred_batch.repeat((1, params["n_petals"]))).abs())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if loss < 0.99 * best_loss:
        torch.save(dcfnet.state_dict(), f"trained_models/dcfnet_{dcfnet.name}.pt")
        best_loss = loss

    losses.append(loss.detach().cpu().item())
    if step % 10 == 0:
        plt.cla()
        plt.semilogy(losses, linewidth=0.7)
        plt.show(block=False)
        plt.pause(0.001)
        print("Step:", step)
        print("Loss:", loss.detach().cpu().item())
        print("Time", time() - t0)

print("FINAL LOSS", np.mean(losses[-100:]).item())

plt.cla()
plt.semilogy(losses)

for i in range(dcf_batch.shape[0]):
    fig, ax = plt.subplots(2, 1, figsize=(7, 8))
    ax[0].plot(traj_batch[i, 0, :].detach().cpu(), traj_batch[i, 1, :].detach().cpu(), linewidth=0.7, marker=".", markersize=3)
    ax[1].plot(dcf_batch[i, : params["timesteps"]].detach().cpu(), label="DCF (kbnufft)", linewidth=0.7)
    ax[1].plot(dcf_pred_batch[i, :].detach().cpu(), label="DCF (predicted)", linewidth=0.7)
    ax[1].legend()
    plt.show()
