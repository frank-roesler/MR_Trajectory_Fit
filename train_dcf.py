import torch
from models import DCFNet, FourierCurve, Ellipse, FCN1D, UNet1D
from params import *
from utils_3 import make_rosette, get_rotation_matrix, LossCollection
from torchkbnufft import calc_density_compensation_function
import matplotlib.pyplot as plt
from time import time
from glob import glob
import numpy as np

compute_dcfs = False
train_data = glob("train_data/*")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps")
print("Device:", device)

t = torch.linspace(0, params["duration"], steps=params["timesteps"]).unsqueeze(1).to(device)  # (timesteps, 1)
ft = 2 * torch.pi / params["duration"] * t[:-1, :]

# dcfnet = DCFNet(input_size=2 * (params["timesteps"] - 1), output_size=params["timesteps"] - 1, n_hidden=8, n_features=256).to(device)
# dcfnet = FCN1D(channels=[2, 128, 256, 512, 256, 128, 1], kernel_size=21).to(device)
dcfnet = UNet1D(in_channels=2, out_channels=1, features=[16, 32, 64, 128, 256]).to(device)

optimizer = torch.optim.Adam(dcfnet.parameters(), lr=5e-4)
loss_fcns = LossCollection(params["loss_function"])

losses = []
best_loss = float("inf")
for step in range(2000):
    if step >= len(train_data):
        compute_dcfs = True
    t0 = time()
    if compute_dcfs:
        with torch.no_grad():
            rosette_batch = []
            dcf_batch = []
            while len(rosette_batch) < 64:
                model = FourierCurve(tmin=0, tmax=params["duration"], initial_max=kmax_traj, n_coeffs=params["model_size"], coeff_lvl=0.5, angle_lvl=0.0).to(device)
                # a = 1 + 0.2 * torch.randn(1)
                # b = 1 + 0.2 * torch.randn(1)
                # traj = torch.cat([0.5 * a * kmax_traj * (torch.cos(ft) - 1), 0.5 * b * kmax_traj * torch.sin(ft)], dim=-1)
                traj, angles = model(t)
                rosette, *derivatives = make_rosette(angles, traj, params["n_petals"], kmax_img, dt, zero_filling=False)
                slew = 1000 / params["gamma"] * derivatives[1]
                # print(slew)
                # grad_loss, slew_loss = loss_fcns.grad_slew_loss(*derivatives, params, grad_mode="threshold", slew_mode="threshold")
                if torch.max(slew) > 300:
                    continue
                # plt.plot(traj[:, 0].detach().cpu(), traj[:, 1].detach().cpu(), linewidth=0.7, marker=".", markersize=3)
                # plt.show()
                rosette = rosette / kmax_img * torch.pi
                dcf = calc_density_compensation_function(rosette.T.cpu(), (params["img_size"], params["img_size"])).abs().squeeze()
                # rosette = torch.cat([rosette[:, 0], rosette[:, 1]], dim=0)
                rosette_batch.append(rosette[: params["timesteps"] - 1].detach())
                # rosette_batch.append(rosette[:-2, :])
                dcf_batch.append(dcf[: params["timesteps"] - 1].detach())
            rosette_batch = torch.stack(rosette_batch, dim=0)
            dcf_batch = torch.stack(dcf_batch, dim=0)
            torch.save((rosette_batch, dcf_batch), f"train_data/{torch.randint(0,1000000,(1,)).item()}.pt")
    else:
        rosette_batch, dcf_batch = torch.load(train_data[step])
    rosette_batch = rosette_batch.to(device).permute(0, 2, 1)
    dcf_batch = dcf_batch.to(device)
    dcf_pred_batch = dcfnet(rosette_batch).squeeze(1)  # Remove only channel dim, keep batch dim
    # loss = torch.mean((dcf_batch - dcf_pred_batch.repeat((1, params["n_petals"]))).abs())
    loss = torch.mean((dcf_batch - dcf_pred_batch).abs())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if loss < 0.99 * best_loss:
        torch.save(dcfnet.state_dict(), f"trained_models/dcfnet_general_{dcfnet.name}.pt")
        best_loss = loss

    losses.append(loss.detach().cpu().item())
    if step % 10 == 0:
        plt.cla()
        plt.semilogy(losses)
        plt.show(block=False)
        plt.pause(0.001)
        print("Step:", step)
        print("Loss:", loss.detach().cpu().item())
        print("Time", time() - t0)

print("FINAL LOSS", np.mean(losses[-100:]).item())

plt.cla()
plt.semilogy(losses)
plt.show()


for i in range(dcf_batch.shape[0]):
    plt.figure()
    plt.plot(dcf_batch[i, :].detach().cpu(), label="True DCF")
    plt.plot(dcf_pred_batch[i, :].detach().cpu(), label="Predicted DCF")
    plt.legend()
    plt.show()


# 1e-3: 0.00876
# 5e-4:
