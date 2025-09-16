import torch
from models import DCFNet, FourierCurve, Ellipse
from params import *
from utils import make_rosette, get_rotation_matrix
from torchkbnufft import calc_density_compensation_function
import matplotlib.pyplot as plt
from time import time

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)

rotation_matrix = get_rotation_matrix(params["n_petals"], device=device)
t = torch.linspace(0, params["duration"], steps=params["timesteps"]).unsqueeze(1).to(device)  # (timesteps, 1)
ft = 2 * torch.pi / params["duration"] * t[:-1, :]

dcfnet = DCFNet(input_size=2 * (params["timesteps"] - 1), output_size=params["timesteps"] - 1, n_hidden=10, n_features=256).to(device)
optimizer = torch.optim.Adam(dcfnet.parameters(), lr=1e-4)

losses = []
for step in range(1000):
    t0 = time()
    with torch.no_grad():
        traj_batch = []
        dcf_batch = []
        t0 = time()
        for b in range(64):
            # model = FourierCurve(tmin=0, tmax=params["duration"], initial_max=kmax_traj, n_coeffs=params["model_size"])
            a = 1 + 0.1 * torch.randn(1).to(device)
            b = 1 + 0.1 * torch.randn(1).to(device)
            traj = torch.cat([0.5 * a * kmax_traj * (torch.cos(ft) - 1), 0.5 * b * kmax_traj * torch.sin(ft)], dim=-1)
            # plt.plot(traj[:, 0].detach().cpu(), traj[:, 1].detach().cpu(), linewidth=0.7, marker=".", markersize=3)
            # plt.show()
            rosette, _, _ = make_rosette(traj, rotation_matrix, params["n_petals"], kmax_img, dt, zero_filling=params["zero_filling"])
            rosette = rosette.squeeze().permute(1, 0) / kmax_img * torch.pi
            dcf = calc_density_compensation_function(rosette, (params["img_size"], params["img_size"])).abs().squeeze()
            traj = torch.cat([traj[:, 0], traj[:, 1]], dim=0)
            traj_batch.append(traj)
            dcf_batch.append(dcf[:-2])
        print("batch time:", time() - t0)
        traj_batch = torch.stack(traj_batch, dim=0)
        dcf_batch = torch.stack(dcf_batch, dim=0)
    dcf_pred_batch = dcfnet(traj_batch)
    loss = torch.mean((dcf_batch - dcf_pred_batch.repeat((1, params["n_petals"]))).abs())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("Step:", step)
    print("Loss:", loss.detach().cpu().item())
    print("Time", time() - t0)
    losses.append(loss.detach().cpu().item())

torch.save(dcfnet.state_dict(), "dcfnet.pt")


plt.plot(losses)
plt.show()


for i in range(dcf_batch.shape[0]):
    plt.figure()
    plt.plot(dcf_batch[i, : params["timesteps"]].detach().cpu())
    plt.plot(dcf_pred_batch[i, :].detach().cpu())
    plt.show()
