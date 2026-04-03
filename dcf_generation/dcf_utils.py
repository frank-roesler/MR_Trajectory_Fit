from torch.utils.data import Dataset, DataLoader
from glob import glob
import torch
from models import DCFNet, FourierCurve, Ellipse, FCN1D, UNet1D
from params import *
from utils_3 import make_rosette, get_rotation_matrix
from torchkbnufft import calc_density_compensation_function
import matplotlib.pyplot as plt
from time import time


class TrajectoryDCFDataset(Dataset):
    def __init__(self, data_dir):
        self.files = sorted(glob(f"{data_dir}/*.pt"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        petal, dcf = torch.load(self.files[idx])
        return petal, dcf


def get_rosette_batch(batch_size, device=torch.device("cpu")):
    rosette_batch = []
    t = torch.linspace(0, params["duration"], steps=params["timesteps"]).unsqueeze(1).to(device)  # (timesteps, 1)
    n_trash = 0
    while len(rosette_batch) < batch_size:
        model = FourierCurve(
            tmin=0,
            tmax=params["duration"],
            initial_max=kmax_traj,
            n_coeffs=params["model_size"],
            coeff_lvl=0.5,
            angle_lvl=0.0,
        ).to(device)
        petal, angles = model(t)
        rosette, *derivatives = make_rosette(angles, petal, params["n_petals"], kmax_img, dt, zero_filling=False)
        slew = 1000 / params["gamma"] * derivatives[1]

        if torch.max(slew) > 300:
            n_trash += 1
            continue
        # print(slew.max().item())
        # plot_example(petal)

        rosette_batch.append(rosette / kmax_img * torch.pi)
    print("Thrown away:", n_trash)
    rosette_batch = torch.stack(rosette_batch, dim=0)
    return rosette_batch


def get_dcf_batch(rosette_batch, device=torch.device("cpu")):
    rosette_batch = rosette_batch.cpu() if device == "mps" else rosette_batch
    dcf_batch = calc_density_compensation_function(rosette_batch.permute(0, 2, 1), (params["img_size"], params["img_size"])).abs().squeeze()
    return dcf_batch


def get_dcf_petal_batch_from_dcf(dcf_batch):
    return dcf_batch[:, : params["timesteps"] - 1]


def get_petal_batch_from_rosette(rosette_batch):
    return rosette_batch[:, : params["timesteps"] - 1, :]


def plot_example(petal):
    plt.plot(petal[:, 0].detach().cpu(), petal[:, 1].detach().cpu(), linewidth=0.7, marker=".", markersize=3)
    plt.show()


def plot_loss(losses, step, t0, block=False):
    plt.cla()
    plt.semilogy(losses)
    plt.show(block=block)
    plt.pause(0.001)
    plt.savefig(f"dcf_generation/loss_{step}.png")
    print("Step:", step)
    print("Loss:", losses[-1])
    print("Time", time() - t0)
    t0 = time()


def plot_final_examples(dcf_batch, dcf_pred_batch):
    n_plots = min(8, dcf_batch.shape[0])
    fig, ax = plt.subplots(2, 4, figsize=(13, 6))
    ax_flat = ax.ravel()
    for i in range(n_plots):
        ax_flat[i].plot(dcf_batch[i, :].detach().cpu(), label="True DCF")
        ax_flat[i].plot(dcf_pred_batch[i, :].detach().cpu(), label="Predicted DCF")
        ax_flat[i].legend()
    plt.show()


def train_step(dcfnet, optimizer, petal_batch, dcf_batch, device=torch.device("cpu")):
    petal_batch = petal_batch.to(device).permute(0, 2, 1)
    dcf_batch = dcf_batch.to(device)
    dcf_pred_batch = dcfnet(petal_batch).squeeze(1)  # Remove only channel dim, keep batch dim
    # loss = torch.mean((dcf_batch - dcf_pred_batch.repeat((1, params["n_petals"]))).abs())
    loss = torch.mean((dcf_batch - dcf_pred_batch).abs())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item(), dcf_pred_batch
