import torch
import odl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchkbnufft import calc_density_compensation_function
from mirtorch.linear import NuSense
import matplotlib.pyplot as plt
import Nufftbindings.nufftbindings.kbnufft as kbnufft
from kornia.losses import SSIMLoss
import os
from PIL import Image
import numpy as np


def get_phantom(size=(1024, 1024), type="shepp_logan"):
    """Generate a Shepp-Logan phantom as a PyTorch tensor."""
    if type.lower() == "shepp_logan":
        phantom = odl.phantom.shepp_logan(odl.uniform_discr([0, 0], [1, 1], size), modified=True)
        phantom_np = phantom.asarray()
    elif type.lower() == "guitar":
        phantom = Image.open("phantom_images/guitar.jpg").convert("L").resize(size)
        phantom_np = 1.0 - np.array(phantom) / 255.0
    else:
        phantom = Image.open("phantom_images/GLPU/GLPU.png").convert("L").resize(size)
        phantom_np = np.array(phantom) / 255.0
    phantom_tensor = torch.from_numpy(phantom_np).float()
    return phantom_tensor


def rotate_trajectory(traj, angle_radians):
    angle_radians = torch.Tensor([angle_radians])
    rotation_matrix = torch.tensor([[torch.cos(angle_radians), -torch.sin(angle_radians)], [torch.sin(angle_radians), torch.cos(angle_radians)]])
    rotated_traj = traj @ rotation_matrix.T
    return rotated_traj


def make_rosette(traj, n_petals, kmax_img, dt, zero_filling=True):
    rotated_trajectories = [traj]
    angle = 2 * torch.pi / n_petals
    for i in range(n_petals - 1):
        traj_tmp = rotate_trajectory(traj, angle)
        rotated_trajectories.append(traj_tmp)
        traj = traj_tmp
    d_max, dd_max = torch.zeros(1, 2), torch.zeros(1, 2)
    for t in rotated_trajectories[: n_petals // 4 + 1]:
        d, dd = compute_derivatives(t, dt)
        d_max = torch.maximum(d.abs().max(dim=0).values, d_max)
        dd_max = torch.maximum(dd.abs().max(dim=0).values, d_max)
    if zero_filling:
        corners = torch.ones(2, 2)
        corners[0] *= kmax_img
        corners[1] *= -kmax_img
        rotated_trajectories.append(corners)
    rosette = torch.cat(rotated_trajectories, dim=0)
    return rosette, d_max, dd_max


def sample_k_space_values(fft, rosette, kmax_img, zero_filling):
    rosette = rosette.reshape(1, 1, rosette.shape[0], 2)
    sampled_r = F.grid_sample(fft.real, rosette / kmax_img, mode="bicubic", align_corners=True)
    sampled_i = F.grid_sample(fft.imag, rosette / kmax_img, mode="bicubic", align_corners=True)
    sampled = torch.complex(sampled_r, sampled_i).squeeze(0)
    if zero_filling:
        sampled[:, :, -2:] *= 0  # zero filling
    return rosette, sampled


def reconstruct_img(rosette, sampled, img_size, scaling):
    s0 = torch.ones(1, 1, img_size, img_size) + 0j
    rosette0 = rosette.squeeze().permute(1, 0) / torch.max(torch.abs(rosette)) * torch.pi
    k0 = sampled.reshape(1, 1, -1)
    dcf = calc_density_compensation_function(rosette0, (img_size, img_size))
    Nop = NuSense(s0, rosette0, norm=None)
    I0 = Nop.H * (dcf * k0)
    I0 = torch.flip(torch.rot90(I0.abs(), k=1, dims=(2, 3)), dims=[2]).squeeze()
    return I0


def reconstruct_img2(rosette, sampled, img_size, scaling):
    rosette = rosette.squeeze().permute(1, 0) / torch.max(torch.abs(rosette)) * torch.pi
    k0 = sampled.reshape(1, 1, -1)
    dcf = calc_density_compensation_function(rosette[:, :-2], (img_size, img_size))
    dcf = torch.cat([dcf, torch.zeros(1, 1, 2)], dim=-1)
    rosette = rosette.permute(1, 0)
    kbnufft.nufft.set_dims(sampled.shape[-1], (img_size, img_size), torch.device("cpu"), Nb=1)
    kbnufft.nufft.precompute(rosette)
    I0 = kbnufft.adjoint(rosette, (k0 * dcf).squeeze(0)).unsqueeze(0)
    I0 = torch.flip(torch.rot90(I0, k=1, dims=(2, 3)), dims=[2]).squeeze()
    I0 = I0 * scaling
    return I0.abs()


def compute_derivatives(traj, dt):
    """Compute the first and second derivatives of a trajectory.
    traj: (timesteps, 2) tensor
    dt: time step size (scalar)
    Returns: d_traj, dd_traj
    """
    # traj = traj[:-1, :]
    d_traj = (torch.roll(traj, shifts=-1, dims=0) - traj) / dt
    dd_traj = (torch.roll(d_traj, shifts=-1, dims=0) - d_traj) / dt
    return d_traj, dd_traj


def threshold_loss(x, threshold):
    threshold_loss = torch.max(torch.abs(x), dim=0).values - threshold
    threshold_loss[threshold_loss < 0] = 0.0
    return threshold_loss**2


def grad_slew_loss(
    d_max_rosette,
    dd_max_rosette,
    grad_max,
    slew_rate_max,
    gamma,
    grad_loss_weight,
    slew_loss_weight,
):
    """Compute a loss based on the slew rate of the trajectory.
    traj: (timesteps, 2) tensor
    dt: time step size (scalar)
    Returns: slew_loss (scalar)
    """
    grad_loss = torch.exp(d_max_rosette - grad_max * gamma / 1000)
    slew_loss = torch.exp(dd_max_rosette - slew_rate_max * gamma / 1000)
    # grad_loss = threshold_loss(grad, grad_max * gamma / 1000)
    # slew_loss = threshold_loss(slew, slew_rate * gamma / 1000)
    return grad_loss_weight * grad_loss.sum(), slew_loss_weight * slew_loss.sum()


def mse_loss(img, target):
    return torch.mean((img - target) ** 2)


def l1_loss(img, target):
    return torch.mean((img - target).abs())


class MySSIMLoss(nn.Module):
    def __init__(self, window_size=11, reduction="mean", max_val=1.0):
        super(MySSIMLoss, self).__init__()
        self.ssim = SSIMLoss(window_size=window_size, reduction=reduction, max_val=max_val)
        self.L1_loss = nn.L1Loss()

    def forward(self, img, target):
        return 1.0 - self.ssim(img.unsqueeze(0).unsqueeze(0), target.unsqueeze(0).unsqueeze(0)) + 0.1 * self.L1_loss(img, target)


def get_loss_fcn(name):
    if name.upper() == "L1":
        return l1_loss
    if name.upper() == "SSIM":
        return MySSIMLoss(window_size=11, reduction="mean", max_val=1.0)
    if name.lower() == "combined":
        return lambda x: MySSIMLoss(window_size=11, reduction="mean", max_val=1.0)(x) + l1_loss(x)
    return mse_loss


class TrainPlotter:
    def __init__(self, img_size):
        plt.rc("xtick", labelsize=8)
        plt.rc("ytick", labelsize=8)
        self.fig, (ax_loss, ax_img, ax_traj) = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
        ax_total_loss = ax_loss.twinx()
        (grad_loss_line,) = ax_loss.semilogy([], [], label="Grad Loss", linewidth=0.7, color="r")
        (slew_loss_line,) = ax_loss.semilogy([], [], label="Slew Loss", linewidth=0.7, color="g")
        (img_loss_line,) = ax_loss.semilogy([], [], label="Image Loss", linewidth=0.7, color="b")
        (total_loss_line,) = ax_total_loss.semilogy([], [], label="Total Loss", linewidth=0.7, color="k")
        ax_loss.set_xlabel("Step")
        ax_loss.set_ylabel("Individual Losses")
        ax_loss.set_title("Running Loss")
        ax_total_loss.set_ylabel("Total Loss")
        lns = [grad_loss_line, slew_loss_line, img_loss_line, total_loss_line]
        labs = [l.get_label() for l in lns]
        ax_loss.legend(lns, labs, loc=0, prop={"size": 6})
        im_recon = ax_img.imshow(torch.zeros((img_size, img_size)).numpy(), cmap="gray")
        ax_img.set_title("Recon (abs)")
        ax_img.axis("off")
        (traj_line,) = ax_traj.plot([], [], label="trajectory", linewidth=0.7, marker=".", markersize=3)
        ax_traj.set_title("Trajectory")
        self.cbar = self.fig.colorbar(im_recon, ax=ax_img)
        plt.show(block=False)
        self.grad_loss_line = grad_loss_line
        self.slew_loss_line = slew_loss_line
        self.img_loss_line = img_loss_line
        self.total_loss_line = total_loss_line
        self.im_recon = im_recon
        self.traj_line = traj_line
        self.ax_loss = ax_loss
        self.ax_total_loss = ax_total_loss
        self.ax_traj = ax_traj
        self.ax_img = ax_img
        self.grad_losses = []
        self.slew_losses = []
        self.img_losses = []
        self.total_losses = []

    def update(self, step, grad_loss, img_loss, slew_loss, total_loss, recon, traj):
        self.grad_losses.append(grad_loss)
        self.img_losses.append(img_loss)
        self.slew_losses.append(slew_loss)
        self.total_losses.append(total_loss)
        if step % 20 == 0:
            self.img_loss_line.set_data(range(len(self.img_losses)), self.img_losses)
            self.grad_loss_line.set_data(range(len(self.grad_losses)), self.grad_losses)
            self.slew_loss_line.set_data(range(len(self.slew_losses)), self.slew_losses)
            self.total_loss_line.set_data(range(len(self.img_losses)), self.total_losses)
            self.ax_total_loss.relim()
            self.ax_total_loss.autoscale_view()
            self.ax_loss.set_ylim(0.9 * min(self.img_losses), 1.1 * max(self.img_losses))
            img = recon.abs().detach().cpu().numpy()
            self.im_recon.set_data(img)
            self.im_recon.set_clim(vmin=0, vmax=img.max())
            self.traj_line.set_data(traj[:, 0].detach().numpy(), traj[:, 1].detach().numpy())
            self.ax_traj.relim()
            self.ax_traj.autoscale_view()
            self.ax_traj.set_aspect("equal", "box")
            self.ax_img.set_title(f"Recon (abs) Step {step+1}")
            plt.pause(0.01)

    def print_info(self, step, train_steps, image_loss, grad_loss, slew_loss, best_loss):
        print(f"Step {step+1}/{train_steps}")
        print(f"  Image loss: {image_loss:.6f}")
        print(f"  Gradient loss: {grad_loss:.6f}")
        print(f"  Slew rate loss: {slew_loss:.6f}")
        print(f"  Total loss: {image_loss+grad_loss+slew_loss:.6f}")
        print(f"  Best loss: {best_loss:.6f}")
        print("-" * 100)

    def export_figure(self, path):
        self.fig.savefig(os.path.join(path, "train_figure.png"))


def save_checkpoint(path, model, d_max_rosette, dd_max_rosette, params):
    os.makedirs(path, exist_ok=True)
    grad = 1000 / params["gamma"] * d_max_rosette
    slew = 1000 / params["gamma"] * dd_max_rosette
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "params": params,
            "slew_rate": slew,
            "gradient": grad,
        },
        os.path.join(path, "checkpoint.pt"),
    )
    print("=" * 100)
    print("CHECKPOINT SAVED")
    print("Slew Rate:", slew)
    print("=" * 100)
    return slew


def plot_pixel_rosette(rosette, fft, img_size, ax=None):
    """Sample FFT at trajectory locations"""
    sampled_coord_x = torch.round((rosette[0, 0, :, 0] + 1) * (img_size - 1) / 2)
    sampled_coord_y = torch.round((rosette[0, 0, :, 1] + 1) * (img_size - 1) / 2)
    pixel_curve = torch.zeros(img_size, img_size)
    pixel_curve[sampled_coord_y.long(), sampled_coord_x.long()] = 1.0
    sampled_from_pixels = fft[0, 0, sampled_coord_y.long(), sampled_coord_x.long()]
    sampled_from_pixels[-2:] *= 0
    if ax is not None:
        ax.imshow(pixel_curve[img_size // 4 : 3 * img_size // 4, img_size // 4 : 3 * img_size // 4].detach().numpy())
    else:
        plt.figure()
        plt.imshow(pixel_curve.detach().numpy())
        plt.show()
    return sampled_from_pixels


def final_plots(phantom, recon, initial_recon, losses, traj, slew_rate, show=True, export=False, export_path=None):
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    im1 = ax[0, 0].imshow(phantom.numpy(), cmap="gray")
    ax[0, 0].set_title("Phantom")
    ax[0, 0].axis("off")
    fig.colorbar(im1, ax=ax[0, 0])

    im2 = ax[0, 1].imshow(recon.abs().detach().numpy(), cmap="gray")
    ax[0, 1].set_title("Recon")
    ax[0, 1].axis("off")
    fig.colorbar(im2, ax=ax[0, 1])

    im3 = ax[0, 2].imshow(initial_recon.squeeze().detach().numpy(), cmap="gray")
    ax[0, 2].set_title("Initial Recon")
    ax[0, 2].axis("off")
    fig.colorbar(im3, ax=ax[0, 2])

    im4 = ax[1, 0].imshow((recon.abs() - phantom).detach().numpy(), cmap="gray")
    ax[1, 0].set_title("Phantom - Recon")
    ax[1, 0].axis("off")
    fig.colorbar(im4, ax=ax[1, 0])

    im5 = ax[1, 1].semilogy(range(len(losses)), losses, linewidth=0.7)
    ax[1, 1].set_title("Loss")

    im6 = ax[1, 2].plot(traj[:, 0].detach().numpy(), traj[:, 1].detach().numpy(), linewidth=0.7, marker=".", markersize=3)
    ax[1, 2].set_title(f"Trajectory. Slew Rate: {slew_rate.abs().max().item():.2f}")
    if show:
        plt.show()
    if export and export_path is not None:
        plt.savefig(os.path.join(export_path, "final_figure.png"), dpi=300)
        plt.close()
    return im1, im2, im3, im4, im5, im6
