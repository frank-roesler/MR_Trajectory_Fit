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
from mirtorch_pkg import NuSense_om, NuSense
from models import DCFNet, UNet1D, FCN1D
import json
from os.path import join, dirname


class ImageRecon:
    def __init__(self, params, kmax_img, normalization, dcfnet="unet"):
        """medhod must be one of: ['kbnufft', 'mirtorch', 'nudft'].
        'kbnufft': uses dcfnet for fast differentiable dcf computation,
        'mirtorch': uses calc_density_compensation_function from torchkbnufft. Not differentiable.
        'nudft': very slow and should only be used for debugging."""
        self.kmax_img = kmax_img
        self.img_size = params["img_size"]
        self.normalization = normalization
        self.timesteps = params["timesteps"]
        self.zero_filling = params["zero_filling"]
        self.n_petals = params["n_petals"]
        if dcfnet == "unet":
            self.dcfnet = UNet1D(in_channels=2, out_channels=1, features=[16, 32, 64, 128, 256])
        else:
            self.dcfnet = FCN1D(channels=[2, 128, 256, 512, 256, 128, 1], kernel_size=21)
        dcfdict = torch.load(f"trained_models/dcfnet_{self.dcfnet.name}.pt")
        self.dcfnet.load_state_dict(dcfdict)

    def sample_k_space_values(self, fft, rosette):
        rosette = rosette.reshape(1, 1, rosette.shape[0], 2)
        sampled_r = F.grid_sample(fft.real.detach(), rosette / self.kmax_img, mode="bicubic", align_corners=True)
        sampled_i = F.grid_sample(fft.imag.detach(), rosette / self.kmax_img, mode="bicubic", align_corners=True)
        sampled = torch.complex(sampled_r, sampled_i).squeeze(0)
        if self.zero_filling:
            sampled[:, :, -2:] *= 0
        return rosette, sampled

    def reconstruct_img(self, fft, rosette, method="kbnufft"):
        rosette, sampled = self.sample_k_space_values(fft, rosette)
        if method == "mirtorch":
            return self.reconstruct_img_mirtorch(rosette, sampled)
        if method == "kbnufft":
            return self.reconstruct_img_kbnufft(rosette, sampled)
        return self.reconstruct_img_nudft(rosette, sampled)

    def reconstruct_img_mirtorch(self, rosette, sampled):
        s0 = torch.ones(1, 1, self.img_size, self.img_size) + 0j
        rosette = rosette.squeeze().permute(1, 0) / self.kmax_img * torch.pi
        sampled = sampled.reshape(1, 1, -1)
        dcf = calc_density_compensation_function(rosette, (self.img_size, self.img_size)) + 0j
        Nop = NuSense_om(s0, rosette.reshape(1, 2, -1), norm=None)
        I0 = Nop.H * (dcf * sampled)
        I0 = torch.flip(torch.rot90(I0.abs(), k=1, dims=(2, 3)), dims=[2]).squeeze()
        return I0 * self.normalization

    def reconstruct_img_kbnufft(self, rosette, sampled):
        rosette = rosette.squeeze().permute(1, 0) / self.kmax_img * torch.pi
        sampled = sampled.reshape(1, 1, -1)
        dcf = self.dcfnet(rosette[:, : self.timesteps - 1].unsqueeze(0)).squeeze() + 0j
        dcf = torch.cat([dcf.repeat((1, self.n_petals)), torch.zeros(1, 2)], dim=-1).unsqueeze(0)
        rosette = rosette.permute(1, 0)
        kbnufft.nufft.set_dims(sampled.shape[-1], (self.img_size, self.img_size), device=rosette.device, Nb=1)
        kbnufft.nufft.precompute(rosette)
        I0 = kbnufft.adjoint(rosette, (sampled * dcf).squeeze(0)).unsqueeze(0)
        I0 = torch.flip(torch.rot90(I0.abs(), k=1, dims=(2, 3)), dims=[2]).squeeze()
        I0 = I0 * self.normalization
        return I0

    def reconstruct_img_nudft(self, rosette, sampled):
        rosette = rosette.squeeze().permute(1, 0)
        sampled = sampled.reshape(-1)
        dcf = self.pipe_density_compensation(rosette, self.timesteps - 1).reshape(sampled.shape)
        sampled = sampled * dcf.squeeze()
        coords = torch.linspace(-1, 1, self.img_size, device=rosette.device) * 112
        grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
        img_coords = torch.stack([grid_x, grid_y], dim=-1)
        img_coords_flat = img_coords.reshape(-1, 2)
        exponent = img_coords_flat @ rosette
        exponent = 2 * torch.pi * 1j * exponent
        nudft = torch.exp(exponent) @ sampled
        img = nudft.view(self.img_size, self.img_size).abs() * self.normalization
        return img

    def pipe_density_compensation(self, rosette, num_iters=10):
        rosette = rosette[:, :-2]
        traj = rosette[..., : self.timesteps]
        n_petals = rosette.shape[-1] // self.timesteps
        N = traj.shape[-1]
        w = torch.ones(n_petals * N)
        for _ in range(num_iters):
            psf = torch.zeros(N)
            for i in range(N):
                dist_sq = torch.sum((rosette - rosette[:, i : i + 1]) ** 2, axis=0)
                psf[i] = torch.sum(w / (dist_sq + 1e-6))
            w = w / psf.repeat(n_petals)
        w = 50 * N * n_petals * torch.cat([w, torch.zeros((2))], dim=0)
        return w


class MySSIMLoss(nn.Module):
    def __init__(self, window_size=11, reduction="mean", max_val=1.0):
        super(MySSIMLoss, self).__init__()
        self.ssim = SSIMLoss(window_size=window_size, reduction=reduction, max_val=max_val)
        self.L1_loss = nn.L1Loss()

    def forward(self, img, target):
        loss = 1.0 - self.ssim(img.unsqueeze(0).unsqueeze(0), target.unsqueeze(0).unsqueeze(0)) + 0.1 * self.L1_loss(img, target)
        return loss


class LossCollection:
    def __init__(self, template="L2"):
        self.use_loss(template)

    def set_default_loss(self):
        if self.template.upper() == "L2":
            self.loss_fn = self.mse_loss
        if self.template.upper() == "L1":
            self.loss_fn = self.l1_loss
        if self.template.upper() == "SSIM":
            self.loss_fn = MySSIMLoss(window_size=11, reduction="mean", max_val=1.0)
        if self.template.lower() == "combined":
            ssim_loss = MySSIMLoss(window_size=11, reduction="mean", max_val=1.0)
            self.loss_fn = lambda x, y: 0.1 * ssim_loss(x, y) + self.l1_loss(x, y)

    def use_loss(self, template):
        self.template = template
        self.set_default_loss()

    def mse_loss(self, img, target):
        return torch.mean((img - target) ** 2)

    def l1_loss(self, img, target):
        loss = torch.mean((img - target).abs())
        return loss

    def threshold_loss(self, x, threshold):
        threshold_loss = torch.max(torch.abs(x), dim=0).values - threshold
        threshold_loss[threshold_loss < 0] = 0.0
        return threshold_loss**2

    def grad_slew_loss(self, d_max_rosette, dd_max_rosette, params, mode="exp"):
        """Compute a loss self,based on the slew rate of the trajectory.
        traj: (timesteps, 2) tensor
        dt: time step size (scalar)
        Returns: slew_loss (scalar)
        """
        unit = params["gamma"] / 1000
        if mode == "exp":
            grad_loss = torch.exp(0.1 * (d_max_rosette - params["grad_max"] * unit))
            slew_loss = torch.exp(0.1 * (dd_max_rosette - params["slew_rate"] * unit))
        elif mode == "threshold":
            grad_loss = self.threshold_loss(d_max_rosette, params["grad_max"] * unit)
            slew_loss = self.threshold_loss(dd_max_rosette, params["slew_rate"] * unit)
        else:
            grad_loss = torch.zeros(1)
            slew_loss = torch.zeros(1)
        return params["grad_loss_weight"] * grad_loss.sum(), params["slew_loss_weight"] * slew_loss.sum()


class TrainPlotter:
    def __init__(self, params, fft, reconstructor, phantom, loss_fn, optimizer):
        self.best_loss = float("inf")
        self.train_steps = params["train_steps"]
        self.gamma = params["gamma"]
        self.fft = fft
        self.reconstructor = reconstructor
        self.phantom = phantom
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        plt.rc("xtick", labelsize=8)
        plt.rc("ytick", labelsize=8)
        self.fig, (ax_loss, ax_img, ax_traj) = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
        ax_total_loss = ax_loss.twinx()
        (grad_loss_line,) = ax_loss.semilogy([], [], label="Grad Loss", linewidth=0.7, color="r")
        (slew_loss_line,) = ax_loss.semilogy([], [], label="Slew Loss", linewidth=0.7, color="g")
        (img_loss_line,) = ax_loss.semilogy([], [], label="Image Loss", linewidth=0.7, color="b")
        (img_loss_line_mirtorch,) = ax_loss.semilogy([], [], label="Image Loss MIRTorch", linewidth=0.7, color="orange")
        (total_loss_line,) = ax_total_loss.semilogy([], [], label="Total Loss", linewidth=0.7, color="k")
        ax_loss.set_xlabel("Step")
        ax_loss.set_ylabel("Individual Losses")
        ax_loss.set_title("Running Loss")
        ax_total_loss.set_ylabel("Total Loss")
        lns = [grad_loss_line, slew_loss_line, img_loss_line, total_loss_line, img_loss_line_mirtorch]
        labs = [l.get_label() for l in lns]
        ax_loss.legend(lns, labs, loc=0, prop={"size": 6})
        im_recon = ax_img.imshow(torch.zeros((params["img_size"], params["img_size"])).numpy(), cmap="gray")
        ax_img.set_title("Recon (abs)")
        ax_img.axis("off")
        (traj_line,) = ax_traj.plot([], [], label="trajectory", linewidth=0.7, marker=".", markersize=3)
        ax_traj.set_title("Trajectory")
        self.cbar = self.fig.colorbar(im_recon, ax=ax_img)
        plt.show(block=False)
        self.grad_loss_line = grad_loss_line
        self.slew_loss_line = slew_loss_line
        self.img_loss_line = img_loss_line
        self.img_loss_line_mirtorch = img_loss_line_mirtorch
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
        self.img_losses_mirtorch = []
        self.total_losses = []

    def update(self, step, grad_loss, img_loss, slew_loss, total_loss, recon, traj, rosette):
        self.grad_losses.append(grad_loss.detach().item())
        self.img_losses.append(img_loss.detach().item())
        self.slew_losses.append(slew_loss.detach().item())
        self.total_losses.append(total_loss.detach().item())
        if step % 10 == 0:
            recon_mirtorch = self.reconstructor.reconstruct_img(self.fft, rosette, method="mirtorch")
            image_loss_mirtorch = self.loss_fn(recon_mirtorch, self.phantom)
            self.img_losses_mirtorch.append(image_loss_mirtorch.detach().item())
            self.img_loss_line_mirtorch.set_data(range(0, 10 * len(self.img_losses_mirtorch), 10), self.img_losses_mirtorch)
            self.img_loss_line.set_data(range(len(self.img_losses)), self.img_losses)
            self.grad_loss_line.set_data(range(len(self.grad_losses)), self.grad_losses)
            self.slew_loss_line.set_data(range(len(self.slew_losses)), self.slew_losses)
            self.total_loss_line.set_data(range(len(self.img_losses)), self.total_losses)
            self.ax_total_loss.relim()
            self.ax_total_loss.autoscale_view()
            self.ax_loss.set_ylim(0.9 * min(self.img_losses), 1.1 * max(self.img_losses))
            img = recon.abs().detach().cpu().cpu().numpy()
            self.im_recon.set_data(img)
            self.im_recon.set_clim(vmin=0, vmax=img.max())
            self.traj_line.set_data(traj[:, 0].detach().cpu().numpy(), traj[:, 1].detach().cpu().numpy())
            self.ax_traj.relim()
            self.ax_traj.autoscale_view()
            self.ax_img.set_title(f"Recon (abs) Step {step+1}")
            plt.pause(0.01)

    def print_info(self, step, image_loss, grad_loss, slew_loss, d_max, dd_max):
        if step % 10 == 0:
            print(f"Step {step+1}/{self.train_steps}")
            for i, param_group in enumerate(self.optimizer.param_groups):
                print(f"  Learning rate {i}: {param_group['lr']:.7f}")
            print(f"  Image loss: {image_loss.detach().item():.6f}")
            print(f"  Gradient loss: {grad_loss.detach().item():.6f}")
            print(f"  Slew rate loss: {slew_loss.detach().item():.6f}")
            print(f"  Total loss: {image_loss.detach().item()+grad_loss.detach().item()+slew_loss.detach().item():.6f}")
            print(f"  Best loss: {self.best_loss:.6f}")
            print("-" * 100)
            print(f"Gradient:{1000 / self.gamma * d_max.max().item():.2f}")
            print(f"Slew Rate:{1000 / self.gamma * dd_max.max().item():.2f}")
            print("=" * 100)

    def export_figure(self, path):
        self.fig.savefig(os.path.join(path, "train_figure.png"))


class Checkpointer:
    def __init__(self, path, params, dt):
        self.path = path
        self.params = params
        self.dt = dt

    def export_json(self, rosette):
        shift = self.params["timesteps"] - 1

        petals = [rosette[i * shift : (i + 1) * shift, :] for i in range(self.params["n_petals"])]

        grads = [1000 / self.params["gamma"] * compute_derivatives(p, self.dt)[0] for p in petals]
        slews = [1000 / self.params["gamma"] * compute_derivatives(p, self.dt)[1] for p in petals]

        grad_norms = [g.abs().max(dim=0).values for g in grads]

        grads_normalized = [g / n for g, n in zip(grads, grad_norms)]

        slew_max = torch.zeros(2)
        for s in slews:
            slew_max = torch.maximum(slew_max, s.abs().max(dim=0).values)

        maxSlewX = slew_max[0].item()
        maxSlewY = slew_max[1].item()
        GxMax = [max(g[:, 0].abs()).item() for g in grads]
        GyMax = [max(g[:, 1].abs()).item() for g in grads]
        Gx = [p[:, 0].tolist() for p in grads_normalized]
        Gy = [p[:, 1].tolist() for p in grads_normalized]

        export_data = {
            "specBW": 2000,
            "specRes": 325,
            "FoV": self.params["FoV"],
            "res": self.params["res"],
            "preEmMom": 24,
            "preEmPts": 10,
            "pdlNo": self.params["n_petals"],
            "pdlPts": self.params["timesteps"] - 1,
            "info": {
                "maxSlewX": maxSlewX,
                "maxSlewY": maxSlewY,
            },
            "trj": {
                "GxMax": GxMax,
                "GyMax": GyMax,
                "Gx": Gx,
                "Gy": Gy,
            },
        }
        with open(join(self.path, "traj_data.json"), "w") as f:
            json.dump(export_data, f, indent=4)

    def save_checkpoint(self, model, d_max_rosette, dd_max_rosette, rosette):
        os.makedirs(self.path, exist_ok=True)
        grad = 1000 / self.params["gamma"] * d_max_rosette
        slew = 1000 / self.params["gamma"] * dd_max_rosette
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_name": model.name,
                "params": self.params,
                "slew_rate": slew,
                "gradient": grad,
            },
            os.path.join(self.path, "checkpoint.pt"),
        )
        self.export_json(rosette)

        print("=" * 100)
        print("CHECKPOINT SAVED")
        print("Slew Rate:", slew)
        print("=" * 100)
        return slew

    def load_checkpoint(self):
        checkpoint = torch.load(os.path.join(self.path, "checkpoint.pt"))
        self.params = checkpoint["params"]
        self.dt = self.params["duration"] / (self.params["timesteps"] - 1)
        return checkpoint


def get_phantom(size=(512, 512), type="shepp_logan"):
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


def get_rotation_matrix(n_petals, device=torch.device("cpu")):
    angle_radians = 2 * torch.pi / n_petals
    angle_radians = torch.Tensor([angle_radians])
    rotation_matrix = torch.tensor(
        [
            [torch.cos(angle_radians), -torch.sin(angle_radians)],
            [torch.sin(angle_radians), torch.cos(angle_radians)],
        ]
    ).to(device)
    return rotation_matrix


def make_rosette(traj, rotation_matrix, n_petals, kmax_img, dt, zero_filling=True):
    rotated_trajectories = [traj]
    for i in range(n_petals - 1):
        traj = traj @ rotation_matrix.T
        rotated_trajectories.append(traj)
    d_max, dd_max = torch.zeros(1, 2, device=traj.device), torch.zeros(1, 2, device=traj.device)
    for t in rotated_trajectories[:n_petals]:
        d, dd = compute_derivatives(t, dt)
        d_max = torch.maximum(d.abs().max(dim=0).values, d_max)
        dd_max = torch.maximum(dd.abs().max(dim=0).values, dd_max)
    if zero_filling:
        corners = torch.ones(2, 2, device=traj.device)
        corners[0] *= kmax_img
        corners[1] *= -kmax_img
        rotated_trajectories.append(corners)
    rosette = torch.cat(rotated_trajectories, dim=0)
    return rosette, d_max, dd_max


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


def plot_pixel_rosette(rosette, fft, img_size, ax=None):
    """Sample FFT at trajectory locations"""
    sampled_coord_x = torch.round((rosette[0, 0, :, 0] + 1) * (img_size - 1) / 2)
    sampled_coord_y = torch.round((rosette[0, 0, :, 1] + 1) * (img_size - 1) / 2)
    pixel_curve = torch.zeros(img_size, img_size)
    pixel_curve[sampled_coord_y.long(), sampled_coord_x.long()] = 1.0
    sampled_from_pixels = fft[0, 0, sampled_coord_y.long(), sampled_coord_x.long()]
    sampled_from_pixels[-2:] *= 0
    if ax is not None:
        ax.imshow(pixel_curve[img_size // 4 : 3 * img_size // 4, img_size // 4 : 3 * img_size // 4].detach().cpu().numpy())
    else:
        plt.figure()
        plt.imshow(pixel_curve.detach().cpu().numpy())
        plt.show()
    return sampled_from_pixels


def final_plots(phantom, recon, initial_recon, losses, traj, slew_rate, show=True, export=False, export_path=None):
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    im1 = ax[0, 0].imshow(phantom.detach().cpu().numpy(), cmap="gray")
    ax[0, 0].set_title("Phantom")
    ax[0, 0].axis("off")
    fig.colorbar(im1, ax=ax[0, 0])

    im2 = ax[0, 1].imshow(recon.abs().detach().cpu().numpy(), cmap="gray")
    ax[0, 1].set_title("Recon")
    ax[0, 1].axis("off")
    fig.colorbar(im2, ax=ax[0, 1])

    im3 = ax[0, 2].imshow(initial_recon.squeeze().detach().cpu().numpy(), cmap="gray")
    ax[0, 2].set_title("Initial Recon")
    ax[0, 2].axis("off")
    fig.colorbar(im3, ax=ax[0, 2])

    im4 = ax[1, 0].imshow((recon.abs() - phantom).detach().cpu().numpy(), cmap="gray")
    ax[1, 0].set_title("Phantom - Recon")
    ax[1, 0].axis("off")
    fig.colorbar(im4, ax=ax[1, 0])

    im5 = ax[1, 1].semilogy(range(len(losses)), losses, linewidth=0.7)
    ax[1, 1].set_title("Loss")

    im6 = ax[1, 2].plot(traj[:, 0].detach().cpu().numpy(), traj[:, 1].detach().cpu().numpy(), linewidth=0.7, marker=".", markersize=3)
    ax[1, 2].set_title(f"Trajectory. Slew Rate: {slew_rate.abs().max().detach().item():.2f}")
    if export and export_path is not None:
        os.makedirs(export_path, exist_ok=True)
        plt.savefig(os.path.join(export_path, "final_figure.png"), dpi=300)
    if show:
        plt.show()
    else:
        plt.close()
    return im1, im2, im3, im4, im5, im6


def export_k_as_csv(traj, path):
    os.makedirs(path, exist_ok=True)
    traj_np = traj.detach().cpu().numpy()
    traj_abs = np.linalg.norm(traj_np, axis=1)
    max_abs_point = np.argmax(traj_abs)
    np.savetxt(os.path.join(path, "k_trajectory.csv"), traj_np, delimiter=",")
    np.savetxt(os.path.join(path, "k_trajectory_maxpt.csv"), np.array([[max_abs_point, 0], [max_abs_point, 1]]), delimiter=",")
