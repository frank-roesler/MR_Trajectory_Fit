import torch

# import odl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchkbnufft import calc_density_compensation_function
import torchkbnufft as tkbn
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
from scipy.signal import find_peaks
from mirtorch.linear import FFTCn
from params import params

# import safe_hw_from_asc


class ImageRecon:
    def __init__(self, params, kmax_img, normalization, dcfnet="unet"):
        """medhod must be one of: ['kbnufft', 'mirtorch', 'nudft'].
        'kbnufft': uses dcfnet for fast differentiable dcf computation,
        'mirtorch': uses calc_density_compensation_function from torchkbnufft. Not differentiable.
        'nudft': very slow and should only be used for debugging."""
        self.device = get_device()
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
        dcfdict = torch.load(f"trained_models/dcfnet_general_{self.dcfnet.name}.pt", map_location=self.device)
        self.dcfnet.load_state_dict(dcfdict)
        self.dcfnet.to(self.device)

    def sample_k_space_values(self, fft, rosette):
        rosette = rosette.reshape(1, 1, rosette.shape[0], 2)
        sampled_r = F.grid_sample(fft.real.detach(), rosette / self.kmax_img, mode="bicubic", align_corners=True)
        sampled_i = F.grid_sample(fft.imag.detach(), rosette / self.kmax_img, mode="bicubic", align_corners=True)
        sampled = torch.complex(sampled_r, sampled_i).squeeze(0)
        if self.zero_filling:
            sampled[:, :, -2:] *= 0
        return rosette, sampled

    def reconstruct_img(self, fft, rosette, method="kbnufft"):
        if fft.dim() == 4 and fft.shape[0] > 1:
            recons = []
            for b in range(fft.shape[0]):
                recons.append(self.reconstruct_img(fft[b : b + 1], rosette, method=method))
            return torch.stack(recons, dim=0)
        rosette, sampled = self.sample_k_space_values(fft, rosette)
        if method == "mirtorch":
            return self.reconstruct_img_mirtorch(rosette, sampled)
        if method == "kbnufft":
            return self.reconstruct_img_dcfnet(rosette, sampled)
        return self.reconstruct_img_nudft(rosette, sampled)

    # def reconstruct_img_mirtorch(self, rosette, sampled):
    #     s0 = torch.ones(1, 1, self.img_size, self.img_size, device=sampled.device) + 0j
    #     rosette = rosette.squeeze() / self.kmax_img * torch.pi
    #     dcf = calc_density_compensation_function(rosette.T, (self.img_size, self.img_size)) + 0j
    #     Nop = NuSense_om(s0, rosette.T.reshape(1, 2, -1), norm=None)
    #     sampled = sampled.reshape(1, 1, -1)
    #     I0 = Nop.H * (dcf * sampled)
    #     I0 = torch.flip(torch.rot90(I0.abs(), k=1, dims=(2, 3)), dims=[2]).squeeze()
    #     return I0 * self.normalization

    def reconstruct_img_mirtorch(self, rosette, sampled):
        s0 = torch.ones(1, 1, self.img_size, self.img_size, device=sampled.device) + 0j
        rosette = rosette.squeeze().permute(1, 0) / self.kmax_img * torch.pi
        sampled = sampled.reshape(1, 1, -1)
        dcf = calc_density_compensation_function(rosette, (self.img_size, self.img_size)) + 0j
        Nop = NuSense_om(s0, rosette.reshape(1, 2, -1), norm=None)
        I0 = Nop.H * (dcf * sampled)
        I0 = torch.flip(torch.rot90(I0.abs(), k=1, dims=(2, 3)), dims=[2]).squeeze()
        return I0 * self.normalization

    def reconstruct_img_dcfnet(self, rosette, sampled):
        rosette = rosette.squeeze() / self.kmax_img * torch.pi
        # dcf = self.smooth_dcf(rosette).unsqueeze(0).unsqueeze(0) + 0j
        sampled = sampled.reshape(1, 1, -1)
        dcf = self.dcfnet(rosette.permute(1, 0)[:, : self.timesteps - 1].unsqueeze(0)).squeeze()
        dcf = torch.cat([dcf.repeat((1, self.n_petals)), torch.zeros(1, 2, device=dcf.device)], dim=-1).unsqueeze(0) + 0j
        # plt.figure()
        # plt.plot(dcf_gpt.real.squeeze()[:-2].cpu().numpy())
        # plt.plot(dcf.real.squeeze()[:-2].cpu().numpy())
        # plt.show()

        kbnufft.nufft.set_dims(sampled.shape[-1], (self.img_size, self.img_size), device=rosette.device, Nb=1)
        kbnufft.nufft.precompute(rosette)
        I0 = kbnufft.adjoint(rosette, (sampled * dcf).squeeze(0)).unsqueeze(0)
        I0 = torch.flip(torch.rot90(I0.abs(), k=1, dims=(2, 3)), dims=[2]).squeeze()
        I0 = I0 * self.normalization
        return I0

    def reconstruct_img_nudft(self, rosette, sampled):
        rosette = rosette.squeeze().permute(1, 0)
        sampled = sampled.reshape(-1)
        dcf = self.knn_density_compensation(rosette, self.timesteps - 1).reshape(sampled.shape)
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
        traj = rosette[..., : self.timesteps - 1]
        n_petals = rosette.shape[-1] // (self.timesteps - 1)
        N = traj.shape[-1]
        w = torch.ones(n_petals * N, device=rosette.device)
        for _ in range(num_iters):
            psf = torch.zeros(N, device=rosette.device)
            for i in range(N):
                dist_sq = torch.sum((rosette - rosette[:, i : i + 1]) ** 2, axis=0)
                psf[i] = torch.sum(w / (dist_sq + 1e-6))
            w = w / psf.repeat(n_petals)
        w = 50 * N * n_petals * torch.cat([w, torch.zeros(2, device=rosette.device)], dim=0)
        return w

    def smooth_dcf(self, traj, sigma=0.01):
        dist = torch.cdist(traj, traj)
        weights = torch.exp(-(dist**2) / sigma**2)
        return 0.05 / np.sqrt(sigma) / (weights.sum(dim=-1) + 1e-6)


class MySSIMLoss(nn.Module):
    def __init__(self, window_size=11, reduction="mean", max_val=1.0):
        super(MySSIMLoss, self).__init__()
        self.ssim = SSIMLoss(window_size=window_size, reduction=reduction, max_val=max_val)
        self.L1_loss = nn.L1Loss()

    def forward(self, img, target):
        if img.dim() == 2:
            img_ssim = img.unsqueeze(0).unsqueeze(0)
        elif img.dim() == 3:
            img_ssim = img.unsqueeze(1)
        else:
            img_ssim = img

        if target.dim() == 2:
            target_ssim = target.unsqueeze(0).unsqueeze(0)
        elif target.dim() == 3:
            target_ssim = target.unsqueeze(1)
        else:
            target_ssim = target

        loss = 1.0 - self.ssim(img_ssim, target_ssim) + 0.1 * self.L1_loss(img, target)
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
        excess = torch.clamp(x - threshold, min=0.0)
        return excess**2

    def logb_star_loss(self, x, threshold, delta=1):
        """(arXviv:2505.07117v1)"""
        x_delta = threshold - delta

        # Safe x prevents NaNs in the log function during the forward pass
        safe_x = torch.where(x <= x_delta, x, torch.full_like(x, x_delta))

        branch_1 = -torch.log(threshold - safe_x)

        # Ensure delta scalar is a tensor on the correct device for computation
        delta_tensor = torch.tensor(delta, device=x.device, dtype=x.dtype)
        branch_2 = ((x - x_delta) / delta_tensor) - torch.log(delta_tensor)

        elementwise_loss = torch.where(x <= x_delta, branch_1, branch_2)
        return elementwise_loss

    def effective_potential_loss(self, x, threshold, A=40.0, B=30.0, epsilon=0.5):
        r = torch.clamp(threshold - x, min=epsilon)

        repulsive_core = A / (r**2)
        attractive_tail = B / r
        potential = repulsive_core - attractive_tail

        overshoot = torch.clamp(x - threshold, min=0.0)
        penalty = 100.0 * (overshoot**2)

        return potential + penalty

    def grad_loss(self, d_max_rosette, params, mode="exp"):
        unit = params["gamma"] / 1000
        threshold = params["grad_max"] * unit

        if mode == "exp":
            loss = torch.exp(0.1 * (d_max_rosette - threshold))
        elif mode == "threshold":
            loss = self.threshold_loss(d_max_rosette, threshold)
        else:
            loss = torch.zeros(1, device=d_max_rosette.device)

        return params["grad_loss_weight"] * loss.sum()

    def slew_loss(self, dd_max_rosette, params, mode="exp", delta=1):
        unit = params["gamma"] / 1000
        threshold = params["slew_rate"] * unit

        if mode == "exp":
            loss = torch.exp(0.1 * (dd_max_rosette - threshold))
        elif mode == "threshold":
            loss = self.threshold_loss(dd_max_rosette, threshold)
        elif mode == "logb_star":
            loss = self.logb_star_loss(dd_max_rosette, threshold, delta=delta)
        elif mode == "effective_potential":
            loss = self.effective_potential_loss(dd_max_rosette, threshold)
        else:
            loss = torch.zeros(1, device=dd_max_rosette.device)

        return params["slew_loss_weight"] * loss.sum()

    def grad_slew_loss(self, d_max_rosette, dd_max_rosette, params, grad_mode="exp", slew_mode="exp", delta=1):
        """
        Compute a loss based on the gradient and slew rate of the trajectory.
        Returns: grad_loss (scalar), slew_loss (scalar)
        """
        grad_loss_val = self.grad_loss(d_max_rosette, params, mode=grad_mode)
        slew_loss_val = self.slew_loss(dd_max_rosette, params, mode=slew_mode, delta=delta)

        return grad_loss_val, slew_loss_val

    def pns_loss(self, max_pns, params, mode="exp", delta=1):
        threshold = params["pns_threshold"]

        if mode == "exp":
            loss = torch.exp(0.1 * (max_pns - threshold))
        elif mode == "threshold":
            loss = self.threshold_loss(max_pns, threshold)
        elif mode == "logb_star":
            loss = self.logb_star_loss(max_pns, threshold, delta=delta)
        elif mode == "effective_potential":
            loss = self.effective_potential_loss(max_pns, threshold)
        else:
            loss = torch.zeros(1, device=max_pns.device)

        return params["pns_loss_weight"] * loss.sum()


class TrainPlotter:
    def __init__(self, params, fft, reconstructor, phantoms, loss_fn, optimizer):
        self.best_loss = float("inf")
        self.train_steps = params["train_steps"]
        self.gamma = params["gamma"]
        # Calculate dt for gradient computation: duration / (steps - 1)
        self.dt = params["duration"] / (params["timesteps"] - 1)

        self.fft = fft
        self.reconstructor = reconstructor
        self.phantoms = phantoms
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        plt.rc("xtick", labelsize=8)
        plt.rc("ytick", labelsize=8)

        # --- 0. Figure Setup ---
        # Returns a 2x3 array of axes
        self.fig, axs = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)

        # Unpack axes for easier access
        ax_loss = axs[0, 0]
        ax_img = axs[0, 1]
        ax_traj = axs[0, 2]
        ax_grad = axs[1, 0]  # New: Gradient Plot
        ax_angles = axs[1, 1]  # New: PNS Plot
        ax_pns_norm = axs[1, 2]  # New: max PNS Norm Plot

        # --- 1. Loss Plot (Top Left) ---

        # Constraint losses on a secondary y-axis - not anymore
        # ax_loss = ax_single_losses.twinx()
        (grad_loss_line,) = ax_loss.semilogy([], [], label="Grad Loss", linewidth=0.7, color="r")
        (slew_loss_line,) = ax_loss.semilogy([], [], label="Slew Loss", linewidth=0.7, color="g")
        (pns_loss_line,) = ax_loss.semilogy([], [], label="PNS Loss", linewidth=0.7, color="m")

        # Image loss and total loss on the primary y-axis
        (img_loss_line,) = ax_loss.semilogy([], [], label="Image Loss", linewidth=0.7, color="b")
        (img_loss_line_mirtorch,) = ax_loss.semilogy([], [], label="Image Loss MIRTorch", linewidth=0.7, color="orange")
        (total_loss_line,) = ax_loss.semilogy([], [], label="Total Loss", linewidth=0.7, color="k")

        ax_loss.set_xlabel("Step")
        ax_loss.set_ylabel("Loss")
        ax_loss.set_title("Running Loss")
        # ax_single_losses.set_ylabel("Grad-Slew-PNS Losses")
        lns = [grad_loss_line, slew_loss_line, pns_loss_line, img_loss_line, total_loss_line, img_loss_line_mirtorch]
        labs = [l.get_label() for l in lns]
        ax_loss.legend(lns, labs, loc=0, prop={"size": 6})

        # --- 2. Reconstructed Image (Top Middle) ---
        im_recon = ax_img.imshow(torch.zeros((params["img_size"], params["img_size"])).numpy(), cmap="gray")
        ax_img.set_title("Recon[0] (abs)")
        ax_img.axis("off")
        self.cbar = self.fig.colorbar(im_recon, ax=ax_img)

        # --- 3. Trajectory (Top Right) ---
        (traj_line,) = ax_traj.plot([], [], label="trajectory", linewidth=0.5)
        ax_traj.set_title("Trajectory")

        # --- 4. Gradients Gx/Gy (Bottom Left) ---
        (gx_line,) = ax_grad.plot([], [], label="Gx", linewidth=1.0, color="tab:blue")
        (gy_line,) = ax_grad.plot([], [], label="Gy", linewidth=1.0, color="tab:orange")
        ax_grad.set_title("Gradients for shown traj.")
        ax_grad.set_xlabel("Time (ms)")
        ax_grad.set_ylabel("Amplitude (mT/m)")
        ax_grad.legend(loc="upper right", prop={"size": 8})
        ax_grad.grid(True, linestyle="--", alpha=0.5)

        # --- 5. Angle Plots (Bottom Middle) ---
        ax_angles = axs[1, 1]
        (angles_line,) = ax_angles.plot([], [], label="Angles", linewidth=0.7, color="tab:blue", marker=".", markersize=3)
        ax_angles.set_title("Angles between adjacent petals.")
        ax_angles.set_ylabel("Angle (rad)")
        ax_angles.legend(loc="upper right", prop={"size": 8})

        # --- 6. PNS Norm Plot (Bottom Right) ---
        ax_pns_norm = axs[1, 2]
        (pns_norm_max_line,) = ax_pns_norm.plot([], [], label="Max PNS Norm", linewidth=1.0, color="k")
        ax_pns_norm.set_title("Max PNS Norm over Steps")
        ax_pns_norm.set_xlabel("Step")
        ax_pns_norm.set_ylabel("Max Stimulation (%)")
        ax_pns_norm.legend(loc="upper right", prop={"size": 8})
        ax_pns_norm.grid(True, linestyle="--", alpha=0.5)

        plt.show(block=False)

        # Store references
        self.grad_loss_line = grad_loss_line
        self.slew_loss_line = slew_loss_line
        self.pns_loss_line = pns_loss_line
        self.img_loss_line = img_loss_line
        self.img_loss_line_mirtorch = img_loss_line_mirtorch
        self.total_loss_line = total_loss_line
        self.im_recon = im_recon
        self.traj_line = traj_line

        self.gx_line = gx_line
        self.gy_line = gy_line

        self.angles_line = angles_line
        self.pns_norm_max_line = pns_norm_max_line

        self.ax_loss = ax_loss
        # self.ax_single_losses = ax_single_losses
        self.ax_traj = ax_traj
        self.ax_img = ax_img
        self.ax_grad = ax_grad
        self.ax_angles = ax_angles
        self.ax_pns_norm = ax_pns_norm

        self.grad_losses = []
        self.slew_losses = []
        self.img_losses = []
        self.img_losses_mirtorch = []
        self.total_losses = []
        self.max_pns_norms = []
        self.pns_losses = []

    def update(
        self,
        step,
        grad_loss,
        img_loss,
        slew_loss,
        pns_loss,
        total_loss,
        recon,
        traj,
        rosette,
        gx,
        gy,
        t_axis,
        angles,
        pns_norm,
        t_pns,
    ):

        # Update for loss plot (final_figure.png and train_figure.png), evaluation, checkpoint..
        self.grad_losses.append(grad_loss.detach().item())
        self.img_losses.append(img_loss.detach().item())
        self.slew_losses.append(slew_loss.detach().item())
        self.pns_losses.append(pns_loss.detach().item())
        self.total_losses.append(total_loss.detach().item())
        max_pns = pns_norm.max().item()
        self.max_pns_norms.append(max_pns)

        plotting_freq = 10
        if step % plotting_freq == 0:
            # --- Update Recon and Losses (for the image) ---
            recon_mirtorch = self.reconstructor.reconstruct_img(self.fft, rosette, method="mirtorch")
            image_loss_mirtorch = self.loss_fn(recon_mirtorch, self.phantoms)
            self.img_losses_mirtorch.append(image_loss_mirtorch.detach().item())

            self.img_loss_line_mirtorch.set_data(range(0, plotting_freq * len(self.img_losses_mirtorch), plotting_freq), self.img_losses_mirtorch)
            self.img_loss_line.set_data(range(len(self.img_losses)), self.img_losses)
            self.grad_loss_line.set_data(range(len(self.grad_losses)), self.grad_losses)
            self.pns_loss_line.set_data(range(len(self.pns_losses)), self.pns_losses)
            self.slew_loss_line.set_data(range(len(self.slew_losses)), self.slew_losses)
            self.total_loss_line.set_data(range(len(self.img_losses)), self.total_losses)

            # self.ax_single_losses.relim()
            # self.ax_single_losses.autoscale_view()
            self.ax_loss.relim()
            self.ax_loss.autoscale_view()
            self.ax_loss.set_ylim(min(self.img_losses) * 0.7, max(self.img_losses) * 1.5)

            recon_to_plot = recon[0] if recon.dim() == 3 else recon
            img = recon_to_plot.abs().detach().cpu().numpy()
            self.im_recon.set_data(img)
            self.im_recon.set_clim(vmin=0, vmax=img.max())

            # --- Update Trajectory ---
            self.traj_line.set_data(rosette[:-2, 0].detach().cpu().numpy(), rosette[:-2, 1].detach().cpu().numpy())
            self.ax_traj.relim()
            self.ax_traj.autoscale_view()

            # 4. Plot (convert to numpy for plotting)
            self.gx_line.set_data(t_axis.detach().cpu().numpy(), gx.detach().cpu().numpy())
            self.gy_line.set_data(t_axis.detach().cpu().numpy(), gy.detach().cpu().numpy())
            self.ax_grad.relim()
            self.ax_grad.autoscale_view()

            self.ax_img.set_title(f"Recon[0] (abs) Step {step+1}")

            # 3. Plot Angles (convert to numpy for plotting)
            self.angles_line.set_data(np.arange(angles.shape[0]), angles.detach().cpu().numpy())
            self.ax_angles.relim()
            self.ax_angles.autoscale_view()
            self.ax_angles.set_ylim(0, 8 * np.pi / params["n_petals"])

            # 4. Track and plot maximum PNS norm over steps
            self.pns_norm_max_line.set_data(range(len(self.max_pns_norms)), self.max_pns_norms)
            self.ax_pns_norm.relim()
            self.ax_pns_norm.autoscale_view()

            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

            plt.pause(0.01)

    def print_info(self, step, image_loss, grad_loss, slew_loss, pns_loss, d_max, dd_max, pns_max, g_x, g_y):
        if step % 10 == 0:
            print(f"Step {step+1}/{self.train_steps}")
            for i, param_group in enumerate(self.optimizer.param_groups):
                print(f"  Learning rate {i}: {param_group['lr']:.7f}")
            print(f"  Image loss: {image_loss.detach().item():.6f}")
            print(f"  Gradient loss: {grad_loss.detach().item():.6f}")
            print(f"  Slew rate loss: {slew_loss.detach().item():.6f}")
            print(f"  PNS loss: {pns_loss.detach().item():.6f}")
            print(f"  Total loss: {image_loss.detach().item()+grad_loss.detach().item()+slew_loss.detach().item()+pns_loss.detach().item():.6f}")
            print(f"  Best loss: {self.best_loss:.6f}")
            print("-" * 100)
            print(f"  Gradient: {1000 / self.gamma * d_max.max().item():.2f}")
            print(f"  Slew Rate: {1000 / self.gamma * dd_max.max().item():.2f}")
            # Handle both tensor and scalar inputs for pns_max
            pns_val = pns_max.item() if torch.is_tensor(pns_max) else pns_max
            print(f"  Max PNS Norm: {pns_val:.2f}%")
            print(f"  Max Gx: {g_x.abs().max().item():.2f} mT/m")
            print(f"  Max Gy: {g_y.abs().max().item():.2f} mT/m")

            print("=" * 100)

    def export_figure(self, path):
        self.fig.savefig(os.path.join(path, "train_figure.png"))


class Checkpointer:
    def __init__(self, path, params, dt):
        self.device = get_device()
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

        slew_max = torch.zeros(2, device=grads[0].device)
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
        checkpoint = torch.load(os.path.join(self.path, "checkpoint.pt"), map_location=self.device)
        self.params = checkpoint["params"]
        self.dt = self.params["duration"] / (self.params["timesteps"] - 1)
        return checkpoint


def get_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    # Disable MPS due to memory constraints - use CPU instead
    # elif torch.backends.mps.is_available():
    #     return torch.device("mps")
    else:
        return torch.device("cpu")


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


def get_batch_of_phantoms(batch_size, size=(512, 512), type="shepp_logan"):
    phantoms = []
    for _ in range(batch_size):
        phantom = get_phantom(size=size, type=type)
        phantoms.append(phantom)
    return torch.stack(phantoms)


def get_rotation_matrix(angle_radians, device=torch.device("cpu")):
    # angle_radians = 2 * torch.pi / n_petals
    # angle_radians = torch.tensor([angle_radians], device=device)
    c = torch.cos(angle_radians)
    s = torch.sin(angle_radians)

    rotation_matrix = torch.stack(
        [
            torch.stack([c, -s]),
            torch.stack([s, c]),
        ]
    )
    return rotation_matrix


def make_rosette(angles, traj, n_petals, kmax_img, dt, zero_filling=True):
    rotated_trajectories = [traj]
    for i in range(n_petals - 1):
        rotation_matrix = get_rotation_matrix(angles[i])
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


def compute_gradients_from_traj(traj, dt, gamma):
    """Compute the gradient waveforms from a trajectory.
    traj: (timesteps, 2) tensor
    dt: time step size (scalar)
    gamma: gyromagnetic ratio (scalar)
    Returns: gx, gy, t_axis (timesteps,) torch tensors in physical units (e.g. mT/m)
    """
    d_traj, _ = compute_derivatives(traj, dt)
    grad_waveform = d_traj * (1000 / gamma)  # Convert to mT/m
    gx = grad_waveform[:, 0]
    gy = grad_waveform[:, 1]
    # gz = torch.zeros_like(gx)  # Assuming 2D trajectory, gz is zero
    t_axis = torch.arange(len(gx), device=traj.device, dtype=traj.dtype) * dt  # Time in ms
    return gx, gy, t_axis


def compute_pns_from_gradients(safe_model, gx, gy, gradPreEmphPts=10, specRes=325, Ramp=False):
    """Compute PNS from gradient waveforms using the differentiable safe_gwf_to_pns_torch.

    gx, gy: (timesteps,) torch tensors of gradient waveforms in mT/m
    dt: time step size in ms
    Ramp: if True add ramp up/down, otherwise use only the repeated waveform

    Returns:
        pns_x, pns_y, pns_norm, t_axis
    """
    device = gx.device
    dtype = gx.dtype

    # Repeat waveform for spectral resolution
    full_ro_x = gx.repeat(specRes)
    full_ro_y = gy.repeat(specRes)

    if Ramp:
        # Ramp up/down
        ramp_up_x = gx[0] * torch.linspace(0, 1, gradPreEmphPts, device=device, dtype=dtype)
        ramp_dn_x = gx[-1] * torch.linspace(1, 0, gradPreEmphPts, device=device, dtype=dtype)

        ramp_up_y = gy[0] * torch.linspace(0, 1, gradPreEmphPts, device=device, dtype=dtype)
        ramp_dn_y = gy[-1] * torch.linspace(1, 0, gradPreEmphPts, device=device, dtype=dtype)

        gVecX = torch.cat([ramp_up_x[:-1], full_ro_x, ramp_dn_x[1:]])
        gVecY = torch.cat([ramp_up_y[:-1], full_ro_y, ramp_dn_y[1:]])

    else:
        gVecX = full_ro_x
        gVecY = full_ro_y

    gVecZ = torch.zeros_like(gVecX)
    gVec = torch.stack([gVecX, gVecY, gVecZ], dim=1)  # (time, 3)

    # Compute PNS
    pns = safe_model.safe_gwf_to_pns(gVec)

    pns_x = pns[:, 0]
    pns_y = pns[:, 1]
    pns_norm = torch.norm(pns, dim=1)

    t_pns = torch.arange(len(pns), device=device, dtype=dtype) * safe_model.dt  # ms

    return pns_x, pns_y, pns_norm, t_pns


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
    pixel_curve = torch.zeros(img_size, img_size, device=rosette.device)
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
    def _to_batch_3d(x):
        if x.dim() == 2:
            return x.unsqueeze(0)
        if x.dim() == 3:
            return x
        if x.dim() == 4 and x.shape[1] == 1:
            return x[:, 0]
        raise ValueError(f"Unsupported image tensor shape: {x.shape}")

    phantom_b = _to_batch_3d(phantom)
    recon_b = _to_batch_3d(recon).abs()
    init_b = _to_batch_3d(initial_recon).abs()

    n_rows = max(phantom_b.shape[0], recon_b.shape[0], init_b.shape[0])

    if phantom_b.shape[0] == 1 and n_rows > 1:
        phantom_b = phantom_b.repeat(n_rows, 1, 1)
    if recon_b.shape[0] == 1 and n_rows > 1:
        recon_b = recon_b.repeat(n_rows, 1, 1)
    if init_b.shape[0] == 1 and n_rows > 1:
        init_b = init_b.repeat(n_rows, 1, 1)

    if not (phantom_b.shape[0] == recon_b.shape[0] == init_b.shape[0]):
        raise ValueError("Batch dimensions of phantom, recon, and initial_recon are incompatible")

    fig, ax = plt.subplots(n_rows, 5, figsize=(20, 4 * n_rows))
    if n_rows == 1:
        ax = np.expand_dims(ax, axis=0)

    col_titles = ["Phantom", "Recon", "Phantom - Recon", "Initial Recon", "Phantom - Initial Recon"]

    for r in range(n_rows):
        phantom_np = phantom_b[r].detach().cpu().numpy()
        recon_np = recon_b[r].detach().cpu().numpy()
        init_np = init_b[r].detach().cpu().numpy()

        images = [
            phantom_np,
            recon_np,
            phantom_np - recon_np,
            init_np,
            phantom_np - init_np,
        ]

        for c in range(5):
            im = ax[r, c].imshow(images[c], cmap="gray")
            if r == 0:
                ax[r, c].set_title(col_titles[c])
            cbar = fig.colorbar(im, ax=ax[r, c], orientation="horizontal", fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=6)
            ax[r, c].axis("off")

    fig.tight_layout()

    fig_traj, ax_traj = plt.subplots(1, 1, figsize=(6, 6))
    x_data = traj[:, 0].detach().cpu().numpy()
    y_data = traj[:, 1].detach().cpu().numpy()
    ax_traj.plot(x_data, y_data, linewidth=0.7, marker=".", markersize=3)
    x_min, x_max = x_data.min(), x_data.max()
    y_min, y_max = y_data.min(), y_data.max()
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    cross_frac = 0.25
    dx = (x_max - x_min) * cross_frac / 2
    dy = (y_max - y_min) * cross_frac / 2
    cross_x_min, cross_x_max = x_center - dx, x_center + dx
    cross_y_min, cross_y_max = y_center - dy, y_center + dy
    cross_style = {"color": "red", "linestyle": "-", "linewidth": 1.2, "alpha": 0.8}
    ax_traj.plot([cross_x_min, cross_x_max], [y_center, y_center], **cross_style)
    ax_traj.plot([x_center, x_center], [cross_y_min, cross_y_max], **cross_style)
    text_style = {
        "color": "red",
        "fontsize": 9,
        "fontweight": "bold",
        "bbox": dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1),
    }
    ax_traj.text(cross_x_min, y_center, f"{x_min:.3f}", ha="right", va="center", **text_style)
    ax_traj.text(cross_x_max, y_center, f"{x_max:.3f}", ha="left", va="center", **text_style)
    ax_traj.text(x_center, cross_y_min, f"{y_min:.3f}", ha="center", va="top", **text_style)
    ax_traj.text(x_center, cross_y_max, f"{y_max:.3f}", ha="center", va="bottom", **text_style)
    ax_traj.text(
        x_center,
        y_center,
        "max",
        ha="center",
        va="center",
        color="red",
        fontweight="bold",
        bbox=dict(facecolor="white", alpha=1.0, edgecolor="none", pad=2),
    )
    ax_traj.set_aspect("equal", adjustable="box")
    ax_traj.set_title(f"Final Trajectory. Slew Rate: {slew_rate.abs().max().detach().item():.2f}")
    ax_traj.grid(True, linestyle="--", alpha=0.5)
    fig_traj.tight_layout()

    if export and export_path is not None:
        os.makedirs(export_path, exist_ok=True)
        fig.savefig(os.path.join(export_path, "final_figure.png"), dpi=300)
        fig_traj.savefig(os.path.join(export_path, "final_traj.png"), dpi=300)

    if show:
        plt.show()
    else:
        plt.close(fig)
        plt.close(fig_traj)

    return fig, fig_traj


def export_k_as_csv(traj, path):
    os.makedirs(path, exist_ok=True)
    traj_np = traj.detach().cpu().numpy()
    traj_abs = np.linalg.norm(traj_np, axis=1)
    max_abs_point = np.argmax(traj_abs)
    np.savetxt(os.path.join(path, "k_trajectory.csv"), traj_np, delimiter=",")
    np.savetxt(os.path.join(path, "k_trajectory_maxpt.csv"), np.array([[max_abs_point, 0], [max_abs_point, 1]]), delimiter=",")


def calculate_fwhm(profile):
    half_max = np.max(profile) / 2.0
    above_half = np.where(profile >= half_max)[0]

    return above_half[-1] - above_half[0] + 1


def calculate_pslr(profile, distance=10, prominence=None):
    """
    Calculates Peak Side Lobe Ratio (PSLR) using scipy.signal.find_peaks.
    """
    peaks, _ = find_peaks(profile, distance=distance, prominence=prominence)
    peak_values = profile[peaks]
    main_lobe_idx_in_peaks = np.argmax(peak_values)
    peak_main = peak_values[main_lobe_idx_in_peaks]
    side_lobe_values = np.delete(peak_values, main_lobe_idx_in_peaks)
    max_sidelobe = np.max(side_lobe_values)
    pslr_db = 20 * np.log10(peak_main / max_sidelobe)

    return pslr_db


def calculate_islr(psf):
    center = torch.tensor(psf.shape) // 2
    r = 1  # radius of main lobe
    yy, xx = torch.meshgrid(torch.arange(psf.shape[0]), torch.arange(psf.shape[1]), indexing="ij")
    mask_main = (xx - center[1]) ** 2 + (yy - center[0]) ** 2 <= r**2
    main_energy = (psf[mask_main] ** 2).sum()
    side_energy = (psf[~mask_main] ** 2).sum()
    islr = side_energy / main_energy
    return islr


def psf(reconstructor, fft_template, rosette_init, rosette_final, device, export_path):
    """
    Computes, compares, and plots the Point Spread Function (PSF) for the initial and final trajectories.

    Args:
        reconstructor: The ImageRecon object used to reconstruct the images.
        fft_template: A tensor with the same shape as your k-space data (used to generate the ones).
        rosette_init: The initial trajectory tensor.
        rosette_final: The final optimized trajectory tensor.
        device: The PyTorch device (CPU or CUDA).
        export_path: The directory string where the plots will be saved.
    """

    with torch.no_grad():
        # Point-like image
        fft_ones = torch.ones_like(fft_template, dtype=torch.complex64, device=device)

        # Reconstruct the PSF
        psf_init = reconstructor.reconstruct_img(fft_ones, rosette_init, method="kbnufft")
        psf_final = reconstructor.reconstruct_img(fft_ones, rosette_final, method="kbnufft")

        # Move to CPU for plotting
        psf_init_np = psf_init.abs().cpu().numpy()
        psf_final_np = psf_final.abs().cpu().numpy()

        # Normalization
        psf_init_np /= psf_init_np.max()
        psf_final_np /= psf_final_np.max()

        # log scale to visualize side-lobes
        epsilon = 1e-7
        psf_init_log = np.log10(psf_init_np + epsilon)
        psf_final_log = np.log10(psf_final_np + epsilon)

        # Center index for 1D profiles
        center_idx = psf_init_np.shape[0] // 2

        fig, axs = plt.subplots(2, 3, figsize=(18, 8))

        fwhm_init = calculate_fwhm(psf_init_np[center_idx, :])
        fwhm_final = calculate_fwhm(psf_final_np[center_idx, :])
        pslr_init = calculate_pslr(psf_init_np[center_idx, :])
        pslr_final = calculate_pslr(psf_final_np[center_idx, :])
        islr_init = calculate_islr(psf_init)
        islr_final = calculate_islr(psf_final)

        # --- Linear Scale ---
        im0 = axs[0, 0].imshow(psf_init_np, cmap="gist_gray")
        axs[0, 0].set_title("Initial PSF")
        fig.colorbar(im0, ax=axs[0, 0])

        im1 = axs[0, 1].imshow(psf_final_np, cmap="gist_gray")
        axs[0, 1].set_title("Final PSF")
        fig.colorbar(im1, ax=axs[0, 1])

        axs[0, 2].plot(psf_init_np[center_idx, :], alpha=0.8, label=f"Initial PSF: \nFWHM={fwhm_init:.1f}px, \nPSLR={pslr_init:.1f}dB, \nISLR={islr_init:.2f}")
        axs[0, 2].plot(psf_final_np[center_idx, :], alpha=0.8, label=f"Final PSF: \nFWHM={fwhm_final:.1f}px, \nPSLR={pslr_final:.1f}dB, \nISLR={islr_final:.2f}")
        axs[0, 2].set_title("PSF Profile (Linear)")
        axs[0, 2].set_xlabel("Pixel")
        axs[0, 2].set_ylabel("PSF (A.U.)")
        axs[0, 2].legend()
        axs[0, 2].grid(True, linestyle="--", alpha=0.6)

        # --- Log Scale ---
        im2 = axs[1, 0].imshow(psf_init_log, cmap="gist_gray", vmin=-3, vmax=0)
        axs[1, 0].set_title("Initial PSF (Log scale)")
        fig.colorbar(im2, ax=axs[1, 0])

        im3 = axs[1, 1].imshow(psf_final_log, cmap="gist_gray", vmin=-3, vmax=0)
        axs[1, 1].set_title("Final PSF (Log scale)")
        fig.colorbar(im3, ax=axs[1, 1])

        axs[1, 2].plot(psf_init_log[center_idx, :], label="Initial PSF (Log)", alpha=0.8)
        axs[1, 2].plot(psf_final_log[center_idx, :], label="Final Optimized PSF (Log)", alpha=0.8)
        axs[1, 2].set_title("PSF Profile (Log scale)")
        axs[1, 2].set_xlabel("Pixel")
        axs[1, 2].set_ylabel("Log10(PSF)")
        axs[1, 2].legend()
        axs[1, 2].grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()
        os.makedirs(export_path, exist_ok=True)
        plt.savefig(os.path.join(export_path, "psf.png"), dpi=300)
        plt.show()


def compute_initial_fft(phantoms, padding):
    Fop_shape = (phantoms.shape[-2] + 2 * padding, phantoms.shape[-1] + 2 * padding)
    Fop = FFTCn(Fop_shape, Fop_shape, (0, 1), norm=None)
    ffts = []
    for b in range(phantoms.shape[0]):
        phantom_padded = F.pad(phantoms[b], pad=(padding, padding, padding, padding))
        ffts.append(Fop * phantom_padded)
    fft = torch.stack(ffts, dim=0)
    fft = fft.unsqueeze(1)
    return fft
