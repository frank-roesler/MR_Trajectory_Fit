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

import safe_gwf_to_pns
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
        dcfdict = torch.load(f"trained_models/dcfnet_{self.dcfnet.name}.pt", map_location=self.device)
        self.dcfnet.load_state_dict(dcfdict)
        self.dcfnet.to(self.device)

    def sample_k_space_values(self, fft, rosette):
        rosette = rosette.reshape(1, 1, rosette.shape[0], 2)
        sampled_r = F.grid_sample(fft.real.detach(), rosette / self.kmax_img, mode="bilinear", align_corners=True)
        sampled_i = F.grid_sample(fft.imag.detach(), rosette / self.kmax_img, mode="bilinear", align_corners=True)
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
        s0 = torch.ones(1, 1, self.img_size, self.img_size, device=sampled.device) + 0j
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
        dcf = self.dcfnet(rosette[:, : self.timesteps - 1].unsqueeze(0)).squeeze()
        dcf = torch.cat([dcf.repeat((1, self.n_petals)), torch.zeros(1, 2, device=dcf.device)], dim=-1).unsqueeze(0) + 0j
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
        w = torch.ones(n_petals * N, device=rosette.device)
        for _ in range(num_iters):
            psf = torch.zeros(N, device=rosette.device)
            for i in range(N):
                dist_sq = torch.sum((rosette - rosette[:, i : i + 1]) ** 2, axis=0)
                psf[i] = torch.sum(w / (dist_sq + 1e-6))
            w = w / psf.repeat(n_petals)
        w = 50 * N * n_petals * torch.cat([w, torch.zeros(2, device=rosette.device)], dim=0)
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
        excess = torch.clamp(x - threshold, min=0.0)
        return excess ** 2

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
        
        repulsive_core = A / (r ** 2)
        attractive_tail = B / r
        potential = repulsive_core - attractive_tail

        overshoot = torch.clamp(x - threshold, min=0.0)
        penalty = 100.0 * (overshoot ** 2)
        
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
    def __init__(self, params, fft, reconstructor, phantom, loss_fn, optimizer):
        self.best_loss = float("inf")
        self.train_steps = params["train_steps"]
        self.gamma = params["gamma"]
        # Calculate dt for gradient computation: duration / (steps - 1)
        self.dt = params["duration"] / (params["timesteps"] - 1)
        
        self.fft = fft
        self.reconstructor = reconstructor
        self.phantom = phantom
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        plt.rc("xtick", labelsize=8)
        plt.rc("ytick", labelsize=8)

        # --- 0. Figure Setup ---
        # Returns a 2x3 array of axes
        self.fig, axs = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
        
        # Unpack axes for easier access
        ax_single_losses = axs[0, 0]
        ax_img  = axs[0, 1]
        ax_traj = axs[0, 2]
        ax_grad = axs[1, 0] # New: Gradient Plot
        ax_pns = axs[1, 1] # New: PNS Plot
        ax_pns_norm = axs[1, 2] # New: max PNS Norm Plot

        # --- 1. Loss Plot (Top Left) ---

        # Constraint losses on a secondary y-axis
        ax_loss = ax_single_losses.twinx()
        (grad_loss_line,) = ax_single_losses.semilogy([], [], label="Grad Loss", linewidth=0.7, color="r")
        (slew_loss_line,) = ax_single_losses.semilogy([], [], label="Slew Loss", linewidth=0.7, color="g")
        (pns_loss_line,) = ax_single_losses.semilogy([], [], label="PNS Loss", linewidth=0.7, color="m")

        # Image loss and total loss on the primary y-axis
        (img_loss_line,) = ax_loss.semilogy([], [], label="Image Loss", linewidth=0.7, color="b")
        (img_loss_line_mirtorch,) = ax_loss.semilogy([], [], label="Image Loss MIRTorch", linewidth=0.7, color="orange")
        (total_loss_line,) = ax_loss.semilogy([], [], label="Total Loss", linewidth=0.7, color="k")
        
        ax_loss.set_xlabel("Step")
        ax_loss.set_ylabel("Image and tot. Losses")
        ax_loss.set_title("Running Loss")
        ax_single_losses.set_ylabel("Grad-Slew-PNS Losses")
        lns = [grad_loss_line, slew_loss_line, pns_loss_line, img_loss_line, total_loss_line, img_loss_line_mirtorch]
        labs = [l.get_label() for l in lns]
        ax_loss.legend(lns, labs, loc=0, prop={"size": 6})

        # --- 2. Reconstructed Image (Top Middle) ---
        im_recon = ax_img.imshow(torch.zeros((params["img_size"], params["img_size"])).numpy(), cmap="gray")
        ax_img.set_title("Recon (abs)")
        ax_img.axis("off")
        self.cbar = self.fig.colorbar(im_recon, ax=ax_img)

        # --- 3. Trajectory (Top Right) ---
        (traj_line,) = ax_traj.plot([], [], label="trajectory", linewidth=0.7, marker=".", markersize=3)
        ax_traj.set_title("Trajectory")

        # --- 4. Gradients Gx/Gy (Bottom Left) ---
        (gx_line,) = ax_grad.plot([], [], label="Gx", linewidth=1.0, color="tab:blue")
        (gy_line,) = ax_grad.plot([], [], label="Gy", linewidth=1.0, color="tab:orange")
        ax_grad.set_title("Gradients for shown traj.")
        ax_grad.set_xlabel("Time (ms)")
        ax_grad.set_ylabel("Amplitude (mT/m)")
        ax_grad.legend(loc="upper right", prop={"size": 8})
        ax_grad.grid(True, linestyle='--', alpha=0.5)

        # --- 5. PNS Plots (Bottom Middle) ---
        ax_pns = axs[1, 1]
        (pns_x_line,) = ax_pns.plot([], [], label="PNS X", linewidth=0.7, color="tab:blue")
        (pns_y_line,) = ax_pns.plot([], [], label="PNS Y", linewidth=0.7, color="tab:orange")
        (pns_norm_line,) = ax_pns.plot([], [], label="PNS Norm", linewidth=1.0, color="k")
        ax_pns.set_title("PNS")
        ax_pns.set_xlabel("Time (ms)")
        ax_pns.set_ylabel("Stimulation (%)")
        ax_pns.legend(loc="upper right", prop={"size": 8})
        ax_pns.grid(True, linestyle='--', alpha=0.5)

        # --- 6. PNS Norm Plot (Bottom Right) ---
        ax_pns_norm = axs[1, 2]
        (pns_norm_max_line,) = ax_pns_norm.plot([], [], label="Max PNS Norm", linewidth=1.0, color="k")
        ax_pns_norm.set_title("Max PNS Norm over Steps")
        ax_pns_norm.set_xlabel("Step")
        ax_pns_norm.set_ylabel("Max Stimulation (%)")
        ax_pns_norm.legend(loc="upper right", prop={"size": 8})
        ax_pns_norm.grid(True, linestyle='--', alpha=0.5)

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

        self.pns_x_line = pns_x_line
        self.pns_y_line = pns_y_line
        self.pns_norm_line = pns_norm_line
        self.pns_norm_max_line = pns_norm_max_line
        
        self.ax_loss = ax_loss
        self.ax_single_losses = ax_single_losses
        self.ax_traj = ax_traj
        self.ax_img = ax_img
        self.ax_grad = ax_grad
        self.ax_pns = ax_pns
        self.ax_pns_norm = ax_pns_norm

        self.grad_losses = []
        self.slew_losses = []
        self.img_losses = []
        self.img_losses_mirtorch = []
        self.total_losses = []
        self.max_pns_norms = []
        self.pns_losses = []

    def update(self, step, grad_loss, img_loss, slew_loss, pns_loss, total_loss, recon, traj, rosette, gx, gy, t_axis, pns_x, pns_y, pns_norm, t_pns):
        
        # Update for loss plot (final_figure.png and train_figure.png), evaluation, checkpoint..
        self.grad_losses.append(grad_loss.detach().item())
        self.img_losses.append(img_loss.detach().item())
        self.slew_losses.append(slew_loss.detach().item())
        self.pns_losses.append(pns_loss.detach().item())
        self.total_losses.append(total_loss.detach().item())
        
        if step % 10 == 0:
            # --- Update Recon and Losses (for the image) ---
            recon_mirtorch = self.reconstructor.reconstruct_img(self.fft, rosette, method="mirtorch")
            image_loss_mirtorch = self.loss_fn(recon_mirtorch, self.phantom)
            self.img_losses_mirtorch.append(image_loss_mirtorch.detach().item())
            
            self.img_loss_line_mirtorch.set_data(range(0, 10 * len(self.img_losses_mirtorch), 10), self.img_losses_mirtorch)
            self.img_loss_line.set_data(range(len(self.img_losses)), self.img_losses)
            self.grad_loss_line.set_data(range(len(self.grad_losses)), self.grad_losses)
            self.pns_loss_line.set_data(range(len(self.pns_losses)), self.pns_losses)
            self.slew_loss_line.set_data(range(len(self.slew_losses)), self.slew_losses)
            self.total_loss_line.set_data(range(len(self.img_losses)), self.total_losses)
            
            self.ax_single_losses.relim()
            self.ax_single_losses.autoscale_view()
            self.ax_loss.relim()
            self.ax_loss.autoscale_view()
            
            img = recon.abs().detach().cpu().numpy()
            self.im_recon.set_data(img)
            self.im_recon.set_clim(vmin=0, vmax=img.max())
            
            # --- Update Trajectory ---
            self.traj_line.set_data(traj[:, 0].detach().cpu().numpy(), traj[:, 1].detach().cpu().numpy())
            self.ax_traj.relim()
            self.ax_traj.autoscale_view()
            
            """
            # --- Update Gradients (New) ---
            gx, gy, t_axis = compute_gradients_from_traj(traj, self.dt, self.gamma)
            """

            # 4. Plot (convert to numpy for plotting)
            self.gx_line.set_data(t_axis.detach().cpu().numpy(), gx.detach().cpu().numpy())
            self.gy_line.set_data(t_axis.detach().cpu().numpy(), gy.detach().cpu().numpy())
            self.ax_grad.relim()
            self.ax_grad.autoscale_view()

            self.ax_img.set_title(f"Recon (abs) Step {step+1}")
            
            """
            # --- Update PNS (New) ---
            pns_x, pns_y, pns_norm, t_pns = compute_pns_from_gradients(gx, gy, self.dt)
            """

            # 3. Plot PNS components and norm (convert to numpy for plotting)
            self.pns_x_line.set_data(t_pns.detach().cpu().numpy(), pns_x.detach().cpu().numpy())
            self.pns_y_line.set_data(t_pns.detach().cpu().numpy(), pns_y.detach().cpu().numpy())
            self.pns_norm_line.set_data(t_pns.detach().cpu().numpy(), pns_norm.detach().cpu().numpy())
            self.ax_pns.relim()
            self.ax_pns.autoscale_view()
            
            # 4. Track and plot maximum PNS norm over steps
            max_pns = pns_norm.max().item()
            self.max_pns_norms.append(max_pns)
            self.pns_norm_max_line.set_data(range(0, 10 * len(self.max_pns_norms), 10), self.max_pns_norms)
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
            print(f"Gradient:{1000 / self.gamma * d_max.max().item():.2f}")
            print(f"Slew Rate:{1000 / self.gamma * dd_max.max().item():.2f}")
            # Handle both tensor and scalar inputs for pns_max
            pns_val = pns_max.item() if torch.is_tensor(pns_max) else pns_max
            print(f"Max PNS Norm: {pns_val:.2f}%")
            print(f"Max Gx: {g_x.abs().max().item():.2f} mT/m")
            print(f"Max Gy: {g_y.abs().max().item():.2f} mT/m")

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


def get_rotation_matrix(n_petals, device=torch.device("cpu")):
    angle_radians = 2 * torch.pi / n_petals
    angle_radians = torch.tensor([angle_radians], device=device)
    rotation_matrix = torch.tensor(
        [
            [torch.cos(angle_radians), -torch.sin(angle_radians)],
            [torch.sin(angle_radians), torch.cos(angle_radians)],
        ],
        device=device
    )
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


def compute_pns_from_gradients(hw, gx, gy, dt, gradPreEmphPts=10, specRes=325, Ramp=False):
    """Compute PNS from gradient waveforms using the differentiable safe_gwf_to_pns_torch.
    
    gx, gy: (timesteps,) torch tensors of gradient waveforms in mT/m
    dt: time step size in ms
    Ramp: if True add ramp up/down, otherwise use only the repeated waveform
    
    Returns:
        pns_x, pns_y, pns_norm, t_pns
    """
    device = gx.device
    dtype = gx.dtype

    dt_seconds = dt / 1000  # Convert ms → s

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

    # Convert mT/m in T/m
    gVecX = gVecX * 1e-3
    gVecY = gVecY * 1e-3

    gVecZ = torch.zeros_like(gVecX)
    gVec = torch.stack([gVecX, gVecY, gVecZ], dim=1)  # (time, 3)

    rfVec = torch.ones(len(gVec), device=device, dtype=dtype)

    # Compute PNS
    _, res = safe_gwf_to_pns.safe_gwf_to_pns_torch(
        gVec, rfVec, dt_seconds, hw, do_padding=False
    )

    pns_x = res['pns'][:, 0]
    pns_y = res['pns'][:, 1]
    pns_norm = torch.norm(res['pns'], dim=1)

    t_pns = torch.arange(len(res['pns']), device=device, dtype=dtype) * dt_seconds * 1000  # ms

    return pns_x, pns_y, pns_norm, t_pns


def compute_fast_pns_from_gradients(hw, gx, gy, dt):
    """Compute PNS from gradient waveforms using the fast FFT-based method
    
    gx, gy: (timesteps,) torch tensors of gradient waveforms in mT/m
    dt: time step size in ms
    
    Returns:
        pns_x, pns_y, pns_norm, t_pns
    """
    device = gx.device
    dtype = gx.dtype

    dt_seconds = dt / 1000  # Convert ms → s

    gVecX = gx * 1e-3  # Convert mT/m to T/m
    gVecY = gy * 1e-3
    gVecZ = torch.zeros_like(gVecX)
    gVec = torch.stack([gVecX, gVecY, gVecZ], dim=1)  # (time, 3)

    rfVec = torch.ones(len(gVec), device=device, dtype=dtype)

    # Compute PNS
    _, res = safe_gwf_to_pns.fft_gwf_to_pns_torch(
        gVec, rfVec, dt_seconds, hw
    )

    pns_x = res['pns'][:, 0]
    pns_y = res['pns'][:, 1]
    pns_norm = torch.norm(res['pns'], dim=1)

    t_pns = torch.arange(len(res['pns']), device=device, dtype=dtype) * dt_seconds * 1000  # ms

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
