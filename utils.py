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


def get_phantom(size=(1024, 1024)):
    """Generate a Shepp-Logan phantom as a PyTorch tensor."""
    phantom = odl.phantom.shepp_logan(odl.uniform_discr([0, 0], [1, 1], size), modified=True)
    phantom_np = phantom.asarray()
    phantom_tensor = torch.from_numpy(phantom_np).float()
    return phantom_tensor


def rotate_trajectory(traj, angle_radians):
    angle_radians = torch.Tensor([angle_radians])
    rotation_matrix = torch.tensor([[torch.cos(angle_radians), -torch.sin(angle_radians)], [torch.sin(angle_radians), torch.cos(angle_radians)]])
    rotated_traj = traj @ rotation_matrix.T
    return rotated_traj


def make_rosette(traj, n_petals, kmax_img, zero_filling=True):
    rotated_trajectories = [traj]
    angle = 2 * torch.pi / n_petals
    for i in range(n_petals - 1):
        traj_tmp = rotate_trajectory(traj, angle)
        rotated_trajectories.append(traj_tmp)
        traj = traj_tmp
    if zero_filling:
        corners = torch.ones(2, 2)
        corners[0] *= kmax_img
        corners[1] *= -kmax_img
        rotated_trajectories.append(corners)
    rosette = torch.cat(rotated_trajectories, dim=0)
    max_norm = traj.norm(dim=1).max()
    return rosette, max_norm


def sample_k_space_values(fft, rosette, kmax_img, zero_filling):
    rosette = rosette.reshape(1, 1, rosette.shape[0], 2)
    img_size = fft.shape[-1]
    fft = fft.reshape(1, 1, img_size, img_size)
    sampled_r = F.grid_sample(fft.real, rosette / kmax_img, mode="bicubic", align_corners=True)
    sampled_i = F.grid_sample(fft.imag, rosette / kmax_img, mode="bicubic", align_corners=True)
    sampled = torch.complex(sampled_r, sampled_i).squeeze(0)
    if zero_filling:
        sampled[:, :, -2:] *= 0  # zero filling
    return rosette, sampled, fft


def reconstruct_img(rosette, sampled, img_size, scaling):
    s0 = torch.ones(1, 1, img_size, img_size) + 0j
    rosette0 = rosette.squeeze().permute(1, 0) / torch.max(torch.abs(rosette)) * torch.pi
    k0 = sampled.reshape(1, 1, -1)
    dcf = calc_density_compensation_function(rosette0, (img_size, img_size))
    Nop = NuSense(s0, rosette0, norm=None)
    I0 = Nop.H * (dcf * k0)
    I0 = torch.flip(torch.rot90(I0.abs(), k=1, dims=(2, 3)), dims=[2]).squeeze()
    return I0 * scaling


def reconstruct_img2(rosette, sampled, img_size, scaling):
    rosette = rosette.squeeze().permute(1, 0) / torch.max(torch.abs(rosette)) * torch.pi
    k0 = sampled.reshape(1, 1, -1)
    dcf = calc_density_compensation_function(rosette, (img_size, img_size))
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
    traj = traj[:-1, :]
    d_traj = (torch.roll(traj, shifts=-1, dims=0) - traj) / dt
    dd_traj = (torch.roll(d_traj, shifts=-1, dims=0) - d_traj) / dt
    return d_traj, dd_traj


def threshold_loss(x, threshold):
    threshold_loss = torch.max(torch.abs(x), dim=0).values - threshold
    threshold_loss[threshold_loss < 0] = 0.0
    return threshold_loss**2


def grad_slew_loss(traj, dt, grad_max, slew_rate, gamma, grad_loss_weight, slew_loss_weight):
    """Compute a loss based on the slew rate of the trajectory.
    traj: (timesteps, 2) tensor
    dt: time step size (scalar)
    Returns: slew_loss (scalar)
    """
    grad, slew = compute_derivatives(traj, dt)
    grad_loss = threshold_loss(1000 / gamma * grad, grad_max)
    slew_loss = threshold_loss(1000 / gamma * slew, slew_rate)
    return grad_loss_weight * grad_loss.sum(), slew_loss_weight * slew_loss.sum()


def mse_loss(img, target):
    return torch.mean((img - target) ** 2)


class MySSIMLoss(nn.Module):
    def __init__(self, window_size=11, reduction="mean", max_val=1.0):
        super(MySSIMLoss, self).__init__()
        self.ssim = SSIMLoss(window_size=window_size, reduction=reduction, max_val=max_val)
        self.L1_loss = nn.L1Loss()

    def forward(self, img, target):
        return 1.0 - self.ssim(img.unsqueeze(0).unsqueeze(0), target.unsqueeze(0).unsqueeze(0)) + 0.1 * self.L1_loss(img, target)


class TrainPlotter:
    def __init__(self, img_size):
        plt.rc("xtick", labelsize=8)
        plt.rc("ytick", labelsize=8)
        fig, (ax_loss, ax_img, ax_traj) = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
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
        (traj_line,) = ax_traj.plot([], [], label="trajectory", linewidth=0.7)
        ax_traj.set_title("Trajectory")
        fig.colorbar(im_recon, ax=ax_img)
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
        if step % 10 == 0:
            self.img_loss_line.set_data(range(len(self.img_losses)), self.img_losses)
            self.grad_loss_line.set_data(range(len(self.grad_losses)), self.grad_losses)
            self.slew_loss_line.set_data(range(len(self.slew_losses)), self.slew_losses)
            self.total_loss_line.set_data(range(len(self.img_losses)), self.total_losses)
            self.ax_total_loss.relim()
            self.ax_total_loss.autoscale_view()
            self.ax_loss.set_ylim(0.9 * min(self.img_losses), 1.1 * max(self.img_losses))
            # self.ax_loss.set_xlim(0, len(self.img_losses))
            # self.ax_total_loss.set_xlim(0, len(self.img_losses))
            img = recon.abs().detach().cpu().numpy()
            self.im_recon.set_data(img)
            self.im_recon.set_clim(vmin=0, vmax=1)
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


def plot_pixel_rosette(traj, fft, img_size, ax=None):
    # Sample FFT at trajectory locations:
    sampled_coord_x = torch.round((traj[0, 0, :, 0] / 2 - 1) * (img_size - 1) + img_size / 2)
    sampled_coord_y = torch.round((traj[0, 0, :, 1] / 2 - 1) * (img_size - 1) + img_size / 2)
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


def final_plots(phantom, recon, initial_recon, losses, rosette, kmax_img, final_FT_scaling, fft):
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    im = ax[0, 0].imshow(phantom.numpy(), cmap="gray")
    ax[0, 0].set_title("Phantom")
    ax[0, 0].axis("off")
    fig.colorbar(im, ax=ax[0, 0])

    im = ax[0, 1].imshow(recon.abs().detach().numpy(), cmap="gray")
    ax[0, 1].set_title("Recon")
    ax[0, 1].axis("off")
    fig.colorbar(im, ax=ax[0, 1])

    im = ax[0, 2].imshow(initial_recon.squeeze().detach().numpy(), cmap="gray")
    ax[0, 2].set_title("Initial Recon")
    ax[0, 2].axis("off")
    fig.colorbar(im, ax=ax[0, 2])

    im = ax[1, 0].imshow((recon.abs() - phantom).detach().numpy(), cmap="gray")
    ax[1, 0].set_title("Phantom - Recon")
    ax[1, 0].axis("off")
    fig.colorbar(im, ax=ax[1, 0])

    im = ax[1, 1].semilogy(range(len(losses)), losses, linewidth=0.7)
    ax[1, 1].set_title("Loss")

    ax[1, 2].set_title("Pixel rosette (zoomed)")
    ax[1, 2].axis("off")
    sampled_from_pixels = plot_pixel_rosette(rosette / kmax_img, fft, phantom.shape[-1], ax=ax[1, 2])
    plt.show()
    return sampled_from_pixels
