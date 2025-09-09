import torch
import odl
import torch
import torch.nn.functional as F
from torchkbnufft import calc_density_compensation_function
from Bjork.sys_op import NuSense_om
from mirtorch.linear import NuSense
import matplotlib.pyplot as plt
import Nufftbindings.nufftbindings.kbnufft as kbnufft


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
    return rosette


def sample_k_space_values(fft, traj, kmax_img, zero_filling):
    traj = traj.reshape(1, 1, traj.shape[0], 2)
    img_size = fft.shape[-1]
    fft = fft.reshape(1, 1, img_size, img_size)
    sampled_r = F.grid_sample(fft.real, traj / kmax_img, mode="bicubic", align_corners=True)
    sampled_i = F.grid_sample(fft.imag, traj / kmax_img, mode="bicubic", align_corners=True)
    sampled = torch.complex(sampled_r, sampled_i).squeeze(0)
    if zero_filling:
        sampled[:, :, -2:] *= 0  # zero filling
    return traj, sampled, fft


def reconstruct_img(rosette, sampled, img_size):
    s0 = torch.ones(1, 1, img_size, img_size) + 0j
    rosette0 = rosette.squeeze().permute(1, 0) / torch.max(torch.abs(rosette)) * torch.pi
    k0 = sampled.reshape(1, 1, -1)
    dcf = calc_density_compensation_function(rosette0, (img_size, img_size))
    Nop = NuSense_om(s0, rosette0)
    # Nop = NuSense(s0, rosette0)
    I0 = Nop.H * (dcf * k0)
    return I0


def reconstruct_img2(rosette, sampled, img_size):
    rosette = rosette.squeeze().permute(1, 0) / torch.max(torch.abs(rosette)) * torch.pi
    k0 = sampled.reshape(1, 1, -1)
    dcf = calc_density_compensation_function(rosette, (img_size, img_size))
    rosette = rosette.permute(1, 0)
    kbnufft.nufft.set_dims(sampled.shape[-1], (img_size, img_size), torch.device("cpu"), Nb=1)
    kbnufft.nufft.precompute(rosette)
    I0 = kbnufft.adjoint(rosette, (k0 * dcf).squeeze(0))
    return I0.unsqueeze(0)


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


def grad_slew_loss(traj, dt, grad_max, slew_rate, gamma):
    """Compute a loss based on the slew rate of the trajectory.
    traj: (timesteps, 2) tensor
    dt: time step size (scalar)
    Returns: slew_loss (scalar)
    """
    grad, slew = compute_derivatives(traj, dt)
    grad_loss = threshold_loss(1000 / gamma * grad, grad_max)
    slew_loss = threshold_loss(1000 / gamma * slew, slew_rate)
    return grad_loss.sum(), slew_loss.sum()


def img_loss(img, target):
    return torch.mean((img - target) ** 2)


class TrainPlotter:
    def __init__(self, img_size):
        fig_loss, (ax_loss, ax_img, ax_traj) = plt.subplots(1, 3, figsize=(15, 4))
        (loss_line,) = ax_loss.semilogy([], [], label="Total Loss", linewidth=0.7)
        ax_loss.set_xlabel("Step")
        ax_loss.set_ylabel("Loss")
        ax_loss.set_title("Running Loss")
        ax_loss.legend()
        im_recon = ax_img.imshow(torch.zeros((img_size, img_size)).numpy(), cmap="gray")
        ax_img.set_title("Recon (abs)")
        ax_img.axis("off")
        (traj_line,) = ax_traj.plot([], [], label="trajectory", linewidth=0.7)
        ax_traj.set_title("Trajectory")
        plt.show(block=False)
        self.loss_line = loss_line
        self.im_recon = im_recon
        self.traj_line = traj_line
        self.ax_loss = ax_loss
        self.ax_traj = ax_traj
        self.ax_img = ax_img

    def update(self, step, losses, recon, traj):
        self.loss_line.set_data(range(len(losses)), losses)
        self.ax_loss.relim()
        self.ax_loss.autoscale_view()
        img = recon.abs().detach().cpu().numpy()
        self.im_recon.set_data(img)
        self.im_recon.set_clim(vmin=0, vmax=1)
        self.traj_line.set_data(traj[:, 0].detach().numpy(), traj[:, 1].detach().numpy())
        self.ax_traj.relim()
        self.ax_traj.autoscale_view()
        self.ax_img.set_title(f"Recon (abs) Step {step+1}")
        plt.pause(0.01)
