from utils import get_phantom, rotate_trajectory
import matplotlib.pyplot as plt
from mirtorch.linear import FFTCn, NuSense
from nets import FourierCurve
import torch
import torch.nn.functional as F
from torchkbnufft import calc_density_compensation_function
import math
from params import *


rel_traj_size = kmax_traj / kmax_img

phantom = get_phantom(size=(img_size, img_size))

# Compute FFT:
Fop = FFTCn(phantom.shape, phantom.shape, (0, 1), norm="ortho")
fft = Fop * phantom

# Make rosette trajectory:
fc = FourierCurve(tmin=0, tmax=1, n_coeffs=model_size)
t = torch.linspace(0, duration, steps=timesteps).unsqueeze(1)  # (timesteps, 1)
traj = fc(t)  # (timesteps, 2)
rotated_trajectories = [traj]
angle = 360 / n_petals
for i in range(n_petals - 1):
    traj = rotate_trajectory(traj, angle)
    rotated_trajectories.append(traj)

# Zero filling:
corners = torch.ones(2, 2)
corners[0] *= kmax_img / kmax_traj
corners[1] *= -kmax_img / kmax_traj
rotated_trajectories.append(corners)
traj = torch.cat(rotated_trajectories, dim=0)

# Sample FFT at trajectory locations:
traj = rel_traj_size * traj.reshape(1, 1, traj.shape[0], 2)
fft = fft.reshape(1, 1, img_size, img_size)
sampled_r = F.grid_sample(fft.real, traj, mode="bicubic", align_corners=True)
sampled_i = F.grid_sample(fft.imag, traj, mode="bicubic", align_corners=True)
sampled = torch.complex(sampled_r, sampled_i).squeeze(0)
sampled[:, :, -2:] *= 0  # zero filling

sampled_coord_x = torch.round((traj[0, 0, :, 0] / 2 - 1) * (img_size - 1) + img_size / 2)
sampled_coord_y = torch.round((traj[0, 0, :, 1] / 2 - 1) * (img_size - 1) + img_size / 2)
pixel_curve = torch.zeros(img_size, img_size)
pixel_curve[sampled_coord_y.long(), sampled_coord_x.long()] = 1.0
sampled_from_pixels = fft[0, 0, sampled_coord_y.long(), sampled_coord_x.long()]
sampled_from_pixels[-2:] *= 0


# Reconstruction with NUFFT:
s0 = torch.ones(1, 1, img_size, img_size) + 0j
traj0 = traj.squeeze().permute(1, 0) / torch.max(torch.abs(traj)) * torch.pi
k0 = sampled.reshape(1, 1, -1)
dcf = calc_density_compensation_function(traj0, (img_size, img_size))
Nop = NuSense(s0, traj0)
I0 = Nop.H * (dcf * k0)


I0 = torch.flip(torch.rot90(I0.squeeze(), k=1, dims=(0, 1)), dims=[0]) * math.sqrt(2 * img_size / res) * 2
# phantom_resized = F.interpolate(phantom.unsqueeze(0).unsqueeze(0), size=(img_size, img_size), mode="bilinear", align_corners=False).squeeze()
fig, ax = plt.subplots(2, 3)
im = ax[0, 0].imshow(phantom.numpy(), cmap="gray")
ax[0, 0].set_title("Phantom")
ax[0, 0].axis("off")
fig.colorbar(im, ax=ax[0, 0])
im = ax[0, 1].imshow(I0.abs().detach().numpy(), cmap="gray")
ax[0, 1].set_title("Recon")
ax[0, 1].axis("off")
fig.colorbar(im, ax=ax[0, 1])
im = ax[0, 2].imshow((I0.abs() - phantom).detach().numpy(), cmap="gray")
ax[0, 2].set_title("Phantom - Recon")
ax[0, 2].axis("off")
fig.colorbar(im, ax=ax[0, 2])
im = ax[1, 1].imshow(torch.angle(I0).detach().numpy(), cmap="gray")
ax[1, 1].set_title("Phase")
ax[1, 1].axis("off")
fig.colorbar(im, ax=ax[1, 1])
plt.show()

plt.imshow(pixel_curve.detach().numpy())
plt.show()

# plt.plot(sampled.squeeze().detach().numpy(), label="F.grid_sample", linewidth=0.7)
# plt.plot(sampled_from_pixels.squeeze().detach().numpy(), label="Indexing", linewidth=0.7)
# plt.legend()
# plt.show()

# plt.plot(traj[0, 0, :, 0].detach().numpy(), traj[0, 0, :, 1].detach().numpy(), linewidth=0.7)
# plt.show()

# fig, ax = plt.subplots(1, 2)
# im = ax[0].imshow(phantom[:, :].numpy(), cmap="gray")
# ax[0].set_title("Phantom")
# ax[0].axis("off")
# fig.colorbar(im, ax=ax[0])
# ax[1].imshow(torch.log(torch.abs(fft.squeeze()) + 1e-9).numpy(), cmap="gray")
# ax[1].set_title("Log FFT Magnitude")
# ax[1].axis("off")
# plt.show()
