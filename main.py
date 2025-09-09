from utils import (
    get_phantom,
    make_rosette,
    grad_slew_loss,
    mse_loss,
    MySSIMLoss,
    sample_k_space_values,
    reconstruct_img,
    reconstruct_img2,
    TrainPlotter,
)
import matplotlib.pyplot as plt
from mirtorch.linear import FFTCn
from nets import FourierCurve
import torch
from params import *
from aux.plot_pixel_rosette import plot_pixel_rosette


phantom = get_phantom(size=(img_size, img_size))

# Compute FFT:
Fop = FFTCn(phantom.shape, phantom.shape, (0, 1), norm="ortho")
fft = Fop * phantom

t = torch.linspace(0, duration, steps=timesteps).unsqueeze(1)  # (timesteps, 1)
model = FourierCurve(tmin=0, tmax=torch.max(t), initial_max=kmax_traj, n_coeffs=model_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
img_loss = mse_loss
# img_loss = MySSIMLoss(window_size=11, reduction="mean", max_val=1.0)

with torch.no_grad():
    traj = model(t)
    rosette = make_rosette(traj, n_petals, kmax_img, zero_filling=zero_filling)
    rosette, sampled, _ = sample_k_space_values(fft, rosette, kmax_img, zero_filling)
    initial_recon = reconstruct_img2(rosette, sampled, img_size)

losses = []
plotter = TrainPlotter(img_size)
for step in range(train_steps):
    traj = model(t)  # (timesteps, 2)
    rosette = make_rosette(traj, n_petals, kmax_img, zero_filling=zero_filling)
    grad_loss, slew_loss = grad_slew_loss(traj, dt, grad_max, slew_rate, gamma)

    rosette, sampled, fft = sample_k_space_values(fft, rosette, kmax_img, zero_filling)
    recon = reconstruct_img2(rosette, sampled, img_size)
    recon = 0.01 * torch.flip(torch.rot90(recon.abs(), k=1, dims=(2, 3)), dims=[2]).squeeze()
    print(f"Recon max: {torch.max(recon)}")
    print(f"Recon min: {torch.min(recon)}")

    image_loss = img_loss(recon, phantom)
    total_loss = image_loss + 1e-6 * grad_loss + 1e-7 * slew_loss

    losses.append(total_loss.item())
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if step % 10 == 0:
        plotter.update(step, losses, recon, traj)
    print(f"Step {step+1}/{train_steps}")
    print(f"  Image loss: {image_loss.item():.6f}")
    print(f"  Gradient loss: {grad_loss.item():.6f}")
    print(f"  Slew rate loss: {slew_loss.item():.6f}")
    print(f"  Total loss: {total_loss.item():.6f}")


# region Plot results:
fig, ax = plt.subplots(2, 3, figsize=(15, 8))
im = ax[0, 0].imshow(phantom.numpy(), cmap="gray")
ax[0, 0].set_title("Phantom")
ax[0, 0].axis("off")
fig.colorbar(im, ax=ax[0, 0])

im = ax[0, 1].imshow(recon.abs().detach().numpy(), cmap="gray")
ax[0, 1].set_title("Recon")
ax[0, 1].axis("off")
fig.colorbar(im, ax=ax[0, 1])

im = ax[0, 2].imshow(0.01 * torch.flip(torch.rot90(initial_recon.abs(), k=1, dims=(2, 3)), dims=[2]).squeeze().detach().numpy(), cmap="gray")
ax[0, 2].set_title("Initial Recon")
ax[0, 2].axis("off")
fig.colorbar(im, ax=ax[0, 2])

im = ax[1, 0].imshow((recon.abs() - phantom).detach().numpy(), cmap="gray")
ax[1, 0].set_title("Phantom - Recon")
ax[1, 0].axis("off")
fig.colorbar(im, ax=ax[1, 0])

im = ax[1, 1].semilogy(range(len(losses)), losses)
ax[1, 1].set_title("Loss")
ax[1, 1].axis("off")

ax[1, 2].set_title("Pixel rosette")
ax[1, 2].axis("off")
sampled_from_pixels = plot_pixel_rosette(rosette / kmax_img, fft, img_size, ax=ax[1, 2])
plt.show()

# plt.plot(sampled.squeeze().detach().numpy(), label="F.grid_sample", linewidth=0.7)
# plt.plot(sampled_from_pixels.squeeze().detach().numpy(), label="Indexing", linewidth=0.7)
# plt.legend()
# plt.show()

fig, ax = plt.subplots(1, 2, figsize=(11, 5))
ax[0].plot(rosette[0, 0, :-2, 0].detach().numpy(), rosette[0, 0, :-2, 1].detach().numpy(), linewidth=0.7)
ax[1].plot(traj[:, 0].detach().numpy(), traj[:, 1].detach().numpy(), linewidth=0.7)
plt.show()

# fig, ax = plt.subplots(1, 2)
# im = ax[0].imshow(phantom[:, :].numpy(), cmap="gray")
# ax[0].set_title("Phantom")
# ax[0].axis("off")
# fig.colorbar(im, ax=ax[0])
# ax[1].imshow(torch.log(torch.abs(fft.squeeze()) + 1e-9).numpy(), cmap="gray")
# ax[1].set_title("Log FFT Magnitude")
# ax[1].axis("off")
# plt.show()
# endregion
