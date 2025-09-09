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
    final_plots,
    compute_derivatives,
)
import matplotlib.pyplot as plt
from mirtorch.linear import FFTCn
from nets import FourierCurve
import torch
from params import *


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
    rosette, kmax_traj = make_rosette(traj, n_petals, kmax_img, zero_filling=zero_filling)
    rosette, sampled, _ = sample_k_space_values(fft, rosette, kmax_img, zero_filling)
    initial_recon = reconstruct_img2(rosette, sampled, img_size, final_FT_scaling)


plotter = TrainPlotter(img_size)
best_loss = float("inf")
for step in range(10 * train_steps):
    traj = model(t)  # (timesteps, 2)
    grad_loss, slew_loss = grad_slew_loss(traj, dt, grad_max, slew_rate, gamma, grad_loss_weight, slew_loss_weight)
    rosette, kmax_traj = make_rosette(traj, n_petals, kmax_img, zero_filling=zero_filling)

    rosette, sampled, fft = sample_k_space_values(fft, rosette, kmax_img, zero_filling)
    recon = reconstruct_img2(rosette, sampled, img_size, 2 * math.sqrt(2 * img_size / (kmax_traj * 2 * FoV)))

    image_loss = img_loss(recon, phantom)
    total_loss = image_loss + grad_loss + slew_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    plotter.print_info(step, train_steps, image_loss.item(), grad_loss.item(), slew_loss.item(), best_loss)
    if total_loss.item() < best_loss:
        if step >= train_steps:
            break
        best_loss = total_loss.item()

    plotter.update(step, grad_loss.item(), image_loss.item(), slew_loss.item(), total_loss.item(), recon, traj)

grad, slew = compute_derivatives(traj, dt)
print("=" * 100)
print("SLEW RATE:", 1000 / gamma * slew.max(dim=0).values)
print("=" * 100)

# region Plot results:
sampled_from_pixels = final_plots(
    phantom,
    recon,
    initial_recon,
    plotter.total_losses,
    rosette,
    kmax_img,
    final_FT_scaling,
    fft,
)

# plt.plot(sampled.squeeze().detach().numpy(), label="F.grid_sample", linewidth=0.7)
# plt.plot(sampled_from_pixels.squeeze().detach().numpy(), label="Indexing", linewidth=0.7)
# plt.legend()
# plt.show()

fig, ax = plt.subplots(1, 2, figsize=(11, 5))
ax[0].plot(rosette[0, 0, :-2, 0].detach().numpy(), rosette[0, 0, :-2, 1].detach().numpy(), linewidth=0.7)
ax[1].plot(traj[:, 0].detach().numpy(), traj[:, 1].detach().numpy(), linewidth=0.7)
ax[1].set_aspect("equal", "box")
ax[0].set_aspect("equal", "box")
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
