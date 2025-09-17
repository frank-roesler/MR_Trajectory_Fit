from utils import (
    get_phantom,
    make_rosette,
    grad_slew_loss,
    get_loss_fcn,
    ImageRecon,
    TrainPlotter,
    final_plots,
    save_checkpoint,
    get_rotation_matrix,
)
import matplotlib.pyplot as plt
from mirtorch.linear import FFTCn
from models import FourierCurve, Ellipse
import torch
from params import *

torch.set_printoptions(threshold=100000)

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

phantom = get_phantom(size=(params["img_size"], params["img_size"]), type="glpu").to(device)

# Compute FFT:
Fop = FFTCn(phantom.shape, phantom.shape, (0, 1), norm=None)
fft = Fop * phantom
fft = fft.reshape(1, 1, params["img_size"], params["img_size"])
rotation_matrix = get_rotation_matrix(params["n_petals"], device=device).detach()

t = torch.linspace(0, params["duration"], steps=params["timesteps"]).unsqueeze(1).to(device)  # (timesteps, 1)

model = FourierCurve(tmin=0, tmax=params["duration"], initial_max=kmax_traj, n_coeffs=params["model_size"])
# model = Ellipse(tmin=0, tmax=params["duration"], initial_max=kmax_traj).to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=params["lr"])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=200, factor=0.5, min_lr=1e-6, threshold=1e-6, cooldown=100)
img_loss = get_loss_fcn(params["loss_function"])

reconstructor = ImageRecon(params, kmax_img, normalization, dcfnet="unet", method="kbnufft")
plotter = TrainPlotter(params["img_size"])

with torch.no_grad():
    traj = model(t)
    rosette, d_max_rosette, dd_max_rosette = make_rosette(traj, rotation_matrix, params["n_petals"], kmax_img, dt, zero_filling=params["zero_filling"])
    initial_recon = reconstructor.reconstruct_img(fft, rosette)

best_loss = float("inf")
for step in range(params["train_steps"]):
    traj = model(t)  # (timesteps, 2)
    rosette, d_max_rosette, dd_max_rosette = make_rosette(traj, rotation_matrix, params["n_petals"], kmax_img, dt, zero_filling=params["zero_filling"])

    grad_loss, slew_loss = grad_slew_loss(d_max_rosette, dd_max_rosette, params, mode="exp")

    recon = reconstructor.reconstruct_img(fft, rosette)
    image_loss = img_loss(recon, phantom)
    total_loss = image_loss + grad_loss + slew_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    scheduler.step(total_loss.detach().item())

    plotter.print_info(step, params["train_steps"], image_loss.detach().item(), grad_loss.detach().item(), slew_loss.detach().item(), best_loss, optimizer)
    if total_loss.detach().item() < 0.99 * best_loss and step > 50:
        best_loss = total_loss.detach().item()
        slew_rate = save_checkpoint(export_path, model, d_max_rosette, dd_max_rosette, params)
        plotter.export_figure(export_path)
        final_plots(phantom, recon, initial_recon, plotter.total_losses, traj, slew_rate, show=False, export=True, export_path=export_path)

    plotter.update(step, grad_loss.detach().item(), image_loss.detach().item(), slew_loss.detach().item(), total_loss.detach().item(), recon, traj)


plotter.export_figure(export_path)

# region Plot results:
reconstructor = ImageRecon(params, kmax_img, normalization, dcfnet="unet", method="mirtorch")
recon = reconstructor.reconstruct_img(fft, rosette)
final_plots(phantom, recon, initial_recon, plotter.total_losses, traj, slew_rate, export=True, export_path=export_path + "/mirtorch")

# sampled_from_pixels = plot_pixel_rosette(rosette / kmax_img, fft, phantom.shape[-1], ax=ax[1, 2])

# plt.plot(sampled.squeeze().detach().cpu().numpy(), label="F.grid_sample", linewidth=0.7)
# plt.plot(sampled_from_pixels.squeeze().detach().cpu().numpy(), label="Indexing", linewidth=0.7)
# plt.legend()
# plt.show()

# fig, ax = plt.subplots(1, 2, figsize=(11, 5))
# ax[0].plot(rosette[0, :-2, 0].detach().cpu().numpy(), rosette[0, 0, :-2, 1].detach().cpu().numpy(), linewidth=0.7)
# ax[1].plot(traj[:, 0].detach().cpu().numpy(), traj[:, 1].detach().cpu().numpy(), linewidth=0.7)
# ax[1].set_aspect("equal", "box")
# ax[0].set_aspect("equal", "box")
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
# endregion
