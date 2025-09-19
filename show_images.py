from utils import (
    get_phantom,
    make_rosette,
    ImageRecon,
    final_plots,
    get_rotation_matrix,
    plot_pixel_rosette,
    compute_derivatives,
    Checkpointer,
)
import matplotlib.pyplot as plt
from mirtorch.linear import FFTCn
from models import FourierCurve
import torch
from os.path import join, dirname


recon_method = "mirtorch"
path = "results/2025-09-18_22-31"
checkpointer = Checkpointer(path, {}, None)
checkpoint = checkpointer.load_checkpoint()
params = checkpoint["params"]
kmax_traj = params["res"] / (2 * params["FoV"])  # 1/m
kmax_img = params["img_size"] / (2 * params["FoV"])  # 1/m
normalization = 4 / params["img_size"] ** 2

params["timesteps"] = params["timesteps"] // 2 + 1
dt = params["duration"] / (params["timesteps"] - 1)
checkpointer.dt = dt

phantom = get_phantom(size=(params["img_size"], params["img_size"]), type="glpu")

Fop = FFTCn(phantom.shape, phantom.shape, (0, 1), norm=None)
fft = Fop * phantom
fft = fft.reshape(1, 1, params["img_size"], params["img_size"])
rotation_matrix = get_rotation_matrix(params["n_petals"])

t = torch.linspace(0, params["duration"], steps=params["timesteps"]).unsqueeze(1)  # (timesteps, 1)
model = FourierCurve(tmin=0, tmax=params["duration"], initial_max=kmax_traj, n_coeffs=params["model_size"])
reconstructor = ImageRecon(params, kmax_img, normalization, dcfnet="unet")

with torch.no_grad():
    rosette, _, _ = make_rosette(model(t), rotation_matrix, params["n_petals"], kmax_img, dt, zero_filling=params["zero_filling"])
    initial_recon = reconstructor.reconstruct_img(fft, rosette, method=recon_method)

model.load_state_dict(checkpoint["model_state_dict"])

with torch.no_grad():
    traj = model(t)  # (timesteps, 2)
    rosette, d, dd = make_rosette(traj, rotation_matrix, params["n_petals"], kmax_img, dt, zero_filling=params["zero_filling"])
    recon = reconstructor.reconstruct_img(fft, rosette, method=recon_method)

checkpointer.export_json(rosette)


im1, im2, im3, im4, im5, im6 = final_plots(phantom, recon, initial_recon, [], traj, checkpoint["slew_rate"], show=False, export=True, export_path=join(dirname(path), recon_method))
# im2.set_clim(0, 1)
# im3.set_clim(0, 1)
# im4.set_clim(0, 1)

# rosette, sampled = reconstructor.sample_k_space_values(fft, rosette)
# sampled_from_pixels = plot_pixel_rosette(rosette / kmax_img, fft, phantom.shape[-1])
# plt.figure()
# plt.plot(sampled.squeeze().detach().cpu(), label="F.grid_sample", linewidth=0.7)
# plt.plot(sampled_from_pixels.squeeze().detach().cpu(), label="Indexing", linewidth=0.7)
# plt.legend()

# fig, ax = plt.subplots(1, 2, figsize=(11, 5))
# ax[0].plot(rosette[0, 0, :-2, 0].detach().cpu(), rosette[0, 0, :-2, 1].detach().cpu(), linewidth=0.7)
# ax[1].plot(traj[:, 0].detach().cpu(), traj[:, 1].detach().cpu(), linewidth=0.7, marker=".", markersize=3)
# ax[1].set_aspect("equal", "box")
# ax[0].set_aspect("equal", "box")

# fig, ax = plt.subplots(1, 2)
# im = ax[0].imshow(phantom[:, :].numpy(), cmap="gray")
# ax[0].set_title("Phantom")
# ax[0].axis("off")
# fig.colorbar(im, ax=ax[0])
# ax[1].imshow(torch.log(torch.abs(fft.squeeze()) + 1e-9).numpy(), cmap="gray")
# ax[1].set_title("Log FFT Magnitude")
# ax[1].axis("off")
# plt.show()
