from utils import (
    get_phantom,
    make_rosette,
    grad_slew_loss,
    get_loss_fcn,
    sample_k_space_values,
    reconstruct_img,
    reconstruct_img2,
    TrainPlotter,
    final_plots,
    save_checkpoint,
)
import matplotlib.pyplot as plt
from mirtorch.linear import FFTCn
from nets import FourierCurve
import torch


path = "/Users/frankrosler/Desktop/PhD/Python/MIRTorch/results/2025-09-11_12-11/checkpoint.pt"
checkpoint = torch.load(path)
params = checkpoint["params"]
kmax_traj = params["res"] / (2 * params["FoV"])
kmax_img = params["img_size"] / (2 * params["FoV"])
final_FT_scaling = 4 / params["img_size"] ** 2


phantom = get_phantom(size=(params["img_size"], params["img_size"]), type="glpu")

# Compute FFT:
Fop = FFTCn(phantom.shape, phantom.shape, (0, 1), norm=None)
fft = Fop * phantom
fft = fft.reshape(1, 1, params["img_size"], params["img_size"])

t = torch.linspace(0, params["duration"], steps=params["timesteps"]).unsqueeze(1)  # (timesteps, 1)
model = FourierCurve(tmin=0, tmax=torch.max(t), initial_max=kmax_traj, n_coeffs=params["model_size"])

with torch.no_grad():
    traj = model(t)
    rosette, _ = make_rosette(traj, params["n_petals"], kmax_img, zero_filling=params["zero_filling"])
    rosette, sampled = sample_k_space_values(fft, rosette, kmax_img, params["zero_filling"])
    initial_recon = reconstruct_img2(rosette, sampled, params["img_size"], final_FT_scaling)

model.load_state_dict(checkpoint["model_state_dict"])

with torch.no_grad():
    traj = model(t)
    rosette, _ = make_rosette(traj, params["n_petals"], kmax_img, zero_filling=params["zero_filling"])
    rosette, sampled, _ = sample_k_space_values(fft, rosette, kmax_img, params["zero_filling"])
    recon = reconstruct_img2(rosette, sampled, params["img_size"], final_FT_scaling)

im1, im2, im3, im4, im5, im6 = final_plots(phantom, recon, initial_recon, [], traj, checkpoint["slew_rate"], show=False, export=True, export_path="")
im2.set_clim(0, 1)
im3.set_clim(0, 1)
im4.set_clim(0, 1)
plt.show()
