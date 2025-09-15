from mrinufft.operators.interfaces.torchkbnufft import TorchKbNUFFTcpu


from utils import (
    get_phantom,
    make_rosette,
    grad_slew_loss,
    get_loss_fcn,
    sample_k_space_values,
    reconstruct_img,
    reconstruct_img2,
    reconstruct_img_nudft,
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

phantom = get_phantom(size=(params["img_size"], params["img_size"]), type="shepp_logan").to(device)

# Compute FFT:
Fop = FFTCn(phantom.shape, phantom.shape, (0, 1), norm=None)
fft = Fop * phantom
fft = fft.reshape(1, 1, params["img_size"], params["img_size"])
rotation_matrix = get_rotation_matrix(params["n_petals"], device=device).detach()

t = torch.linspace(0, params["duration"], steps=params["timesteps"]).unsqueeze(1).to(device)  # (timesteps, 1)

model = Ellipse(tmin=0, tmax=params["duration"], initial_max=kmax_traj)

model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=params["lr"])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=200, factor=0.5, min_lr=1e-6, threshold=1e-6, cooldown=100)
img_loss = get_loss_fcn(params["loss_function"])

with torch.no_grad():
    traj = model(t)
    rosette, d_max_rosette, dd_max_rosette = make_rosette(traj, rotation_matrix, params["n_petals"], kmax_img, dt, zero_filling=params["zero_filling"])
    rosette, sampled = sample_k_space_values(fft, rosette, kmax_img, params["zero_filling"])
    rosette = rosette.squeeze().permute(1, 0)
    nufft = TorchKbNUFFTcpu(rosette, (params["img_size"], params["img_size"]), density=True, smaps=None, squeeze_dims=True)
    plt.figure()
    plt.plot(nufft.density)
    plt.show()
