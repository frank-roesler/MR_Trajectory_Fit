from utils import (
    get_phantom,
    make_rosette,
    LossCollection,
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
Fop = FFTCn(phantom.shape, phantom.shape, (0, 1), norm=None)
fft = Fop * phantom
fft = fft.reshape(1, 1, params["img_size"], params["img_size"])
rotation_matrix = get_rotation_matrix(params["n_petals"], device=device).detach()
t = torch.linspace(0, params["duration"], steps=params["timesteps"]).unsqueeze(1).to(device)  # (timesteps, 1)

model = FourierCurve(tmin=0, tmax=params["duration"], initial_max=kmax_traj, n_coeffs=params["model_size"])
# model = Ellipse(tmin=0, tmax=params["duration"], initial_max=kmax_traj).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=params["lr"])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=300, factor=0.5, min_lr=1e-6, threshold=1e-6, cooldown=100)

loss_fcns = LossCollection(params["loss_function"])
reconstructor = ImageRecon(params, kmax_img, normalization, dcfnet="unet")
plotter = TrainPlotter(params["img_size"])

with torch.no_grad():
    rosette, _, _ = make_rosette(model(t), rotation_matrix, params["n_petals"], kmax_img, dt, zero_filling=params["zero_filling"])
    initial_recon = reconstructor.reconstruct_img(fft, rosette, method="kbnufft")

best_loss = float("inf")
for step in range(params["train_steps"]):
    traj = model(t)  # (timesteps, 2)
    rosette, d_max_rosette, dd_max_rosette = make_rosette(traj, rotation_matrix, params["n_petals"], kmax_img, dt, zero_filling=params["zero_filling"])
    recon = reconstructor.reconstruct_img(fft, rosette, method="kbnufft")

    grad_loss, slew_loss = loss_fcns.grad_slew_loss(d_max_rosette, dd_max_rosette, params, mode="exp")
    image_loss = loss_fcns.loss_fn(recon, phantom)
    total_loss = image_loss + grad_loss + slew_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    scheduler.step(total_loss.detach().item())

    plotter.print_info(step, params["train_steps"], image_loss, grad_loss, slew_loss, best_loss, optimizer)
    if total_loss.detach().item() < 0.999 * best_loss and step > 100:
        best_loss = total_loss.detach().item()
        slew_rate = save_checkpoint(export_path, model, d_max_rosette, dd_max_rosette, params, rosette)
        plotter.export_figure(export_path)
        final_plots(phantom, recon, initial_recon, plotter.total_losses, traj, slew_rate, show=False, export=True, export_path=export_path)

    plotter.update(step, grad_loss, image_loss, slew_loss, total_loss, recon, traj, phantom, fft, rosette, reconstructor, loss_fcns.loss_fn)


plotter.export_figure(export_path)

recon_mirtorch = reconstructor.reconstruct_img(fft, rosette, method="mirtorch")
final_plots(phantom, recon_mirtorch, initial_recon, plotter.total_losses, traj, slew_rate, export=True, export_path=export_path + "/mirtorch")
