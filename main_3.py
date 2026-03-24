from utils_3 import (
    LossCollection,
    ImageRecon,
    TrainPlotter,
    Checkpointer,
    compute_gradients_from_traj,
    compute_pns_from_gradients,
    get_phantom,
    get_batch_of_phantoms,
    make_rosette,
    final_plots,
    get_rotation_matrix,
    get_device,
    psf,
)
import matplotlib.pyplot as plt
from mirtorch.linear import FFTCn
from models import FourierCurve, Ellipse
import torch
from params import *
from safe_gwf_to_pns import SAFE_PNS

torch.set_printoptions(threshold=100000)

device = get_device()

batch_size = 2
phantoms = get_batch_of_phantoms(batch_size, size=(params["img_size"], params["img_size"]), type="glpu").to(device)
Fop = FFTCn(phantoms.shape[-2:], phantoms.shape[-2:], (0, 1), norm=None)
fft = torch.stack([Fop * phantoms[b] for b in range(batch_size)], dim=0)  # Image -> k-space
fft = fft.unsqueeze(1)
rotation_matrix = get_rotation_matrix(params["n_petals"], device=device).detach()
t = torch.linspace(0, params["duration"], steps=params["timesteps"], device=device).unsqueeze(1)  # (timesteps, 1)

model = FourierCurve(tmin=0, tmax=params["duration"], initial_max=kmax_traj, n_coeffs=params["model_size"], coeff_lvl=1e-2).to(device)  # 1e-2
# model = Ellipse(tmin=0, tmax=params["duration"], initial_max=kmax_traj).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1000, factor=0.5, min_lr=1e-6, threshold=1e-6, cooldown=100)

loss_fcns = LossCollection(params["loss_function"])
reconstructor = ImageRecon(params, kmax_img, normalization, dcfnet="unet")  # k-space -> Image
plotter = TrainPlotter(params, fft, reconstructor, phantoms, loss_fcns.loss_fn, optimizer)
checkpointer = Checkpointer(export_path, params, dt)
safe_model = SAFE_PNS(dt=dt, hw_path="safe_pns_prediction/MP_GradSys_K2298_2250V_1250A_W60_SC72CD.asc", mode="full")
best_rosette = None  # for PSF analysis at the end
best_traj = None
best_slew_rate = None

with torch.no_grad():
    rosette, _, _ = make_rosette(model(t), rotation_matrix, params["n_petals"], kmax_img, dt, zero_filling=params["zero_filling"])
    rosette_init = rosette.clone().to(device)  # for later PSF analysis
    initial_recon = reconstructor.reconstruct_img(fft, rosette, method="kbnufft")


# for step in range(params["train_steps"]):
for step in range(1000):
    traj = model(t)  # (timesteps, 2)
    rosette, *derivatives = make_rosette(traj, rotation_matrix, params["n_petals"], kmax_img, dt, zero_filling=params["zero_filling"])
    recon = reconstructor.reconstruct_img(fft, rosette, method="kbnufft")
    # Compute PNS from gradients - fully differentiable
    gx, gy, t_axis = compute_gradients_from_traj(traj, dt, params["gamma"])
    pns_x, pns_y, pns_norm, t_pns = compute_pns_from_gradients(safe_model, gx, gy)
    max_pns = pns_norm.max()

    # Losses
    # pns_loss = loss_fcns.pns_loss(max_pns, params, mode=args.pns_mode, delta=1)
    pns_loss = loss_fcns.pns_loss(max_pns, params, mode="threshold")
    # grad_loss, slew_loss = loss_fcns.grad_slew_loss(*derivatives, params, grad_mode="exp", slew_mode=args.slew_mode, delta=1)
    grad_loss, slew_loss = loss_fcns.grad_slew_loss(*derivatives, params, grad_mode="exp", slew_mode="threshold")
    image_loss = loss_fcns.loss_fn(recon, phantoms)
    total_loss = image_loss + grad_loss + slew_loss + pns_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    scheduler.step(total_loss.detach().item())

    plotter.print_info(step, image_loss, grad_loss, slew_loss, pns_loss, *derivatives, max_pns, gx, gy)
    if total_loss.detach().item() < 0.999 * plotter.best_loss and step > 100:
        plotter.best_loss = total_loss.detach().item()
        best_rosette = rosette.detach().clone()
        best_traj = traj.detach().clone()
        slew_rate = checkpointer.save_checkpoint(model, *derivatives, rosette)
        best_slew_rate = slew_rate.detach().clone()
        plotter.export_figure(export_path)
    plotter.update(step, grad_loss, image_loss, slew_loss, pns_loss, total_loss, recon, traj, rosette, gx, gy, t_axis, pns_x, pns_y, pns_norm, t_pns)

plotter.export_figure(export_path)

rosette_for_final = best_rosette if best_rosette is not None else rosette
traj_for_final = best_traj if best_traj is not None else traj
slew_for_final = best_slew_rate if best_slew_rate is not None else slew_rate

recon_best_kbnufft = reconstructor.reconstruct_img(fft, rosette_for_final, method="kbnufft")
final_plots(phantoms, recon_best_kbnufft, initial_recon, plotter.total_losses, traj_for_final, slew_for_final, export=True, export_path=export_path)

recon_best_mirtorch = reconstructor.reconstruct_img(fft, rosette_for_final, method="mirtorch")
final_plots(phantoms, recon_best_mirtorch, initial_recon, plotter.total_losses, traj_for_final, slew_for_final, export=True, export_path=export_path + "/mirtorch")

psf(reconstructor, fft[0:1], rosette_init, rosette_for_final, device, export_path)
