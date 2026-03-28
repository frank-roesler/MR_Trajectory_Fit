from utils_3 import (
    LossCollection,
    ImageRecon,
    TrainPlotter,
    Checkpointer,
    compute_gradients_from_traj,
    compute_pns_from_gradients,
    get_batch_of_phantoms,
    make_rosette,
    final_plots,
    get_rotation_matrix,
    get_device,
    psf,
    compute_initial_fft,
)
import matplotlib.pyplot as plt
from models import FourierCurve, Ellipse
import torch
from params import *
from safe_gwf_to_pns import SAFE_PNS

torch.set_printoptions(threshold=100000)

device = get_device()

batch_size = 1
phantoms = get_batch_of_phantoms(batch_size, size=(params["img_size"], params["img_size"]), type="glpu").to(device)
fft = compute_initial_fft(phantoms, padding=params["img_size"])
t = torch.linspace(0, params["duration"], steps=params["timesteps"], device=device).unsqueeze(1)  # (timesteps, 1)

model1 = FourierCurve(tmin=0, tmax=params["duration"], initial_max=kmax_traj, n_coeffs=params["model_size"], coeff_lvl=1e-2).to(device)  # 1e-2
model2 = FourierCurve(tmin=0, tmax=params["duration"], initial_max=kmax_traj, n_coeffs=params["model_size"], coeff_lvl=1e-2).to(device)  # 1e-2
# model = Ellipse(tmin=0, tmax=params["duration"], initial_max=kmax_traj).to(device)
optimizer = torch.optim.Adam(list(model1.parameters()) + list(model2.parameters()), lr=params["lr"])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1000, factor=0.5, min_lr=1e-6, threshold=1e-6, cooldown=100)

loss_fcns = LossCollection(params["loss_function"])
reconstructor = ImageRecon(params, kmax_img, normalization, dcfnet="unet")  # k-space -> Image
plotter = TrainPlotter(params, fft, reconstructor, phantoms, loss_fcns.loss_fn, optimizer)
checkpointer = Checkpointer(export_path, params, dt)
safe_model = SAFE_PNS(dt=dt, hw_path="safe_pns_prediction/MP_GradSys_K2298_2250V_1250A_W60_SC72CD.asc", method="fourier")
best_rosette = None  # for PSF analysis at the end
best_traj = None
best_slew_rate = None

with torch.no_grad():
    traj1, angles1, radii1 = model1(t)
    traj2, angles2, radii2 = model2(t)
    rosette1, _, _ = make_rosette(angles1, radii1, traj1, params["n_petals"] // 2, kmax_img, dt, zero_filling=False)
    rosette2, _, _ = make_rosette(angles2, radii2, traj2, params["n_petals"] // 2, kmax_img, dt, zero_filling=params["zero_filling"])
    rosette = torch.cat([rosette1, rosette2], dim=0)
    rosette_init = rosette.clone().to(device)  # for later PSF analysis
    initial_recon = reconstructor.reconstruct_img(fft, rosette, method="nudft")


# for step in range(params["train_steps"]):
for step in range(10000):
    traj1, angles1, radii1 = model1(t)
    traj2, angles2, radii2 = model2(t)  # (timesteps, 2)
    rosette1, *derivatives1 = make_rosette(angles1, radii1, traj1, params["n_petals"] // 2, kmax_img, dt, zero_filling=False)
    rosette2, *derivatives2 = make_rosette(angles2, radii2, traj2, params["n_petals"] // 2, kmax_img, dt, zero_filling=params["zero_filling"])
    rosette = torch.cat([rosette1, rosette2], dim=0)
    derivatives = [torch.maximum(derivatives1[0], derivatives2[0]), torch.maximum(derivatives1[1], derivatives2[1])]

    recon = reconstructor.reconstruct_img(fft, rosette, method="nudft")
    # Compute PNS from gradients - fully differentiable
    gx1, gy1, t_axis = compute_gradients_from_traj(traj1, dt, params["gamma"])
    gx2, gy2, _ = compute_gradients_from_traj(traj2, dt, params["gamma"])
    gx, gy = torch.cat([gx1, gx2], dim=0), torch.cat([gy1, gy2], dim=0)

    # --------- !!! SWAP x AND y FOR PNS BECAUSE THEY ARE SWAPPED IN IDEA !!! ---------
    pns_x, pns_y, pns_norm, t_pns = compute_pns_from_gradients(safe_model, gy, gx)
    # ---------------------------------------------------------------------------------

    max_pns = pns_norm.max()

    # Losses
    # pns_loss = loss_fcns.pns_loss(max_pns, params, mode=args.pns_mode, delta=1)
    pns_loss = loss_fcns.pns_loss(max_pns, params, mode="exp")
    # grad_loss, slew_loss = loss_fcns.grad_slew_loss(*derivatives, params, grad_mode="exp", slew_mode=args.slew_mode, delta=1)
    grad_loss, slew_loss = loss_fcns.grad_slew_loss(*derivatives, params, grad_mode="exp", slew_mode="exp")
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
        best_traj = traj1.detach().clone()
        slew_rate = checkpointer.save_checkpoint(model1, *derivatives, rosette)
        best_slew_rate = slew_rate.detach().clone()
        plotter.export_figure(export_path)
    plotter.update(step, grad_loss, image_loss, slew_loss, pns_loss, total_loss, recon, traj1, rosette, gx, gy, t_axis, torch.cat([angles1, angles2], dim=0), pns_norm, t_pns)

plotter.export_figure(export_path)

rosette_for_final = best_rosette if best_rosette is not None else rosette
traj_for_final = best_traj if best_traj is not None else traj1
slew_for_final = best_slew_rate if best_slew_rate is not None else slew_rate

recon_best_kbnufft = reconstructor.reconstruct_img(fft, rosette_for_final, method="kbnufft")
final_plots(phantoms, recon_best_kbnufft, initial_recon, plotter.total_losses, traj1, slew_for_final, export=True, export_path=export_path)

recon_best_mirtorch = reconstructor.reconstruct_img(fft, rosette_for_final, method="mirtorch")
final_plots(phantoms, recon_best_mirtorch, initial_recon, plotter.total_losses, traj2, slew_for_final, export=True, export_path=export_path + "/mirtorch")

psf(reconstructor, fft[0:1], rosette_init, rosette_for_final, device, export_path)
