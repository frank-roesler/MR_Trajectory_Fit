from utils import (
    get_phantom,
    make_rosette,
    get_loss_fcn,
    sample_k_space_values,
    reconstruct_img2,
    reconstruct_img,
    reconstruct_img_nudft,
    get_rotation_matrix,
)
import matplotlib.pyplot as plt
from matplotlib import cm
from mirtorch.linear import FFTCn
from models import FourierCurve, Ellipse, DCFNet
import torch
from params import *

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

phantom = get_phantom(size=(params["img_size"], params["img_size"]), type="shepp_logan").to(device)

# Compute FFT:
Fop = FFTCn(phantom.shape, phantom.shape, (0, 1), norm=None)
fft = Fop * phantom
fft = fft.reshape(1, 1, params["img_size"], params["img_size"])
rotation_matrix = get_rotation_matrix(params["n_petals"], device=device)

t = torch.linspace(0, params["duration"], steps=params["timesteps"]).unsqueeze(1).to(device)  # (timesteps, 1)
model = Ellipse(tmin=0, tmax=params["duration"], initial_max=kmax_traj)
model.to(device)
img_loss = get_loss_fcn(params["loss_function"])

dcfnet = DCFNet(input_size=2 * (params["timesteps"] - 1), output_size=params["timesteps"] - 1, n_hidden=10, n_features=256).to(device)
dcfdict = torch.load("dcfnet2.pt")
dcfnet.load_state_dict(dcfdict)

axes_range = torch.arange(0.05, 0.06, 0.0001)
loss_values = torch.zeros(len(axes_range), len(axes_range)).to(device)

(x, y, loss_values) = torch.load("landscape_l1_nudft_dcfnet.pt")
# x, y, loss_values = x[: -x.shape[0] // 3, : -x.shape[0] // 3], y[: -x.shape[0] // 3, : -x.shape[0] // 3], loss_values[: -x.shape[0] // 3, : -x.shape[0] // 3]

# region
# with torch.no_grad():
#     for i, a in enumerate(axes_range):
#         print(i, "/", len(axes_range))
#         for j, b in enumerate(axes_range):
#             model.axes[0] = a
#             model.axes[1] = b
#             traj = model(t)

#             rotated_trajectories = [traj]
#             for _ in range(params["n_petals"] - 1):
#                 traj_tmp = traj @ rotation_matrix.T
#                 rotated_trajectories.append(traj_tmp)
#                 traj = traj_tmp
#             corners = torch.ones(2, 2, device=traj.device)
#             corners[0] *= kmax_img
#             corners[1] *= -kmax_img
#             rotated_trajectories.append(corners)
#             rosette = torch.cat(rotated_trajectories, dim=0)

#             rosette, sampled = sample_k_space_values(fft, rosette, kmax_img, True)
#             recon = reconstruct_img2(rosette, sampled, params["img_size"], kmax_img, final_FT_scaling, dcfnet)
#             image_loss = img_loss(recon, phantom)
#             loss_values[i, j] = image_loss.detach().item()

# x, y = torch.meshgrid(axes_range, axes_range)
# torch.save((x, y, loss_values), "landscape_l1_nudft_dcfnet.pt")
# endregion

# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# im = ax.plot_surface(x, y, loss_values.detach().cpu(), cmap=cm.viridis)
# fig.colorbar(im, ax=ax)

fig2, ax2 = plt.subplots()
im2 = ax2.contour(x, y, loss_values.detach().cpu(), cmap=cm.viridis, levels=100, linewidths=0.5)
fig2.colorbar(im2, ax=ax2)

(a, b) = torch.load("results/2025-09-16_20-23/train_path.pt")
a = a[::1]
b = b[::1]

path_a = [apair[0] for apair in a]
path_b = [bpair[0] for bpair in b]

grad_a = [apair[1] for apair in a]
grad_b = [bpair[1] for bpair in b]

# ax2.plot(path_a, path_b, color="r", linewidth=0.9, marker=".", markersize=1)

for i, (ga, gb) in enumerate(zip(grad_a, grad_b)):
    dx = -params["lr"] * ga / 8
    dy = -params["lr"] * gb / 8
    ax2.arrow(path_a[i], path_b[i], dx, dy, color="r", head_length=0.00001, head_width=0.00001, width=0.0000001)


plt.show()
