import math


train_steps = 1000
lr = 1e-4
grad_loss_weight = 1e-7
slew_loss_weight = 1e-8
zero_filling = True

timesteps = 128
duration = 0.5  # ms
dt = duration / (timesteps - 1)

grad_max = 135
slew_rate = 250
gamma = 42.575575  # Gyromagnetic ratio in MHz/T

img_size = 512  # pixels
FoV = 224  # mm
res = 50
n_petals = 80

model_size = 11  # number of Fourier coefficients

kmax_traj = res / (2 * FoV)  # 1/m
kmax_img = img_size / (2 * FoV)  # 1/m

final_FT_scaling = math.sqrt(2 / (img_size * res))
