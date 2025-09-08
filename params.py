timesteps = 128
duration = 0.5  # seconds
dt = duration / (timesteps - 1)

grad_max = 135
slew_rate = 280

img_size = 512  # pixels
FoV = 224  # mm
res = 50
n_petals = 80

model_size = 21  # number of Fourier coefficients

kmax_traj = res / (2 * FoV)  # 1/m
kmax_img = img_size / (2 * FoV)  # 1/m
