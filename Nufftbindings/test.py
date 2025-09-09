# Author: Alban Gossard
# Last modification: 2021/22/09

import torch
import numpy as np
# import nufftbindings.pykeops as nufft
import nufftbindings.kbnufft as nufft

nx = ny = 320
K = int(nx*ny)
Nb = 3

# device = torch.device("cuda:0")
device = torch.device("cpu")

xi = torch.rand(K, 2, device=device)*2*np.pi-np.pi
xi.requires_grad = True

nufft.nufft.set_dims(K, (nx, ny), device, Nb=Nb)
nufft.nufft.precompute(xi)

f = torch.randn(Nb, nx, ny, 2, device=device)
f = torch.zeros(Nb, nx, ny, 2, device=device)
import matplotlib.pyplot as plt
f[0,:nx//2,:ny//2,0]=1
y = nufft.forward(xi, f)
g = nufft.adjoint(xi, y)
plt.figure(1)
plt.imshow(f[0,:,:,0].detach().numpy())
plt.figure(2)
plt.imshow(g[0,:,:,0].detach().numpy())
plt.show()
l = g.abs().pow(2).sum()
print(l)
# l.backward()
