import matplotlib.pyplot as plt
import numpy as np


t = np.linspace(0, 100, 1000)
g = np.abs(np.cos(np.pi * t))
dt = t[1] - t[0]  # sampling interval
shift = int(np.round(1 / dt))  # samples ≃ π seconds


tau = 1
alpha = dt / (tau + dt)
pns = np.zeros(len(t))
pns[0] = alpha * g[0]
for i in range(1, len(t)):
    pns[i] = alpha * g[i] + (1 - alpha) * pns[i - 1]

pns_shifted = np.roll(pns, -shift)
phi = (pns[shift] - pns[0]) * np.exp(-t / tau)

fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
ax[0].plot(t, g, linewidth=0.7)
ax[1].plot(t, pns, linewidth=0.7)
ax[1].plot(t, pns_shifted - pns, linewidth=0.7)
ax[1].plot(t, phi, linewidth=0.7)
ax[0].set_title("dG/dt")
ax[1].set_title("PNS (qualitative)")
plt.show()
