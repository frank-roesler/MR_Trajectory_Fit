import numpy as np
import matplotlib.pyplot as plt
from params import *

def threshold_loss(x, threshold):
    excess = x - threshold 
    return np.maximum(0.0, excess) ** 2

# From arXviv:2505.07117v1 (OPTIKS), Eq. 3
def logb_star_loss(pns, params, delta = 0.1):
    x = pns
    x_max = params["pns_threshold"]
    x_delta = x_max - delta
    
    safe_x = np.where(x <= x_delta, x, x_delta)
    
    branch_1 = -np.log(x_max - safe_x)
    branch_2 = ((x - x_delta) / delta) - np.log(delta)
    
    elementwise_loss = np.where(x <= x_delta, branch_1, branch_2)
    
    return elementwise_loss


# V_eff(r) = L^2 / (2 * m * r^2) - k / r as an orbiting planet
def effective_potential_loss(x, threshold, A=40.0, B=30.0, epsilon=0.5):
    r = np.maximum(threshold - x, epsilon)

    repulsive_core = A / (r**2)
    attractive_tail = B / r

    potential = repulsive_core - attractive_tail

    overshoot = np.maximum(0.0, x - threshold)
    penalty = 100.0 * (overshoot ** 2)

    return potential + penalty

def pns_loss(pns, params, mode="exp", delta=0.1):
    if mode == "exp":
        pns_loss = np.exp(0.1 * (pns - params["pns_threshold"]))
    elif mode == "threshold":
        pns_loss = threshold_loss(pns, params["pns_threshold"])
    elif mode == "logb_star":
        pns_loss = logb_star_loss(pns, params, delta = delta)
    elif mode == "effective_potential":
        pns_loss = effective_potential_loss(pns, params["pns_threshold"])
    return pns_loss

pns_example = np.linspace(94, 102, 100)

pns_loss_exp = pns_loss(pns_example, params, mode="exp")
pns_loss_threshold = pns_loss(pns_example, params, mode="threshold")
pns_loss_logb_star_01 = pns_loss(pns_example, params, mode="logb_star", delta=1)
pns_loss_logb_star_05 = pns_loss(pns_example, params, mode="logb_star", delta=0.5)
pns_loss_effective_potential = pns_loss(pns_example, params, mode="effective_potential")

plt.figure(figsize=(10, 6))
plt.plot(pns_example, pns_loss_exp, label='Exponential Loss')
plt.plot(pns_example, pns_loss_threshold, label='Threshold Loss')
plt.plot(pns_example, pns_loss_logb_star_01, label='LogB* Loss (Delta=1)')
plt.plot(pns_example, pns_loss_logb_star_05, label='LogB* Loss (Delta=0.5)')
plt.plot(pns_example, pns_loss_effective_potential, label='Eff. Potential Loss')

plt.axvline(params["pns_threshold"], color='red', linestyle='--', label='PNS Thr = 98', alpha = 0.5)
plt.title('Comparison of PNS Loss Functions')
plt.xlabel('PNS Value')
plt.ylabel('Loss')
plt.ylim(-10,20)
plt.legend()
plt.grid()
plt.savefig("pns_loss_comparison.png")
plt.show()


