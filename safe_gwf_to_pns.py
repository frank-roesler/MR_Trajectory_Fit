import math
import torch
import matplotlib.pyplot as plt
import safe_hw_from_asc
from torch.fft import rfft, irfft

from params import *


def safe_example_hw():
    """
    SAFE model parameters for EXAMPLE scanner hardware (not a real scanner).

    Returns
    -------
    hw : dict
        Dictionary containing hardware parameters for x, y, and z axes.
    """
    hw = {
        "name": "MP_GPA_EXAMPLE",
        "checksum": "1234567890",
        "dependency": "",
        "x": {"tau1": 0.20, "tau2": 0.03, "tau3": 3.00, "a1": 0.40, "a2": 0.10, "a3": 0.50, "stim_limit": 30.0, "stim_thresh": 24.0, "g_scale": 0.35},
        "y": {"tau1": 1.50, "tau2": 2.50, "tau3": 0.15, "a1": 0.55, "a2": 0.15, "a3": 0.30, "stim_limit": 15.0, "stim_thresh": 12.0, "g_scale": 0.31},
        "z": {"tau1": 2.00, "tau2": 0.12, "tau3": 1.00, "a1": 0.42, "a2": 0.40, "a3": 0.18, "stim_limit": 25.0, "stim_thresh": 20.0, "g_scale": 0.25},
    }
    return hw


def safe_longest_time_const(hw):
    taus = []
    for axis in ["x", "y", "z"]:
        if axis in hw:
            taus.extend([hw[axis]["tau1"], hw[axis]["tau2"], hw[axis]["tau3"]])
    return max(taus)


def safe_hw_check(hw):
    for axis in ["x", "y", "z"]:
        total_a = hw[axis]["a1"] + hw[axis]["a2"] + hw[axis]["a3"]
        if abs(total_a - 1.0) > 0.001:
            raise ValueError(f"Hardware specification {axis}: a1+a2+a3 must be equal to 1!")
    required_params = ["stim_limit", "stim_thresh", "tau1", "tau2", "tau3", "a1", "a2", "a3", "g_scale"]
    for axis in ["x", "y", "z"]:
        for param in required_params:
            if param not in hw[axis] or hw[axis][param] is None:
                raise ValueError(f"Hardware specification {axis}.{param} is empty or missing!")


def safe_pns_model_fourier(dgdt: torch.Tensor, dt: float, hw_axis: dict) -> torch.Tensor:
    tau_tensor = torch.Tensor([hw_axis["tau1"], hw_axis["tau2"], hw_axis["tau3"]]).unsqueeze(0)
    a_tensor = torch.Tensor([hw_axis["a1"], hw_axis["a2"], hw_axis["a3"]]).unsqueeze(0)
    pad = dgdt.shape[0] // 2
    dgdt_padded = torch.nn.functional.pad(dgdt, (pad, pad))
    dgdt_padded = torch.stack((dgdt_padded, dgdt_padded.abs(), dgdt_padded), dim=1)
    dgdt_hat = torch.fft.fft(dgdt_padded, dim=0)
    K = torch.fft.fftfreq(dgdt.shape[0] + 2 * pad, d=dt).unsqueeze(1) * 2 * torch.pi
    lp_hat = dgdt_hat / (1 + 1j * K * tau_tensor)
    lp = torch.fft.ifft(lp_hat, dim=0)[pad:-pad].real.abs()
    stim = torch.sum(a_tensor * lp, dim=1) / hw_axis["stim_thresh"] * hw_axis["g_scale"] * 100.0
    return stim


def safe_gwf_to_pns_torch(gwf: torch.Tensor, rf: torch.Tensor, dt: float, hw: dict):
    gwf = gwf.to(torch.float32)
    rf = rf.to(torch.float32)

    # Hardware check
    safe_hw_check(hw)

    # Slew rate
    dgdt = torch.gradient(gwf, spacing=dt / 1000, dim=0)[0]

    # PNS calculation
    pns_x = safe_pns_model_fourier(dgdt[:, 0], dt, hw["x"])
    pns_y = safe_pns_model_fourier(dgdt[:, 1], dt, hw["y"])
    pns_z = safe_pns_model_fourier(dgdt[:, 2], dt, hw["z"])
    pns = torch.stack([pns_x, pns_y, pns_z], dim=1)

    res = {"pns": pns, "gwf": gwf, "rf": rf, "dgdt": dgdt, "dt": dt, "hw": hw}
    return pns, res


def fp_from_one_Gamma_period(dgdt: torch.Tensor, tau: float, dt: float) -> torch.Tensor:
    """
    Computes the periodic part f_p using only a single period of Gamma (dgdt).
    Based on the formula: f_p = 1 / (tau*(1 - e^{-T/tau})) * F^-1( F(Gamma) * F(e^{-t/tau}) )

    Args:
        dgdt: (time,) tensor of real values
        tau: time constant in ms
        dt: sampling interval in ms
    """
    N = len(dgdt)
    T = dt * N
    t = torch.arange(N, device=dgdt.device) * dt

    const = dt / (tau * (1 - math.exp(-T / tau)))
    decay = torch.exp(-t / tau)

    # rfft/irfft to maximize memory efficiency and ensure a purely real output.
    fw = const * irfft(rfft(dgdt) * rfft(decay), n=N)

    return fw


def fft_pns_model_torch(dgdt: torch.Tensor, dt: float, hw_axis: dict) -> torch.Tensor:

    dt_ms = dt * 1000.0  # convert to ms

    lp1 = fp_from_one_Gamma_period(dgdt, hw_axis["tau1"], dt_ms)
    stim1 = hw_axis["a1"] * torch.abs(lp1)
    lp2 = fp_from_one_Gamma_period(torch.abs(dgdt), hw_axis["tau2"], dt_ms)
    stim2 = hw_axis["a2"] * lp2
    lp3 = fp_from_one_Gamma_period(dgdt, hw_axis["tau3"], dt_ms)
    stim3 = hw_axis["a3"] * torch.abs(lp3)
    stim = (stim1 + stim2 + stim3) / hw_axis["stim_thresh"] * hw_axis["g_scale"] * 100.0

    return stim


def fft_gwf_to_pns_torch(
    gwf: torch.Tensor,
    rf: torch.Tensor,
    dt: float,
    hw: dict,
) -> torch.Tensor:
    gwf = gwf.to(torch.float32)
    rf = rf.to(torch.float32)

    safe_hw_check(hw)

    dgdt = (gwf[1:] - gwf[:-1]) / dt

    # PNS calculation
    pns_x = fft_pns_model_torch(dgdt[:, 0], dt, hw["x"])
    pns_y = fft_pns_model_torch(dgdt[:, 1], dt, hw["y"])
    pns_z = fft_pns_model_torch(dgdt[:, 2], dt, hw["z"])
    pns = torch.stack([pns_x, pns_y, pns_z], dim=1)

    res = {"pns": pns, "gwf": gwf, "rf": rf, "dgdt": dgdt, "dt": dt, "hw": hw}
    return pns, res


# Example usage
if __name__ == "__main__":
    import torch
    import json
    import numpy as np
    import matplotlib.pyplot as plt

    from utils_3 import compute_pns_from_gradients, compute_fast_pns_from_gradients

    import safe_hw_from_asc

    # Setup Device and Hardware
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    hw = safe_hw_from_asc.safe_hw_from_asc("safe_pns_prediction/MP_GradSys_K2298_2250V_1250A_W60_SC72CD.asc")

    # Load Data from JSON
    json_path = "results/2026-03-10_12-33/traj_data.json"
    with open(json_path, "r") as f:
        data = json.load(f)

    gx_list = data["trj"]["Gx"][0]
    gy_list = data["trj"]["Gy"][0]

    gx_max = data["trj"]["GxMax"][0]
    gy_max = data["trj"]["GyMax"][0]

    gx = [val * gx_max for val in gx_list]
    gy = [val * gy_max for val in gy_list]

    dt_ms = 0.005  # ms
    gx = torch.tensor(gx, device=device, dtype=torch.float32)
    gy = torch.tensor(gy, device=device, dtype=torch.float32)

    num_steps = len(gx)
    period_ms = num_steps * dt_ms
    t = torch.arange(num_steps, device=device) * dt_ms
    print(f"Loaded trajectory: dt = {dt_ms:.4f} ms, period = {period_ms:.2f} ms, steps = {num_steps}")

    # Standard PNS Computation (Time-Domain)
    specRes = 325
    pns_x_std, pns_y_std, pns_norm_std, t_pns_std = compute_pns_from_gradients(hw, gx, gy, dt_ms, specRes=specRes, Ramp=False)

    # Fast PNS Computation (FFT-Based)
    pns_x_fast, pns_y_fast, pns_norm_fast, t_pns_fast = compute_fast_pns_from_gradients(gx, gy, dt_ms, hw)

    max_pns_std = pns_norm_std.max()
    max_pns_fast = pns_norm_fast.max()

    # Plotting
    fig = plt.figure(figsize=(14, 8))

    # --- Plot 1: Input Gradients ---
    ax0 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
    ax0.plot(t.cpu(), gx.cpu(), label="Gx", color="r", linewidth=2)
    ax0.plot(t.cpu(), gy.cpu(), label="Gy", color="g", linewidth=2)
    ax0.set_ylabel("Gradient [mT/m]")
    ax0.set_xlabel("Time [ms]")
    ax0.set_title("Input Gradient Waveforms (1 Period)")
    ax0.grid(True, linestyle="--", alpha=0.6)
    ax0.legend(loc="upper right")

    # --- Plot 2: Standard PNS Computation ---
    ax1 = plt.subplot2grid((2, 2), (1, 0))
    ax1.plot(t_pns_std.cpu(), pns_x_std.cpu(), label="PNS x", color="r")
    ax1.plot(t_pns_std.cpu(), pns_y_std.cpu(), label="PNS y", color="g")
    ax1.plot(t_pns_std.cpu(), pns_norm_std.cpu(), label="PNS Norm", color="k", linestyle="--")
    ax1.set_ylabel("Stimulation [%]")
    ax1.set_xlabel("Time [ms]")
    ax1.set_title("Cutoff-Based PNS, max {:.2f}%".format(max_pns_std))
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper right")

    # --- Plot 3: Fast FFT PNS Computation ---
    ax2 = plt.subplot2grid((2, 2), (1, 1))
    ax2.plot(t_pns_fast.cpu(), pns_x_fast.cpu(), label="PNS x", color="r", linewidth=2)
    ax2.plot(t_pns_fast.cpu(), pns_y_fast.cpu(), label="PNS y", color="g", linewidth=2)
    ax2.plot(t_pns_fast.cpu(), pns_norm_fast.cpu(), label="PNS Norm", color="k", linestyle="--", linewidth=2)
    ax2.set_ylabel("Stimulation [%]")
    ax2.set_xlabel("Time [ms]")
    ax2.set_title("Fast FFT-Based PNS, max {:.2f}%".format(max_pns_fast))
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig("pns_fft_vs_std_comparison.png", dpi=300)
    plt.show()
