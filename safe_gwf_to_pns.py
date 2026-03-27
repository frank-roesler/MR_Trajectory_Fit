import math
import torch
import matplotlib.pyplot as plt
import safe_hw_from_asc
from torch.fft import rfft, irfft, fft, ifft, fftfreq
from params import *


def cumulative_trapezoid(y, dx=1.0, initial=0.0, dim=0):
    """
    PyTorch version of SciPy's cumulative_trapezoid with `initial` argument.

    Parameters
    ----------
    y : torch.Tensor
        Input values to integrate along `dim`.
    dx : float
        Uniform spacing between points.
    initial : float or tensor
        Value to prepend at the start (matches SciPy's initial=0 behavior).
    dim : int
        Dimension along which to integrate.

    Returns
    -------
    torch.Tensor
        Cumulative trapezoidal integral of `y`, same length as `y`.
    """
    cumsum = torch.cumulative_trapezoid(y, dx=dx, dim=dim)
    if not torch.is_tensor(initial):
        initial_tensor = torch.tensor([initial], device=y.device, dtype=y.dtype)
    else:
        initial_tensor = initial.to(y.device).type(y.dtype)
        if initial_tensor.ndim == 0:
            initial_tensor = initial_tensor.unsqueeze(0)
    slices = [slice(None)] * y.ndim
    slices[dim] = slice(0, 1)
    return torch.cat([initial_tensor, cumsum], dim=dim)


class SAFE_PNS:
    """
    implementation of a range of methods for PNS computation from the field gradient gwf.
    The computation is always composed of 3 lowpass filters with varying parameter tau
    and scaling. The methods differ only in the computation of the lowpass filters and are specified
    by the 'method' argument. Possible values for 'method' are ['euler', 'fourier', 'fixed_point', 'fourier_plateau'].
    'euler', 'fourier', 'fixed_point' are different methods of solving the lowpass ODE.
    'fourier_plateau' should only be used when dgdt is periodic and only computes the long time asymptotics of the low pass.
    """

    def __init__(self, dt, hw_path, method="full"):
        self.hw = safe_hw_from_asc.safe_hw_from_asc(hw_path)
        self.hw_check()
        self.dt = dt
        self.method = method

    def example_hw(self):
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

    def longest_time_const(self):
        taus = []
        for axis in ["x", "y", "z"]:
            if axis in self.hw:
                taus.extend([hw[axis]["tau1"], hw[axis]["tau2"], hw[axis]["tau3"]])
        return max(taus)

    def hw_check(self):
        for axis in ["x", "y", "z"]:
            total_a = self.hw[axis]["a1"] + self.hw[axis]["a2"] + self.hw[axis]["a3"]
            if abs(total_a - 1.0) > 0.001:
                raise ValueError(f"Hardware specification {axis}: a1+a2+a3 must be equal to 1!")
        required_params = ["stim_limit", "stim_thresh", "tau1", "tau2", "tau3", "a1", "a2", "a3", "g_scale"]
        for axis in ["x", "y", "z"]:
            for param in required_params:
                if param not in self.hw[axis] or self.hw[axis][param] is None:
                    raise ValueError(f"Hardware specification {axis}.{param} is empty or missing!")

    def safe_pns_model_fourier(self, dgdt: torch.Tensor, hw_axis: dict) -> torch.Tensor:
        tau_tensor = torch.Tensor([hw_axis["tau1"], hw_axis["tau2"], hw_axis["tau3"]]).unsqueeze(0)
        a_tensor = torch.Tensor([hw_axis["a1"], hw_axis["a2"], hw_axis["a3"]]).unsqueeze(0)
        pad = dgdt.shape[0] // 2
        dgdt_padded = torch.nn.functional.pad(dgdt, (pad, pad))
        dgdt_padded = torch.stack((dgdt_padded, dgdt_padded.abs(), dgdt_padded), dim=1)
        dgdt_hat = fft(dgdt_padded, dim=0)
        K = fftfreq(dgdt.shape[0] + 2 * pad, d=self.dt).unsqueeze(1) * 2 * torch.pi
        lp_hat = dgdt_hat / (1 + 1j * K * tau_tensor)
        lp = ifft(lp_hat, dim=0)[pad:-pad].real.abs()
        stim = torch.sum(a_tensor * lp, dim=1) / hw_axis["stim_thresh"] * hw_axis["g_scale"] * 100.0
        return stim

    def fp_from_one_Gamma_period(self, dgdt: torch.Tensor, tau: float) -> torch.Tensor:
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
        t = torch.arange(N, device=dgdt.device) * self.dt
        const = dt / (tau * (1 - math.exp(-T / tau)))
        decay = torch.exp(-t / tau)
        fw = const * irfft(rfft(dgdt) * rfft(decay), n=N)
        return fw

    def fft_pns_plateau(self, dgdt: torch.Tensor, hw_axis: dict) -> torch.Tensor:
        lp1 = self.fp_from_one_Gamma_period(dgdt, hw_axis["tau1"], self.dt)
        stim1 = hw_axis["a1"] * torch.abs(lp1)
        lp2 = self.fp_from_one_Gamma_period(torch.abs(dgdt), hw_axis["tau2"], self.dt)
        stim2 = hw_axis["a2"] * lp2
        lp3 = self.fp_from_one_Gamma_period(dgdt, hw_axis["tau3"], self.dt)
        stim3 = hw_axis["a3"] * torch.abs(lp3)
        stim = (stim1 + stim2 + stim3) / hw_axis["stim_thresh"] * hw_axis["g_scale"] * 100.0
        return stim

    def log_norm_T(self, tmax, tau, m):
        m = torch.as_tensor(m)
        f1 = -0.5 * torch.log10(2 * torch.pi * m)
        f2 = m * torch.log10(math.exp(1) * tmax / (m * tau))
        return f1 + f2

    def get_optimal_split(self, tau, full_length, threshold=1e-5):
        n_sections = 1
        while True:
            m = 0
            n = float("inf")
            section_length = full_length // n_sections
            while n > math.log10(threshold):
                m += 1
                n = self.log_norm_T(self.dt * section_length, tau, m)
                if n > 4:
                    n_sections += 1
                    break
            if n < math.log10(threshold):
                break
        print("-" * 100)
        print(f"Optimal section length: {section_length}")
        print(f"Steps per section: {m}")
        print(f"Number of sections: {n_sections}")
        print(f"Total steps: {n_sections*m}")
        print("-" * 100)
        return section_length, m

    def fixed_point_method(self, tau, dgdt, steps_per_section, section_length):
        pref = self.dt / tau
        fw = torch.zeros_like(dgdt)
        full_length = len(dgdt)
        n_full_sections = full_length // section_length
        for i in range(n_full_sections + 1):
            starting_pos = i * section_length
            prev_pos = max(starting_pos - 1, 0)
            if i < n_full_sections:
                current_range = range(prev_pos, (i + 1) * section_length)
            else:
                current_range = range(prev_pos, full_length)
            dgdt_i = dgdt[current_range]
            prev_value = fw[prev_pos]
            for _ in range(steps_per_section):
                Tf = pref * cumulative_trapezoid(dgdt_i - fw[current_range], initial=0)
                fw[current_range] = prev_value + Tf
        return fw

    def pns_fixed_point(self, dgdt: torch.Tensor, hw_axis: dict) -> torch.Tensor:
        threshold = 1e-4
        section_length, steps_per_section = self.get_optimal_split(hw_axis["tau1"], len(dgdt), threshold=threshold)
        lp1 = self.fixed_point_method(hw_axis["tau1"], dgdt, steps_per_section, section_length)
        stim1 = hw_axis["a1"] * torch.abs(lp1)
        section_length, steps_per_section = self.get_optimal_split(hw_axis["tau2"], len(dgdt), threshold=threshold)
        lp2 = self.fixed_point_method(hw_axis["tau2"], dgdt.abs(), steps_per_section, section_length)
        stim2 = hw_axis["a2"] * lp2
        section_length, steps_per_section = self.get_optimal_split(hw_axis["tau3"], len(dgdt), threshold=threshold)
        lp3 = self.fixed_point_method(hw_axis["tau3"], dgdt, steps_per_section, section_length)
        stim3 = hw_axis["a3"] * torch.abs(lp3)
        stim = (stim1 + stim2 + stim3) / hw_axis["stim_thresh"] * hw_axis["g_scale"] * 100.0
        return stim

    def safe_gwf_to_pns(self, gwf: torch.Tensor):
        dgdt = torch.gradient(gwf, spacing=self.dt, dim=0)[0]
        if self.method == "fourier_plateau":
            pns_x = self.fft_pns_plateau(dgdt[:, 0], self.hw["x"])
            pns_y = self.fft_pns_plateau(dgdt[:, 1], self.hw["y"])
            pns_z = self.fft_pns_plateau(dgdt[:, 2], self.hw["z"])
            return torch.stack([pns_x, pns_y, pns_z], dim=1)
        elif self.method == "fixed_point":
            pns_x = self.pns_fixed_point(dgdt[:, 0], self.hw["x"])
            pns_y = self.pns_fixed_point(dgdt[:, 1], self.hw["y"])
            pns_z = self.pns_fixed_point(dgdt[:, 2], self.hw["z"])
            return torch.stack([pns_x, pns_y, pns_z], dim=1)
        else:
            pns_x = self.safe_pns_model_fourier(dgdt[:, 0], self.hw["x"])
            pns_y = self.safe_pns_model_fourier(dgdt[:, 1], self.hw["y"])
            pns_z = self.safe_pns_model_fourier(dgdt[:, 2], self.hw["z"])
            return torch.stack([pns_x, pns_y, pns_z], dim=1)


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
