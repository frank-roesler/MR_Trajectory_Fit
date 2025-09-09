import torch
import torch.nn as nn


class FourierPulseOpt(nn.Module):
    """Auxiliary 1d Fourier series. Used in FourierSeries class."""

    def __init__(self, t_min, t_max, n_coeffs=101, initialization="cos"):
        super().__init__()
        p = 1e-4 * torch.randn((2 * n_coeffs + 1, 2))
        if initialization == "cos":
            p[n_coeffs + 1, 1] = 1.0
        elif initialization == "sin":
            p[n_coeffs + 1, 0] = 1.0
        weights = torch.exp(-0.01 * torch.arange(-n_coeffs, n_coeffs + 1) ** 2)
        p = p * weights.unsqueeze(1)
        self.params = torch.nn.Parameter(p)
        k = torch.arange(-n_coeffs, n_coeffs + 1).unsqueeze(0)
        self.freqs = 2 * torch.pi * k / (t_max - t_min)

    def to(self, device):
        self.k = self.k.to(device)
        return super().to(device)

    def forward(self, x):
        fx = self.freqs * x  # (batch, k)
        sin_fx = torch.sin(fx)
        cos_fx = torch.cos(fx)
        y_sin = sin_fx @ self.params[:, 0:1]
        y_cos = cos_fx @ self.params[:, 1:]
        return y_sin + y_cos


class FourierCurve(nn.Module):
    def __init__(self, tmin, tmax, initial_max=1.0, n_coeffs=51):
        super().__init__()
        self.scaling = initial_max * 0.5
        self.pulses = nn.ModuleList(
            [
                FourierPulseOpt(tmin, tmax, n_coeffs=n_coeffs, initialization="cos"),
                FourierPulseOpt(tmin, tmax, n_coeffs=n_coeffs, initialization="sin"),
            ]
        )
        self.name = "FourierCurve"

    def to(self, device):
        for pulse in self.pulses:
            pulse.to(device)
        return super().to(device)

    def forward(self, x):
        out = torch.cat([self.pulses[0](x) - self.pulses[0](0), self.pulses[1](x) - self.pulses[1](0)], dim=-1)
        return out * self.scaling
