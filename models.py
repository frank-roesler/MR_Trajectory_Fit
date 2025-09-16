import torch
import torch.nn as nn


class FourierPulseOpt(nn.Module):
    """Auxiliary 1d Fourier series. Used in FourierSeries class."""

    def __init__(self, t_min, t_max, n_coeffs=101, initialization="cos"):
        super().__init__()
        p = 1e-1 * torch.randn((2 * n_coeffs + 1, 2))
        if initialization == "cos":
            p[n_coeffs + 1, 1] = 1.0
        elif initialization == "sin":
            p[n_coeffs + 1, 0] = 1.0
        weights = torch.exp(-0.1 * torch.arange(-n_coeffs, n_coeffs + 1) ** 2)
        p = p * weights.unsqueeze(1)
        self.params = torch.nn.Parameter(p)
        k = torch.arange(-n_coeffs, n_coeffs + 1).unsqueeze(0)
        self.freqs = 2 * torch.pi * k / (t_max - t_min)

    def to(self, device):
        self.freqs = self.freqs.to(device)
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
        x = x[:-1, :]
        out = torch.cat([self.pulses[0](x) - self.pulses[0](0), self.pulses[1](x) - self.pulses[1](0)], dim=-1)
        return out * self.scaling


class Ellipse(nn.Module):
    def __init__(self, tmin, tmax, initial_a=1.0, initial_b=1.0):
        super().__init__()
        self.scaling_a = initial_a * 0.5
        self.scaling_b = initial_b * 0.5
        self.axes = torch.nn.Parameter(torch.Tensor([self.scaling_a, self.scaling_b]))
        self.name = "Ellipse"
        self.k = 2 * torch.pi / (tmax - tmin)

    def to(self, device):
        return super().to(device)

    def forward(self, x):
        x = x[:-1, :]
        fx = self.k * x
        out = torch.cat([self.axes[0] * (torch.cos(fx) - 1), self.axes[1] * torch.sin(fx)], dim=-1)
        return out


class DCFNet(nn.Module):
    def __init__(self, input_size=200, output_size=100, n_hidden=2, n_features=128):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.n_features = n_features
        self.relu = nn.ELU()
        self.input_layer = nn.Linear(self.input_size, n_features)
        layers = []
        for i in range(n_hidden // 2):
            layers += self.up_block(i)
        for i in range(n_hidden // 2 - 1, -1, -1):
            layers += self.down_block(i)
        self.hidden_layers = nn.Sequential(*layers)
        self.out_layer = nn.Linear(n_features, self.output_size)

    def up_block(self, i):
        return [nn.Linear(self.n_features * 2**i, self.n_features * 2 ** (i + 1)), self.relu]

    def down_block(self, i):
        return [nn.Linear(self.n_features * 2 ** (i + 1), self.n_features * 2**i), self.relu]

    def forward(self, x):
        x = self.relu(self.input_layer(x))
        x = self.hidden_layers(x)
        x = self.relu(self.out_layer(x))
        return x


# x = torch.randn((1, 200))
# net = DCFNet(n_hidden=4)
# y = net(x)
# print("OUTPUT:", y.shape)
