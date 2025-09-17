import torch
import torch.nn as nn


class FourierPulseOpt(nn.Module):
    """Auxiliary 1d Fourier series. Used in FourierSeries class."""

    def __init__(self, t_min, t_max, n_coeffs=101, initialization="cos", coeff_lvl=1e-5):
        super().__init__()
        p = coeff_lvl * torch.randn((2 * n_coeffs + 1, 2))
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
    def __init__(self, tmin, tmax, initial_max=1.0, n_coeffs=51, coeff_lvl=1e-5):
        super().__init__()
        self.scaling = initial_max * 0.5
        self.pulses = nn.ModuleList(
            [
                FourierPulseOpt(tmin, tmax, n_coeffs=n_coeffs, initialization="cos", coeff_lvl=coeff_lvl),
                FourierPulseOpt(tmin, tmax, n_coeffs=n_coeffs, initialization="sin", coeff_lvl=coeff_lvl),
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
    def __init__(self, tmin, tmax, initial_max=1.0):
        super().__init__()
        self.scaling = initial_max * 0.5
        self.axes = torch.nn.Parameter(self.scaling * torch.ones(2))
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
        self.name = "fc"
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


class FCN1D(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.name = "fcn"
        assert kernel_size % 2 == 1
        padding = kernel_size // 2

        layers = []
        for in_ch, out_ch in zip(channels[:-1], channels[1:]):
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=padding))
            if out_ch != channels[-1]:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding), nn.ReLU(), nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding), nn.ReLU())

    def forward(self, x):
        return self.block(x)


class UNet1D(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, features=[16, 32, 64]):
        """
        Simple 1D U-Net.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        features : list of int
            Channel sizes for encoder layers.
        """
        super().__init__()
        self.name = "unet"
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.pools = nn.ModuleList()

        # Encoder path
        prev_ch = in_channels
        for f in features:
            self.encoders.append(ConvBlock(prev_ch, f))
            self.pools.append(nn.MaxPool1d(kernel_size=2, stride=2))
            prev_ch = f

        # Bottleneck
        self.bottleneck = ConvBlock(prev_ch, prev_ch * 2)

        # Decoder path
        rev_features = features[::-1]
        prev_ch = prev_ch * 2
        for f in rev_features:
            self.decoders.append(nn.ConvTranspose1d(prev_ch, f, kernel_size=2, stride=2))
            self.decoders.append(ConvBlock(prev_ch, f))
            prev_ch = f

        # Final output conv
        self.final_conv = nn.Conv1d(prev_ch, out_channels, kernel_size=1)

    def forward(self, x):
        skips = []

        # Encoder
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        skips = skips[::-1]
        for i in range(0, len(self.decoders), 2):
            up = self.decoders[i]
            conv = self.decoders[i + 1]
            x = up(x)
            skip = skips[i // 2]

            # pad if needed (in case lengths mismatch by 1)
            if x.shape[-1] != skip.shape[-1]:
                diff = skip.shape[-1] - x.shape[-1]
                x = nn.functional.pad(x, (0, diff))

            x = torch.cat([skip, x], dim=1)
            x = conv(x)

        return self.final_conv(x)


# x = torch.randn((2, 100))
# net = FCN1D(channels=[2, 16, 32, 16, 1])
# y = net(x)
# print("OUTPUT:", y.shape)
