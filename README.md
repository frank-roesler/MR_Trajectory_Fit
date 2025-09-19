# MR_Trajectory_Fit

**MR_Trajectory_Fit** is a Python project for MRI trajectory optimization and image reconstruction using differentiable Non-uniform Fast Fourier Transforms (NUFFT) and deep learning. It provides tools for designing, training, and evaluating MRI sampling trajectories, with support for automatic differentiation and integration with PyTorch.

## Features

- Trajectory optimization using Fourier series and ellipses
- Deep learning-based density compensation (DCF) networks (UNet, FCN)
- Image reconstruction pipelines with multiple backends
- Visualization and plotting utilities for training and results
- Modular design for research and experimentation

![Example output](https://github.com/frank-roesler/MR_Trajectory_Fit/blob/main/final_figure.png)

## Project Structure

- `main.py` — Main training and evaluation script
- `models.py` — Trajectory models (FourierCurve, Ellipse, etc.)
- `params.py` — Configuration parameters
- `utils.py` — Utility functions (image recon, plotting, checkpointing)
- `phantom_images/` — Example phantom images
- `trained_models/` — Pretrained DCF networks
- `results/` — Output results and figures
- **External packages used:**
    - `Nufftbindings/` — External package providing NUFFT backends and bindings (kbnufft, cufinufft, nfft, pykeops)
    - `mirtorch_pkg.py` — External package for core MRI operators and differentiable layers
    - `MIRTorch` — External library for MRI reconstruction (see [MIRTorch](https://github.com/mmuckley/mirtorch))

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/frank-roesler/MR_Trajectory_Fit.git
   cd MR_Trajectory_Fit
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   Additional requirements for NUFFT backends:
   - [torchkbnufft](https://github.com/mmuckley/torchkbnufft)
   - [cufinufft](https://github.com/flatironinstitute/cufinufft/)
   - [PyNFFT](https://github.com/pyNFFT/pyNFFT)
   - [pykeops](https://www.kernel-operations.io/keops/python/installation.html)
   - [MIRTorch](https://github.com/mmuckley/mirtorch)

## Usage

1. Configure parameters in `params.py` as needed.
2. Run the main script:
   ```bash
   python main.py
   ```
3. Training results, figures, and checkpoints will be saved in the `results/` directory.

## Example

See `Nufftbindings/README.md` for usage examples of the NUFFT backends.

## Requirements

Tested with:
- Python 3.12+
- CUDA 12.6+
- PyTorch 2.8+
- Numpy, Matplotlib, Kornia, ODL, etc.

See `requirements.txt` for the full list.

## License

MIT License. See `Nufftbindings/LICENSE` for details.

## References

- [Off-the-grid data-driven optimization of sampling schemes in MRI](https://arxiv.org/pdf/2010.01817.pdf)
- [Bayesian Optimization of Sampling Densities in MRI](https://arxiv.org/pdf/2209.07170.pdf)

---

**Note:** This project uses external packages and libraries for core MRI operations and NUFFT functionality. Please cite the respective repositories if you use them in your work.
