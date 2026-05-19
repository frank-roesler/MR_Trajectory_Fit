import torch
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from utils_3 import get_device, make_rosette
from models import FourierCurve, Ellipse
from params import *

def export_kspace_json(checkpoint_path, output_json_path):
    device = get_device()

    # checkpoint
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    params = checkpoint["params"]
    model_name = checkpoint.get("model_name", "FourierCurve")
    
    print("Loading model with params:", params)

    # model reconstruction
    tmin = 0
    tmax = params["duration"]
    state_dict = checkpoint["model_state_dict"]
    n_petals = params["n_petals"]

    angles_shape = state_dict.get("angles", torch.empty(0)).shape[0]
    
    variable_angles = False 
    # False if angles are not present in the checkpoint, True if they are present 

    # model based on checkpoint
    if "Fourier" in model_name or model_name == "FourierCurve":
        n_coeffs = params.get("model_size", 51)
        model = FourierCurve(
            tmin=tmin, 
            tmax=tmax, 
            n_petals=n_petals, 
            initial_max=1.0, 
            n_coeffs=n_coeffs,
            variable_angles=variable_angles
        ).to(device)
    else:
        model = Ellipse(tmin=tmin, tmax=tmax, initial_max=1.0).to(device)

    model.load_state_dict(state_dict)
    model.eval()

    # main traj
    timesteps = params["timesteps"]
    dt = params["duration"] / (timesteps - 1)
    t = torch.linspace(0, params["duration"], steps=timesteps, device=device).unsqueeze(1)

    with torch.no_grad():
        traj = model(t)

        # complete rosette
        kmax_img = params.get("kmax_img", 1.0) 
        zero_filling = params.get("zero_filling", True)

        rosette, _, _ = make_rosette(
            incremental_angles=model.angles, 
            base_traj=traj, 
            n_petals=n_petals, 
            kmax_img=kmax_img, 
            dt=dt, 
            zero_filling=zero_filling
        )


    if hasattr(model, 'angles'):
        inc_angles_deg = model.angles.detach().cpu().numpy() * 180 / np.pi
        abs_angles_deg = torch.cumsum(model.angles, dim=0).detach().cpu().numpy() * 180 / np.pi
        
        print("\n--- Analysis of Angles ---")
        print(f"Incremental angles (degrees):\n {np.round(inc_angles_deg, 2)}")
        print(f"Absolute angles (degrees):\n {np.round(abs_angles_deg, 2)}")
        print("----------------------\n")

    # k space points per petal
    shift = timesteps - 1
    petals = [rosette[i * shift : (i + 1) * shift, :] for i in range(n_petals)]

    kx_data = [p[:, 0].tolist() for p in petals]
    ky_data = [p[:, 1].tolist() for p in petals]

    plt.figure(figsize=(6, 6))
    for i in range(n_petals):
        plt.plot(kx_data[i], ky_data[i], label=f'Petal {i+1}' if n_petals <= 10 else "")
    
    plt.title('K-space Trajectory (Extracted from Checkpoint)')
    plt.xlabel('Kx (1/m)')
    plt.ylabel('Ky (1/m)')
    plt.axis("equal")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()

    # JSON
    export_data = {
        "info": {
            "description": "K-space trajectory extracted from checkpoint",
            "n_petals": n_petals,
            "timesteps_per_petal": shift
        },
        "k_space": {
            "Kx": kx_data,
            "Ky": ky_data,
        }
    }

    output_dir = os.path.dirname(output_json_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    with open(output_json_path, "w") as f:
        json.dump(export_data, f, indent=4)

    print(f"Successfully exported {n_petals} petals of k-space data to {output_json_path}")


if __name__ == "__main__":
    CHECKPOINT_FILE = "results/2026-05-19_19-20/checkpoint.pt"  
    OUTPUT_JSON = "results/2026-05-19_19-20/kspace_traj.json" 

    export_kspace_json(CHECKPOINT_FILE, OUTPUT_JSON)