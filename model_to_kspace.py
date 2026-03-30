import torch
import json
import os
from utils_3 import get_device, get_rotation_matrix, make_rosette
from models import FourierCurve, Ellipse
import matplotlib.pyplot as plt

def export_kspace_json(checkpoint_path, output_json_path):
    device = get_device()

    # 1. Load the checkpoint
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    params = checkpoint["params"]
    model_name = checkpoint.get("model_name", "FourierCurve")
    
    print("Loading model with params:", params)

    # 2. Reconstruct the model
    tmin = 0
    tmax = params["duration"]
    state_dict = checkpoint["model_state_dict"]

    # Initialize the appropriate model based on the checkpoint
    if "Fourier" in model_name or model_name == "FourierCurve":
        n_coeffs = params.get("model_size", len(state_dict.get('coeffs_x', [])))
        model = FourierCurve(tmin=tmin, tmax=tmax, initial_max=1.0, n_coeffs=n_coeffs).to(device)
    else:
        model = Ellipse(tmin=tmin, tmax=tmax, initial_max=1.0).to(device)

    model.load_state_dict(state_dict)
    model.eval()

    # 3. Generate the trajectory
    timesteps = params["timesteps"]
    dt = params["duration"] / (timesteps - 1)
    t = torch.linspace(0, params["duration"], steps=timesteps, device=device).unsqueeze(1)

    with torch.no_grad():
        traj = model(t)

        # 4. Full rosette
        n_petals = params["n_petals"]
        rotation_matrix = get_rotation_matrix(n_petals, device=device)
        
        kmax_img = params.get("kmax_img", 1.0) 
        zero_filling = params.get("zero_filling", True)

        rosette, _, _ = make_rosette(
            traj, rotation_matrix, n_petals, kmax_img, dt, zero_filling=zero_filling
        )

    # 5. Extract K-space points per petal
    shift = timesteps - 1
    petals = [rosette[i * shift : (i + 1) * shift, :] for i in range(n_petals)]

    # Python floats for JSON file
    kx_data = [p[:, 0].tolist() for p in petals]
    ky_data = [p[:, 1].tolist() for p in petals]

    plt.figure(figsize=(6, 6))
    for i in range(n_petals):
        plt.plot(kx_data[i], ky_data[i])
    plt.title('K-space Trajectory')
    plt.xlabel('Kx (1/m)')
    plt.ylabel('Ky (1/m)')
    plt.grid(True)
    plt.show()

    # 6. JSON file
    export_data = {
        "info": {
            "description": "K-space trajectory extracted from checkpoint"
        },
        "k_space": {
            "Kx": kx_data,
            "Ky": ky_data,
        }
    }

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(export_data, f, indent=4)

    print(f"Successfully exported {n_petals} petals of k-space data to {output_json_path}")


if __name__ == "__main__":
    CHECKPOINT_FILE = "results_loss_compare/pns_threshold_slew_threshold_2026-03-16_17-40/checkpoint.pt"  
    OUTPUT_JSON = "results_loss_compare/pns_threshold_slew_threshold_2026-03-16_17-40/kspace_traj.json" 

    export_kspace_json(CHECKPOINT_FILE, OUTPUT_JSON)