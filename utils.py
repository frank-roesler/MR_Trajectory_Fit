import torch
import odl
import torch


def get_phantom(size=(1024, 1024)):
    """Generate a Shepp-Logan phantom as a PyTorch tensor."""
    phantom = odl.phantom.shepp_logan(odl.uniform_discr([0, 0], [1, 1], size), modified=True)
    phantom_np = phantom.asarray()
    phantom_tensor = torch.from_numpy(phantom_np).float()
    return phantom_tensor


def get_phantom2(size=(1024, 1024), phantom_fraction=0.5):
    """
    Generate a Shepp-Logan phantom as a PyTorch tensor, centered in a larger black image.
    phantom_fraction: fraction of the output size to fill with the phantom (0 < phantom_fraction <= 1)
    """
    # Compute phantom size
    phantom_size = (int(size[0] * phantom_fraction), int(size[1] * phantom_fraction))
    phantom = odl.phantom.shepp_logan(odl.uniform_discr([0, 0], [1, 1], phantom_size), modified=True)
    phantom_np = phantom.asarray()
    phantom_tensor = torch.from_numpy(phantom_np).float()

    # Create larger black image
    result = torch.zeros(size, dtype=phantom_tensor.dtype)
    y0 = (size[0] - phantom_size[0]) // 2
    x0 = (size[1] - phantom_size[1]) // 2
    result[y0 : y0 + phantom_size[0], x0 : x0 + phantom_size[1]] = phantom_tensor
    return result


def rotate_trajectory(traj, angle_degrees):
    """Rotate a 2D trajectory by a given angle in degrees."""
    angle_radians = torch.deg2rad(torch.tensor(angle_degrees))
    rotation_matrix = torch.tensor([[torch.cos(angle_radians), -torch.sin(angle_radians)], [torch.sin(angle_radians), torch.cos(angle_radians)]])
    rotated_traj = traj @ rotation_matrix.T
    return rotated_traj


# region k-space interpolation (not used)
# def fast_kspace_interpolation(kspace_data, rosette_traj):
#     kx_traj = np.real(rosette_traj)
#     ky_traj = np.imag(rosette_traj)

#     # Filter grid points to only those near trajectory (spatial optimization)
#     trajectory_points = np.vstack((kx_traj, ky_traj)).T
#     nearby_indices = set()
#     for traj_point in trajectory_points:
#         indices = tree.query_ball_point(traj_point, max_dist)
#         nearby_indices.update(indices)

#     nearby_indices = list(nearby_indices)
#     # print(f"  Using {len(nearby_indices)} grid points (vs {len(grid_points)} original)")

#     # Use only nearby points for interpolation
#     filtered_points = grid_points[nearby_indices]
#     filtered_values = kspace_data.flatten()[nearby_indices]

#     interp_points = np.vstack((kx_traj, ky_traj)).T
#     kspace_sampled = griddata(filtered_points, filtered_values, interp_points, method="linear", fill_value=0)

#     return kspace_sampled
# endregion
