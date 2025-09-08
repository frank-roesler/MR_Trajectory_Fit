import torch
import odl
import torch


def get_phantom(size=(1024, 1024)):
    """Generate a Shepp-Logan phantom as a PyTorch tensor."""
    phantom = odl.phantom.shepp_logan(odl.uniform_discr([0, 0], [1, 1], size), modified=True)
    phantom_np = phantom.asarray()
    phantom_tensor = torch.from_numpy(phantom_np).float()
    return phantom_tensor


def rotate_trajectory(traj, angle_degrees):
    """Rotate a 2D trajectory by a given angle in degrees."""
    angle_radians = torch.deg2rad(torch.tensor(angle_degrees))
    rotation_matrix = torch.tensor([[torch.cos(angle_radians), -torch.sin(angle_radians)], [torch.sin(angle_radians), torch.cos(angle_radians)]])
    rotated_traj = traj @ rotation_matrix.T
    return rotated_traj
