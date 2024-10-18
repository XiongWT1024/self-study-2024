import math
import numpy as np
import torch
import torch.nn.functional as F
import open3d as o3d
from scipy.spatial.transform import Rotation

def get_02v_bone_transforms(Jtr,):
    rot45p = Rotation.from_euler('z', 45, degrees=True).as_matrix()
    rot45n = Rotation.from_euler('z', -45, degrees=True).as_matrix()

    # Specify the bone transformations that transform a SMPL A-pose mesh
    # to a star-shaped A-pose (i.e. Vitruvian A-pose)
    bone_transforms_02v = np.tile(np.eye(4), (24, 1, 1))

    # First chain: L-hip (1), L-knee (4), L-ankle (7), L-foot (10)
    chain = [1, 4, 7, 10]
    rot = rot45p.copy()
    for i, j_idx in enumerate(chain):
        bone_transforms_02v[j_idx, :3, :3] = rot
        t = Jtr[j_idx].copy()
        if i > 0:
            parent = chain[i-1]
            t_p = Jtr[parent].copy()
            t = np.dot(rot, t - t_p)
            t += bone_transforms_02v[parent, :3, -1].copy()

        bone_transforms_02v[j_idx, :3, -1] = t

    bone_transforms_02v[chain, :3, -1] -= np.dot(Jtr[chain], rot.T)
    # Second chain: R-hip (2), R-knee (5), R-ankle (8), R-foot (11)
    chain = [2, 5, 8, 11]
    rot = rot45n.copy()
    for i, j_idx in enumerate(chain):
        bone_transforms_02v[j_idx, :3, :3] = rot
        t = Jtr[j_idx].copy()
        if i > 0:
            parent = chain[i-1]
            t_p = Jtr[parent].copy()
            t = np.dot(rot, t - t_p)
            t += bone_transforms_02v[parent, :3, -1].copy()

        bone_transforms_02v[j_idx, :3, -1] = t

    bone_transforms_02v[chain, :3, -1] -= np.dot(Jtr[chain], rot.T)

    return bone_transforms_02v

def save_vertices_to_ply(vertices, file_name):
    point_cloud = o3d.geometry.PointCloud()

    point_cloud.points = o3d.utility.Vector3dVector(vertices)
    
    o3d.io.write_point_cloud(file_name, point_cloud)

def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)