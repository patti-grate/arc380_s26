import numpy as np
import os
import math
import copy
try:
    import open3d as o3d
except ImportError:
    pass


def create_plane_mesh(plane, size=1.0):
    """
    Create a mesh plane based on the given plane parameters.

    Args:
        plane (tuple): A tuple containing the plane parameters (normal, d).
            - normal (numpy.ndarray): The normal vector of the plane.
            - d (float): The distance from the origin to the plane.
        size (float, optional): The size of the plane mesh. Defaults to 1.0.

    Returns:
        open3d.geometry.TriangleMesh: The created plane mesh.
    """

    # Create a mesh plane centered at the origin
    mesh = o3d.geometry.TriangleMesh.create_box(width=size, height=size, depth=0.001)
    mesh.translate([-size / 2, -size / 2, -0.005])  # Centering the plane mesh

    # Calculate rotation
    normal = plane[0]
    # Rotation between z-axis and plane normal
    rotation_axis = np.cross([0, 0, 1], normal)
    if np.linalg.norm(rotation_axis) < 1e-6:  # Check if the vectors are parallel
        rotation_axis = [1, 0, 0]
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
    rotation_angle = np.arccos(np.dot([0, 0, 1], normal))
    R = mesh.get_rotation_matrix_from_axis_angle(rotation_axis * rotation_angle)

    # Apply transformations
    mesh.rotate(R, center=[0, 0, 0])
    # Move the mesh to the correct distance from the origin
    d = plane[1]
    translation_distance = -d / np.linalg.norm(normal)
    mesh.translate(normal * translation_distance)

    return mesh


def random_transformation(scale=1.0):
    """
    Generates a random 3D rotation and translation.

    :param scale: Scale for the translation vector magnitude.
    :return: 4x4 transformation matrix (numpy array).
    """
    # Random rotation matrix
    angle_x = np.random.uniform(0, np.pi/4)
    angle_y = np.random.uniform(0, np.pi/4)
    angle_z = np.random.uniform(0, np.pi/4)
    R = o3d.geometry.get_rotation_matrix_from_xyz((angle_x, angle_y, angle_z)) # Euler Angles

    # Random translation vector
    t = np.random.uniform(-scale, scale, size=(3,))

    # Combine into a homogeneous transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    return T

