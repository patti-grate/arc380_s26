import numpy as np
from scipy.spatial.transform import Rotation as R
import itertools


class Brick:
    # Default dimensions (x, y, z) as requested
    DEFAULT_DIMS = np.array([0.051, 0.023, 0.014])

    def __init__(self, corners=None, pose_7d=None):
        """
        Initialize the Brick.
        :param corners: 8x3 numpy array of corner coordinates.
        :param pose_7d: 7-element list/array [x, y, z, qx, qy, qz, qw].
        """
        self.dimensions = self.DEFAULT_DIMS.copy()

        if pose_7d is not None:
            self.from_7d_pose(pose_7d)
        elif corners is not None:
            self._init_from_corners(np.array(corners))
        else:
            # Default pose at origin, identity rotation
            self.pose_7d = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

    def _init_from_corners(self, corners):
        """
        Initialize the 7D pose from 8 unordered corners.
        Finds the orientation with the highest cosine similarity to the world frame.
        """
        center = np.mean(corners, axis=0)
        centered = corners - center
        cov = np.cov(centered.T)

        # Eigenvalues correspond to the variance along the principal axes.
        # The axes with the smallest to largest variances correspond to
        # the shortest to longest dimensions (Z, Y, X).
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # eigenvectors[:, i] is the i-th principal axis
        # Sort order from eigh is ascending eigenvalues, so:
        # idx 0 -> shortest (Z)
        # idx 1 -> middle (Y)
        # idx 2 -> longest (X)
        axis_z_raw = eigenvectors[:, 0]
        axis_y_raw = eigenvectors[:, 1]
        axis_x_raw = eigenvectors[:, 2]

        # Ensure we pick a right-handed coordinate frame that is as close to
        # the identity world frame as possible.
        best_trace = -np.inf
        best_R = np.eye(3)

        # Test all 8 possible sign combinations
        for sx, sy, sz in itertools.product([1, -1], repeat=3):
            vx = sx * axis_x_raw
            vy = sy * axis_y_raw
            vz = sz * axis_z_raw

            rot_mat = np.column_stack((vx, vy, vz))

            # Must be a valid rotation matrix (det == 1)
            if np.isclose(np.linalg.det(rot_mat), 1.0):
                # Trace is proportional to cosine similarity with identity matrix
                trace = np.trace(rot_mat)
                if trace > best_trace:
                    best_trace = trace
                    best_R = rot_mat

        quat = R.from_matrix(best_R).as_quat()  # returns [x, y, z, w] by default
        self.pose_7d = np.concatenate([center, quat])

    def get_7d_pose(self):
        """Returns [x, y, z, qx, qy, qz, qw]"""
        return self.pose_7d.copy()

    def from_7d_pose(self, pose_7d):
        self.pose_7d = np.array(pose_7d)

    def to_homogeneous_matrix(self):
        """Converts the 7D pose to a 4x4 homogeneous transformation matrix"""
        mat = np.eye(4)
        mat[:3, :3] = R.from_quat(self.pose_7d[3:]).as_matrix()
        mat[:3, 3] = self.pose_7d[:3]
        return mat

    def from_homogeneous_matrix(self, matrix):
        """Updates the internal 7D pose from a 4x4 matrix"""
        center = matrix[:3, 3]
        quat = R.from_matrix(matrix[:3, :3]).as_quat()
        self.pose_7d = np.concatenate([center, quat])

    def to_5d_pose(self):
        """
        Converts to [x, y, z, state, yaw].
        State is 0 if laying on largest face (Z is vertical),
        1 if standing on a smaller face (X or Y is vertical).
        Returns a string error message if not convertible via defensive programming.
        """
        rot_mat = R.from_quat(self.pose_7d[3:]).as_matrix()
        world_z = np.array([0, 0, 1])

        # Check alignment of local axes with world Z
        dots = np.abs(
            rot_mat.T @ world_z
        )  # [|dot(X_loc, Z_w)|, |dot(Y_loc, Z_w)|, |dot(Z_loc, Z_w)|]

        if not np.any(dots > 0.99):
            return "Error: Brick configuration is not aligned to the vertical axis (it is tilted). Cannot convert to 5D pose."

        # Determine standing (1) vs laying (0)
        # Z is shortest dim. If Z is vertical, it's laying on the largest face.
        if dots[2] > 0.99:
            state = 0
            # Yaw is rotation of local X in world XY plane
            yaw = np.arctan2(rot_mat[1, 0], rot_mat[0, 0])
        else:
            state = 1
            # If Y is vertical, yaw is based on X. If X is vertical, yaw is based on Y.
            if dots[1] > 0.99:  # Y is vertical
                yaw = np.arctan2(rot_mat[1, 0], rot_mat[0, 0])
            else:  # X is vertical
                yaw = np.arctan2(rot_mat[1, 1], rot_mat[0, 1])

        return np.array([self.pose_7d[0], self.pose_7d[1], self.pose_7d[2], state, yaw])

    def from_5d_pose(self, pose_5d):
        """
        Reconstructs the 7D pose from a 5D pose [x, y, z, state, yaw].
        """
        x, y, z, state, yaw = pose_5d

        if state == 0:
            # Laying: local Z is along world Z. local X is rotated by yaw.
            rot = R.from_euler("z", yaw)
        else:
            # Standing: We default to Y being vertical (standing on long edge).
            # Local Y corresponds to world Z. Local X is rotated by yaw in world XY.
            base_mat = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
            rot = R.from_euler("z", yaw) * R.from_matrix(base_mat)

        quat = rot.as_quat()
        self.pose_7d = np.array([x, y, z, quat[0], quat[1], quat[2], quat[3]])


if __name__ == "__main__":
    print("Running Tests for Brick Pose Conversion...\n")

    # 1. Create a dummy brick perfectly aligned (Identity Rotation)
    b1 = Brick()
    print("Initial 7D Pose:", b1.get_7d_pose())

    # Generate 8 corners for this brick centered at origin
    dx, dy, dz = Brick.DEFAULT_DIMS / 2.0
    corners = np.array(
        [
            [dx, dy, dz],
            [-dx, dy, dz],
            [dx, -dy, dz],
            [-dx, -dy, dz],
            [dx, dy, -dz],
            [-dx, dy, -dz],
            [dx, -dy, -dz],
            [-dx, -dy, -dz],
        ]
    )

    # Randomly shuffle corners to test PCA initialization
    np.random.shuffle(corners)

    b2 = Brick(corners=corners)
    print("\nReconstructed from shuffled corners 7D Pose:")
    print(np.round(b2.get_7d_pose(), 4))

    # 2. Test 4x4 matrix conversion
    mat = b2.to_homogeneous_matrix()
    b3 = Brick()
    b3.from_homogeneous_matrix(mat)
    print("\nReconstructed from 4x4 Matrix 7D Pose:")
    print(np.round(b3.get_7d_pose(), 4))

    # 3. Test 5D Pose (Laying flat with a yaw)
    yaw_test = np.radians(135)
    b4 = Brick()
    b4.from_5d_pose([0.1, 0.2, 0.3, 0, yaw_test])
    pose_5d = b4.to_5d_pose()
    print(f"\n5D Pose Conversion Test (Laying, Yaw={yaw_test:.2f} rad):")
    print(
        "Recovered 5D Pose:",
        np.round(pose_5d[:5] if isinstance(pose_5d, np.ndarray) else pose_5d, 4),
    )

    # 4. Test 5D Pose (Standing with a yaw)
    b5 = Brick()
    b5.from_5d_pose([0.1, 0.2, 0.3, 1, yaw_test])
    pose_5d_stand = b5.to_5d_pose()
    print(f"\n5D Pose Conversion Test (Standing, Yaw={yaw_test:.2f} rad):")
    print(
        "Recovered 5D Pose:",
        np.round(
            pose_5d_stand[:5]
            if isinstance(pose_5d_stand, np.ndarray)
            else pose_5d_stand,
            4,
        ),
    )

    # 5. Defensive Programming Test (Tilted Brick)
    b6 = Brick()
    # Apply a rotation around X-axis by 30 degrees (tilted, not aligned with Z)
    mat_tilted = np.eye(4)
    mat_tilted[:3, :3] = R.from_euler("x", 30, degrees=True).as_matrix()
    b6.from_homogeneous_matrix(mat_tilted)
    result = b6.to_5d_pose()
    print("\nTilted Brick 5D Conversion Result:")
    print(result)
    print("\nTests completed successfully.")
