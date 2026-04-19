"""
Gripper grasp optimization module.

Flat brick grasp candidates:
  Grasp 1: straight down         (0 deg tilt around Y)
  Grasp 2: 45 deg tilt from X-  (+45 deg around Y)
  Grasp 3: 45 deg tilt from X+  (-45 deg around Y, mirror of grasp 2)

Standing brick: 
  Same 3 grasps + 90 deg roll around X = 6 total

Fingers always close across Y (short axis) since that fits in the 8cm opening.
Z is negated on all grasp quaternions to match the gripper TCP frame convention.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROBOT_BASE_XYZ    = np.array([0.0, 0.0, 0.0])  # world frame
GRIPPER_OPEN_WIDTH = 0.08                        # metres
APPROACH_TILT_DEG  = 45.0                        # tilt for grasp 2 and 3
GRASP_STANDOFF_M   = 0.04                        # TCP offset from brick centre

# ---------------------------------------------------------------------------
# Rotation matrices
# ---------------------------------------------------------------------------

def _rot_x_180() -> R:
    """180 deg around X: flips +Z to -Z, flips +Y to -Y."""
    return R.from_matrix(np.array([
        [1,  0,  0],
        [0, -1,  0],
        [0,  0, -1],
    ], dtype=float))


def _rot_x_90() -> R:
    """90 deg around X: used for standing brick roll."""
    return R.from_matrix(np.array([
        [1, 0,  0],
        [0, 0, -1],
        [0, 1,  0],
    ], dtype=float))


def _rot_y(deg: float) -> R:
    """Rotation around Y axis by deg degrees (positive = tilt from X-)."""
    rad = np.radians(deg)
    c, s = np.cos(rad), np.sin(rad)
    return R.from_matrix(np.array([
        [ c, 0, s],
        [ 0, 1, 0],
        [-s, 0, c],
    ], dtype=float))


# ---------------------------------------------------------------------------
# Matrix conversion  
# ---------------------------------------------------------------------------

def matrix_to_pos_quat(matrix_4x4) -> tuple:
    """
    4x4 homogeneous matrix -> (position [x,y,z], quaternion [x,y,z,w])
    scipy / geometry_msgs / MoveIt all use [x,y,z,w].
    """

    m = np.array(matrix_4x4, dtype=float)
    if m.shape != (4, 4):
        raise ValueError(f"Expected (4,4) matrix, got {m.shape}")
    return m[:3, 3], R.from_matrix(m[:3, :3]).as_quat()


def from_transformation_matrices(matrices: list) -> list:
    """
    Convert a list of 4x4 matrices into (position, quat_xyzw) candidates.
    Use this once he sends the Rhino poses.
    """
    return [matrix_to_pos_quat(m) for m in matrices]


# ---------------------------------------------------------------------------
# Grasp candidate generation
# ---------------------------------------------------------------------------

def generate_grasp_candidates(
    brick_pos: np.ndarray,
    brick_quat_xyzw: np.ndarray,
    is_standing: bool = False,
) -> list:
    """
    Generate grasp candidates around a brick.

    Brick frame:
      X = long axis
      Y = short axis  
      Z = up (flat)

    Flat (3 candidates):
      Grasp 1 -- straight down         tilt =   0 deg
      Grasp 2 -- 45 deg lean from X-   tilt = +45 deg
      Grasp 3 -- 45 deg lean from X+   tilt = -45 deg

    Standing adds the same 3 with a 90 deg roll around X (6 total).

    All candidates:
      - Flip Z to -Z so gripper points into the brick (rot_x_180)
      - Z component of quaternion negated (TCP frame convention)
      - TCP offset from brick centre along the approach direction
    """
    brick_rot = R.from_quat(brick_quat_xyzw)

    # Base: flip +Z to -Z so gripper points downward into brick
    base_down = _rot_x_180()

    tilt_angles = [0.0, APPROACH_TILT_DEG, -APPROACH_TILT_DEG]
    roll_rots   = [R.identity(), _rot_x_90()] if is_standing else [R.identity()]

    candidates = []

    for roll_rot in roll_rots:
        for tilt_deg in tilt_angles:
            tilt_rot = _rot_y(tilt_deg)

            # Local gripper orientation: roll -> tilt -> flip Z down
            gripper_rot_local = roll_rot * tilt_rot * base_down

            # Transform into world frame using brick orientation
            gripper_rot_world = brick_rot * gripper_rot_local
            grasp_quat = gripper_rot_world.as_quat()  # [x, y, z, w]

            # Negate Z component -- required for all grasp poses
            grasp_quat[2] = -grasp_quat[2]

            # TCP offset from brick centre:
            #   straight down -> purely above in Z
            #   tilted        -> shifts in X toward approach side
            tilt_rad = np.radians(tilt_deg)
            offset_local = np.array([
                -np.sin(tilt_rad) * GRASP_STANDOFF_M,  # X: toward approach side
                0.0,                                     # Y: centred on short axis
                 np.cos(tilt_rad) * GRASP_STANDOFF_M,  # Z: above brick
            ])
            grasp_pos = brick_pos + brick_rot.apply(offset_local)

            candidates.append((grasp_pos, grasp_quat))

    return candidates


# ---------------------------------------------------------------------------
# Scoring  (lower = better = tried first)
# ---------------------------------------------------------------------------

def score_grasp(
    grasp_pos: np.ndarray,
    grasp_quat_xyzw: np.ndarray,
    brick_pos: np.ndarray,
) -> float:
    """
    Score a candidate. Lower is better.

    Criterion 1 -- distance from robot base (prefer closer TCP positions)
    Criterion 2 -- overshoot past the brick (penalise grasps that reach
                   further from the robot than the brick itself)
    """
    dist      = float(np.linalg.norm(grasp_pos - ROBOT_BASE_XYZ))
    to_b      = brick_pos - ROBOT_BASE_XYZ
    b_dir     = to_b / (np.linalg.norm(to_b) + 1e-9)
    overshoot = float(np.dot(grasp_pos - ROBOT_BASE_XYZ, b_dir))
    return dist + 0.5 * overshoot


# ---------------------------------------------------------------------------
# Main optimisation function
# ---------------------------------------------------------------------------

def get_best_grasp(
    candidates: list,
    brick_pos: np.ndarray,
    node=None,
    group_name: str = "arm",
    link_name: str  = "gripper_tcp",
    frame_id: str   = "world",
) -> Optional[tuple]:
    """
    Return the best valid grasp candidate.

    node=None  ->  offline mode: returns highest-scored without IK check.
    node=...   ->  live mode: tries each candidate via MoveIt in score order,
                   returns first one that plans successfully (collision-aware).

    Returns (position, quaternion_xyzw) or None if everything fails.
    """
    if not candidates:
        print("[grasp_optimizer] No candidates.")
        return None

    scored = sorted(candidates, key=lambda c: score_grasp(c[0], c[1], brick_pos))

    if node is None:
        print("[grasp_optimizer] Offline -- returning best scored candidate.")
        return scored[0]

    for i, (pos, quat) in enumerate(scored):
        print(f"[grasp_optimizer] Testing {i+1}/{len(scored)} ...")
        traj = node.plan_arm_to_pose_constraints(
            group_name=group_name,
            link_name=link_name,
            frame_id=frame_id,
            goal_xyz=tuple(pos),
            goal_quat_xyzw=tuple(quat),
        )
        if traj is not None:
            print(f"[grasp_optimizer] Candidate {i+1} VALID")
            return pos, quat
        print(f"[grasp_optimizer] Candidate {i+1} failed")

    print("[grasp_optimizer] All candidates failed.")
    return None


# ---------------------------------------------------------------------------
# Debug / testing utility
# ---------------------------------------------------------------------------

def test_all_grasps(
    candidates: list,
    brick_pos: np.ndarray,
    node=None,
) -> None:
    """
    Print a ranked summary. node=None = offline (no IK check).
    """
    names = [
        "Flat  -- straight down   (  0 deg)",
        "Flat  -- tilt from X-   (+45 deg)",
        "Flat  -- tilt from X+   (-45 deg)",
        "Stand -- straight down   (  0 deg)",
        "Stand -- tilt from X-   (+45 deg)",
        "Stand -- tilt from X+   (-45 deg)",
    ]

    scored = sorted(
        enumerate(candidates),
        key=lambda ic: score_grasp(ic[1][0], ic[1][1], brick_pos),
    )

    print(f"\n{'='*58}")
    print(f"  Grasp candidates ({len(candidates)} total)")
    print(f"{'='*58}")

    for rank, (idx, (pos, quat)) in enumerate(scored):
        score = score_grasp(pos, quat, brick_pos)
        label = names[idx] if idx < len(names) else f"Candidate {idx}"

        print(f"\n  Rank {rank+1} -- {label}")
        print(f"    Score:    {score:.4f}")
        print(f"    Position: x={pos[0]:.4f}  y={pos[1]:.4f}  z={pos[2]:.4f}")
        print(f"    Quat:     x={quat[0]:.4f}  y={quat[1]:.4f}  z={quat[2]:.4f}  w={quat[3]:.4f}")

        if node is not None:
            traj = node.plan_arm_to_pose_constraints(
                group_name=group_name, link_name="gripper_tcp",
                frame_id="world", goal_xyz=tuple(pos), goal_quat_xyzw=tuple(quat),
            )
            print(f"    MoveIt:   {'VALID' if traj else 'FAILED'}")

    print(f"\n{'='*58}\n")


# ---------------------------------------------------------------------------
# Integration with Irene's sequence()
# ---------------------------------------------------------------------------
#
# Replace the hardcoded supply grasp:
#
#     q_supply = [0, 1, 0, 0]                        <- delete this
#
# With:
#
#     from grasp_optimizer import generate_grasp_candidates, get_best_grasp
#
#     brick_pos  = np.array(p_supply)
#     brick_quat = np.array([0.0, 0.0, 0.0, 1.0])   # flat supply brick
#
#     candidates = generate_grasp_candidates(brick_pos, brick_quat, is_standing=False)
#     result     = get_best_grasp(candidates, brick_pos=brick_pos, node=node)
#
#     if result is None:
#         print(f"Step {step}: no valid grasp -- skipping.")
#         continue
#
#     p_supply, q_supply_xyzw = result
#     # pass q_supply_xyzw into move() with quat_as_wxyz=False
#
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print("Offline test -- flat brick\n")

    brick_pos  = np.array([0.35, -0.26, 0.04])
    brick_quat = np.array([0.0, 0.0, 0.0, 1.0])

    candidates = generate_grasp_candidates(brick_pos, brick_quat, is_standing=False)
    test_all_grasps(candidates, brick_pos=brick_pos, node=None)

    best = get_best_grasp(candidates, brick_pos=brick_pos, node=None)
    if best:
        pos, quat = best
        print(f"Best position:   {np.round(pos, 4)}")
        print(f"Best quaternion: {np.round(quat, 4)}")