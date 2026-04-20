"""
construct_using_validated.py
============================
Pick-and-place construction script that replays a validated demo sequence
using the ABB IRB120 robot arm.

Workflow
--------
1. Load a validated 7D brick sequence (from training_data/batch1/validated_simPhysics).
2. For each brick in the sequence:
   a. Try each of the 6 pre-defined grasping poses (extracted from grasping_poses.3dm).
   b. For each grasp candidate, plan a full pick-and-place trajectory via
      trajectory_planner_draft.PlanAndExecuteClient.
   c. Use the first grasp that plans collision-free for all phases.
   d. In --dry-run mode: only plan and report; no motion is executed.
   e. In --real mode: execute via MoveIt (/execute_trajectory action).
3. After each successful placement, register the placed brick as a collision
   object so subsequent plans avoid it.

Usage (inside Docker with MoveIt running)
------------------------------------------
  # Dry-run (planning only, no execution -- safe to run anywhere):
  python3 scripts/construct_using_validated.py --demo demo_0

  # Real robot execution (run ONLY inside ros2_real Docker container):
  python3 scripts/construct_using_validated.py --demo demo_0 --real

CLI arguments
-------------
  --demo     : demo name inside validated_simPhysics/ (default: demo_0)
  --data-dir : override path to validated_simPhysics root
  --real     : enable real robot execution (default: dry-run)
  --grasp-id : force a specific grasp (grasp1..grasp6); default tries all
  --supply-xyz: supply pallet XYZ as "x,y,z" (default: 0.0,0.48,0.030)
  --hover-z  : additional Z height for pre/post grasp hover in metres (default: 0.12)

Grasping Poses
--------------
Extracted from src/grasping_poses/grasping_poses.3dm.
Each entry is a 4x4 homogeneous matrix T_grasp_in_brick:
  T_tcp_world = T_brick_world @ T_grasp_in_brick

The brick_pose reference frame in the Rhino model:
  origin  : [0, 0.42, 0.0315]
  X-axis  : [+1,  0,  0]   (longest brick dimension)
  Y-axis  : [ 0, -1,  0]   (medium brick dimension, note flipped)
  Z-axis  : [ 0,  0, +1]   (shortest brick dimension / vertical)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

# ---------------------------------------------------------------------------
# ROS 2 / MoveIt availability guard -- allows offline syntax-check / dry-run
# ---------------------------------------------------------------------------
try:
    import rclpy
    from rclpy.node import Node

    # Import the planner client from the sibling script
    sys.path.insert(0, os.path.dirname(__file__))
    from trajectory_planner_draft_JG import PlanAndExecuteClient

    HAVE_ROS2 = True
except ImportError:
    HAVE_ROS2 = False
    print(
        "WARNING: rclpy / trajectory_planner_draft not importable. "
        "Script will validate logic only (planning calls are no-ops)."
    )

# ---------------------------------------------------------------------------
# Pose conversion utility (Brick class)
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))
from pose_conversion import Brick


# ===========================================================================
# Constants
# ===========================================================================

BRICK_SIZE_XYZ: tuple[float, float, float] = (0.051, 0.023, 0.014)

# Relative transforms from the brick center frame to each gripper TCP pose.
# Derived from src/grasping_poses/grasping_poses.3dm (see module docstring).
#
# Convention:
#   - Axis lengths in Rhino: longest=X (0.020 m), mid=Y (0.015 m), shortest=Z (0.010 m)
#   - The TCP frame Z-axis is built as the *shortest* Rhino axis so it
#     aligns with the gripper approach direction (pointing toward the brick).
#   - T_tcp_world = T_brick_world @ T_GRASP_OFFSETS[name]
#
# Data extracted by the Python script run on the .3dm file:
#   brick_pose T:
#     [[ 1,  0,  0,  0   ],
#      [ 0, -1,  0,  0.42],
#      [ 0,  0,  1,  0.0315],
#      [ 0,  0,  0,  1   ]]
#
# Each T_grasp_in_brick = T_brick_inv @ T_grasp_world


def _t(rot_cols: list[list[float]], pos: list[float]) -> np.ndarray:
    """Build a 4x4 homogeneous matrix from column vectors and position."""
    T = np.eye(4)
    T[:3, :3] = np.column_stack(rot_cols)
    T[:3, 3] = pos
    return T


# grasp5 is the identity-like top-down approach (same orientation as brick frame)
# grasp6 is 180 deg Z-rotated top-down
# grasp1/2 approach from the +X side of brick; grasp3/4 from -X side
# Axis columns are [X_col, Y_col, Z_col] reconstructed from the Rhino extraction.

T_GRASP_OFFSETS: dict[str, np.ndarray] = {
    # Derived from grasping_poses.3dm
    "grasp1": _t(
        rot_cols=[
            [0.7071, 0.0000, -0.7071],
            [0.0000, -1.0000, 0.0000],
            [-0.7071, 0.0000, -0.7071],
        ],
        pos=[0.0176, 0.0000, 0.0028],
    ),
    "grasp2": _t(
        rot_cols=[
            [-0.7071, 0.0000, 0.7071],
            [0.0000, 1.0000, 0.0000],
            [-0.7071, 0.0000, -0.7071],
        ],
        pos=[0.0176, 0.0000, 0.0028],
    ),
    "grasp3": _t(
        rot_cols=[
            [1.0000, 0.0000, 0.0000],
            [0.0000, -1.0000, 0.0000],
            [0.0000, 0.0000, -1.0000],
        ],
        pos=[0.0000, 0.0000, 0.0000],
    ),
    "grasp4": _t(
        rot_cols=[
            [-1.0000, 0.0000, 0.0000],
            [0.0000, 1.0000, 0.0000],
            [0.0000, 0.0000, -1.0000],
        ],
        pos=[0.0000, 0.0000, 0.0000],
    ),
}

# Ordered list for sequential trial
GRASP_ORDER: list[str] = ["grasp1", "grasp2", "grasp3", "grasp4"]

# Default supply pallet pose -- x, y, z (flat brick, no rotation for now)
DEFAULT_SUPPLY_XYZ: tuple[float, float, float] = (0.0, -0.48, 0.030)
DEFAULT_SUPPLY_QUAT_XYZW: tuple[float, float, float, float] = (
    0.0,
    0.0,
    0.0,
    1.0,
)  # flat, identity

# Extra Z lift above pickup / placement for hover waypoints (metres)
DEFAULT_HOVER_Z: float = 0.12

# Gazebo SDF path for bricks (used in --sim mode for visual spawning)
_SDF_PATH = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "src",
        "abb_irb120_gazebo",
        "models",
        "brick",
        "model.sdf",
    )
)
_WORLD_NAME = "irb120_workcell"

# Execution modes
MODE_DRY_RUN = "dry_run"  # plan only, no motion
MODE_SIM = "sim"  # plan + execute in Gazebo simulation
MODE_REAL = "real"  # plan + execute on real robot hardware


# ===========================================================================
# Gazebo helpers (sim mode -- subprocess, no ROS spin)
# ===========================================================================


def _gz_spawn(name: str, pose_7d: np.ndarray) -> bool:
    """Spawn a brick SDF into Gazebo.  Returns True on success."""
    if not os.path.isfile(_SDF_PATH):
        print(f"  [gz] WARNING: SDF not found at {_SDF_PATH}, skipping visual spawn.")
        return False
    x, y, z = pose_7d[0], pose_7d[1], pose_7d[2] + 0.001
    qx, qy, qz, qw = pose_7d[3], pose_7d[4], pose_7d[5], pose_7d[6]
    req = (
        f'sdf_filename: \\"{_SDF_PATH}\\" '
        f'name: \\"{name}\\" '
        f"pose: {{ position: {{x: {x} y: {y} z: {z}}} "
        f"orientation: {{x: {qx} y: {qy} z: {qz} w: {qw}}} }}"
    )
    cmd = (
        f"gz service -s /world/{_WORLD_NAME}/create "
        f"--reqtype gz.msgs.EntityFactory --reptype gz.msgs.Boolean "
        f'--timeout 2000 --req "{req}"'
    )
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    ok = result.returncode == 0
    if not ok:
        print(f"  [gz] Spawn failed for {name}: {result.stderr.strip()[:120]}")
    return ok


def _gz_remove(name: str) -> None:
    """Remove a named model from Gazebo."""
    cmd = (
        f"gz service -s /world/{_WORLD_NAME}/remove "
        f"--reqtype gz.msgs.Entity --reptype gz.msgs.Boolean "
        f"--timeout 500 --req 'name: \"{name}\" type: MODEL'"
    )
    subprocess.run(
        cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )


_SUPPLY_GZ_NAME = "supply_brick_gz"  # Gazebo model name for the feeding brick


def _gz_fetch_model_names() -> list[str]:
    """Return all top-level model names visible in the Gazebo pose stream."""
    names: list[str] = []
    try:
        output = subprocess.check_output(
            f"gz topic -e -n 1 -t /world/{_WORLD_NAME}/pose/info",
            shell=True,
            text=True,
            timeout=5.0,
        )
        for block in output.split("pose {"):
            m = re.search(r'name:\s+"([^"]+)"', block)
            if not m:
                continue
            name = m.group(1)
            # Skip link/visual sub-entries that live inside models
            if name not in ("brick_link", "visual"):
                names.append(name)
    except Exception:
        pass
    return names


def _gz_clean_scene(node: "PlanAndExecuteClient | None") -> None:
    """
    Fully clean the Gazebo world and MoveIt planning scene before construction.

    Mirrors demo_validation.py reset_world():
      1. Trigger ControlWorld reset (respawns default bricks -- we remove them next).
      2. Remove brick_00 .. brick_24  (the 20 pre-existing bricks that respawn).
      3. Remove construct_brick_* and supply_brick_gz from any prior run.
      4. Clear the MoveIt planning scene via remove_all_world_collision_objects().
    """
    print("[gz] Cleaning Gazebo scene (mirroring demo_validation reset_world)...")

    # Step 1: world reset via the node's existing helper (ControlWorld service)
    if node is not None:
        print("[gz]   Step 1: Triggering ControlWorld reset...")
        ok = node.reset_gazebo_simulation()  # calls /reset_simulation service
        if ok:
            print("[gz]   Reset acknowledged. Waiting for world to settle...")
            time.sleep(1.5)
        else:
            print("[gz]   WARNING: ControlWorld reset failed or timed out.")

    # Step 2: purge brick_00 .. brick_24 (pre-existing bricks that respawn)
    print("[gz]   Step 2: Removing pre-existing brick_00..brick_24...")
    for i in range(25):
        _gz_remove(f"brick_{i:02d}")

    # Step 3: purge our own construct_brick_* and supply_brick_gz
    print("[gz]   Step 3: Removing construct_brick_* and supply_brick_gz...")
    names = _gz_fetch_model_names()
    to_remove = [
        n for n in names if n.startswith("construct_brick_") or n == _SUPPLY_GZ_NAME
    ]
    for name in to_remove:
        _gz_remove(name)
    if to_remove:
        print(f"[gz]   Removed {len(to_remove)} leftover construct model(s).")

    time.sleep(1.0)  # let Gazebo finish all removals

    # Step 4: clear MoveIt planning scene
    if node is not None:
        print("[gz]   Step 4: Clearing MoveIt planning scene...")
        node.remove_all_world_collision_objects()

    print("[gz] Scene cleanup complete.")


# ===========================================================================
# Helpers
# ===========================================================================


def apply_local_rotation(
    pose_7d: np.ndarray, rx_deg: float, ry_deg: float, rz_deg: float
) -> np.ndarray:
    """Apply local euler rotations to a 7D pose (XYZ + Quat) and return new 7D pose."""
    brick = Brick(pose_7d=pose_7d)
    T_world = brick.to_homogeneous_matrix()
    R_local = R.from_euler("xyz", [rx_deg, ry_deg, rz_deg], degrees=True).as_matrix()
    T_local = np.eye(4)
    T_local[:3, :3] = R_local
    T_new = T_world @ T_local
    b_new = Brick()
    b_new.from_homogeneous_matrix(T_new)
    new_pose = b_new.get_7d_pose()
    return new_pose if isinstance(new_pose, np.ndarray) else np.array(new_pose)


def is_standing_brick(pose_7d: np.ndarray) -> bool:
    """True if the local X-axis is mostly vertical."""
    brick = Brick(pose_7d=pose_7d)
    mat = brick.to_homogeneous_matrix()
    return abs(mat[2, 0]) > 0.9


def generate_fallback_poses(original_goal_7d: np.ndarray, supply_7d_base: np.ndarray):
    """
    Generator yielding tuples of (fallback_goal_7d, fallback_supply_7d, description).
    Laying bricks: X flips (0,180) and Z flips (0,180).
    Standing bricks: X rotations (0, 45, 90, 135, 180, 225, 270, 315) and Z flips (0,180).
    Pickup Pose: Z flips (0, 180).
    """
    is_standing = is_standing_brick(original_goal_7d)

    x_angles = [0, 90, 180, 270, 45, 135, 225, 315] if is_standing else [0, 180]
    z_angles = [0, 180]
    supply_z_angles = [0, 180]

    for sz in supply_z_angles:
        fb_supply_7d = apply_local_rotation(supply_7d_base, 0, 0, sz)
        for rz in z_angles:
            for rx in x_angles:
                desc = (
                    "Original"
                    if rx == 0 and rz == 0 and sz == 0
                    else f"Target=X{rx}_Z{rz}_Supply=Z{sz}"
                )
                yield (
                    apply_local_rotation(original_goal_7d, rx, 0, rz),
                    fb_supply_7d,
                    desc,
                )


def load_demo_sequence(
    demo_name: str,
    data_dir: str,
) -> list[np.ndarray]:
    """
    Load a validated 7D brick sequence from disk.

    Returns a list of np.ndarray, each [x, y, z, qx, qy, qz, qw].
    Looks for:
      <data_dir>/<demo_name>/7d_sequence/sequence.json
    """
    seq_path = os.path.join(data_dir, demo_name, "7d_sequence", "sequence.json")
    if not os.path.isfile(seq_path):
        raise FileNotFoundError(
            f"Validated sequence not found: {seq_path}\n"
            f"Run demo_validation.py first to generate validated sequences."
        )
    with open(seq_path) as f:
        raw = json.load(f)
    poses = [np.array(p, dtype=float) for p in raw]
    print(f"[construct] Loaded {len(poses)} bricks from {seq_path}")
    return poses


def apply_grasp_offset(
    brick_pose_7d: np.ndarray,
    T_grasp_in_brick: np.ndarray,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """
    Compute the gripper TCP pose in the world frame.

    T_tcp_world = T_brick_world @ T_grasp_in_brick

    Returns
    -------
    tcp_xyz        : (x, y, z) position in world frame
    tcp_quat_xyzw  : (qx, qy, qz, qw) orientation in world frame
    """
    brick = Brick(pose_7d=brick_pose_7d)
    T_brick_world = brick.to_homogeneous_matrix()
    T_tcp_world = T_brick_world @ T_grasp_in_brick

    tcp_xyz = tuple(float(v) for v in T_tcp_world[:3, 3])
    tcp_quat_xyzw = tuple(
        float(v) for v in R.from_matrix(T_tcp_world[:3, :3]).as_quat()
    )
    return tcp_xyz, tcp_quat_xyzw  # type: ignore[return-value]


def hover_above(
    xyz: tuple[float, float, float],
    hover_z: float,
) -> tuple[float, float, float]:
    """Return a pose directly above xyz by hover_z metres."""
    return (xyz[0], xyz[1], xyz[2] + hover_z)


# ===========================================================================
# Per-brick pick-and-place planner
# ===========================================================================


def _extract_last_state(traj) -> tuple[list[str], list[float]] | None:
    if traj is None:
        return None
    jt = traj.joint_trajectory
    if not jt.points:
        return None
    return list(jt.joint_names), list(jt.points[-1].positions)


def plan_brick_sequence(
    node: "PlanAndExecuteClient | None",
    *,
    supply_7d: np.ndarray,
    goal_7d: np.ndarray,
    grasp_id: str,
    T_rel: np.ndarray,
    hover_z: float,
) -> Optional[list["RobotTrajectory"]]:
    """
    Try to plan all phases (Hover Supply -> Grasp Supply -> ... -> Retract Goal).
    Chains the trajectory end-state as the start-state of the next phase.
    Returns the list of 6 planned trajectories if ALL succeed, else None.
    """
    log_info = node.get_logger().info if node else print

    # -- Compute goal TCP pose (apply grasp offset to goal brick frame) ------
    goal_tcp_xyz, goal_tcp_quat = apply_grasp_offset(goal_7d, T_rel)

    # -- Compute supply TCP pose (apply SAME grasp offset to supply brick frame)
    supply_tcp_xyz, supply_tcp_quat = apply_grasp_offset(supply_7d, T_rel)

    supply_hover = hover_above(supply_tcp_xyz, hover_z)
    goal_hover = hover_above(goal_tcp_xyz, hover_z)

    if node is None:
        # Offline mode
        return [None] * 6  # type: ignore

    plans = []
    current_start_state = None

    def _plan_phase(label: str, xyz, quat, lock_wrist: bool = False) -> bool:
        nonlocal current_start_state
        start_names, start_positions = (
            current_start_state if current_start_state else (None, None)
        )

        traj = node.plan_arm_to_pose_constraints(
            group_name="arm",
            link_name="gripper_tcp",
            frame_id="world",
            goal_xyz=tuple(float(v) for v in xyz),
            goal_quat_xyzw=tuple(float(v) for v in quat),
            joint_4_constraints=3.14,  # Expanded to 180 deg. +/- 90 degrees breaks IK workspace routing.
            joint_6_constraints=3.14,
            allowed_planning_time=6.0,
            num_attempts=5,
            start_joint_names=start_names,
            start_joint_positions=start_positions,
            lock_wrist_to_start=lock_wrist,
        )
        if traj is None:
            log_info(f"    [FAIL] Plan failed at {label} for {grasp_id}")
            return False

        plans.append(traj)
        next_state = _extract_last_state(traj)
        if next_state:
            current_start_state = next_state
        return True

    # Offline MoveIt attachment object id
    ghost_id = "ghost_brick"

    if not _plan_phase("hover_supply", supply_hover, supply_tcp_quat):
        return None
    if not _plan_phase(
        "grasp_supply", supply_tcp_xyz, supply_tcp_quat, lock_wrist=True
    ):
        return None

    # GHOST ATTACH: For the transit phases, tell MoveIt the robot is holding the brick
    if node:
        node.attach_box_to_gripper(ghost_id, BRICK_SIZE_XYZ)

    try:
        if not _plan_phase(
            "lift_supply", supply_hover, supply_tcp_quat, lock_wrist=True
        ):
            return None
        if not _plan_phase("hover_goal", goal_hover, goal_tcp_quat):
            return None
        if not _plan_phase("place_goal", goal_tcp_xyz, goal_tcp_quat, lock_wrist=True):
            return None
    finally:
        # GHOST DETACH: Ensure we always release the ghost object, even on failure
        if node:
            node.detach_box_from_gripper(ghost_id)

    if not _plan_phase("retract_goal", goal_hover, goal_tcp_quat, lock_wrist=True):
        return None

    return plans


def execute_brick_sequence(
    node: "PlanAndExecuteClient | None",
    plans: list["RobotTrajectory"],
    mode: str,
    gz_spawn_callable,
    gz_spawn_args,
) -> bool:
    """
    Execute the pre-planned trajectories.
    Handles gripper commands and Gazebo visual spawning.
    """
    dry_run = mode == MODE_DRY_RUN
    if node is None:
        return True

    log = node.get_logger()

    def _exec(label: str, traj) -> bool:
        if dry_run:
            return True
        return node.execute_moveit_trajectory(traj)

    # 0. Open gripper
    if not dry_run:
        node.send_gripper_command(position=0.0, max_velocity=0.05)

    # 1. Hover above supply
    if not _exec("hover_supply", plans[0]):
        return False

    # In sim mode, spawn the physical block at the supply right before grasping!
    # Gazebo string/physics friction will handle moving it.
    if mode == MODE_SIM and gz_spawn_callable:
        gz_spawn_callable(*gz_spawn_args)
        time.sleep(0.3)

    # 2. Move to grasp supply
    if not _exec("grasp_supply", plans[1]):
        return False

    # 3. Close gripper
    if not dry_run:
        node.send_gripper_command(position=0.01, max_velocity=0.05)
    if not dry_run:
        time.sleep(0.5)

    # 4. Lift out from supply
    if not _exec("lift_supply", plans[2]):
        return False

    # 5. Hover above goal
    if not _exec("hover_goal", plans[3]):
        return False

    # 6. Move to goal grasp pose
    if not _exec("place_goal", plans[4]):
        return False

    # 7. Open gripper
    if not dry_run:
        node.send_gripper_command(position=0.0, max_velocity=0.05)
    if not dry_run:
        time.sleep(0.5)

    # 8. Retract above goal
    if not _exec("retract_goal", plans[5]):
        return False

    return True


# ===========================================================================
# Main construction loop
# ===========================================================================


def run_construction(
    node: Optional["PlanAndExecuteClient"],
    demo_poses: list[np.ndarray],
    *,
    supply_xyz: tuple[float, float, float],
    supply_quat_xyzw: tuple[float, float, float, float],
    hover_z: float,
    mode: str,
    forced_grasp: Optional[str],
) -> None:
    """
    Iterate through the demo sequence and pick-and-place each brick.

    For each brick:
    - Try grasps in GRASP_ORDER (or only forced_grasp if specified).
    - Use the first grasp where all planning phases succeed.
    - Register the successfully placed brick as a MoveIt collision object.
    - In sim mode, also spawn the placed brick SDF into Gazebo visually.
    - Log a warning and skip if no grasp succeeds.
    """

    placed_count = 0
    failed_bricks: list[int] = []
    spawned_gz_names: list[str] = []  # tracked for potential cleanup

    print(f"\n[construct] Starting construction -- {len(demo_poses)} bricks")
    mode_label = {
        MODE_DRY_RUN: "DRY-RUN (planning only, no execution)",
        MODE_SIM: "SIM    (plan + execute in Gazebo simulation)",
        MODE_REAL: "REAL   (plan + execute on real robot)",
    }.get(mode, mode)
    print(f"[construct] Mode: {mode_label}\n")

    # In sim mode, perform a full scene cleanup (mirrors demo_validation reset_world)
    # before starting the construction sequence.
    if mode == MODE_SIM:
        _gz_clean_scene(node)

    # Pre-publish the table surface as a collision box so MoveIt avoids it.
    if node is not None:
        node.publish_scene_box(
            object_id="table_surface",
            frame_id="world",
            size_xyz=(2.0, 2.0, 0.02),
            position_xyz=(0.0, 0.0, -0.02),
        )

    supply_7d_base = np.array([*supply_xyz, *supply_quat_xyzw])

    for brick_idx, original_goal_7d in enumerate(demo_poses):
        print(f"\n[construct] -- Brick {brick_idx + 1}/{len(demo_poses)} --")

        is_standing = is_standing_brick(original_goal_7d)
        if forced_grasp:
            active_grasps = [forced_grasp]
        else:
            active_grasps = (
                ["grasp1", "grasp2"] if is_standing else ["grasp3", "grasp4"]
            )

        print(
            f"  [info] Target is {'STANDING' if is_standing else 'LAYING'}. Using grasps: {active_grasps}"
        )

        best_plans = None
        best_goal_7d = None
        best_supply_7d = None
        best_desc = None

        gz_name = f"construct_brick_{brick_idx:03d}"

        for fb_goal_7d, fb_supply_7d, fallback_desc in generate_fallback_poses(
            original_goal_7d, supply_7d_base
        ):
            if best_plans is not None:
                break

            for grasp_id in active_grasps:
                T_rel = T_GRASP_OFFSETS[grasp_id]

                plans = plan_brick_sequence(
                    node,
                    supply_7d=fb_supply_7d,
                    goal_7d=fb_goal_7d,
                    grasp_id=grasp_id,
                    T_rel=T_rel,
                    hover_z=hover_z,
                )

                if plans is not None:
                    print(
                        f"  [OK] Found valid sequence: {fallback_desc}, Grasp=[{grasp_id}]"
                    )
                    best_plans = plans
                    best_goal_7d = fb_goal_7d
                    best_supply_7d = fb_supply_7d
                    best_desc = fallback_desc
                    break

        if best_plans is None:
            print(
                f"  [FAIL] Exhausted all fallback poses and grasps for Brick {brick_idx + 1}. Terminating."
            )
            failed_bricks.append(brick_idx)
            break  # Stop construction entirely

        # Execute the cleanly generated plan
        exec_ok = execute_brick_sequence(
            node,
            best_plans,
            mode,
            gz_spawn_callable=_gz_spawn,
            gz_spawn_args=(gz_name, best_supply_7d),
        )

        if not exec_ok:
            print(f"  [FAIL] Plan executed poorly, terminating early.")
            failed_bricks.append(brick_idx)
            break

        placed_count += 1
        if mode == MODE_SIM:
            spawned_gz_names.append(gz_name)

        # Register the placed brick as a permanent MoveIt collision object.
        if node is not None:
            brick_obj_id = f"placed_brick_{brick_idx:03d}"
            brick_quat_xyzw = tuple(float(v) for v in best_goal_7d[3:])
            node.publish_scene_box(
                object_id=brick_obj_id,
                frame_id="world",
                size_xyz=BRICK_SIZE_XYZ,
                position_xyz=tuple(float(v) for v in best_goal_7d[:3]),
                quat_xyzw=brick_quat_xyzw,
            )
            print(f"  [scene] Added collision object '{brick_obj_id}'")

        # Optional: rest slightly
        if not (mode == MODE_DRY_RUN):
            time.sleep(1.0)

    # -- Summary -----------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"[construct] Construction complete.")
    print(f"  Placed  : {placed_count}/{len(demo_poses)} bricks")
    print(f"  Skipped : {len(failed_bricks)} bricks (indices: {failed_bricks})")
    if spawned_gz_names:
        print(f"  Gazebo bricks spawned: {len(spawned_gz_names)}")
    print(f"{'=' * 60}\n")


# ===========================================================================
# CLI
# ===========================================================================


def parse_args() -> argparse.Namespace:
    default_data = os.path.normpath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "training_data",
            "batch1",
            "validated_simPhysics",
        )
    )

    p = argparse.ArgumentParser(
        description=(
            "Pick-and-place construction from a validated demo sequence.\n"
            "Modes:\n"
            "  (default)  dry-run: plan trajectories, report pass/fail, no motion\n"
            "  --sim    : plan + execute in Gazebo simulation\n"
            "  --real   : plan + execute on the real robot hardware"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--demo",
        default="demo_0",
        help="Demo name in validated_simPhysics/ (default: demo_0)",
    )
    p.add_argument(
        "--data-dir",
        default=default_data,
        help="Root directory of validated sequences (default: training_data/batch1/validated_simPhysics)",
    )

    # Mutually exclusive execution modes
    mode_group = p.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--sim",
        action="store_true",
        default=False,
        help="Execute trajectories in Gazebo simulation and spawn bricks visually",
    )
    mode_group.add_argument(
        "--real",
        action="store_true",
        default=False,
        help="Execute trajectories on the real robot (CAUTION: moves hardware)",
    )

    p.add_argument(
        "--grasp-id",
        default=None,
        choices=list(T_GRASP_OFFSETS.keys()),
        metavar="GRASP_ID",
        help=(
            "Force a specific grasp variant (grasp1..grasp6). "
            "Default: try all in order " + str(GRASP_ORDER)
        ),
    )
    p.add_argument(
        "--supply-xyz",
        default=None,
        metavar="X,Y,Z",
        help=(
            "Supply pallet XYZ as comma-separated floats "
            f"(default: {DEFAULT_SUPPLY_XYZ[0]},{DEFAULT_SUPPLY_XYZ[1]},{DEFAULT_SUPPLY_XYZ[2]})"
        ),
    )
    p.add_argument(
        "--hover-z",
        type=float,
        default=DEFAULT_HOVER_Z,
        metavar="METRES",
        help=f"Hover height above pick/place poses in metres (default: {DEFAULT_HOVER_Z})",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # -- Parse supply pose --------------------------------------------------
    if args.supply_xyz is not None:
        parts = args.supply_xyz.split(",")
        if len(parts) != 3:
            print("ERROR: --supply-xyz must be X,Y,Z (3 comma-separated floats)")
            sys.exit(1)
        supply_xyz: tuple[float, float, float] = tuple(float(v) for v in parts)  # type: ignore
    else:
        supply_xyz = DEFAULT_SUPPLY_XYZ

    # -- Determine execution mode -------------------------------------------
    if args.real:
        mode = MODE_REAL
    elif args.sim:
        mode = MODE_SIM
    else:
        mode = MODE_DRY_RUN

    # -- Load demo sequence -------------------------------------------------
    demo_poses = load_demo_sequence(args.demo, args.data_dir)

    # -- ROS 2 / MoveIt setup -----------------------------------------------
    node = None
    if HAVE_ROS2:
        rclpy.init()
        node = PlanAndExecuteClient()
        print("[construct] PlanAndExecuteClient ready.")
        print(f"[construct] Mode: {mode}")

        # In non-sim modes, clear any leftover MoveIt collision objects.
        # (In sim mode _gz_clean_scene() handles this inside run_construction.)
        if mode != MODE_SIM:
            node.remove_all_world_collision_objects()
    else:
        print(
            "[construct] ROS 2 not available -- running in offline/print-only mode.\n"
            "            Install rclpy and trajectory_planner_draft dependencies "
            "for actual planning."
        )
        mode = MODE_DRY_RUN  # force dry-run if no ROS

    # -- Run construction ---------------------------------------------------
    try:
        run_construction(
            node,
            demo_poses,
            supply_xyz=supply_xyz,
            supply_quat_xyzw=DEFAULT_SUPPLY_QUAT_XYZW,
            hover_z=args.hover_z,
            mode=mode,
            forced_grasp=args.grasp_id,
        )
    finally:
        if node is not None:
            # Spin briefly to flush any pending DDS messages
            for _ in range(60):
                rclpy.spin_once(node, timeout_sec=0.05)
            node.destroy_node()
            rclpy.shutdown()
            print("[construct] ROS 2 shutdown complete.")


if __name__ == "__main__":
    main()
