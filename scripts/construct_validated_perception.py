"""
construct_validated_perception.py
==================================
Perception-integrated pick-and-place construction script.
Detects the supply brick pose via camera each cycle, then plans and
executes brick placement using the ABB IRB120 robot arm.

Workflow
--------
1. Load a validated 7D brick sequence.
2. Home the robot to SAFE_HOME.
3. For each brick:
   a. Prompt the operator to place a brick in the supply area.
   b. Call perception_simple.py to capture a frame and detect the brick pose.
   c. Try grasp candidates; plan collision-free trajectories with MoveIt.
   d. If planning succeeds, show the plan summary and ask for execution confirmation.
   e. On confirmation, execute on real robot (or dry-run if no --real flag).
   f. Register the placed brick as a MoveIt collision object.

Usage (inside Docker with MoveIt running)
------------------------------------------
  # Dry-run (plan only, perception active, no motion):
  python3 scripts/construct_validated_perception.py --demo demo_0

  # Real robot execution:
  python3 scripts/construct_validated_perception.py --demo demo_0 --real

CLI arguments
-------------
  --demo        : demo name inside validated_simPhysics/ (default: demo_0)
  --data-dir    : override path to validated_simPhysics root
  --real        : enable real robot execution (default: dry-run)
  --grasp-id    : force a specific grasp (grasp1..grasp6); default tries all
  --hover-z     : additional Z height for pre/post grasp hover in metres (default: 0.12)
  --skip-perception : use --supply-xyz instead of camera detection
  --supply-xyz  : fallback supply XYZ when --skip-perception is set

Notes on collision/joint checking in real mode
----------------------------------------------
MoveIt planning (OMPL + IK + collision scene) runs identically in the real
container. The planning scene contains the table surface and all previously
placed bricks, so collision checking is meaningful without Gazebo.
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

    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "examples"))
        from egm_example import EGMClient

        HAVE_EGM = True
    except ImportError as e2:
        HAVE_EGM = False
        print(f"WARNING: egm_example not importable ({e2}). Real robot mode disabled.")

    HAVE_ROS2 = True
except ImportError as e:
    HAVE_ROS2 = False
    print(
        f"WARNING: rclpy / trajectory_planner_draft not importable: {e}\n"
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


# Axis columns are [X_col, Y_col, Z_col] reconstructed from the Rhino extraction.

import os

def _load_dynamic_grasps(filepath: str):
    import rhino3dm
    if not os.path.exists(filepath):
        print(f"[warning] Grasp file not found: {filepath}. Using empty grasps.")
        return {}, []
    
    m = rhino3dm.File3dm.Read(filepath)
    if not m:
        return {}, []

    grasps = {}
    for layer in m.Layers:
        name = layer.Name
        if name.startswith("grasping_pose"):
            grasps[name] = None
            
    brick_center = np.array([0.0, 0.42, 0.0315])  # fallback
    for o in m.Objects:
        if m.Layers[o.Attributes.LayerIndex].Name == "brick_pose":
            if isinstance(o.Geometry, rhino3dm.Point):
                brick_center = np.array([
                    o.Geometry.Location.X,
                    o.Geometry.Location.Y,
                    o.Geometry.Location.Z
                ])
                break

    for name in list(grasps.keys()):
        pt = None
        curves = []
        for o in m.Objects:
            if m.Layers[o.Attributes.LayerIndex].Name == name:
                if isinstance(o.Geometry, rhino3dm.Point):
                    pt = o.Geometry
                elif isinstance(o.Geometry, rhino3dm.PolylineCurve):
                    curves.append(o.Geometry)
        
        if pt and len(curves) == 3:
            origin = np.array([pt.Location.X, pt.Location.Y, pt.Location.Z])
            pos = origin - brick_center
            
            vecs = []
            for c in curves:
                start = np.array([c.Point(0).X, c.Point(0).Y, c.Point(0).Z])
                end = np.array([c.Point(c.PointCount-1).X, c.Point(c.PointCount-1).Y, c.Point(c.PointCount-1).Z])
                vec = end - start
                vecs.append(vec)
            
            vecs.sort(key=np.linalg.norm)
            z_vec = vecs[0] / np.linalg.norm(vecs[0])
            y_vec = vecs[1] / np.linalg.norm(vecs[1])
            x_vec = vecs[2] / np.linalg.norm(vecs[2])
            
            y_vec = np.cross(z_vec, x_vec)
            
            T = np.eye(4)
            T[:3, 0] = x_vec
            T[:3, 1] = y_vec
            T[:3, 2] = z_vec
            T[:3, 3] = pos
            
            short_name = name.replace("ing_pose", "")
            grasps[short_name] = T

    valid_grasps = {k: v for k, v in grasps.items() if v is not None}
    order = sorted(valid_grasps.keys())
    return valid_grasps, order

_RHINO_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "src", "grasping_poses", "grasping_poses.3dm"))
T_GRASP_OFFSETS, GRASP_ORDER = _load_dynamic_grasps(_RHINO_PATH)

# Default supply pallet pose -- x, y, z (flat brick, no rotation for now)
DEFAULT_SUPPLY_XYZ: tuple[float, float, float] = (-0.20, 0.40, 0.030)
DEFAULT_SUPPLY_QUAT_XYZW: tuple[float, float, float, float] = (
    0.0,
    0.0,
    0.0,
    1.0,
)  # flat, identity

# Extra Z lift above pickup / placement for hover waypoints (metres)
DEFAULT_HOVER_Z: float = 0.12

# Safe home joint configuration -- elevated slightly above table to avoid
# MoveIt goal-state collision rejection at the Gazebo spawn frame.
SAFE_HOME_NAMES: list[str] = [
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
]
SAFE_HOME_POSITIONS: list[float] = [1.57, 0.00, 0.00, 0.00, 1.57, 1.57]

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

# Real robot gripper is physically mounted +45° (π/4 rad) around the joint_6 axis
# relative to the sim URDF model. Subtract this offset from all joint_6 positions
# when replaying sim-planned trajectories on the real robot.
REAL_GRIPPER_J6_OFFSET_RAD: float = -np.pi / 4

# Real-mode Cartesian z corrections applied at PLANNING TIME.
# These compensate for differences between sim and physical table/brick heights.
# Physical supply bricks sit ~5 mm higher than sim (z=0.025 sim → z=0.030 real).
REAL_SUPPLY_Z: float = 0.030
# Lower goal placement by 2 mm so bricks seat flush on the real surface.
REAL_GOAL_Z_OFFSET_M: float = -0.002


# ===========================================================================
# Gazebo helpers (sim mode -- subprocess, no ROS spin)
# ===========================================================================


def _gz_spawn(name: str, pose_7d: np.ndarray) -> bool:
    """Spawn a brick SDF into Gazebo.  Returns True on success."""
    if not os.path.isfile(_SDF_PATH):
        print(f"  [gz] WARNING: SDF not found at {_SDF_PATH}, skipping visual spawn.")
        return False
    x, y, z = pose_7d[0], pose_7d[1], pose_7d[2] + 0.005
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


def _gz_get_pose(name: str):
    """Query Gazebo for the true 7D pose [x,y,z,qx,qy,qz,qw] of a model."""
    import re
    cmd = f'gz model -m {name}'
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=2.0)
        if result.returncode != 0:
            return None
        out = result.stdout
        
        pos_match = re.search(r"position\s*\{.*?x:\s*([-\d\.e]+).*?y:\s*([-\d\.e]+).*?z:\s*([-\d\.e]+)", out, re.DOTALL)
        ori_match = re.search(r"orientation\s*\{.*?x:\s*([-\d\.e]+).*?y:\s*([-\d\.e]+).*?z:\s*([-\d\.e]+).*?w:\s*([-\d\.e]+)", out, re.DOTALL)
        
        if pos_match and ori_match:
            x, y, z = float(pos_match.group(1)), float(pos_match.group(2)), float(pos_match.group(3))
            qx, qy, qz, qw = float(ori_match.group(1)), float(ori_match.group(2)), float(ori_match.group(3)), float(ori_match.group(4))
            return np.array([x, y, z, qx, qy, qz, qw])
    except Exception:
        pass
    return None

def _gz_remove_batch(brick_names: list[str], settle_sec: float = 1.0) -> None:
    """Remove a list of named models from Gazebo in parallel."""
    processes = []
    for name in brick_names:
        cmd = f'source /opt/ros/jazzy/setup.bash && gz service -s /world/irb120_workcell/remove --reqtype gz.msgs.Entity --reptype gz.msgs.Boolean --timeout 500 --req "name: \\"{name}\\" type: 2"'
        # Use executable='/bin/bash' because 'source' is a bash built-in
        p = subprocess.Popen(
            cmd,
            shell=True,
            executable="/bin/bash",
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        processes.append(p)

    # Wait for all removals to finish
    for p in processes:
        p.wait()

    if settle_sec > 0:
        time.sleep(settle_sec)


_SUPPLY_GZ_NAME = "supply_brick_gz"  # Gazebo model name for the feeding brick
_PLACED_STRUCTURE_NAME = "placed_structure"  # single static model consolidating all placed bricks


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
    """
    print("[gz] Cleaning Gazebo scene...")

    # Step 1: purge brick_00 .. brick_24
    print("[gz]   Removing pre-existing brick_00..brick_24...")
    bricks_to_remove = [f"brick_{i:02d}" for i in range(25)]

    # Step 2: purge our own construct_brick_*, supply_brick_gz, and any stale grip weld
    names = _gz_fetch_model_names()
    to_remove = [
        n for n in names
        if n.startswith("construct_brick_") or n in (_SUPPLY_GZ_NAME, _PLACED_STRUCTURE_NAME, "grip_weld")
    ]
    bricks_to_remove.extend(to_remove)

    if node is not None:
        print("[gz]   Calling Gazebo Reset service...")
        # Use a short timeout so we fail fast when simulation_reset_node isn't running
        node.reset_gazebo_simulation(service_wait_sec=5.0)
        time.sleep(1.0)

        if bricks_to_remove:
            print(
                f"[gz]   Removing {len(bricks_to_remove)} models via gz service subprocesses..."
            )
            _gz_remove_batch(bricks_to_remove)

        # Give controllers time to settle after model removals / robot respawn
        time.sleep(3.0)
    else:
        print("[gz]   Calling Gazebo Reset service via subprocess...")
        subprocess.run(
            "source /opt/ros/jazzy/setup.bash && gz service -s /world/irb120_workcell/control --reqtype gz.msgs.WorldControl --req 'reset: {all: true}'",
            shell=True,
            executable="/bin/bash",
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(1.0)
        _gz_remove_batch(bricks_to_remove)

    # Step 3: clear MoveIt planning scene
    if node is not None:
        print("[gz]   Clearing MoveIt planning scene...")
        node.remove_all_world_collision_objects()

    print("[gz] Scene cleanup complete.")


def _gz_spawn_static_structure(placed_poses: list[np.ndarray]) -> None:
    """
    Spawn (or replace) a single static Gazebo model containing all placed
    bricks as individual links.  Replaces the previous placed_structure model
    so Gazebo only tracks one rigid body regardless of brick count.
    """
    # Remove the previous version and wait for Gazebo to commit the removal
    # before issuing the spawn.  With settle_sec=0 the spawn request can arrive
    # while Gazebo is still processing the deletion and will be rejected.
    _gz_remove_batch([_PLACED_STRUCTURE_NAME], settle_sec=0.5)
    if not placed_poses:
        return

    w, h, d = BRICK_SIZE_XYZ
    links_sdf = ""
    for i, pose_7d in enumerate(placed_poses):
        x, y, z = float(pose_7d[0]), float(pose_7d[1]), float(pose_7d[2])
        roll, pitch, yaw = R.from_quat(pose_7d[3:7]).as_euler("xyz")
        links_sdf += (
            f'<link name="b{i:03d}">'
            f"<pose>{x:.6f} {y:.6f} {z:.6f} {roll:.6f} {pitch:.6f} {yaw:.6f}</pose>"
            f'<collision name="c"><geometry><box><size>{w} {h} {d}</size></box></geometry></collision>'
            f'<visual name="v">'
            f"<geometry><box><size>{w} {h} {d}</size></box></geometry>"
            f"<material><ambient>0.76 0.45 0.18 1</ambient><diffuse>0.76 0.45 0.18 1</diffuse></material>"
            f"</visual></link>"
        )
    sdf_str = (
        '<?xml version="1.0"?>'
        '<sdf version="1.8">'
        f'<model name="{_PLACED_STRUCTURE_NAME}">'
        "<static>true</static>"
        f"{links_sdf}"
        "</model></sdf>"
    )

    # Use a fixed, well-known path so Gazebo's file-loader can always find it.
    # Random NamedTemporaryFile paths have caused load failures when Gazebo
    # reads the file asynchronously after the service call returns.
    sdf_path = "/tmp/placed_structure.sdf"
    with open(sdf_path, "w") as _f:
        _f.write(sdf_str)

    # Source ROS so that 'gz' is on PATH in the subprocess environment,
    # matching the pattern used in _gz_remove_batch.
    req = f'sdf_filename: \\"{sdf_path}\\"'
    cmd = (
        f"source /opt/ros/jazzy/setup.bash && "
        f"gz service -s /world/{_WORLD_NAME}/create "
        f"--reqtype gz.msgs.EntityFactory --reptype gz.msgs.Boolean "
        f'--timeout 5000 --req "{req}"'
    )

    for attempt in range(3):
        result = subprocess.run(
            cmd, shell=True, executable="/bin/bash", capture_output=True, text=True
        )
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        if result.returncode == 0:
            print(
                f"  [gz] placed_structure spawned OK "
                f"({len(placed_poses)} brick(s)) — {stdout or 'no output'}"
            )
            # Give Gazebo a moment to finish loading the entity before the
            # next command (e.g. supply brick spawn for the next brick).
            time.sleep(0.5)
            return
        print(
            f"  [gz] Static structure spawn failed (attempt {attempt + 1}/3): "
            f"rc={result.returncode} stderr={stderr[:200]}"
        )
        time.sleep(1.0 + attempt)  # back-off before retry

    print("  [gz] WARNING: placed_structure could not be spawned after 3 attempts.")


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

    # seq_path_supply = os.path.join(data_dir, demo_name, "7d_sequence", "supply.json")
    # if not os.path.isfile(seq_path_supply):
    #     raise FileNotFoundError(
    #         f"Validated sequence not found: {seq_path_supply}\n"
    #         f"Run demo_validation.py first to generate validated sequences."
    #     )
    # with open(seq_path_supply) as f_s:
    #     raw = json.load(f_s)

    # pose_supply = [np.array(p, dtype=float) for p in raw]
    # poses = pose_supply + poses
    print(f"[construct] Loaded {len(poses)} bricks from {seq_path}")
    return poses

def load_supply(
    supply_dir: str,
    data_dir: str,
) -> list[np.ndarray]:
    """
    Load a validated 7D brick sequence from disk.

    Returns a list of np.ndarray, each [x, y, z, qx, qy, qz, qw].
    Looks for:
      <data_dir>/supply_dir/7d_sequence/supply.json
    """
    seq_path = os.path.join(data_dir, supply_dir, "7d_sequence", "supply.json")
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
# Trajectory serialization  (export / replay)
# ===========================================================================

_PHASE_LABELS = [
    "hover_supply",
    "grasp_supply",
    "lift_supply",
    "hover_goal",
    "place_goal",
    "retract_goal",
    "return_home",
]


def _serialize_traj(traj) -> dict:
    """Convert a RobotTrajectory to a plain dict suitable for JSON."""
    jt = traj.joint_trajectory
    pts = []
    for p in jt.points:
        pts.append(
            {
                "positions": list(p.positions),
                "velocities": list(p.velocities),
                "accelerations": list(p.accelerations),
                "t_sec": p.time_from_start.sec,
                "t_nanosec": p.time_from_start.nanosec,
            }
        )
    return {"joint_names": list(jt.joint_names), "points": pts}


def _deserialize_traj(data: dict):
    """Reconstruct a RobotTrajectory from a serialized dict."""
    if not HAVE_ROS2:
        return None
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    from moveit_msgs.msg import RobotTrajectory as _RT
    from builtin_interfaces.msg import Duration

    jt = JointTrajectory()
    jt.joint_names = data["joint_names"]
    for d in data["points"]:
        pt = JointTrajectoryPoint()
        pt.positions = d["positions"]
        pt.velocities = d.get("velocities", [])
        pt.accelerations = d.get("accelerations", [])
        pt.time_from_start = Duration(sec=d["t_sec"], nanosec=d["t_nanosec"])
        jt.points.append(pt)

    rt = _RT()
    rt.joint_trajectory = jt
    return rt


def _scale_trajectory_speed(traj, speed_factor: float):
    """Scale the timestamps, velocities, and accelerations of a RobotTrajectory."""
    if speed_factor == 1.0 or traj is None:
        return traj
    jt = traj.joint_trajectory
    for pt in jt.points:
        t_total = pt.time_from_start.sec + pt.time_from_start.nanosec * 1e-9
        t_scaled = t_total / speed_factor
        pt.time_from_start.sec = int(t_scaled)
        pt.time_from_start.nanosec = int((t_scaled - int(t_scaled)) * 1e9)
        pt.velocities = [v * speed_factor for v in pt.velocities]
        pt.accelerations = [a * (speed_factor**2) for a in pt.accelerations]
    return traj


def _apply_j6_offset(traj, offset_rad: float):
    """Shift joint_6 in every trajectory point by offset_rad.

    Used to compensate for the real gripper being physically mounted at a
    different angle than the URDF model. Modifies the trajectory in-place.
    """
    jt = traj.joint_trajectory
    if "joint_6" not in jt.joint_names:
        return traj
    j6_idx = list(jt.joint_names).index("joint_6")
    for pt in jt.points:
        positions = list(pt.positions)
        positions[j6_idx] += offset_rad
        pt.positions = tuple(positions)
    return traj


def _build_brick_steps(
    plans: list,
    goal_7d: np.ndarray,
    supply_7d: np.ndarray,
    grasp_id: str,
    fallback_desc: str,
    brick_idx: int,
) -> dict:
    """
    Package a brick's 7 planned trajectories + gripper commands into a
    JSON-serializable dict that _run_replay() can execute without replanning.
    """
    steps = []
    steps.append({"type": "gripper", "action": "open"})
    for label, traj in zip(_PHASE_LABELS, plans):
        if traj is not None:
            steps.append({"type": "traj", "label": label, **_serialize_traj(traj)})
        if label == "grasp_supply":
            steps.append({"type": "sleep", "sec": 0.2})
            steps.append({"type": "gripper", "action": "close"})
            steps.append({"type": "sleep", "sec": 0.5})
        elif label == "place_goal":
            steps.append({"type": "sleep", "sec": 0.2})
            steps.append({"type": "gripper", "action": "open"})
            steps.append({"type": "sleep", "sec": 0.5})
    return {
        "brick_idx": brick_idx,
        "grasp_id": grasp_id,
        "fallback_desc": fallback_desc,
        "goal_7d": goal_7d.tolist(),
        "supply_7d": supply_7d.tolist(),
        "steps": steps,
    }


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
        return [None] * 7  # type: ignore

    # ------------------------------------------------------------------
    # Fast IK pre-check: reject geometrically infeasible poses in ~10 ms
    # before spending up to 6 s on OMPL per phase.
    # ------------------------------------------------------------------
    if not node.check_ik(
        "arm",
        node.tcp_link,
        "world",
        supply_tcp_xyz,
        supply_tcp_quat,
        start_joint_names=SAFE_HOME_NAMES,
        start_joint_positions=SAFE_HOME_POSITIONS,
    ):
        log_info(f"    [IK SKIP] Supply TCP infeasible for {grasp_id}")
        return None
    if not node.check_ik(
        "arm",
        node.tcp_link,
        "world",
        goal_tcp_xyz,
        goal_tcp_quat,
        start_joint_names=SAFE_HOME_NAMES,
        start_joint_positions=SAFE_HOME_POSITIONS,
    ):
        log_info(f"    [IK SKIP] Goal TCP infeasible for {grasp_id}")
        return None

    plans = []
    current_start_state = None

    def _plan_phase(
        label: str,
        xyz,
        quat,
        lock_wrist: bool = False,
        wrist_tolerance: float = 0.4,
        speed_factor: float = 1.0,
        cartesian: bool = False,
    ) -> bool:
        nonlocal current_start_state
        start_names, start_positions = (
            current_start_state if current_start_state else (None, None)
        )

        if cartesian:
            from geometry_msgs.msg import Pose
            target_pose = Pose()
            target_pose.position.x = float(xyz[0])
            target_pose.position.y = float(xyz[1])
            target_pose.position.z = float(xyz[2])
            target_pose.orientation.x = float(quat[0])
            target_pose.orientation.y = float(quat[1])
            target_pose.orientation.z = float(quat[2])
            target_pose.orientation.w = float(quat[3])
            
            traj = node.plan_cartesian_path(
                group_name="arm",
                link_name=node.tcp_link,
                frame_id="world",
                waypoints=[target_pose],
                start_joint_names=start_names,
                start_joint_positions=start_positions,
                avoid_collisions=True,
            )
        else:
            traj = node.plan_arm_to_pose_constraints(
                group_name="arm",
                link_name=node.tcp_link,
                frame_id="world",
                goal_xyz=tuple(float(v) for v in xyz),
                goal_quat_xyzw=tuple(float(v) for v in quat),
                joint_4_constraints=2.79,  # ±160 deg (physical limit)
                joint_5_constraints=2.09,  # ±120 deg (physical limit)
                joint_6_constraints=3.14,  # ±180 deg (keeps wrist within [0, 2pi])
                allowed_planning_time=6.0,
                num_attempts=5,
                start_joint_names=start_names,
                start_joint_positions=start_positions,
                lock_wrist_to_start=lock_wrist,
                lock_wrist_tolerance=wrist_tolerance,
            )
        if traj is None:
            log_info(f"    [FAIL] Plan failed at {label} for {grasp_id}")
            return False

        if speed_factor != 1.0:
            traj = _scale_trajectory_speed(traj, speed_factor)

        plans.append(traj)
        next_state = _extract_last_state(traj)
        if next_state:
            # Warn if j6 rotated more than ~120° during this phase so we can
            # track wrist-spin without blocking planning.
            if current_start_state:
                prev_names, prev_pos = current_start_state
                if "joint_6" in prev_names:
                    j6_prev = prev_pos[prev_names.index("joint_6")]
                    new_names, new_pos = next_state
                    if "joint_6" in new_names:
                        j6_delta = abs(new_pos[new_names.index("joint_6")] - j6_prev)
                        if j6_delta > 2.09:  # 2.09 rad ≈ 120°
                            log_info(
                                f"    [j6-spin] {label}: j6 rotated "
                                f"{j6_delta:.2f} rad ({np.degrees(j6_delta):.0f}°)"
                            )
            current_start_state = next_state
        return True

    # Offline MoveIt attachment object id
    ghost_id = "ghost_brick"

    if not _plan_phase("hover_supply", supply_hover, supply_tcp_quat):
        return None
    if not _plan_phase(
        "grasp_supply", supply_tcp_xyz, supply_tcp_quat, lock_wrist=True, cartesian=True
    ):
        return None

    # GHOST ATTACH: For the transit phases, tell MoveIt the robot is holding the brick
    if node:
        node.attach_box_to_gripper(ghost_id, BRICK_SIZE_XYZ)

    try:
        if not _plan_phase(
            "lift_supply", supply_hover, supply_tcp_quat, lock_wrist=True, cartesian=True
        ):
            return None
        # hover_goal is the long transit: do NOT lock the wrist here.
        # Locking j6 to the lift-supply end value causes OMPL to fail when
        # the natural IK solution requires a wrist rotation > the tolerance
        # (observed: ~3.1 rad change in j6 between lift and hover_goal).
        # The spin is cosmetic; correctness requires leaving the wrist free.
        if not _plan_phase("hover_goal", goal_hover, goal_tcp_quat):
            return None
        if not _plan_phase(
            "place_goal", goal_tcp_xyz, goal_tcp_quat, lock_wrist=True, speed_factor=0.3, cartesian=True
        ):
            return None
    finally:
        # GHOST DETACH: Release the attached object.  MoveIt puts detached objects
        # back into the world scene at the gripper TCP position -- explicitly remove
        # it so it doesn't block subsequent bricks' path planning.
        if node:
            node.detach_box_from_gripper(ghost_id)
            node.remove_scene_object(ghost_id)

    if not _plan_phase(
        "retract_goal",
        goal_hover,
        goal_tcp_quat,
        lock_wrist=True,
        wrist_tolerance=0.8,
        speed_factor=0.5,
        cartesian=True,
    ):
        return None

    def _plan_return_home() -> bool:
        nonlocal current_start_state

        start_names, start_positions = (
            current_start_state if current_start_state else (None, None)
        )
        t_names = SAFE_HOME_NAMES
        t_positions = SAFE_HOME_POSITIONS

        traj = node.plan_gripper_to_joint_positions(
            group_name="arm",
            goal_joint_names=t_names,
            goal_joint_positions=t_positions,
            start_joint_names=start_names,
            start_joint_positions=start_positions,
            tolerance=0.08,
            allowed_planning_time=6.0,
            num_attempts=5,
        )
        if traj is None:
            log_info(f"    [FAIL] Plan failed at return_home for {grasp_id}")
            return False

        plans.append(traj)
        next_state = _extract_last_state(traj)
        if next_state:
            current_start_state = next_state
        return True

    if not _plan_return_home():
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
        time.sleep(0.2)
        node.send_gripper_command(position=0.004, max_velocity=0.03)
    if not dry_run:
        time.sleep(1.0)  # allow contact forces to fully settle before lifting

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
        time.sleep(0.2)
        node.send_gripper_command(position=0.0, max_velocity=0.05)
    if not dry_run:
        time.sleep(0.5)

    # 8. Retract above goal
    if not _exec("retract_goal", plans[5]):
        return False

    # 9. Return to home state
    if len(plans) > 6 and plans[6] is not None:
        if not _exec("return_home", plans[6]):
            return False

    return True


# ===========================================================================
# Perception helper
# ===========================================================================


def detect_supply_pose(
    perception_script: str,
    supply_json: str,
    fallback_xyz: tuple[float, float, float],
    fallback_quat: tuple[float, float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """
    Run perception_simple.py as a subprocess, wait for it to write supply.json,
    then return (xyz, quat_xyzw).  Falls back to the provided defaults on any error.
    """
    import subprocess as _sp

    print("[perception] Running camera detection...")
    try:
        result = _sp.run(
            [sys.executable, perception_script],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            print(f"[perception] WARNING: script exited {result.returncode}")
            print(result.stderr[-800:] if result.stderr else "(no stderr)")
            return fallback_xyz, fallback_quat
    except Exception as exc:
        print(f"[perception] ERROR running perception script: {exc}")
        return fallback_xyz, fallback_quat

    try:
        with open(supply_json) as _f:
            data = json.load(_f)
        xyz = tuple(float(v) for v in data["supply_xyz"])       # type: ignore
        quat = tuple(float(v) for v in data["supply_quat_xyzw"])  # type: ignore
        print(
            f"[perception] Detected supply pose: "
            f"xyz=({xyz[0]:.4f}, {xyz[1]:.4f}, {xyz[2]:.4f})  "
            f"quat=({quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}, {quat[3]:.3f})"
        )
        return xyz, quat  # type: ignore
    except Exception as exc:
        print(f"[perception] ERROR reading supply.json: {exc}")
        return fallback_xyz, fallback_quat


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
    export_dir: Optional[str] = None,
    use_perception: bool = True,
    perception_script: Optional[str] = None,
    supply_json: Optional[str] = None,
) -> None:
    """
    Perception-integrated construction loop.

    For each brick:
    1. Home robot to SAFE_HOME.
    2. Prompt operator to place supply brick.
    3. Run perception to detect supply pose (or use fallback xyz).
    4. Try all grasp candidates; plan collision-free trajectories.
    5. On success, ask operator to confirm execution.
    6. Execute (real / dry-run) and register placed brick as collision object.
    """

    placed_count = 0
    failed_bricks: list[int] = []
    spawned_gz_names: list[str] = []  # tracked for potential cleanup
    placed_gz_poses: list[np.ndarray] = []  # settled poses for static structure (sim only)
    export_sequence: list[dict] = []  # filled when export_dir is set

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

    # Ensure the robot is physically at the exact SAFE_HOME_POSITIONS before any plans run
    if node is not None and mode in [MODE_SIM, MODE_REAL]:
        print(
            "[construct] Physically aligning to SAFE_HOME to initialize geometric baseline..."
        )
        homed = False
        for home_attempt in range(5):
            init_traj = node.plan_gripper_to_joint_positions(
                group_name="arm",
                goal_joint_names=SAFE_HOME_NAMES,
                goal_joint_positions=SAFE_HOME_POSITIONS,
                start_joint_names=None,
                start_joint_positions=None,
                tolerance=0.08,
                allowed_planning_time=10.0,
                num_attempts=5,
            )
            if not init_traj:
                print(f"[construct][WARN] SAFE_HOME plan failed (attempt {home_attempt + 1}/5).")
                break
            ok = node.execute_moveit_trajectory(init_traj)
            if ok:
                node.send_gripper_command(position=0.011, max_velocity=0.05)
                homed = True
                break
            print(
                f"[construct][WARN] Homing execution rejected (attempt {home_attempt + 1}/5), "
                "waiting 3s for controllers to recover..."
            )
            time.sleep(3.0)
        if not homed:
            print("[construct][WARN] Could not home robot; first trajectory might reject.")

    for brick_idx, original_goal_7d in enumerate(demo_poses):
        print(f"\n{'=' * 60}")
        print(f"[construct] -- Brick {brick_idx + 1}/{len(demo_poses)} --")
        print(f"{'=' * 60}")

        # ----------------------------------------------------------------
        # Step 1: Home the robot between bricks (skip for very first brick
        # which was already homed above, but home again for safety on real).
        # ----------------------------------------------------------------
        if brick_idx > 0 and node is not None and mode in [MODE_SIM, MODE_REAL]:
            print("[construct] Homing robot before next brick...")
            home_traj = node.plan_gripper_to_joint_positions(
                group_name="arm",
                goal_joint_names=SAFE_HOME_NAMES,
                goal_joint_positions=SAFE_HOME_POSITIONS,
                start_joint_names=None,
                start_joint_positions=None,
                tolerance=0.08,
                allowed_planning_time=10.0,
                num_attempts=5,
            )
            if home_traj:
                node.execute_moveit_trajectory(home_traj)
            else:
                print("[construct][WARN] Could not plan home trajectory; continuing anyway.")

        # ----------------------------------------------------------------
        # Step 2: Prompt operator to place supply brick
        # ----------------------------------------------------------------
        input(
            f"\n[USER] Place brick {brick_idx + 1}/{len(demo_poses)} in the supply area, "
            f"then press ENTER to capture..."
        )

        # ----------------------------------------------------------------
        # Step 3: Detect supply pose via camera (or use fallback)
        # ----------------------------------------------------------------
        if use_perception and perception_script and supply_json:
            detected_xyz, detected_quat = detect_supply_pose(
                perception_script, supply_json,
                fallback_xyz=supply_xyz,
                fallback_quat=supply_quat_xyzw,
            )
        else:
            detected_xyz, detected_quat = supply_xyz, supply_quat_xyzw
            print(f"[construct] Perception skipped. Using supply_xyz={detected_xyz}")

        # Always use the hardcoded supply Z (brick flat on table)
        detected_xyz = (detected_xyz[0], detected_xyz[1], REAL_SUPPLY_Z)
        current_supply_7d = np.array([*detected_xyz, *detected_quat])

        # Apply real-robot z correction to the goal placement pose.
        if mode == MODE_REAL and REAL_GOAL_Z_OFFSET_M != 0.0:
            original_goal_7d = original_goal_7d.copy()
            original_goal_7d[2] += REAL_GOAL_Z_OFFSET_M

        is_standing = is_standing_brick(original_goal_7d)
        if forced_grasp:
            active_grasps = [forced_grasp]
        else:
            active_grasps = (
                ["grasp1", "grasp2"] if is_standing else ["grasp3", "grasp1", "grasp2"]
            )

        print(
            f"  [info] Target is {'STANDING' if is_standing else 'LAYING'}. "
            f"Trying grasps: {active_grasps}"
        )

        # Sim-mode: spawn the visual supply brick and wait for physics settle
        gz_name = f"construct_brick_{brick_idx:03d}"
        if mode == MODE_SIM:
            _gz_spawn(gz_name, current_supply_7d)
            time.sleep(1.0)
            settled_pose = _gz_get_pose(gz_name)
            if settled_pose is not None:
                print(f"  [gz] Physical resting pose acquired: z={settled_pose[2]:.4f}")
                current_supply_7d = settled_pose
            else:
                print("  [gz] Warning: Could not query resting pose, using default.")

        # Try preferred grasps first, then fallback to all 3 if needed
        grasps_to_try = active_grasps + [
            g for g in GRASP_ORDER if g not in active_grasps
        ]

        for grasp_id in grasps_to_try:
            if best_plans is not None:
                break

            T_rel = T_GRASP_OFFSETS[grasp_id].copy()

            if mode == MODE_REAL:
                # Physical gripper is installed 45 degrees CCW (+45 deg around Z).
                # Compensate by rotating the MoveIt target TCP frame 45 degrees CW (-45 deg).
                rad = np.radians(-45.0)
                Rz_correction = np.array(
                    [
                        [np.cos(rad), -np.sin(rad), 0.0, 0.0],
                        [np.sin(rad), np.cos(rad), 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                )
                T_rel = T_rel @ Rz_correction

            for fb_goal_7d, fb_supply_7d, fallback_desc in generate_fallback_poses(
                original_goal_7d, current_supply_7d
            ):
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
                        f"  [OK] Found valid grasp: {fallback_desc}, Grasp=[{grasp_id}]"
                    )
                    best_plans = plans
                    best_goal_7d = fb_goal_7d
                    best_supply_7d = fb_supply_7d
                    best_desc = fallback_desc
                    best_grasp_id = grasp_id
                    break

        if best_plans is None:
            print(
                f"  [FAIL] Exhausted all fallback poses and grasps for Brick {brick_idx + 1}."
            )
            ans = input("  Skip this brick and continue? [y/N]: ").strip().lower()
            if ans == "y":
                failed_bricks.append(brick_idx)
                continue
            else:
                failed_bricks.append(brick_idx)
                break

        # ----------------------------------------------------------------
        # Step 4: Confirm execution with operator (real/sim only)
        # ----------------------------------------------------------------
        print(
            f"\n  [PLAN OK] Grasp={best_grasp_id}, Variant={best_desc}"
            f"\n  Supply: xyz=({best_supply_7d[0]:.4f}, {best_supply_7d[1]:.4f}, {best_supply_7d[2]:.4f})"
            f"\n  Goal:   xyz=({best_goal_7d[0]:.4f}, {best_goal_7d[1]:.4f}, {best_goal_7d[2]:.4f})"
        )
        if mode in [MODE_REAL, MODE_SIM]:
            confirm = input("  Execute this brick? [y/N/skip/abort]: ").strip().lower()
            if confirm == "skip":
                print("  [USER] Skipping brick.")
                failed_bricks.append(brick_idx)
                continue
            elif confirm == "abort":
                print("  [USER] Aborting construction.")
                break
            elif confirm != "y":
                print("  [USER] Execution declined; skipping brick.")
                failed_bricks.append(brick_idx)
                continue

        # Execute the cleanly generated plan
        exec_ok = execute_brick_sequence(
            node,
            best_plans,
            mode,
            gz_spawn_callable=None,
            gz_spawn_args=None,
        )

        if not exec_ok:
            print(f"  [FAIL] Execution rejected by controller.")
            ans = input("  Continue to next brick? [y/N]: ").strip().lower()
            if ans != "y":
                failed_bricks.append(brick_idx)
                break
            failed_bricks.append(brick_idx)
            continue

        placed_count += 1
        if mode == MODE_SIM:
            spawned_gz_names.append(gz_name)

        # Collect export data (only when all phases are real trajectories)
        if export_dir is not None and all(p is not None for p in best_plans):
            export_sequence.append(
                _build_brick_steps(
                    best_plans,
                    best_goal_7d,
                    best_supply_7d,
                    best_grasp_id,
                    best_desc,
                    brick_idx,
                )
            )

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

        # In sim mode: wait 2 s for the brick to settle after the gripper
        # opens, then consolidate it into a single static model so Gazebo only
        # simulates one rigid body regardless of brick count.
        # NOTE: _gz_get_pose(gz_name) returns the SUPPLY/FEED position because
        # the Gazebo brick model has no physical weld to the gripper and never
        # moves.  We use best_goal_7d (the planner's target) as the authoritative
        # placed position so the structure is correctly built at the goal.
        if mode == MODE_SIM:
            print("  [gz] Waiting 2 s for brick to settle before consolidating...")
            time.sleep(2.0)
            _gz_remove_batch([gz_name], settle_sec=0.0)
            placed_gz_poses.append(best_goal_7d)
            _gz_spawn_static_structure(placed_gz_poses)
            print(f"  [gz] Static structure updated ({len(placed_gz_poses)} brick(s))")
        elif not (mode == MODE_DRY_RUN):
            # Non-sim real-execution: brief rest before next brick
            time.sleep(1.0)

    # -- Export ------------------------------------------------------------
    if export_dir is not None and not failed_bricks and export_sequence:
        os.makedirs(export_dir, exist_ok=True)
        # Use demo name derived from the data path or a generic timestamp
        export_path = os.path.join(export_dir, "planned_sequence.json")
        payload = {
            "supply_xyz": list(supply_xyz),
            "supply_quat_xyzw": list(supply_quat_xyzw),
            "hover_z": hover_z,
            "bricks": export_sequence,
        }
        with open(export_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[construct] Trajectories exported to {export_path}")
    elif export_dir is not None and failed_bricks:
        print("[construct] Export skipped: construction was not fully successful.")

    # -- Summary -----------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"[construct] Construction complete.")
    print(f"  Placed  : {placed_count}/{len(demo_poses)} bricks")
    print(f"  Skipped : {len(failed_bricks)} bricks (indices: {failed_bricks})")
    if spawned_gz_names:
        print(f"  Gazebo bricks spawned: {len(spawned_gz_names)}")
    print(f"{'=' * 60}\n")


# ===========================================================================
# Replay (execute pre-planned trajectories without replanning)
# ===========================================================================


def run_replay(
    node: Optional["PlanAndExecuteClient"],
    sequence_path: str,
    *,
    mode: str,
    speed_replay: float = 1.0,
) -> None:
    """
    Load a JSON sequence exported by run_construction (--export-dir) and
    execute every brick's pre-planned trajectories without calling MoveIt
    planning at all.  Collision objects are still registered after each
    placement so MoveIt's execution monitor stays consistent.
    """
    if mode == MODE_SIM:
        _gz_clean_scene(node)

    with open(sequence_path) as f:
        data = json.load(f)

    bricks = data["bricks"]
    hover_z = data.get("hover_z", DEFAULT_HOVER_Z)
    supply_xyz = tuple(data.get("supply_xyz", DEFAULT_SUPPLY_XYZ))
    print(f"[replay] Loaded {len(bricks)} bricks from {sequence_path}")

    dry_run = mode == MODE_DRY_RUN

    if node is not None:
        node.publish_scene_box(
            object_id="table_surface",
            frame_id="world",
            size_xyz=(2.0, 2.0, 0.02),
            position_xyz=(0.0, 0.0, -0.02),
        )

    # Move to SAFE_HOME first (sim only — in real mode the operator positions the robot manually
    # before running, and routing through EGM requires RAPID to be in the EGM motion segment)
    if node is not None and not dry_run and mode != MODE_REAL:
        print("[replay] Homing to SAFE_HOME before replay...")
        homed = False
        for home_attempt in range(5):
            init_traj = node.plan_gripper_to_joint_positions(
                group_name="arm",
                goal_joint_names=SAFE_HOME_NAMES,
                goal_joint_positions=SAFE_HOME_POSITIONS,
                tolerance=0.08,
                allowed_planning_time=10.0,
                num_attempts=5,
            )
            if not init_traj:
                print(f"[replay][WARN] SAFE_HOME plan failed (attempt {home_attempt + 1}/5).")
                break
            ok = node.execute_moveit_trajectory(init_traj)
            if ok:
                node.send_gripper_command(position=0.011, max_velocity=0.05)
                homed = True
                break
            print(
                f"[replay][WARN] Homing execution rejected (attempt {home_attempt + 1}/5), "
                "waiting 3s for controllers to recover..."
            )
            time.sleep(3.0)
        if not homed:
            print("[replay][WARN] Could not home robot; first execution may reject.")
    elif mode == MODE_REAL:
        print(
            "[replay] Real mode: skipping SAFE_HOME init — position robot manually before replay."
        )

    print(f"\n[replay] Starting replay -- mode={mode}\n")

    placed_gz_poses: list[np.ndarray] = []  # settled poses for static structure (sim only)

    for brick_data in bricks:
        brick_idx = brick_data["brick_idx"]
        print(
            f"\n[replay] -- Brick {brick_idx + 1}/{len(bricks)} "
            f"grasp={brick_data['grasp_id']} ({brick_data['fallback_desc']}) --"
        )

        gz_name = f"construct_brick_{brick_idx:03d}"
        supply_7d = np.array(brick_data["supply_7d"])
        goal_7d = np.array(brick_data["goal_7d"])
        gz_spawned = False
        gripper_just_opened = False  # tracks when the gripper releases the brick

        for step in brick_data["steps"]:
            stype = step["type"]
            if stype == "traj":
                label = step.get("label", "?")

                # Spawn the Gazebo supply brick just before the arm descends to
                # grasp, mirroring the timing used in execute_brick_sequence.
                if label == "grasp_supply" and mode == MODE_SIM and not gz_spawned:
                    _gz_spawn(gz_name, supply_7d)
                    gz_spawned = True
                    time.sleep(0.3)

                if not dry_run and node is not None:
                    traj = _deserialize_traj(step)
                    if mode == MODE_REAL:
                        traj = _apply_j6_offset(traj, REAL_GRIPPER_J6_OFFSET_RAD)

                    if speed_replay != 1.0:
                        jt = traj.joint_trajectory
                        for pt in jt.points:
                            t_total = (
                                pt.time_from_start.sec
                                + pt.time_from_start.nanosec * 1e-9
                            )
                            t_scaled = t_total / speed_replay
                            pt.time_from_start.sec = int(t_scaled)
                            pt.time_from_start.nanosec = int(
                                (t_scaled - int(t_scaled)) * 1e9
                            )
                            pt.velocities = [v * speed_replay for v in pt.velocities]
                            pt.accelerations = [
                                a * (speed_replay**2) for a in pt.accelerations
                            ]

                    print(f"    [replay] executing {label} (speed={speed_replay}x)")
                    # Use replay_arm_trajectory (FollowJointTrajectory directly to
                    # the arm controller) instead of execute_moveit_trajectory
                    # (MoveIt /execute_trajectory action).  The MoveIt executor
                    # validates the trajectory's first waypoint against the live
                    # robot state and rejects with CONTROL_FAILED (-4) whenever
                    # there is any start-state mismatch from the pre-planned
                    # trajectory.  replay_arm_trajectory bypasses that check in
                    # sim mode while the real-mode EGM path is unchanged.
                    if mode == MODE_REAL:
                        ok = node.execute_moveit_trajectory(traj)
                    else:
                        ok = node.replay_arm_trajectory(traj)
                    if not ok:
                        print(
                            f"    [replay][WARN] {label} controller rejected — skipping"
                        )
                else:
                    print(f"    [dry] execute {label}")

            elif stype == "gripper":
                if not dry_run and node is not None:
                    pos = 0.004 if step["action"] == "close" else 0.0
                    node.send_gripper_command(position=pos, max_velocity=0.03)
                    if step["action"] == "close":
                        time.sleep(0.1)  # brief settle before lift
                    elif step["action"] == "open":
                        # Track the moment the gripper releases the brick so we
                        # can trigger the 2-second settle + consolidation below.
                        gripper_just_opened = True
                else:
                    print(f"    [dry] gripper {step['action']}")

            elif stype == "sleep":
                if not dry_run:
                    time.sleep(step["sec"])

        # Re-register the placed brick in MoveIt for execution safety
        if node is not None:
            node.publish_scene_box(
                object_id=f"placed_brick_{brick_idx:03d}",
                frame_id="world",
                size_xyz=BRICK_SIZE_XYZ,
                position_xyz=tuple(float(v) for v in goal_7d[:3]),
                quat_xyzw=tuple(float(v) for v in goal_7d[3:]),
            )
            print(f"  [scene] registered placed_brick_{brick_idx:03d}")

        # In sim mode: wait 2 s after the gripper opened so the brick settles
        # under physics, then consolidate all placed bricks into a single static
        # model.  This keeps Gazebo's physics load constant regardless of how
        # many bricks have been placed.
        # NOTE: The supply brick (gz_name) was spawned at the feed and never
        # physically moves in Gazebo (no gripper weld).  Always use goal_7d as
        # the authoritative placed position so the structure appears at the goal.
        if mode == MODE_SIM and not dry_run:
            print("  [gz] Waiting 2 s for brick to settle before consolidating...")
            time.sleep(2.0)
            _gz_remove_batch([gz_name], settle_sec=0.0)
            placed_gz_poses.append(goal_7d)
            _gz_spawn_static_structure(placed_gz_poses)
            print(f"  [gz] Static structure updated ({len(placed_gz_poses)} brick(s))")
        elif not dry_run:
            time.sleep(1.0)

    print(f"\n[replay] Complete -- {len(bricks)} bricks executed.")


# ===========================================================================
# CLI
# ===========================================================================


def parse_args() -> argparse.Namespace:
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
        "--batch",
        default="batch1",
        help="Batch name inside training_data/ (default: batch1)",
    )
    p.add_argument(
        "--demo",
        default="demo_0",
        help="Demo name in validated_simPhysics/ (default: demo_0)",
    )
    p.add_argument(
        "--data-dir",
        default=None,
        help="Override root directory of validated sequences. If not given, derived from --batch.",
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
    p.add_argument(
        "--structure-z-offset",
        type=float,
        default=0.0,
        metavar="METRES",
        help="Vertical offset to apply to the entire target structure (default: 0.0)",
    )
    p.add_argument(
        "--export-dir",
        default=None,
        metavar="DIR",
        help=(
            "Override the export directory for planned trajectories. "
            "By default, successful sim/real runs auto-export to "
            "<batch>/validated_simPhysics_robot/<demo>/planned_sequence.json."
        ),
    )
    p.add_argument(
        "--no-export",
        action="store_true",
        default=False,
        help="Disable the automatic export of planned trajectories after a successful run.",
    )
    p.add_argument(
        "--speed-sim",
        type=float,
        default=0.5,
        help="Max velocity scaling for simulation (default: 0.5)",
    )
    p.add_argument(
        "--speed-real",
        type=float,
        default=0.13,
        help="Max velocity scaling for real robot planning (default: 0.13)",
    )
    p.add_argument(
        "--speed-replay",
        type=float,
        default=1.0,
        help="Execution speed multiplier for replay mode (default: 1.0)",
    )
    p.add_argument(
        "--replay",
        default=None,
        metavar="JSON_FILE",
        help=(
            "Skip MoveIt planning entirely and execute a pre-planned sequence. "
            "If given a demo name (e.g. 'demo_0') instead of a file path, "
            "loads from <batch>/validated_simPhysics_robot/<demo>/planned_sequence.json."
        ),
    )
    p.add_argument(
        "--skip-perception",
        action="store_true",
        default=False,
        help="Disable camera detection; use --supply-xyz as the fixed supply pose.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # -- Resolve data directory ---------------------------------------------
    if args.data_dir:
        data_dir = args.data_dir
    else:
        data_dir = os.path.normpath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "training_data",
                args.batch,
                "validated_simPhysics",
            )
        )

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

    # -- ROS 2 / MoveIt setup -----------------------------------------------
    node = None
    if HAVE_ROS2:
        rclpy.init()
        if mode == MODE_REAL:
            node = EGMClient(
                max_velocity_scaling=args.speed_real,
                max_acceleration_scaling=args.speed_real,
            )  # type: ignore[name-defined]
            print(f"[construct] EGMClient ready (real mode, speed={args.speed_real}).")
        else:
            v_scale = args.speed_sim if mode == MODE_SIM else 0.2
            node = PlanAndExecuteClient(
                mode=mode,
                max_velocity_scaling=v_scale,
                max_acceleration_scaling=v_scale,
            )  # type: ignore[name-defined]
            print(
                f"[construct] PlanAndExecuteClient ready (mode={mode}, speed={v_scale})."
            )
        print(f"[construct] Mode: {mode}")
    else:
        print(
            "[construct] ROS 2 not available -- running in offline/print-only mode.\n"
            "            Install rclpy and trajectory_planner_draft dependencies "
            "for actual planning."
        )
        mode = MODE_DRY_RUN  # force dry-run if no ROS

    # -- Resolve the validated_simPhysics_robot sibling directory --------------
    # Structure: <root>/<batch>/validated_simPhysics/<demo>  →
    #            <root>/<batch>/validated_simPhysics_robot/<demo>
    robot_dir = os.path.join(
        os.path.dirname(os.path.abspath(data_dir)),
        "validated_simPhysics_robot",
    )

    # -- Replay mode: load JSON and execute without replanning ---------------
    if args.replay:
        # Accept either an explicit file path or a bare demo name
        replay_path = args.replay
        if not os.path.isfile(replay_path):
            replay_path = os.path.join(robot_dir, replay_path, "planned_sequence.json")
        if not os.path.isfile(replay_path):
            print(f"[replay] ERROR: sequence file not found: {replay_path}")
            sys.exit(1)
        try:
            run_replay(node, replay_path, mode=mode, speed_replay=args.speed_replay)
        finally:
            if node is not None:
                for _ in range(60):
                    rclpy.spin_once(node, timeout_sec=0.05)
                node.destroy_node()
                rclpy.shutdown()
                print("[construct] ROS 2 shutdown complete.")
        return

    # -- Planning mode: load demo sequence and plan --------------------------
    demo_poses = load_demo_sequence(args.demo, data_dir)

    # Apply vertical offset to all bricks if requested
    if args.structure_z_offset != 0.0:
        print(
            f"[construct] Applying vertical offset of {args.structure_z_offset:.4f}m to all bricks."
        )
        for pose in demo_poses:
            pose[2] += args.structure_z_offset

    # In non-sim modes, clear any leftover MoveIt collision objects.
    # (In sim mode _gz_clean_scene() handles this inside run_construction.)
    if node is not None and mode != MODE_SIM:
        node.remove_all_world_collision_objects()

    # Resolve export directory: explicit override > auto-derived sibling > disabled
    if args.no_export:
        export_dir = None
    elif args.export_dir:
        export_dir = args.export_dir
    else:
        export_dir = os.path.join(robot_dir, args.demo)

    # -- Resolve perception paths ------------------------------------------
    _scripts_dir = os.path.dirname(os.path.abspath(__file__))
    perception_script = os.path.join(_scripts_dir, "perception_simple.py")
    # supply.json is written by perception_simple.py to SHARED_DIR
    try:
        sys.path.insert(0, _scripts_dir)
        from camera import SHARED_DIR as _SHARED_DIR  # type: ignore
        supply_json = str(_SHARED_DIR / "supply.json")
    except Exception:
        supply_json = "/realsense_shared/supply.json"  # fallback path

    use_perception = not args.skip_perception


    try:
        run_construction(
            node,
            demo_poses,
            supply_xyz=supply_xyz,
            supply_quat_xyzw=DEFAULT_SUPPLY_QUAT_XYZW,
            hover_z=args.hover_z,
            mode=mode,
            forced_grasp=args.grasp_id,
            export_dir=export_dir,
            use_perception=use_perception,
            perception_script=perception_script,
            supply_json=supply_json,
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
