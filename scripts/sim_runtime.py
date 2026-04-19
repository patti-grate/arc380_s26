"""
sim_runtime.py — Shared simulation runtime for brick-stacking prediction models.

Usage (inside the Docker container):
    python3 scripts/sim_runtime.py [--model knn] [--k 5] [--n-bricks 20]
                                   [--data-dir <path>] [--run-id my_run]

Architecture
------------
1. Load & fit a prediction model (the model owns its own dataset pipeline).
2. Open a ROS 2 / Gazebo session (mirrors demo_validation.py primitives).
3. Predict bricks one at a time, spawn each into Gazebo, wait for physics
   to settle, snapshot the resting pose, then check for structural collapse.
4. If collapse is detected, display a Gazebo text marker and stop the run.
5. Results are written to training_data/sim_runtime/<run-id>/.

Adding a new model
------------------
1. Create scripts/<yourmodel>_model.py  with fit_from_dir(data_dir) + predict(history).
2. Import it here and add an entry to MODEL_REGISTRY.
"""

import argparse
import os
import re
import subprocess
import sys
import time
import json
import numpy as np
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# ROS 2 / Gazebo availability guard
# ---------------------------------------------------------------------------
try:
    import rclpy
    from rclpy.node import Node
    from ros_gz_interfaces.srv import ControlWorld
    HAVE_ROS2 = True
except ImportError:
    HAVE_ROS2 = False

# ---------------------------------------------------------------------------
# Pose conversion utility
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))
from pose_conversion import Brick

# ---------------------------------------------------------------------------
# Model registry — add new model classes here
# ---------------------------------------------------------------------------
from knn_model import KNNModel

MODEL_REGISTRY: Dict[str, type] = {
    "knn": KNNModel,
}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SDF_PATH = os.path.join(
    os.path.dirname(__file__), "..", "src",
    "abb_irb120_gazebo", "models", "brick", "model.sdf"
)
WORLD_NAME = "irb120_workcell"
PHYSICS_SETTLE_S = 2.0      # seconds to wait after spawning before snapshotting
COLLAPSE_THRESHOLD_M = 0.01  # bricks that fall more than 1 cm are considered collapsed


# ===========================================================================
# Gazebo helpers  (pure-subprocess, no ROS 2 spin required for Gazebo ops)
# ===========================================================================

def gz_spawn(name: str, pose_7d: np.ndarray, sdf_path: str) -> bool:
    """Spawn a brick into Gazebo via gz service EntityFactory."""
    x, y, z = pose_7d[0], pose_7d[1], pose_7d[2] + 0.001  # tiny lift to avoid floor clip
    qx, qy, qz, qw = pose_7d[3], pose_7d[4], pose_7d[5], pose_7d[6]
    req = (
        f'sdf_filename: \\"{sdf_path}\\" '
        f'name: \\"{name}\\" '
        f'pose: {{ position: {{x: {x} y: {y} z: {z}}} '
        f'orientation: {{x: {qx} y: {qy} z: {qz} w: {qw}}} }}'
    )
    cmd = (
        f'gz service -s /world/{WORLD_NAME}/create '
        f'--reqtype gz.msgs.EntityFactory --reptype gz.msgs.Boolean '
        f'--timeout 2000 --req "{req}"'
    )
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.returncode == 0


def gz_remove(name: str) -> None:
    """Remove a named model from the Gazebo world."""
    cmd = (
        f'gz service -s /world/{WORLD_NAME}/remove '
        f'--reqtype gz.msgs.Entity --reptype gz.msgs.Boolean '
        f'--timeout 500 --req \'name: "{name}" type: MODEL\''
    )
    subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def gz_fetch_poses() -> Dict[str, float]:
    """
    Poll Gazebo's pose topic once.
    Returns dict mapping model_name → z position for every model whose name
    contains 'brick'.  Missing dimensions default to 0.0 (protobuf omits them).
    """
    def _get(pattern, s):
        m = re.search(pattern, s)
        return float(m.group(1)) if m else 0.0

    poses: Dict[str, float] = {}
    try:
        output = subprocess.check_output(
            f"gz topic -e -n 1 -t /world/{WORLD_NAME}/pose/info",
            shell=True, text=True, timeout=3.0,
        )
        for block in output.split("pose {"):
            if "name:" not in block:
                continue
            nm = re.search(r'name:\s+"([^"]+)"', block)
            if not nm:
                continue
            model_name = nm.group(1)
            if "brick" not in model_name:
                continue
            poses[model_name] = _get(r'z:\s+([^\n]+)', block)
    except Exception:
        pass
    return poses


def gz_list_brick_models() -> List[str]:
    """
    Query the live Gazebo scene and return names of all top-level brick models.

    Parses pose/info (a continuous streaming topic) and excludes link/visual
    sub-entries ('brick_link', 'visual') to return only model-level names
    whose string contains 'brick'.
    """
    names: List[str] = []
    try:
        output = subprocess.check_output(
            f"gz topic -e -n 1 -t /world/{WORLD_NAME}/pose/info",
            shell=True, text=True, timeout=5.0,
        )
        for block in output.split("pose {"):
            m = re.search(r'name:\s+"([^"]+)"', block)
            if not m:
                continue
            name = m.group(1)
            # Skip link/visual entries that live inside brick models
            if name in ("brick_link", "visual"):
                continue
            if "brick" in name:
                names.append(name)
    except Exception:
        pass
    return names


def gz_marker_show(text: str = "COLLAPSE DETECTED!") -> None:
    """Render floating red text in the Gazebo GUI."""
    msg = (
        f"ns: 'runtime', id: 200, action: ADD_MODIFY, type: TEXT, "
        f"text: '{text}', "
        f"pose: {{position: {{x: 0.15, y: 0.35, z: 0.5}}}}, "
        f"scale: {{x: 0.08, y: 0.08, z: 0.08}}, "
        f"material: {{ambient: {{r: 1.0, g: 0.0, b: 0.0, a: 1.0}}}}"
    )
    subprocess.run(
        f'gz topic -t /marker -m gz.msgs.Marker -p "{msg}"',
        shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def gz_marker_clear() -> None:
    """Remove the floating text marker."""
    subprocess.run(
        'gz topic -t /marker -m gz.msgs.Marker -p "ns: \'runtime\', id: 200, action: DELETE_MARKER"',
        shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def clean_world(extra_names: List[str] = []) -> None:
    """
    Remove every brick model currently in the scene.

    Queries the live Gazebo scene so it handles any run-id prefix correctly,
    then removes any extra names explicitly supplied (e.g. from a failed run
    where some bricks may not yet appear in the scene topic).
    """
    print("[sim_runtime] Cleaning world...")
    gz_marker_clear()

    # Query what is actually in the scene
    live_bricks = gz_list_brick_models()
    to_remove = list(dict.fromkeys(live_bricks + list(extra_names)))  # deduplicated, ordered

    if to_remove:
        print(f"[sim_runtime] Removing {len(to_remove)} brick(s): {to_remove}")
        for name in to_remove:
            gz_remove(name)
    else:
        print("[sim_runtime] No brick models found in scene.")

    # Allow Gazebo to finish processing the deletions before the next phase
    time.sleep(2.0)
    print("[sim_runtime] World clean.")


# ===========================================================================
# Simulation run
# ===========================================================================

def run_simulation(
    model,
    n_bricks: int,
    run_id: str,
    out_dir: str,
    sdf_path: str,
) -> None:
    """
    Core loop: predict → spawn → settle → snapshot → stability check.

    Args
    ----
    model    : fitted prediction model with .predict(history) → 5D np.ndarray
    n_bricks : number of bricks to place
    run_id   : identifier string used for brick naming and output filenames
    out_dir  : directory to write the output sequence JSON
    sdf_path : absolute path to the brick SDF file
    """
    history: List[np.ndarray] = []      # growing list of PLACED 5D poses (global)
    initial_z: Dict[str, float] = {}    # baseline z per brick name for collapse detection
    spawned_names: List[str] = []

    print(f"\n[sim_runtime] Starting run '{run_id}' — {n_bricks} bricks\n")
    clean_world([])

    for i in range(n_bricks):
        brick_name = f"{run_id}_brick_{i}"

        # ── 1. Predict next pose ─────────────────────────────────────────
        print(f"\n[sim_runtime] --- Brick {i} ---")
        pose_5d = model.predict(history, verbose=True)
        print(f"  [{i:>3}] Predicted 5D: x={pose_5d[0]:.4f}  y={pose_5d[1]:.4f}  "
              f"z={pose_5d[2]:.4f}  b={int(pose_5d[3])}  r={pose_5d[4]:.4f}")

        # ── 2. Convert 5D → 7D using Brick utility ──────────────────────
        b = Brick()
        b.from_5d_pose(pose_5d)
        pose_7d = b.get_7d_pose()

        # ── 3. Spawn into Gazebo ─────────────────────────────────────────
        ok = gz_spawn(brick_name, pose_7d, sdf_path)
        if not ok:
            print(f"  [{i:>3}] ERROR: spawn failed for {brick_name}. Skipping.")
            continue
        spawned_names.append(brick_name)
        print(f"  [{i:>3}] Spawned {brick_name}")

        # ── 4. Physics settle ────────────────────────────────────────────
        time.sleep(PHYSICS_SETTLE_S)

        # ── 5. Snapshot initial resting z ────────────────────────────────
        current_poses = gz_fetch_poses()
        if brick_name in current_poses:
            initial_z[brick_name] = current_poses[brick_name]
        else:
            print(f"  [{i:>3}] WARNING: {brick_name} not found in pose stream after spawn.")

        # ── 6. Stability check over all placed bricks ────────────────────
        current_poses = gz_fetch_poses()
        collapsed = False
        for bname, snap_z in initial_z.items():
            cur_z = current_poses.get(bname, snap_z)
            dz = cur_z - snap_z
            if dz < -COLLAPSE_THRESHOLD_M:
                print(f"\n  *** COLLAPSE: {bname} dropped {abs(dz)*100:.1f} cm ***\n")
                gz_marker_show()
                collapsed = True
                break

        if collapsed:
            break

        # ── 7. Add placed pose to history ────────────────────────────────
        history.append(pose_5d)

    # ── Write output ──────────────────────────────────────────────────────
    os.makedirs(out_dir, exist_ok=True)
    seq_path = os.path.join(out_dir, "sequence.json")
    with open(seq_path, "w") as f:
        json.dump([p.tolist() for p in history], f, indent=2)

    status = "COMPLETE" if not collapsed else "COLLAPSED"
    print(f"\n[sim_runtime] Run '{run_id}' {status}. "
          f"{len(history)}/{n_bricks} bricks placed.")
    print(f"[sim_runtime] Sequence saved to: {seq_path}")

    clean_world(spawned_names)


# ===========================================================================
# CLI
# ===========================================================================

def parse_args() -> argparse.Namespace:
    default_data = os.path.normpath(os.path.join(
        os.path.dirname(__file__), "..", "training_data", "batch1", "validated_simPhysics"
    ))
    default_out = os.path.normpath(os.path.join(
        os.path.dirname(__file__), "..", "training_data", "sim_runtime"
    ))

    p = argparse.ArgumentParser(
        description="Sim runtime: predict bricks with a pluggable model and test in Gazebo."
    )
    p.add_argument("--model", choices=list(MODEL_REGISTRY.keys()), default="knn",
                   help="Prediction model to use (default: knn).")
    p.add_argument("--k", type=int, default=5,
                   help="k value for KNN (default: 5).")
    p.add_argument("--n-bricks", type=int, default=20,
                   help="Number of bricks to place per run (default: 20).")
    p.add_argument("--data-dir", default=default_data,
                   help="Root directory of validated training sequences.")
    p.add_argument("--run-id", default="runtime_run",
                   help="Identifier for this run (used for brick names and output dir).")
    p.add_argument("--out-dir", default=default_out,
                   help="Directory to write output sequences.")
    p.add_argument("--demos", default=None,
                   help="Comma-separated demo names to include as training data "
                        "(e.g. demo_1,demo_2,demo_3,demo_4). Default: all demos.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Build model ───────────────────────────────────────────────────────
    model_cls = MODEL_REGISTRY[args.model]
    if args.model == "knn":
        model = model_cls(k=args.k)
    else:
        model = model_cls()

    allowed_demos = [d.strip() for d in args.demos.split(",")] if args.demos else None
    model.fit_from_dir(args.data_dir, allowed_demos=allowed_demos)
    print(f"[sim_runtime] Model ready: {model}")

    if not HAVE_ROS2:
        print("[sim_runtime] WARNING: ROS 2 not found. Gazebo ops require running inside the Docker container.")
        print("[sim_runtime] Dry-run: printing predictions only.\n")
        history = []
        for i in range(args.n_bricks):
            print(f"\n--- Brick {i} ---")
            pose = model.predict(history, verbose=True)
            print(f"  [{i:>3}] {pose}")
            history.append(pose)
        return

    # ── Gazebo simulation run ─────────────────────────────────────────────
    sdf_path = os.path.normpath(SDF_PATH)
    if not os.path.isfile(sdf_path):
        print(f"[sim_runtime] ERROR: SDF not found at {sdf_path}")
        sys.exit(1)

    out_dir = os.path.join(args.out_dir, args.run_id, "5d_sequence")

    rclpy.init()
    try:
        run_simulation(
            model=model,
            n_bricks=args.n_bricks,
            run_id=args.run_id,
            out_dir=out_dir,
            sdf_path=sdf_path,
        )
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
