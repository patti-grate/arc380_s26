#!/usr/bin/env python3
"""
model_evaluation.py  —  SST model-driven brick assembly in Gazebo simulation.

Loop:
  1. Model proposes N candidate next-brick poses (sorted by log-likelihood).
  2. Each candidate's z is snapped to the nearest known training height.
  3. Candidate is converted 5D → 7D and checked for robot reachability.
  4. Valid candidate is spawned in Gazebo; structural stability is tested.
  5. If stable  → accept, update history, propose next brick.
     If unstable → remove, try next candidate.
  6. If every candidate in a round fails, re-sample and retry (up to --max-rounds).

Usage (from project root):
  python scripts/model_evaluation.py
  python scripts/model_evaluation.py --checkpoint training_data/best_model.pth \\
      --max-bricks 25 --n-candidates 100
"""

import os
import sys
import json
import math
import time
import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Ensure scripts/ is on the import path ─────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pose_conversion import Brick

# ── ROS 2 / Gazebo (mirrors demo_validation.py) ───────────────────────────────
HAVE_ROS2 = False
try:
    import rclpy
    from rclpy.node import Node
    from ros_gz_interfaces.srv import ControlWorld, DeleteEntity
    HAVE_ROS2 = True
except ImportError:
    print("WARNING: rclpy not found — running in dry-run/mock mode.")

# ── MoveIt reachability (mirrors demo_validation.py) ─────────────────────────
HAVE_MOVEIT = False
_MoveitClient: Any = None
T_GRASP_OFFSETS: dict = {}
GRASP_ORDER: list = []
apply_grasp_offset: Any = None
apply_local_rotation: Any = None
is_standing_brick: Any = None
BRICK_SIZE_XYZ: tuple = (0.051, 0.023, 0.014)
SAFE_HOME_NAMES: list = []
SAFE_HOME_POSITIONS: list = []
try:
    from construct_using_validated import (
        T_GRASP_OFFSETS, GRASP_ORDER,
        apply_grasp_offset, apply_local_rotation, is_standing_brick,
        BRICK_SIZE_XYZ, SAFE_HOME_NAMES, SAFE_HOME_POSITIONS,
    )
    if HAVE_ROS2:
        from trajectory_planner_draft_JG import PlanAndExecuteClient as _MoveitClient
        HAVE_MOVEIT = True
except ImportError as _e:
    print(f"WARNING: Reachability checking unavailable: {_e}")

# Import validator classes from demo_validation so we reuse the same logic
from demo_validation import (
    DemoValidator,
    check_placement_reachable,
)
if HAVE_ROS2:
    from demo_validation import DemoValidatorNode


# ══════════════════════════════════════════════════════════════════════════════
# Model architecture — must exactly mirror training_SST.ipynb
# ══════════════════════════════════════════════════════════════════════════════

FEATURE_DIM = 8    # [x, y, z, b, sin_r, cos_r, layer_id_norm, time_norm]
HIDDEN_DIM  = 128
N_HEADS     = 4
N_LAYERS    = 2
FF_DIM      = 256
DROPOUT     = 0.1
K_MIXTURES  = 5
MAX_BRICKS  = 60
POSE_DIM    = 5    # MDN output: [x, y, z, sin_r, cos_r]
_LOG2PI     = math.log(2.0 * math.pi)

# Physical brick dimensions (metres): longest × middle × shortest
BRICK_W = 0.051   # x when laying
BRICK_D = 0.023   # y when laying
BRICK_H = 0.014   # z when laying (thickness)


class NextBrickModel(nn.Module):
    """Transformer encoder + MDN head — identical to training_SST.ipynb."""

    def __init__(
        self,
        feature_dim=FEATURE_DIM, hidden_dim=HIDDEN_DIM,
        nhead=N_HEADS, num_layers=N_LAYERS, ff_dim=FF_DIM,
        dropout=DROPOUT, K=K_MIXTURES,
    ):
        super().__init__()
        self.K        = K
        self.pose_dim = POSE_DIM

        self.input_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.b_head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(64, 2),
        )
        self.mdn_pi        = nn.Linear(hidden_dim, K)
        self.mdn_mu        = nn.Linear(hidden_dim, K * self.pose_dim)
        self.mdn_log_sigma = nn.Linear(hidden_dim, K * self.pose_dim)

    def forward(self, tokens, mask):
        B = tokens.shape[0]
        x   = self.input_proj(tokens)
        cls = self.cls_token.expand(B, 1, -1)
        x   = torch.cat([cls, x], dim=1)
        cls_valid = torch.ones(B, 1, dtype=torch.bool, device=mask.device)
        x = self.transformer(x, src_key_padding_mask=~torch.cat([cls_valid, mask], dim=1))
        scene    = x[:, 0]
        b_logits = self.b_head(scene)
        pi    = F.softmax(self.mdn_pi(scene), dim=-1)
        mu    = self.mdn_mu(scene).view(B, self.K, self.pose_dim)
        sigma = F.softplus(self.mdn_log_sigma(scene)).view(B, self.K, self.pose_dim) + 1e-4
        return b_logits, pi, mu, sigma


# ══════════════════════════════════════════════════════════════════════════════
# Feature-encoding helpers — identical to training_SST.ipynb
# ══════════════════════════════════════════════════════════════════════════════

def assign_layer_ids(poses_5d, z_tol=1e-4):
    unique_z = []
    for p in poses_5d:
        z = p[2]
        if not any(abs(z - uz) < z_tol for uz in unique_z):
            unique_z.append(z)
    unique_z.sort()
    return [
        min(range(len(unique_z)), key=lambda i: abs(unique_z[i] - p[2]))
        for p in poses_5d
    ]


def canonicalize_r(r):
    r = r % math.pi
    return r + math.pi if r < 0 else r


def encode_brick(pose_5d, layer_id, time_index):
    x, y, z, b, r = pose_5d
    r_c = canonicalize_r(r)
    return [x, y, z, b, math.sin(r_c), math.cos(r_c), float(layer_id), float(time_index)]


def history_to_encoded(history_5d):
    """Convert list of accepted 5D poses to 8-dim feature vectors for the model."""
    if not history_5d:
        return []
    layer_ids = assign_layer_ids(history_5d)
    return [
        encode_brick(pose, lid, t)
        for t, (pose, lid) in enumerate(zip(history_5d, layer_ids))
    ]


# ══════════════════════════════════════════════════════════════════════════════
# Z-level utilities
# ══════════════════════════════════════════════════════════════════════════════

def load_z_levels(batch_dirs, seq_subpath="5d_sequence/sequence.json", z_tol=1e-4):
    """
    Collect all unique z values from training sequences.

    Returns:
      z_levels : sorted list of canonical placement heights
      z_to_b   : {z_level: b_state} — the b state (0=laying, 1=standing) observed
                 at each height in training data.  b and z are physically coupled
                 (each valid height has exactly one valid orientation), so this dict
                 lets us enforce consistency at inference time.
    """
    z_vals   = []          # raw z values being collected
    z_b_vote = {}          # {index_into_z_vals: {0: count, 1: count}}

    for bd in batch_dirs:
        bd_path = Path(bd)
        if not bd_path.exists():
            continue
        for demo_dir in sorted(bd_path.iterdir()):
            seq_path = demo_dir / seq_subpath
            if not seq_path.exists():
                continue
            with open(seq_path) as f:
                poses = json.load(f)
            for p in poses:
                z, b = p[2], int(p[3])
                idx = next(
                    (i for i, zv in enumerate(z_vals) if abs(z - zv) < z_tol),
                    None,
                )
                if idx is None:
                    z_vals.append(z)
                    idx = len(z_vals) - 1
                    z_b_vote[idx] = {0: 0, 1: 0}
                z_b_vote[idx][b] += 1

    order    = sorted(range(len(z_vals)), key=lambda i: z_vals[i])
    z_levels = [z_vals[i] for i in order]
    z_to_b   = {z_vals[i]: max(z_b_vote[i], key=z_b_vote[i].get) for i in range(len(z_vals))}
    return z_levels, z_to_b


def snap_z(z_pred, z_levels):
    """Snap a predicted z to the nearest known training height."""
    return min(z_levels, key=lambda z: abs(z - z_pred))


def load_seed_layer(seed_path: str, z_levels: list) -> list:
    """
    Extract the first (lowest) layer of bricks from a 5D sequence file.

    Every brick whose z snaps to z_levels[0] is kept; its z is replaced with
    the exact training value so the history is consistent with how the model
    was trained.  Returns a list of [x, y, z, b, r] poses.
    """
    with open(seed_path) as f:
        poses = json.load(f)
    seed = []
    for p in poses:
        z_snapped = snap_z(float(p[2]), z_levels)
        if z_snapped == z_levels[0]:
            seed.append([float(p[0]), float(p[1]), z_snapped,
                         int(p[3]), float(p[4])])
    return seed


def snap_z_with_b(z_pred, b_pred, z_levels, z_to_b):
    """
    Snap z to the nearest training height AND return the physically correct b state
    for that height.  b and z are tightly coupled (each valid height has exactly one
    valid orientation), so b_pred from the model is overridden by the training-data
    ground truth.  Falls back to b_pred if the height is not in z_to_b.
    """
    z_snapped = snap_z(z_pred, z_levels)
    b_correct = z_to_b.get(z_snapped, b_pred)
    return z_snapped, b_correct


# ══════════════════════════════════════════════════════════════════════════════
# Inference utilities
# ══════════════════════════════════════════════════════════════════════════════

def _build_input_tensors(history_encoded, norm_stats, device):
    ns = norm_stats
    tokens = torch.zeros(1, MAX_BRICKS, FEATURE_DIM)
    mask   = torch.zeros(1, MAX_BRICKS, dtype=torch.bool)
    for i, h in enumerate(history_encoded[:MAX_BRICKS]):
        x, y, z, b, sin_r, cos_r, layer_id, time_index = h
        tokens[0, i] = torch.tensor([
            (x - ns["mean_x"]) / ns["std_x"],
            (y - ns["mean_y"]) / ns["std_y"],
            (z - ns["mean_z"]) / ns["std_z"],
            b, sin_r, cos_r, layer_id / 20.0, time_index / 60.0,
        ])
        mask[0, i] = True
    return tokens.to(device), mask.to(device)


def sample_candidates(model, history_encoded, norm_stats, n_candidates,
                      device, z_levels, z_to_b):
    """
    Sample n_candidates from the MDN, snap z, score by log-likelihood.

    z and b are physically coupled (each valid height has exactly one valid
    orientation), so after snapping z to the nearest training height the
    correct b state is looked up from z_to_b rather than sampled independently.

    Returns a list of candidate dicts sorted best → worst:
      {x, y, z, b, sin_r, cos_r, r, log_prob}
    """
    ns = norm_stats
    model.eval()
    tokens, mask = _build_input_tensors(history_encoded, ns, device)

    with torch.no_grad():
        b_logits, pi, mu, sigma = model(tokens, mask)

    b_prob = F.softmax(b_logits[0], dim=-1).cpu().numpy()
    pi_np  = pi[0].cpu().numpy()
    mu_np  = mu[0].cpu().numpy()     # (K, 5)
    sig_np = sigma[0].cpu().numpy()  # (K, 5)

    candidates = []
    for _ in range(n_candidates):
        # Sample from the mixture
        k = np.random.choice(len(pi_np), p=pi_np / pi_np.sum())
        s = np.random.randn(POSE_DIM) * sig_np[k] + mu_np[k]

        # Denormalise x, y, z
        xr = float(s[0] * ns["std_x"] + ns["mean_x"])
        yr = float(s[1] * ns["std_y"] + ns["mean_y"])
        zr = float(s[2] * ns["std_z"] + ns["mean_z"])

        # Recover rotation (unit-circle normalisation)
        sin_r, cos_r = float(s[3]), float(s[4])
        nrm = math.sqrt(sin_r**2 + cos_r**2 + 1e-8)
        sin_r /= nrm
        cos_r /= nrm
        r = math.atan2(sin_r, cos_r)

        # Snap z to nearest known training height; derive correct b from z_to_b.
        # b_prob from the model is used only as a fallback when the snapped height
        # is not in z_to_b (should not happen with complete training data).
        b_raw     = int(np.random.choice(2, p=b_prob))
        z_snapped, b = snap_z_with_b(zr, b_raw, z_levels, z_to_b)

        # Score this sample under the full mixture (using snapped z)
        norm_sample = np.array([
            (xr        - ns["mean_x"]) / ns["std_x"],
            (yr        - ns["mean_y"]) / ns["std_y"],
            (z_snapped - ns["mean_z"]) / ns["std_z"],
            sin_r, cos_r,
        ])
        log_gauss = (
            -0.5 * (((norm_sample - mu_np) / sig_np) ** 2
                    + 2 * np.log(sig_np) + _LOG2PI)
        ).sum(axis=-1)  # (K,)
        log_prob = float(np.logaddexp.reduce(np.log(pi_np + 1e-8) + log_gauss))

        candidates.append({
            "x": xr, "y": yr, "z": z_snapped,
            "b": b, "sin_r": sin_r, "cos_r": cos_r, "r": r,
            "log_prob": log_prob,
        })

    # Highest-likelihood candidates first
    candidates.sort(key=lambda c: c["log_prob"], reverse=True)
    return candidates, b_prob


def candidate_to_7d(cand):
    """Convert a candidate dict to a 7D pose array via pose_conversion.Brick."""
    brick = Brick()
    brick.from_5d_pose([cand["x"], cand["y"], cand["z"], cand["b"], cand["r"]])
    return brick.get_7d_pose()


# ══════════════════════════════════════════════════════════════════════════════
# Simulation helpers
# ══════════════════════════════════════════════════════════════════════════════

def _remove_brick(validator, brick_name):
    """Remove a single brick from Gazebo (best-effort)."""
    if not HAVE_ROS2:
        validator.current_model_states.pop(brick_name, None)
        return
    try:
        if hasattr(validator, "remove_client") and \
                validator.remove_client.wait_for_service(timeout_sec=1.0):
            req = DeleteEntity.Request()
            req.entity.name = brick_name
            req.entity.type = 2  # MODEL
            future = validator.remove_client.call_async(req)
            rclpy.spin_until_future_complete(validator, future, timeout_sec=2.0)
        time.sleep(0.3)
        validator.fetch_latest_poses_from_gz()
    except Exception as exc:
        print(f"    [warn] Could not remove {brick_name}: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# Spatial collision helpers
# ══════════════════════════════════════════════════════════════════════════════

def _brick_z_half(b: int) -> float:
    """Half-height along world Z: thickness/2 when laying, longest/2 when standing."""
    return (BRICK_H if b == 0 else BRICK_W) / 2.0


def _brick_xy_half_extents(b: int):
    """(hw, hd): half-extents in the local XY plane for state b."""
    return (BRICK_W / 2.0, BRICK_D / 2.0) if b == 0 else (BRICK_D / 2.0, BRICK_H / 2.0)


def _obb2d_intersect(cx1, cy1, hw1, hd1, r1,
                     cx2, cy2, hw2, hd2, r2,
                     margin: float = 0.0) -> bool:
    """
    Separating Axis Theorem test for two 2D oriented bounding boxes.
    Returns True when they overlap (each half-extent is inflated by `margin`).
    """
    hw1 += margin; hd1 += margin
    hw2 += margin; hd2 += margin
    c1, s1 = math.cos(r1), math.sin(r1)
    c2, s2 = math.cos(r2), math.sin(r2)
    dx, dy = cx2 - cx1, cy2 - cy1
    for ax, ay in [(c1, s1), (-s1, c1), (c2, s2), (-s2, c2)]:
        sep  = abs(ax * dx + ay * dy)
        rp1  = hw1 * abs(ax * c1 + ay * s1) + hd1 * abs(-ax * s1 + ay * c1)
        rp2  = hw2 * abs(ax * c2 + ay * s2) + hd2 * abs(-ax * s2 + ay * c2)
        if sep > rp1 + rp2:
            return False  # separating axis found → no overlap
    return True


def check_spatial_collision(cand: dict, history_5d: list,
                             xy_margin: float = 0.001) -> bool:
    """
    Returns True if the candidate brick's volume intersects any placed brick.

    Z check uses NO margin: adjacent layers are supposed to be physically
    touching (the gap between layer-0-top and layer-1-bottom is ~0.04 mm),
    so any positive margin would incorrectly flag valid stacked placements.

    XY check uses xy_margin (default 1 mm) to reject bricks placed too close
    to each other within the same layer.
    """
    cx, cy, cz = cand["x"], cand["y"], cand["z"]
    cb, cr = int(cand["b"]), cand["r"]
    czh = _brick_z_half(cb)
    chw, chd = _brick_xy_half_extents(cb)

    for p in history_5d:
        px, py, pz, pb, pr = p[0], p[1], p[2], int(p[3]), p[4]
        pzh = _brick_z_half(pb)
        # No margin on Z: stacked bricks touch — a positive margin triggers
        # false collisions between adjacent layers.
        if abs(cz - pz) >= czh + pzh:
            continue   # no Z overlap → skip XY test
        phw, phd = _brick_xy_half_extents(pb)
        if _obb2d_intersect(cx, cy, chw, chd, cr,
                            px, py, phw, phd, pr, xy_margin):
            return True
    return False


# ══════════════════════════════════════════════════════════════════════════════
# Main evaluation loop
# ══════════════════════════════════════════════════════════════════════════════

def run_evaluation(args, validator, moveit_node, model, norm_stats,
                   z_levels, z_to_b, device, seed_layer=None):
    """
    Iteratively place bricks using the model.

    If seed_layer is provided, those bricks are spawned first and added to
    history so the model generates on top of them.  --max-bricks counts only
    the bricks generated by the model (not the seed).

    Returns (placed_7d, history_5d) where both lists include seed bricks first.
    """
    history_5d = []   # [x, y, z, b, r] for every brick (seed + generated)
    placed_7d  = []   # 7D np.ndarray  for every brick (seed + generated)
    total_rejected = 0
    brick_idx      = 0

    # ── Initial world setup ──────────────────────────────────────────────────
    validator.reset_world()
    if moveit_node is not None:
        moveit_node.remove_all_world_collision_objects()
        moveit_node.publish_scene_box(
            object_id="table_surface", frame_id="world",
            size_xyz=(2.0, 2.0, 0.02), position_xyz=(0.0, 0.0, -0.02),
        )

    # ── Spawn seed layer ─────────────────────────────────────────────────────
    if seed_layer:
        print(f"\nSpawning {len(seed_layer)} seed bricks (layer 0 from training data) …")
        for s_idx, pose_5d in enumerate(seed_layer):
            seed_name = f"seed_brick_{s_idx:03d}"
            b_obj = Brick()
            b_obj.from_5d_pose(pose_5d)
            seed_7d = b_obj.get_7d_pose()

            validator.spawn_brick(seed_name, seed_7d)

            if moveit_node is not None:
                moveit_node.publish_scene_box(
                    object_id=f"mv_{seed_name}", frame_id="world",
                    size_xyz=BRICK_SIZE_XYZ,
                    position_xyz=tuple(float(v) for v in seed_7d[:3]),
                    quat_xyzw=tuple(float(v) for v in seed_7d[3:]),
                )

            history_5d.append(list(pose_5d))
            placed_7d.append(seed_7d)
            print(f"  [seed {s_idx:03d}]  x={pose_5d[0]:.4f}  y={pose_5d[1]:.4f}"
                  f"  z={pose_5d[2]:.4f}  b={int(pose_5d[3])}")

        # Let physics settle once after all seed bricks are in place
        time.sleep(2.0 if HAVE_ROS2 else 0.05)
        validator.fetch_latest_poses_from_gz()
        print(f"  Seed layer ready — model will generate {args.max_bricks} bricks on top.\n")

    while brick_idx < args.max_bricks:
        print(f"\n{'='*62}")
        print(f"  Brick {brick_idx + 1:3d} / {args.max_bricks}   |   "
              f"history: {len(history_5d)} bricks placed so far")
        print(f"{'='*62}")

        placed  = False
        round_n = 0

        while not placed and round_n < args.max_rounds:
            round_n += 1
            print(f"  [round {round_n}/{args.max_rounds}]  "
                  f"sampling {args.n_candidates} candidates …")

            # Encode current history and sample
            history_encoded = history_to_encoded(history_5d)
            candidates, b_prob = sample_candidates(
                model, history_encoded, norm_stats,
                n_candidates=args.n_candidates,
                device=device, z_levels=z_levels, z_to_b=z_to_b,
            )
            print(f"    b: laying={b_prob[0]:.3f}  standing={b_prob[1]:.3f}  "
                  f"best log_p={candidates[0]['log_prob']:.2f}")

            # Snapshot the clean foundation ONCE before trying any candidate
            validator.fetch_latest_poses_from_gz()
            foundation_states = validator.current_model_states.copy()

            for c_idx, cand in enumerate(candidates):
                brick_name = f"model_brick_{brick_idx:03d}"

                # ── 5D → 7D ─────────────────────────────────────────────────
                pose_7d = candidate_to_7d(cand)

                # ── Spatial collision (cheap, no Gazebo call) ────────────────
                if check_spatial_collision(cand, history_5d):
                    print(f"    [c{c_idx:03d}] REJECT — overlaps a placed brick  "
                          f"x={cand['x']:.4f} y={cand['y']:.4f} z={cand['z']:.4f} b={cand['b']}")
                    total_rejected += 1
                    continue

                # ── Reachability ─────────────────────────────────────────────
                effective_pose = check_placement_reachable(moveit_node, pose_7d)
                if effective_pose is None:
                    print(f"    [c{c_idx:03d}] REJECT — not reachable by robot  "
                          f"x={cand['x']:.4f} y={cand['y']:.4f} z={cand['z']:.4f} b={cand['b']}")
                    total_rejected += 1
                    continue

                # ── Spawn ────────────────────────────────────────────────────
                if not validator.spawn_brick(brick_name, effective_pose):
                    print(f"    [c{c_idx:03d}] REJECT — spawn failed")
                    total_rejected += 1
                    continue

                # Wait for Gazebo physics to settle
                time.sleep(2.0 if HAVE_ROS2 else 0.05)

                # Fetch current world state (used for both placement and stability checks)
                validator.fetch_latest_poses_from_gz()

                # ── Placement check: did the new brick land near its intended z?
                if HAVE_ROS2 and brick_name in validator.current_model_states:
                    settled_z = validator.current_model_states[brick_name].position.z
                    dz_intended = settled_z - cand["z"]
                    if abs(dz_intended) > 0.01:
                        print(f"    [c{c_idx:03d}] REJECT — brick fell {dz_intended:.3f}m from "
                              f"intended z={cand['z']:.4f}")
                        _remove_brick(validator, brick_name)
                        total_rejected += 1
                        time.sleep(0.5)
                        continue

                # Build check_states: pre-spawn positions for existing bricks +
                # settled position for the new brick (so check_stability measures
                # displacement of existing bricks relative to before the spawn).
                check_states = foundation_states.copy()
                if brick_name in validator.current_model_states:
                    check_states[brick_name] = validator.current_model_states[brick_name]

                # ── Stability: existing bricks must not have shifted > 1 cm ──
                if not validator.check_stability(check_states):
                    print(f"    [c{c_idx:03d}] REJECT — existing structure destabilised  "
                          f"z={cand['z']:.4f}  b={cand['b']}")
                    _remove_brick(validator, brick_name)
                    total_rejected += 1
                    # Brief pause so physics settles back before next candidate
                    time.sleep(0.5 if HAVE_ROS2 else 0.01)
                    continue

                # ── Accept ───────────────────────────────────────────────────
                history_5d.append([
                    cand["x"], cand["y"], cand["z"],
                    float(cand["b"]), cand["r"],
                ])
                placed_7d.append(effective_pose)

                # Register in MoveIt as a collision obstacle
                if moveit_node is not None:
                    moveit_node.publish_scene_box(
                        object_id=f"mv_{brick_name}", frame_id="world",
                        size_xyz=BRICK_SIZE_XYZ,
                        position_xyz=tuple(float(v) for v in effective_pose[:3]),
                        quat_xyzw=tuple(float(v) for v in effective_pose[3:]),
                    )

                placed = True
                layer_id = assign_layer_ids(history_5d)[-1]
                print(f"    [c{c_idx:03d}] ACCEPT — "
                      f"x={cand['x']:.4f} y={cand['y']:.4f} z={cand['z']:.4f}  "
                      f"b={cand['b']}  layer={layer_id}  "
                      f"log_p={cand['log_prob']:.2f}")
                break   # proceed to next brick

            if not placed:
                print(f"    [round {round_n}] all {len(candidates)} candidates rejected.")

        if not placed:
            print(f"\n[STOP] Could not place brick {brick_idx + 1} "
                  f"after {args.max_rounds} sampling rounds.")
            break

        brick_idx += 1

    print(f"\n{'='*62}")
    print(f"  Generation finished: {brick_idx} bricks placed, "
          f"{total_rejected} candidates rejected.")
    print(f"{'='*62}")

    return placed_7d, history_5d


# ══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="SST model-driven brick assembly evaluation in Gazebo."
    )
    parser.add_argument(
        "--checkpoint",
        default="training_data/trained_models/best_model.pth",
        help="Path to trained model checkpoint  (default: training_data/trained_models/best_model.pth)",
    )
    parser.add_argument(
        "--max-bricks", type=int, default=30,
        help="Maximum number of bricks to place  (default: 30)",
    )
    parser.add_argument(
        "--n-candidates", type=int, default=100,
        help="MDN samples drawn per placement round  (default: 100)",
    )
    parser.add_argument(
        "--max-rounds", type=int, default=5,
        help="Re-sampling rounds per brick before giving up  (default: 5)",
    )
    parser.add_argument(
        "--no-reachability-check", action="store_true",
        help="Skip MoveIt reachability validation",
    )
    parser.add_argument(
        "--output-dir",
        default="training_data/model_generated",
        help="Output directory for generated sequence  (default: training_data/model_generated)",
    )
    parser.add_argument(
        "--batch-dirs", nargs="+",
        default=[
            "training_data/batch2/validated_simPhysics",
            "training_data/batch3/validated_simPhysics",
        ],
        help="Training batch directories used to build z-level grid",
    )
    parser.add_argument(
        "--seed-sequence",
        default="training_data/batch2/validated_simPhysics/demo_1/5d_sequence/sequence.json",
        help="5D sequence JSON whose first layer is spawned as the starting foundation "
             "(default: batch2/demo_1 layer 0).  Pass '' to disable seeding.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load checkpoint ───────────────────────────────────────────────────────
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}")
        sys.exit(1)

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt       = torch.load(ckpt_path, map_location=device)
    model      = NextBrickModel()
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    norm_stats = ckpt["norm_stats"]
    print(f"  epoch={ckpt.get('epoch','?')}  val_loss={ckpt.get('val_loss', float('nan')):.4f}")

    # ── Build z-level grid ────────────────────────────────────────────────────
    print("Building z-level grid from training data …")
    z_levels, z_to_b = load_z_levels(args.batch_dirs)
    if not z_levels:
        print("ERROR: No z-levels found.  Check --batch-dirs.")
        sys.exit(1)
    print(f"  {len(z_levels)} levels:")
    for z in z_levels:
        print(f"    z={z:.6f}  b={'standing' if z_to_b.get(z)==1 else 'laying'}")

    # ── Load seed layer ───────────────────────────────────────────────────────
    seed_layer = []
    if args.seed_sequence:
        seed_path = Path(args.seed_sequence)
        if not seed_path.exists():
            print(f"ERROR: seed sequence not found: {seed_path}")
            sys.exit(1)
        seed_layer = load_seed_layer(str(seed_path), z_levels)
        if not seed_layer:
            print(f"WARNING: no first-layer bricks found in {seed_path} — starting from scratch.")
        else:
            print(f"Seed layer: {len(seed_layer)} bricks from {seed_path}")

    # ── Initialise Gazebo validator ───────────────────────────────────────────
    if HAVE_ROS2:
        rclpy.init()
        validator = DemoValidatorNode()
    else:
        validator = DemoValidator()

    # ── Initialise MoveIt (optional) ─────────────────────────────────────────
    moveit_node = None
    if HAVE_MOVEIT and not args.no_reachability_check:
        print("Initialising MoveIt reachability checker …")
        moveit_node = _MoveitClient(mode="sim")
        moveit_node.publish_scene_box(
            object_id="table_surface", frame_id="world",
            size_xyz=(2.0, 2.0, 0.02), position_xyz=(0.0, 0.0, -0.02),
        )
        print("MoveIt ready.")

    try:
        placed_7d, history_5d = run_evaluation(
            args, validator, moveit_node, model, norm_stats, z_levels, z_to_b, device,
            seed_layer=seed_layer,
        )

        # ── Export sequences ──────────────────────────────────────────────────
        out_dir = Path(args.output_dir)
        (out_dir / "7d_sequence").mkdir(parents=True, exist_ok=True)
        (out_dir / "5d_sequence").mkdir(parents=True, exist_ok=True)

        seq_7d = [p.tolist() for p in placed_7d]
        seq_5d = []
        for p in placed_7d:
            result = Brick(pose_7d=p).to_5d_pose()
            seq_5d.append(
                result.tolist() if isinstance(result, np.ndarray) else {"error": result}
            )

        with open(out_dir / "7d_sequence" / "sequence.json", "w") as f:
            json.dump(seq_7d, f, indent=2)
        with open(out_dir / "5d_sequence" / "sequence.json", "w") as f:
            json.dump(seq_5d, f, indent=2)

        print(f"\nSaved {len(placed_7d)} poses  →  {out_dir}")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        if moveit_node is not None:
            moveit_node.destroy_node()
        validator.destroy_node()
        if HAVE_ROS2:
            rclpy.shutdown()


if __name__ == "__main__":
    main()
