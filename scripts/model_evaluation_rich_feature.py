#!/usr/bin/env python3
"""
model_evaluation_rich_feature.py  —  COG-aware staged SST brick assembly in Gazebo.

Architecture (mirrors training_SST_sequential2.ipynb):
  Transformer encoder → scene (CLS) + per-token embeddings
  → advance_head (binary: stay / advance layer)
  → derive pred_layer = max_hist_layer + advance
  → derive pred_b = B_FROM_LAYER[pred_layer % 3]   (period-3 deterministic)
  → derive pred_z = z_lookup[pred_layer]            (from training-data mean)
  → ss_head (support state 0/1/2)
  → heuristic s1/s2 selection from previous layer
  → proj_head([scene | s1_emb | s2_emb | layer_norm]) → [alpha_A, perp_A, alpha_B, perp_B]
  → decode critical-point projections → world (x, y, r)

16-dim token per brick:
  base pose    (5): rel_x, rel_y, b, sin_r, cos_r  (coords relative to context centroid)
  rel_age      (1): (len(context) - time_index) / (MAX_LAYER_NORM * 3)
  layer state  (3): rel_to_top, is_top, is_second_top
  same-layer   (5): count_norm, ndx, ndy, ndist, is_frontier
  geometry     (1): crit_span_norm
  support ctx  (1): num_support_layer_norm

Usage:
  python scripts/model_evaluation_rich_feature.py \\
      --checkpoint training_data/trained_models/latest_model_cog_aware.pth \\
      --max-bricks 30 --n-candidates 100 \\
      --seed-sequence training_data/batch2/validated_simPhysics/demo_1/5d_sequence/sequence.json
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pose_conversion import Brick

# ── ROS 2 / Gazebo ───────────────────────────────────────────────────────────
HAVE_ROS2 = False
try:
    import rclpy
    from ros_gz_interfaces.srv import DeleteEntity
    HAVE_ROS2 = True
except ImportError:
    print("WARNING: rclpy not found — running in dry-run/mock mode.")

# ── MoveIt reachability ───────────────────────────────────────────────────────
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

from demo_validation import DemoValidator, check_placement_reachable
if HAVE_ROS2:
    from demo_validation import DemoValidatorNode


# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

FEATURE_DIM      = 16
PROJ_DIM         = 4      # [alpha_A, perp_A, alpha_B, perp_B]
N_SUPPORT_STATES = 3
HIDDEN_DIM       = 128
N_HEADS          = 4
N_LAYERS         = 2
FF_DIM           = 256
DROPOUT          = 0.1
MAX_BRICKS       = 60
MAX_LAYER_NORM   = 20

BRICK_W = 0.051
BRICK_D = 0.023
BRICK_H = 0.014

# Period-3 b pattern: layer%3==0 → laying(0), 1 → standing(1), 2 → laying(0)
B_FROM_LAYER = [0, 1, 0]

# Maximum perpendicular distance (m) for a brick to count as a valid support
SUPPORT_RADIUS_M = 0.03

# Fixed noise for projection sampling at inference
SIGMA_PROJ = [0.15, 0.08, 0.15, 0.08]


# ══════════════════════════════════════════════════════════════════════════════
# Model — mirrors training_SST_sequential2.ipynb
# ══════════════════════════════════════════════════════════════════════════════

def b_from_layer(layer_id: int) -> int:
    """Period-3 deterministic b: 0→laying, 1→standing, 2→laying, repeat."""
    return B_FROM_LAYER[layer_id % len(B_FROM_LAYER)]


class NextBrickModel(nn.Module):
    """
    Simplified COG-aware model matching training_SST_sequential2.ipynb.

    Heads:
      advance_head  scene → binary (stay / advance to next layer)
      ss_head       scene → support-state (0=ground, 1=one-sup, 2=two-sup)
      proj_head     [scene | s1_emb | s2_emb | layer_norm(1)] → PROJ_DIM=4

    b and z are derived deterministically from pred_layer at inference.
    s1/s2 are selected heuristically at inference; teacher-forced in training.
    """

    def __init__(
        self,
        feature_dim=FEATURE_DIM,
        hidden_dim=HIDDEN_DIM,
        nhead=N_HEADS,
        num_layers=N_LAYERS,
        ff_dim=FF_DIM,
        dropout=DROPOUT,
        max_layer_classes=12,
    ):
        super().__init__()
        H = hidden_dim
        self.hidden_dim        = H
        self._max_layer_classes = max_layer_classes

        self.input_proj = nn.Sequential(
            nn.Linear(feature_dim, H), nn.ReLU(), nn.Linear(H, H),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, H) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=H, nhead=nhead, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.advance_head = nn.Sequential(
            nn.Linear(H, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 2),
        )
        self.ss_head = nn.Sequential(
            nn.Linear(H, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, N_SUPPORT_STATES),
        )
        self.no_support_emb = nn.Parameter(torch.randn(H) * 0.02)

        # proj_head input: scene(H) | s1_emb(H) | s2_emb(H) | layer_norm(1)
        self.proj_head = nn.Sequential(
            nn.Linear(3 * H + 1, 128), nn.ReLU(), nn.Linear(128, PROJ_DIM),
        )

    def _encode(self, tokens, mask):
        B = tokens.shape[0]
        x   = self.input_proj(tokens)
        cls = self.cls_token.expand(B, 1, -1)
        x   = torch.cat([cls, x], dim=1)
        cls_valid = torch.ones(B, 1, dtype=torch.bool, device=mask.device)
        x = self.transformer(x, src_key_padding_mask=~torch.cat([cls_valid, mask], dim=1))
        return x[:, 0], x[:, 1:]

    def _gather_support_emb(self, token_embs, idx):
        B, L, _ = token_embs.shape
        absent = idx >= L
        safe   = idx.clamp(0, L - 1)
        emb    = token_embs[torch.arange(B, device=token_embs.device), safe].clone()
        no_emb = self.no_support_emb.unsqueeze(0).expand(B, -1)
        emb[absent] = no_emb[absent]
        return emb

    def forward(
        self,
        tokens,           # (B, L, FEATURE_DIM)
        mask,             # (B, L)  True = real token
        sup_mask,         # (B, L)  True = support-layer candidate (unused in forward)
        cond_layer=None,  # (B,) long — teacher forcing / inference conditioning
        cond_s1=None,     # (B,) long — teacher forcing s1 index
        cond_s2=None,     # (B,) long — teacher forcing s2 index
    ):
        scene, token_embs = self._encode(tokens, mask)

        advance_logits = self.advance_head(scene)   # (B, 2)
        ss_logits      = self.ss_head(scene)         # (B, N_SUPPORT_STATES)

        layer_scalar = (
            cond_layer.float() if cond_layer is not None
            else advance_logits.argmax(-1).float()
        ) / max(self._max_layer_classes - 1, 1)
        layer_scalar = layer_scalar.unsqueeze(-1)    # (B, 1)

        if cond_s1 is not None:
            s1_emb = self._gather_support_emb(token_embs, cond_s1)
            s2_emb = self._gather_support_emb(token_embs, cond_s2)
        else:
            s1_emb = self.no_support_emb.unsqueeze(0).expand(scene.shape[0], -1)
            s2_emb = self.no_support_emb.unsqueeze(0).expand(scene.shape[0], -1)

        proj_inp = torch.cat([scene, s1_emb, s2_emb, layer_scalar], dim=-1)  # (B, 3H+1)
        proj     = self.proj_head(proj_inp)

        return advance_logits, ss_logits, proj


# ══════════════════════════════════════════════════════════════════════════════
# Structural geometry
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


def get_critical_points(pose_5d):
    """Midpoints of the two short ends of the brick XY footprint."""
    x, y, _, b, r = float(pose_5d[0]), float(pose_5d[1]), float(pose_5d[2]), int(pose_5d[3]), float(pose_5d[4])
    half_span = (BRICK_W if b == 0 else BRICK_D) / 2.0
    c, s = math.cos(r), math.sin(r)
    return [x + c * half_span, y + s * half_span], [x - c * half_span, y - s * half_span], half_span


def decode_pose_from_projections(support_state, s1_pose, s2_pose,
                                  alpha_A, perp_A, alpha_B, perp_B):
    """
    Recover world (x, y, r) from critical-point projections.
    Returns [x, y, r] or None if support_state == 0.
    """
    def decode_point(sup_pose, alpha, perp):
        sA, sB, s_half = get_critical_points(sup_pose)
        span = 2.0 * s_half
        dx, dy = sB[0] - sA[0], sB[1] - sA[1]
        d  = math.sqrt(dx * dx + dy * dy + 1e-8)
        ax, ay = dx / d, dy / d
        px, py = -ay, ax
        wx = sA[0] + alpha * span * ax + perp * span * px
        wy = sA[1] + alpha * span * ay + perp * span * py
        return [wx, wy]

    if support_state == 0 or s1_pose is None:
        return None

    tA = decode_point(s1_pose, alpha_A, perp_A)
    tB = decode_point(
        s2_pose if (support_state == 2 and s2_pose is not None) else s1_pose,
        alpha_B, perp_B,
    )

    cx = (tA[0] + tB[0]) / 2.0
    cy = (tA[1] + tB[1]) / 2.0
    r  = canonicalize_r(math.atan2(tA[1] - tB[1], tA[0] - tB[0]))
    return [cx, cy, r]


# ══════════════════════════════════════════════════════════════════════════════
# 17-dim brick token encoder
# ══════════════════════════════════════════════════════════════════════════════

def encode_brick(pose_5d, layer_id, time_index,
                 context_poses, context_layer_ids,
                 norm_stats, max_layer=MAX_LAYER_NORM):
    """16-dim feature vector for one brick token (matches training_SST_sequential2.ipynb)."""
    ns = norm_stats
    x, y, z, b, r = float(pose_5d[0]), float(pose_5d[1]), float(pose_5d[2]), int(pose_5d[3]), float(pose_5d[4])
    r_c    = canonicalize_r(r)
    std_xy = (ns["std_x"] + ns["std_y"]) / 2.0

    max_ctx_lid = max(context_layer_ids, default=layer_id)
    max_ctx_lid = max(max_ctx_lid, layer_id)

    # Relative coordinates — centroid of context bricks (or self if no context)
    if context_poses:
        cx = sum(p[0] for p in context_poses) / len(context_poses)
        cy = sum(p[1] for p in context_poses) / len(context_poses)
    else:
        cx, cy = x, y
    rel_x = (x - cx) / std_xy
    rel_y = (y - cy) / std_xy

    # Base pose (5)
    feat = [rel_x, rel_y, float(b), math.sin(r_c), math.cos(r_c)]
    # Relative age (1): 0 for the most recently placed brick in the current context
    rel_age = (len(context_poses) - time_index) / float(MAX_LAYER_NORM * 3)
    feat.append(rel_age)
    # Layer state (3)
    feat += [
        (max_ctx_lid - layer_id) / max(max_ctx_lid, 1),
        float(layer_id == max_ctx_lid),
        float(layer_id == max_ctx_lid - 1),
    ]
    # Same-layer occupancy (5)
    same = [context_poses[i] for i in range(len(context_poses))
            if context_layer_ids[i] == layer_id]
    if same:
        dxdy = [(p[0]-x, p[1]-y, math.sqrt((p[0]-x)**2+(p[1]-y)**2)) for p in same]
        ndx, ndy, ndist = min(dxdy, key=lambda d: d[2])
        is_frontier = float(len(same) <= 2)
    else:
        ndx = ndy = ndist = 0.0
        is_frontier = 1.0
    feat += [len(same) / 12.0, ndx / ns["std_x"], ndy / ns["std_y"],
             ndist / std_xy, is_frontier]
    # Geometry (1)
    half_span = (BRICK_W if b == 0 else BRICK_D) / 2.0
    feat.append(half_span / std_xy)
    # Support context (1)
    num_sup = sum(1 for lid in context_layer_ids if lid == layer_id - 1)
    feat.append(min(num_sup, 10) / 10.0)

    assert len(feat) == FEATURE_DIM, f"Expected {FEATURE_DIM}, got {len(feat)}"
    return feat


# ══════════════════════════════════════════════════════════════════════════════
# Inference — 3-pass staged sampling
# ══════════════════════════════════════════════════════════════════════════════

def snap_z(layer_id, z_lookup):
    if layer_id in z_lookup:
        return z_lookup[layer_id]
    return z_lookup[max(z_lookup.keys())]


def sample_candidates(model, history_5d, norm_stats, n_candidates, device, z_lookup):
    """
    3-pass staged inference matching training_SST_sequential2.ipynb.

    Pass 1: advance_head → pred_advance → derive pred_layer, pred_b (period-3), pred_z
            ss_head → pred_ss
    Pass 2: heuristic support selection — random s1 from nearby, nearest s2
    Pass 3: proj_head conditioned on (layer, s1, s2) → sample noise → decode pose

    Returns (candidates, pred_b, pred_layer, pred_ss).
    """
    ns = norm_stats
    model.eval()

    layer_ids = assign_layer_ids(history_5d) if history_5d else []
    encoded = [
        encode_brick(history_5d[t], layer_ids[t], t,
                     history_5d[:t], layer_ids[:t], ns)
        for t in range(len(history_5d))
    ]

    tokens = torch.zeros(1, MAX_BRICKS, FEATURE_DIM, dtype=torch.float32)
    mask   = torch.zeros(1, MAX_BRICKS, dtype=torch.bool)
    for i, h in enumerate(encoded[:MAX_BRICKS]):
        tokens[0, i] = torch.tensor(h, dtype=torch.float32)
        mask[0, i]   = True
    tokens = tokens.to(device)
    mask   = mask.to(device)

    # ── Pass 1: advance → derive layer / b / z  +  support state ─────────────
    dummy_sup = torch.zeros(1, MAX_BRICKS, dtype=torch.bool, device=device)
    with torch.no_grad():
        advance_logits, ss_logits, _ = model(tokens, mask, dummy_sup)

    pred_advance   = int(advance_logits[0].argmax())
    max_hist_layer = max(layer_ids) if layer_ids else -1
    pred_layer     = max(0, max_hist_layer + pred_advance)
    pred_b         = b_from_layer(pred_layer)
    pred_z         = snap_z(pred_layer, z_lookup)
    pred_ss        = int(ss_logits[0].argmax())

    # ── Pass 2: heuristic support selection ───────────────────────────────────
    prev_layer  = pred_layer - 1
    sup_indices = [i for i, lid in enumerate(layer_ids[:MAX_BRICKS]) if lid == prev_layer]

    pred_s1, pred_s2 = -1, -1
    s1_pose,  s2_pose = None, None

    if pred_ss > 0 and sup_indices:
        if history_5d:
            avg_x = float(np.mean([p[0] for p in history_5d]))
            avg_y = float(np.mean([p[1] for p in history_5d]))
        else:
            avg_x = avg_y = 0.0

        nearby = [
            i for i in sup_indices
            if math.sqrt((history_5d[i][0] - avg_x) ** 2 +
                         (history_5d[i][1] - avg_y) ** 2) < SUPPORT_RADIUS_M * 3
        ] or sup_indices

        if nearby:
            pred_s1 = int(np.random.choice(nearby))
            s1_pose = history_5d[pred_s1]

            if pred_ss == 2 and len(nearby) > 1:
                s1_xy  = [s1_pose[0], s1_pose[1]]
                others = [
                    (i, (history_5d[i][0] - s1_xy[0]) ** 2 +
                        (history_5d[i][1] - s1_xy[1]) ** 2)
                    for i in nearby if i != pred_s1
                ]
                if others:
                    pred_s2 = min(others, key=lambda kv: kv[1])[0]
                    s2_pose = history_5d[pred_s2]

    # ── Pass 3: proj_head conditioned on s1/s2 → decode ─────────────────────
    sup_mask = torch.zeros(1, MAX_BRICKS, dtype=torch.bool, device=device)
    for i in sup_indices:
        sup_mask[0, i] = True

    layer_t = torch.tensor([pred_layer], dtype=torch.long, device=device)
    s1_t    = torch.tensor(
        [pred_s1 if pred_s1 >= 0 else MAX_BRICKS], dtype=torch.long, device=device)
    s2_t    = torch.tensor(
        [pred_s2 if pred_s2 >= 0 else MAX_BRICKS], dtype=torch.long, device=device)

    with torch.no_grad():
        _, _, proj_mean = model(tokens, mask, sup_mask,
                                cond_layer=layer_t, cond_s1=s1_t, cond_s2=s2_t)

    mu_proj = proj_mean[0].cpu().numpy()
    sigmas  = np.array(SIGMA_PROJ)

    candidates = []
    for _ in range(n_candidates):
        proj_s = mu_proj + np.random.randn(4) * sigmas
        alpha_A, perp_A, alpha_B, perp_B = proj_s.tolist()

        if pred_ss == 0 or s1_pose is None:
            lx = [h[0] for h in history_5d] or [0.0]
            ly = [h[1] for h in history_5d] or [0.0]
            x  = float(np.mean(lx)) + np.random.randn() * 0.02
            y  = float(np.mean(ly)) + np.random.randn() * 0.02
            r  = canonicalize_r(float(np.random.uniform(0, math.pi)))
        else:
            result = decode_pose_from_projections(
                pred_ss, s1_pose, s2_pose, alpha_A, perp_A, alpha_B, perp_B,
            )
            if result is None:
                continue
            x, y, r = result

        candidates.append({
            "x": x, "y": y, "z": pred_z,
            "b": pred_b, "r": r,
            "sin_r": math.sin(r), "cos_r": math.cos(r),
            "layer_id": pred_layer,
            "support_state": pred_ss,
            "s1_idx": pred_s1, "s2_idx": pred_s2,
        })

    return candidates, pred_b, pred_layer, pred_ss


def candidate_to_7d(cand):
    brick = Brick()
    brick.from_5d_pose([cand["x"], cand["y"], cand["z"], cand["b"], cand["r"]])
    return brick.get_7d_pose()


# ══════════════════════════════════════════════════════════════════════════════
# Spatial collision helpers
# ══════════════════════════════════════════════════════════════════════════════

def _brick_z_half(b):
    return (BRICK_H if b == 0 else BRICK_W) / 2.0


def _brick_xy_half_extents(b):
    return (BRICK_W / 2.0, BRICK_D / 2.0) if b == 0 else (BRICK_D / 2.0, BRICK_H / 2.0)


def _obb2d_intersect(cx1, cy1, hw1, hd1, r1,
                     cx2, cy2, hw2, hd2, r2, margin=0.0):
    hw1 += margin; hd1 += margin
    hw2 += margin; hd2 += margin
    c1, s1 = math.cos(r1), math.sin(r1)
    c2, s2 = math.cos(r2), math.sin(r2)
    dx, dy = cx2 - cx1, cy2 - cy1
    for ax, ay in [(c1, s1), (-s1, c1), (c2, s2), (-s2, c2)]:
        sep = abs(ax * dx + ay * dy)
        rp1 = hw1 * abs(ax * c1 + ay * s1) + hd1 * abs(-ax * s1 + ay * c1)
        rp2 = hw2 * abs(ax * c2 + ay * s2) + hd2 * abs(-ax * s2 + ay * c2)
        if sep > rp1 + rp2:
            return False
    return True


def check_spatial_collision(cand, history_5d, xy_margin=0.001):
    cx, cy, cz = cand["x"], cand["y"], cand["z"]
    cb, cr = int(cand["b"]), cand["r"]
    czh = _brick_z_half(cb)
    chw, chd = _brick_xy_half_extents(cb)
    for p in history_5d:
        px, py, pz, pb, pr = p[0], p[1], p[2], int(p[3]), p[4]
        pzh = _brick_z_half(pb)
        if abs(cz - pz) >= czh + pzh:
            continue
        phw, phd = _brick_xy_half_extents(pb)
        if _obb2d_intersect(cx, cy, chw, chd, cr, px, py, phw, phd, pr, xy_margin):
            return True
    return False


# ══════════════════════════════════════════════════════════════════════════════
# Simulation helpers
# ══════════════════════════════════════════════════════════════════════════════

def _remove_brick(validator, brick_name):
    if not HAVE_ROS2:
        validator.current_model_states.pop(brick_name, None)
        return
    try:
        if hasattr(validator, "remove_client") and \
                validator.remove_client.wait_for_service(timeout_sec=1.0):
            req = DeleteEntity.Request()  # type: ignore[name-defined]
            req.entity.name = brick_name
            req.entity.type = 2
            future = validator.remove_client.call_async(req)
            rclpy.spin_until_future_complete(validator, future, timeout_sec=2.0)  # type: ignore[name-defined]
        time.sleep(0.3)
        validator.fetch_latest_poses_from_gz()
    except Exception as exc:
        print(f"    [warn] Could not remove {brick_name}: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# Seed layer
# ══════════════════════════════════════════════════════════════════════════════

def load_seed_layer(seed_path, z_lookup):
    with open(seed_path) as f:
        poses = json.load(f)
    target_z = z_lookup.get(0, None)
    if target_z is None:
        return []
    z_tol = 5e-3
    return [
        [float(p[0]), float(p[1]), target_z, int(p[3]), float(p[4])]
        for p in poses if abs(float(p[2]) - target_z) < z_tol
    ]


# ══════════════════════════════════════════════════════════════════════════════
# Main evaluation loop
# ══════════════════════════════════════════════════════════════════════════════

_SS_NAMES = {0: "ground", 1: "one-sup", 2: "two-sup"}


def run_evaluation(args, validator, moveit_node, model, norm_stats,
                   z_lookup, device, seed_layer=None):
    history_5d     = []
    placed_7d      = []
    total_rejected = 0
    brick_idx      = 0

    validator.reset_world()
    if moveit_node is not None:
        moveit_node.remove_all_world_collision_objects()
        moveit_node.publish_scene_box(
            object_id="table_surface", frame_id="world",
            size_xyz=(2.0, 2.0, 0.02), position_xyz=(0.0, 0.0, -0.02),
        )

    if seed_layer:
        print(f"\nSpawning {len(seed_layer)} seed bricks …")
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
        time.sleep(2.0 if HAVE_ROS2 else 0.05)
        validator.fetch_latest_poses_from_gz()
        print(f"  Seed ready — model will generate {args.max_bricks} bricks on top.\n")

    while brick_idx < args.max_bricks:
        print(f"\n{'='*62}")
        print(f"  Brick {brick_idx + 1:3d} / {args.max_bricks}   |   "
              f"history: {len(history_5d)} bricks")
        print(f"{'='*62}")

        placed  = False
        round_n = 0

        while not placed and round_n < args.max_rounds:
            round_n += 1
            print(f"  [round {round_n}/{args.max_rounds}]  "
                  f"sampling {args.n_candidates} candidates …")

            candidates, pred_b, pred_layer, pred_ss = sample_candidates(
                model, history_5d, norm_stats,
                n_candidates=args.n_candidates,
                device=device,
                z_lookup=z_lookup,
            )
            b_str = "laying" if pred_b == 0 else "standing"
            print(f"    pred: b={b_str}  layer={pred_layer}  "
                  f"ss={_SS_NAMES[pred_ss]}  candidates={len(candidates)}")

            validator.fetch_latest_poses_from_gz()
            foundation_states = validator.current_model_states.copy()

            for c_idx, cand in enumerate(candidates):
                brick_name = f"model_brick_{brick_idx:03d}"
                pose_7d    = candidate_to_7d(cand)

                if check_spatial_collision(cand, history_5d):
                    print(f"    [c{c_idx:03d}] REJECT — overlaps placed brick  "
                          f"x={cand['x']:.4f} y={cand['y']:.4f} z={cand['z']:.4f}")
                    total_rejected += 1
                    continue

                effective_pose = check_placement_reachable(moveit_node, pose_7d)
                if effective_pose is None:
                    print(f"    [c{c_idx:03d}] REJECT — not reachable  "
                          f"x={cand['x']:.4f} y={cand['y']:.4f} z={cand['z']:.4f}")
                    total_rejected += 1
                    continue

                if not validator.spawn_brick(brick_name, effective_pose):
                    print(f"    [c{c_idx:03d}] REJECT — spawn failed")
                    total_rejected += 1
                    continue

                time.sleep(2.0 if HAVE_ROS2 else 0.05)
                validator.fetch_latest_poses_from_gz()

                if HAVE_ROS2 and brick_name in validator.current_model_states:
                    settled_z = validator.current_model_states[brick_name].position.z
                    dz = settled_z - cand["z"]
                    if abs(dz) > 0.01:
                        print(f"    [c{c_idx:03d}] REJECT — brick fell {dz:.3f}m "
                              f"from z={cand['z']:.4f}")
                        _remove_brick(validator, brick_name)
                        total_rejected += 1
                        time.sleep(0.5)
                        continue

                check_states = foundation_states.copy()
                if brick_name in validator.current_model_states:
                    check_states[brick_name] = validator.current_model_states[brick_name]

                if not validator.check_stability(check_states):
                    print(f"    [c{c_idx:03d}] REJECT — structure destabilised  "
                          f"z={cand['z']:.4f}  b={cand['b']}")
                    _remove_brick(validator, brick_name)
                    total_rejected += 1
                    time.sleep(0.5 if HAVE_ROS2 else 0.01)
                    continue

                history_5d.append([cand["x"], cand["y"], cand["z"],
                                    float(cand["b"]), cand["r"]])
                placed_7d.append(effective_pose)
                if moveit_node is not None:
                    moveit_node.publish_scene_box(
                        object_id=f"mv_{brick_name}", frame_id="world",
                        size_xyz=BRICK_SIZE_XYZ,
                        position_xyz=tuple(float(v) for v in effective_pose[:3]),
                        quat_xyzw=tuple(float(v) for v in effective_pose[3:]),
                    )
                placed   = True
                layer_id = assign_layer_ids(history_5d)[-1]
                print(f"    [c{c_idx:03d}] ACCEPT — "
                      f"x={cand['x']:.4f} y={cand['y']:.4f} z={cand['z']:.4f}  "
                      f"b={cand['b']}  layer={layer_id}  ss={_SS_NAMES[cand['support_state']]}")
                break

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
        description="COG-aware staged SST brick assembly evaluation."
    )
    parser.add_argument(
        "--checkpoint",
        default="training_data/trained_models/latest_model_cog_aware.pth",
    )
    parser.add_argument("--max-bricks",   type=int, default=30)
    parser.add_argument("--n-candidates", type=int, default=100)
    parser.add_argument("--max-rounds",   type=int, default=5)
    parser.add_argument("--no-reachability-check", action="store_true")
    parser.add_argument("--output-dir",   default="training_data/model_generated_cog")
    parser.add_argument(
        "--seed-sequence",
        default="training_data/batch2/validated_simPhysics/demo_1/5d_sequence/sequence.json",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}")
        sys.exit(1)

    ckpt              = torch.load(ckpt_path, map_location=device)
    norm_stats        = ckpt["norm_stats"]
    z_lookup          = {int(k): float(v) for k, v in ckpt["z_lookup"].items()}
    max_layer_classes = int(ckpt.get("max_layer_classes", 12))
    feature_dim       = int(ckpt.get("feature_dim", FEATURE_DIM))
    hidden_dim        = int(ckpt.get("hidden_dim", HIDDEN_DIM))

    model = NextBrickModel(
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        max_layer_classes=max_layer_classes,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded {ckpt_path.name}  epoch={ckpt.get('epoch','?')}  "
          f"val_loss={ckpt.get('val_loss', float('nan')):.4f}  "
          f"feature_dim={feature_dim}  max_layer_classes={max_layer_classes}")
    for lid, z_val in sorted(z_lookup.items()):
        print(f"  layer {lid:2d}: z = {z_val:.6f} m")

    seed_layer = []
    if args.seed_sequence:
        seed_path = Path(args.seed_sequence)
        if not seed_path.exists():
            print(f"ERROR: seed sequence not found: {seed_path}")
            sys.exit(1)
        seed_layer = load_seed_layer(str(seed_path), z_lookup)
        print(f"Seed layer: {len(seed_layer)} bricks from {seed_path}"
              if seed_layer else "WARNING: no seed bricks found")

    if HAVE_ROS2:
        rclpy.init()  # type: ignore[name-defined]
        validator = DemoValidatorNode()  # type: ignore[name-defined]
    else:
        validator = DemoValidator()

    moveit_node = None
    if HAVE_MOVEIT and not args.no_reachability_check:
        moveit_node = _MoveitClient(mode="sim")
        moveit_node.publish_scene_box(
            object_id="table_surface", frame_id="world",
            size_xyz=(2.0, 2.0, 0.02), position_xyz=(0.0, 0.0, -0.02),
        )

    try:
        placed_7d, history_5d = run_evaluation(
            args, validator, moveit_node, model, norm_stats,
            z_lookup, device,
            seed_layer=seed_layer,
        )

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
