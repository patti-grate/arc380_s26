#!/usr/bin/env python3
"""
model_evaluation_rich_feature.py  —  SST (z-refactored, rich 33-dim features,
support-relative MDN output) brick assembly evaluation in Gazebo.

Architecture (mirrors training_SST_z_refactoring.ipynb):
  Transformer encoder → b_head (binary) + layer_head (class)
  → MDN over [u, v, sin_r_rel, cos_r_rel] in support frame
  → decode to world pose [x, y, r]
  → z from z_lookup[layer_id]

Rich 33-dim token per brick:
  base pose      (7): x_n, y_n, b, sin_r, cos_r, layer_norm, time_norm
  layer state    (3): rel_to_top, is_top, is_second_top
  support 1      (6): has_s1, dx, dy, sin_dr, cos_dr, dist
  support 2      (6): has_s2, dx, dy, sin_dr, cos_dr, dist
  support pair   (6): has_pair, pair_dist, pair_ax_sin, pair_ax_cos, u, v
  same-layer occ (5): count_norm, ndx, ndy, ndist, is_frontier

MDN output (support-relative):
  [u/std_uv, v/std_uv, sin_r_rel, cos_r_rel]
  where (u, v) are offsets in the support-pair frame, r_rel is rotation
  relative to the support-pair axis.

Usage:
  python scripts/model_evaluation_rich_feature.py \\
      --checkpoint training_data/trained_models/best_model_z_refactored.pth \\
      --max-bricks 30 --n-candidates 100
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

FEATURE_DIM    = 33
HIDDEN_DIM     = 128   # kept small to reduce overfitting
N_HEADS        = 4
N_LAYERS       = 2
FF_DIM         = 256
DROPOUT        = 0.1
K_MIXTURES     = 5
MAX_BRICKS     = 60
POSE_DIM       = 4     # MDN: [u/std_uv, v/std_uv, sin_r_rel, cos_r_rel]
_LOG2PI        = math.log(2.0 * math.pi)
MAX_LAYER_NORM = 20

BRICK_W = 0.051
BRICK_D = 0.023
BRICK_H = 0.014


# ══════════════════════════════════════════════════════════════════════════════
# Model — mirrors training_SST_z_refactoring.ipynb
# ══════════════════════════════════════════════════════════════════════════════

class NextBrickModel(nn.Module):
    """
    Hierarchical: b_head + layer_head + MDN(4D) conditioned on
    (b, layer_id, has_support_pair).

    MDN predicts support-relative pose [u/std_uv, v/std_uv, sin_r_rel, cos_r_rel].
    Teacher forcing during training: ground-truth b, layer_id, has_pair passed as cond_.
    Inference: predicted b / layer_id used; has_pair from scene rule.
    """

    def __init__(
        self,
        feature_dim=FEATURE_DIM, hidden_dim=HIDDEN_DIM,
        nhead=N_HEADS, num_layers=N_LAYERS, ff_dim=FF_DIM,
        dropout=DROPOUT, K=K_MIXTURES, max_layer_classes=13,
    ):
        super().__init__()
        self.K            = K
        self.pose_dim     = POSE_DIM
        self.n_layers_cls = max_layer_classes

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
        self.layer_head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(64, max_layer_classes),
        )

        # MDN conditioned on (b, layer_id_norm, has_support_pair)
        mdn_in = hidden_dim + 3
        self.mdn_pi        = nn.Linear(mdn_in, K)
        self.mdn_mu        = nn.Linear(mdn_in, K * POSE_DIM)
        self.mdn_log_sigma = nn.Linear(mdn_in, K * POSE_DIM)

    def forward(self, tokens, mask,
                cond_b=None, cond_layer=None, cond_has_pair=None):
        B = tokens.shape[0]
        x   = self.input_proj(tokens)
        cls = self.cls_token.expand(B, 1, -1)
        x   = torch.cat([cls, x], dim=1)
        cls_valid = torch.ones(B, 1, dtype=torch.bool, device=mask.device)
        x     = self.transformer(x, src_key_padding_mask=~torch.cat([cls_valid, mask], dim=1))
        scene = x[:, 0]

        b_logits     = self.b_head(scene)
        layer_logits = self.layer_head(scene)

        if cond_b is None:
            b_cond = b_logits.argmax(-1).float().unsqueeze(-1)
        else:
            b_cond = cond_b.float().unsqueeze(-1)

        if cond_layer is None:
            layer_cond = (layer_logits.argmax(-1).float().unsqueeze(-1)
                          / self.n_layers_cls)
        else:
            layer_cond = cond_layer.float().unsqueeze(-1) / self.n_layers_cls

        if cond_has_pair is None:
            has_pair_cond = torch.zeros(B, 1, device=scene.device)
        else:
            has_pair_cond = cond_has_pair.float().unsqueeze(-1)

        scene_cond = torch.cat([scene, b_cond, layer_cond, has_pair_cond], dim=-1)
        pi    = F.softmax(self.mdn_pi(scene_cond), dim=-1)
        mu    = self.mdn_mu(scene_cond).view(B, self.K, self.pose_dim)
        sigma = (F.softplus(self.mdn_log_sigma(scene_cond))
                 .view(B, self.K, self.pose_dim) + 1e-4)
        return b_logits, layer_logits, pi, mu, sigma


# ══════════════════════════════════════════════════════════════════════════════
# Support-frame geometry
# ══════════════════════════════════════════════════════════════════════════════

def get_support_frame(target_xy, context_poses, context_layer_ids, target_layer_id):
    """
    Build a 2-D reference frame from the nearest lower-layer bricks.

    target_xy         : [x, y] of the target (or proxy) brick — used to rank
                        support candidates by distance
    context_poses     : list of [x,y,z,b,r]
    context_layer_ids : matching list of int
    target_layer_id   : layer_id of the brick being placed

    Returns dict:
      origin       : [ox, oy]
      x_axis       : [ax, ay]  unit vector (from s1 toward s2, or support-1 local x)
      y_axis       : [bx, by]  perpendicular unit vector
      axis_angle   : atan2(ay, ax)
      has_support_1: bool
      has_support_2: bool  (True → pair frame; False → single-support or world frame)
      is_first_layer: bool
    """
    tx, ty = target_xy
    below  = target_layer_id - 1
    is_first_layer = (target_layer_id == 0 or below < 0)

    sup_candidates = sorted(
        [(context_poses[i], context_layer_ids[i])
         for i in range(len(context_poses))
         if context_layer_ids[i] == below],
        key=lambda ps: math.sqrt((ps[0][0] - tx) ** 2 + (ps[0][1] - ty) ** 2),
    )

    if is_first_layer or len(sup_candidates) == 0:
        return dict(origin=[0.0, 0.0], x_axis=[1.0, 0.0], y_axis=[0.0, 1.0],
                    axis_angle=0.0,
                    has_support_1=False, has_support_2=False, is_first_layer=True)

    if len(sup_candidates) == 1:
        s1 = sup_candidates[0][0]
        r_c = canonicalize_r(s1[4])
        c, s = math.cos(r_c), math.sin(r_c)
        return dict(origin=[s1[0], s1[1]], x_axis=[c, s], y_axis=[-s, c],
                    axis_angle=r_c,
                    has_support_1=True, has_support_2=False, is_first_layer=False)

    s1, s2 = sup_candidates[0][0], sup_candidates[1][0]
    mid_x  = (s1[0] + s2[0]) / 2.0
    mid_y  = (s1[1] + s2[1]) / 2.0
    pdx, pdy = s2[0] - s1[0], s2[1] - s1[1]
    d    = math.sqrt(pdx * pdx + pdy * pdy + 1e-8)
    ax, ay = pdx / d, pdy / d
    return dict(origin=[mid_x, mid_y], x_axis=[ax, ay], y_axis=[-ay, ax],
                axis_angle=math.atan2(ay, ax),
                has_support_1=True, has_support_2=True, is_first_layer=False)


def world_to_support_frame(tx, ty, tr, sf, std_uv):
    """
    Convert world-space target (tx, ty, tr) to support-frame-relative coords.
    Returns [u_norm, v_norm, sin_r_rel, cos_r_rel].
    """
    ox, oy = sf["origin"]
    ax, ay = sf["x_axis"]
    bx, by = sf["y_axis"]
    dx, dy = tx - ox, ty - oy
    u = ax * dx + ay * dy
    v = bx * dx + by * dy
    r_rel = canonicalize_r(tr - sf["axis_angle"])
    return [u / std_uv, v / std_uv, math.sin(r_rel), math.cos(r_rel)]


def support_frame_to_world(u_norm, v_norm, sin_r_rel, cos_r_rel, sf, std_uv):
    """
    Decode support-frame-relative MDN prediction to world pose (x, y, r).
    """
    ox, oy = sf["origin"]
    ax, ay = sf["x_axis"]
    bx, by = sf["y_axis"]
    u = u_norm * std_uv
    v = v_norm * std_uv
    x = ox + ax * u + bx * v
    y = oy + ay * u + by * v
    nrm = math.sqrt(sin_r_rel ** 2 + cos_r_rel ** 2 + 1e-8)
    r_world = canonicalize_r(sf["axis_angle"] + math.atan2(sin_r_rel / nrm, cos_r_rel / nrm))
    return x, y, r_world


def infer_support_frame(history_5d, layer_ids, pred_layer):
    """
    Deterministically select a support frame at inference time (Option A).

    Uses the centroid of the support-layer bricks as a proxy target position
    to rank support candidates, matching the typical training distribution.
    """
    below = pred_layer - 1
    if pred_layer == 0 or below < 0 or not history_5d:
        return dict(origin=[0.0, 0.0], x_axis=[1.0, 0.0], y_axis=[0.0, 1.0],
                    axis_angle=0.0,
                    has_support_1=False, has_support_2=False, is_first_layer=True)

    sup_bricks = [history_5d[i] for i in range(len(history_5d))
                  if layer_ids[i] == below]
    if not sup_bricks:
        return dict(origin=[0.0, 0.0], x_axis=[1.0, 0.0], y_axis=[0.0, 1.0],
                    axis_angle=0.0,
                    has_support_1=False, has_support_2=False, is_first_layer=True)

    # Proxy target: centroid of support-layer bricks
    cx = sum(p[0] for p in sup_bricks) / len(sup_bricks)
    cy = sum(p[1] for p in sup_bricks) / len(sup_bricks)
    return get_support_frame([cx, cy], history_5d, layer_ids, pred_layer)


# ══════════════════════════════════════════════════════════════════════════════
# Rich 33-dim feature encoder
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


def encode_brick_rich(pose_5d, layer_id, time_index,
                      context_poses, context_layer_ids,
                      norm_stats, max_layer=MAX_LAYER_NORM):
    """33-dim rich feature vector for one brick (input to Transformer)."""
    ns = norm_stats
    x, y, z, b, r = pose_5d
    r_c = canonicalize_r(r)
    sin_r, cos_r = math.sin(r_c), math.cos(r_c)
    max_lid  = max(context_layer_ids, default=layer_id)
    max_lid  = max(max_lid, layer_id)
    std_dist = (ns["std_x"] + ns["std_y"]) / 2.0

    feat = [
        (x - ns["mean_x"]) / ns["std_x"],
        (y - ns["mean_y"]) / ns["std_y"],
        float(b), sin_r, cos_r,
        layer_id / max_layer,
        time_index / 60.0,
    ]
    feat += [
        (max_lid - layer_id) / max(max_lid, 1),
        float(layer_id == max_lid),
        float(layer_id == max_lid - 1),
    ]

    below    = layer_id - 1
    supports = sorted(
        [(context_poses[i], context_layer_ids[i])
         for i in range(len(context_poses))
         if context_layer_ids[i] == below],
        key=lambda ps: math.sqrt((ps[0][0] - x) ** 2 + (ps[0][1] - y) ** 2),
    )

    def _sup_feat(sp):
        sx, sy, _, sb, sr = sp
        dx, dy = sx - x, sy - y
        dist   = math.sqrt(dx * dx + dy * dy + 1e-8)
        dr     = canonicalize_r(canonicalize_r(sr) - r_c)
        return [1.0, dx / ns["std_x"], dy / ns["std_y"],
                math.sin(dr), math.cos(dr), dist / std_dist]

    _no_sup = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    feat += _sup_feat(supports[0][0]) if len(supports) >= 1 else _no_sup
    feat += _sup_feat(supports[1][0]) if len(supports) >= 2 else _no_sup

    if len(supports) >= 2:
        s1x, s1y = supports[0][0][0], supports[0][0][1]
        s2x, s2y = supports[1][0][0], supports[1][0][1]
        mid_x, mid_y = (s1x + s2x) / 2.0, (s1y + s2y) / 2.0
        pdx, pdy  = s2x - s1x, s2y - s1y
        pair_dist = math.sqrt(pdx * pdx + pdy * pdy + 1e-8)
        pax_sin, pax_cos = pdy / pair_dist, pdx / pair_dist
        fmx, fmy  = x - mid_x, y - mid_y
        u =  pax_cos * fmx + pax_sin * fmy
        v = -pax_sin * fmx + pax_cos * fmy
        feat += [1.0, pair_dist / std_dist, pax_sin, pax_cos,
                 u / ns["std_x"], v / ns["std_y"]]
    else:
        feat += [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]

    same = [context_poses[i] for i in range(len(context_poses))
            if context_layer_ids[i] == layer_id]
    count_norm = len(same) / 12.0
    if same:
        dxdy = [(p[0]-x, p[1]-y, math.sqrt((p[0]-x)**2+(p[1]-y)**2)) for p in same]
        ndx, ndy, ndist = min(dxdy, key=lambda d: d[2])
        is_frontier = float(len(same) <= 2)
    else:
        ndx, ndy, ndist, is_frontier = 0.0, 0.0, 0.0, 1.0
    feat += [count_norm, ndx / ns["std_x"], ndy / ns["std_y"],
             ndist / std_dist, is_frontier]

    assert len(feat) == 33
    return feat


def history_to_encoded_rich(history_5d, norm_stats):
    if not history_5d:
        return []
    layer_ids = assign_layer_ids(history_5d)
    return [
        encode_brick_rich(pose, lid, t, history_5d[:t], layer_ids[:t], norm_stats)
        for t, (pose, lid) in enumerate(zip(history_5d, layer_ids))
    ]


# ══════════════════════════════════════════════════════════════════════════════
# Inference utilities
# ══════════════════════════════════════════════════════════════════════════════

def _build_input_tensors(history_encoded_rich, device):
    tokens = torch.zeros(1, MAX_BRICKS, FEATURE_DIM)
    mask   = torch.zeros(1, MAX_BRICKS, dtype=torch.bool)
    for i, h in enumerate(history_encoded_rich[:MAX_BRICKS]):
        tokens[0, i] = torch.tensor(h, dtype=torch.float32)
        mask[0, i]   = True
    return tokens.to(device), mask.to(device)


def sample_candidates(model, history_encoded, history_5d, norm_stats,
                      n_candidates, device, z_lookup, max_layer_classes):
    """
    Sample n_candidates from the z-refactored support-relative model.

    1. Predict b and layer_id from discrete heads.
    2. Build support frame from current scene (deterministic, Option A).
    3. MDN samples [u_norm, v_norm, sin_r_rel, cos_r_rel] in support frame.
    4. Decode to world [x, y, z, b, r].

    Returns (candidates, b_prob, layer_prob).
    """
    ns     = norm_stats
    std_uv = ns.get("std_uv", (ns["std_x"] + ns["std_y"]) / 2.0)
    model.eval()
    tokens, mask = _build_input_tensors(history_encoded, device)

    # First pass: get discrete predictions (no conditioning)
    with torch.no_grad():
        b_logits, layer_logits, _, _, _ = model(tokens, mask)

    b_prob     = F.softmax(b_logits[0],     dim=-1).cpu().numpy()
    layer_prob = F.softmax(layer_logits[0], dim=-1).cpu().numpy()
    pred_layer = int(layer_prob.argmax())
    pred_b     = int(b_prob.argmax())

    z_val = z_lookup.get(pred_layer, z_lookup.get(max(z_lookup.keys()), 0.0))

    # Build support frame for this predicted layer
    layer_ids   = assign_layer_ids(history_5d) if history_5d else []
    support_frame = infer_support_frame(history_5d, layer_ids, pred_layer)
    has_pair    = 1.0 if support_frame["has_support_2"] else 0.0

    # Second pass: sample MDN conditioned on predicted b, layer, has_pair
    has_pair_t = torch.tensor([has_pair], dtype=torch.float32, device=device)  # (1,) → (1,1) after unsqueeze in forward
    b_t        = torch.tensor([pred_b],    dtype=torch.long,    device=device)
    layer_t    = torch.tensor([pred_layer],dtype=torch.long,    device=device)

    with torch.no_grad():
        _, _, pi, mu, sigma = model(tokens, mask,
                                    cond_b=b_t, cond_layer=layer_t,
                                    cond_has_pair=has_pair_t)

    pi_np  = pi[0].cpu().numpy()
    mu_np  = mu[0].cpu().numpy()    # (K, 4) in normalized support-relative space
    sig_np = sigma[0].cpu().numpy()

    candidates = []
    for _ in range(n_candidates):
        k = np.random.choice(len(pi_np), p=pi_np / pi_np.sum())
        s = np.random.randn(POSE_DIM) * sig_np[k] + mu_np[k]
        # s = [u_norm, v_norm, sin_r_rel, cos_r_rel]
        u_norm, v_norm = float(s[0]), float(s[1])
        sin_r_rel, cos_r_rel = float(s[2]), float(s[3])

        # Decode to world
        xr, yr, rr = support_frame_to_world(
            u_norm, v_norm, sin_r_rel, cos_r_rel, support_frame, std_uv
        )
        sin_r = math.sin(rr); cos_r = math.cos(rr)

        # Log-likelihood in normalized output space
        norm_s    = np.array([u_norm, v_norm, sin_r_rel, cos_r_rel])
        log_gauss = (
            -0.5 * (((norm_s - mu_np) / sig_np) ** 2
                    + 2 * np.log(sig_np) + _LOG2PI)
        ).sum(axis=-1)
        log_prob = float(np.logaddexp.reduce(np.log(pi_np + 1e-8) + log_gauss))

        candidates.append({
            "x": xr, "y": yr, "z": z_val,
            "b": pred_b, "sin_r": sin_r, "cos_r": cos_r, "r": rr,
            "layer_id": pred_layer, "log_prob": log_prob,
        })

    candidates.sort(key=lambda c: c["log_prob"], reverse=True)
    return candidates, b_prob, layer_prob


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

def run_evaluation(args, validator, moveit_node, model, norm_stats,
                   z_lookup, max_layer_classes, device, seed_layer=None):
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

            history_encoded = history_to_encoded_rich(history_5d, norm_stats)
            candidates, b_prob, layer_prob = sample_candidates(
                model, history_encoded, history_5d, norm_stats,
                n_candidates=args.n_candidates,
                device=device, z_lookup=z_lookup,
                max_layer_classes=max_layer_classes,
            )
            pred_layer = candidates[0]["layer_id"] if candidates else -1
            print(f"    b: laying={b_prob[0]:.3f}  standing={b_prob[1]:.3f}  "
                  f"pred_layer={pred_layer}  best log_p={candidates[0]['log_prob']:.2f}")

            validator.fetch_latest_poses_from_gz()
            foundation_states = validator.current_model_states.copy()

            for c_idx, cand in enumerate(candidates):
                brick_name = f"model_brick_{brick_idx:03d}"
                pose_7d    = candidate_to_7d(cand)

                if check_spatial_collision(cand, history_5d):
                    print(f"    [c{c_idx:03d}] REJECT — overlaps placed brick  "
                          f"x={cand['x']:.4f} y={cand['y']:.4f} "
                          f"z={cand['z']:.4f} b={cand['b']}")
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
                placed    = True
                layer_id  = assign_layer_ids(history_5d)[-1]
                print(f"    [c{c_idx:03d}] ACCEPT — "
                      f"x={cand['x']:.4f} y={cand['y']:.4f} z={cand['z']:.4f}  "
                      f"b={cand['b']}  layer={layer_id}  log_p={cand['log_prob']:.2f}")
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
        description="Rich-feature SST (support-relative MDN) brick assembly evaluation."
    )
    parser.add_argument(
        "--checkpoint",
        default="training_data/trained_models/best_model_z_refactored.pth",
    )
    parser.add_argument("--max-bricks",   type=int, default=30)
    parser.add_argument("--n-candidates", type=int, default=100)
    parser.add_argument("--max-rounds",   type=int, default=5)
    parser.add_argument("--no-reachability-check", action="store_true")
    parser.add_argument("--output-dir",   default="training_data/model_generated_rich")
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
    max_layer_classes = int(ckpt.get("max_layer_classes", 13))
    # std_uv may be missing in older checkpoints; fall back to (std_x+std_y)/2
    if "std_uv" in ckpt:
        norm_stats["std_uv"] = float(ckpt["std_uv"])
    else:
        norm_stats["std_uv"] = (norm_stats["std_x"] + norm_stats["std_y"]) / 2.0

    model = NextBrickModel(max_layer_classes=max_layer_classes).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded {ckpt_path.name}  epoch={ckpt.get('epoch','?')}  "
          f"val_loss={ckpt.get('val_loss', float('nan')):.4f}  "
          f"max_layer_classes={max_layer_classes}")
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
            z_lookup, max_layer_classes, device,
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
