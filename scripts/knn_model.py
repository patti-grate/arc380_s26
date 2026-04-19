"""
knn_model.py — K-Nearest Neighbours prediction model for brick placement.

Responsibilities (self-contained):
  - Loading 5D sequence datasets from disk
  - Building context/target pairs with:
      * Relative normalisation (all poses expressed relative to the latest
        real brick in the context window so the model is position/orientation
        agnostic)
      * Binary mask vectors (padded slots are zeroed AND masked out of the
        distance metric so phantom history cannot pollute neighbour lookup)
  - Fitting (storing training data)
  - Predicting the next brick pose and returning a regulated global 5D pose

Public interface expected by sim_runtime.py:
    model = KNNModel(k=5)
    model.fit_from_dir(data_dir)                  # load all demos
    model.fit_from_dir(data_dir,                  # or filter specific ones
                       allowed_demos=['demo_1', 'demo_2', 'demo_3', 'demo_4'])
    pose_5d = model.predict(history)              # history: list of raw 5D np.ndarrays
                                                  # returns regulated global 5D pose
"""

import json
import os
import numpy as np
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants (shared with sim_runtime via import if needed)
# ---------------------------------------------------------------------------
CONTEXT_WINDOW = 10
POSE_DIM = 5  # [x, y, z, b, r]
ABS_DIM = (
    2  # absolute anchor features appended to context: [x, y] of latest brick only.
    # z omitted — captured by relative dz. r omitted — σ=1.15 rad dominates distance.
)
# Indices for each field
IDX_X, IDX_Y, IDX_Z, IDX_B, IDX_R = 0, 1, 2, 3, 4


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------


def _normalize_pose(pose: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    """
    Express *pose* relative to *anchor*.

    Continuous fields x, y, z, r are shifted.
    Binary field b is left unchanged.

    anchor treated as [ax, ay, az, ab, ar].
    Returns a new array; does not modify inputs.
    """
    normed = pose.copy()
    normed[IDX_X] -= anchor[IDX_X]
    normed[IDX_Y] -= anchor[IDX_Y]
    normed[IDX_Z] -= anchor[IDX_Z]
    normed[IDX_R] -= anchor[IDX_R]
    return normed


def _denormalize_pose(normed: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    """Inverse of _normalize_pose — back to global coordinates."""
    global_pose = normed.copy()
    global_pose[IDX_X] += anchor[IDX_X]
    global_pose[IDX_Y] += anchor[IDX_Y]
    global_pose[IDX_Z] += anchor[IDX_Z]
    global_pose[IDX_R] += anchor[IDX_R]
    return global_pose


# ---------------------------------------------------------------------------
# Dataset building (called internally by KNNModel.fit_from_dir)
# ---------------------------------------------------------------------------


def load_sequences(
    data_dir: str,
    allowed_demos: Optional[List[str]] = None,
) -> List[List[List[float]]]:
    """
    Recursively find every 5d_sequence/sequence.json under *data_dir*.
    Returns a list of sequences; each sequence is a list of 5-element lists.

    Args
    ----
    allowed_demos : if given, only load demos whose directory name is in this
                   list (e.g. ['demo_1', 'demo_2', 'demo_3', 'demo_4']).
                   Matched against the parent of the '5d_sequence' folder.
    """
    sequences = []
    for root, _dirs, files in os.walk(data_dir):
        if os.path.basename(root) == "5d_sequence" and "sequence.json" in files:
            demo_name = os.path.basename(os.path.dirname(root))
            if allowed_demos is not None and demo_name not in allowed_demos:
                continue
            path = os.path.join(root, "sequence.json")
            with open(path) as f:
                seq = json.load(f)
            if seq:
                sequences.append(seq)
    return sequences


def build_dataset(
    sequences: List[List[List[float]]],
    context_window: int = CONTEXT_WINDOW,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build (X, y, masks, anchors) training arrays from validated sequences.

    For brick i in a sequence:
      anchor  = latest real brick in history (brick i-1), or zero-vector if i==0.
      context = last `context_window` bricks, *normalised relative to anchor*,
                zero-padded on the left when history is shorter than the window.
      mask    = 1-D binary array of length context_window: 1 → real slot, 0 → padded.
      X[i]    = context.flatten()                    shape (context_window * POSE_DIM,)
      y[i]    = brick i normalised relative to anchor shape (POSE_DIM,)

    Returns
    -------
    X       : (N, context_window * POSE_DIM)
    y       : (N, POSE_DIM)  — normalised targets
    masks   : (N, context_window)
    anchors : (N, POSE_DIM)  — raw anchor pose for denormalisation at predict time
    """
    X_rows, y_rows, mask_rows, anchor_rows = [], [], [], []

    for seq in sequences:
        seq_arr = np.array(seq, dtype=float)  # (len_seq, POSE_DIM)

        for i, target_pose in enumerate(seq_arr):
            start = max(0, i - context_window)
            history = seq_arr[start:i]  # (<= context_window, POSE_DIM)

            # --- Anchor: latest real brick if any, else zeros ---------------
            anchor = history[-1].copy() if len(history) > 0 else np.zeros(POSE_DIM)

            # --- Binary mask (before padding so it reflects real data) ------
            pad_len = context_window - len(history)
            mask = np.array([0] * pad_len + [1] * len(history), dtype=float)

            # --- Normalise history relative to anchor -----------------------
            if len(history) > 0:
                norm_history = np.array([_normalize_pose(p, anchor) for p in history])
            else:
                norm_history = np.empty((0, POSE_DIM))

            # --- Zero-pad on the left ---------------------------------------
            if pad_len > 0:
                padding = np.zeros((pad_len, POSE_DIM))
                context_poses = np.vstack([padding, norm_history])
            else:
                context_poses = norm_history

            X_rows.append(
                np.concatenate(
                    [
                        context_poses.flatten(),  # (cw * POSE_DIM,) relative
                        [
                            anchor[IDX_X],
                            anchor[IDX_Y],
                        ],  # (ABS_DIM=2,) absolute x,y only
                    ]
                )
            )
            mask_rows.append(mask)
            y_rows.append(_normalize_pose(target_pose, anchor))
            anchor_rows.append(anchor)

    return (
        np.array(X_rows, dtype=float),
        np.array(y_rows, dtype=float),
        np.array(mask_rows, dtype=float),
        np.array(anchor_rows, dtype=float),
    )


# ---------------------------------------------------------------------------
# KNN Model
# ---------------------------------------------------------------------------


class KNNModel:
    """
    K-Nearest Neighbours model for next-brick prediction.

    Dataset loading, normalisation, masking are all handled internally.

    Distance metric:
        Euclidean over feature space, with each feature masked by the
        *query's* binary mask expanded to POSE_DIM width:
            effective_diff[slot] = (X_train[slot] - query[slot]) * mask[slot]
        Padded slots in the query contribute zero distance regardless of what
        the training sample had in those positions.

    Output regulation:
        b  → majority vote among k neighbours, clamped to {0, 1}
        r  → mean of k neighbours, snapped to 0.01 resolution
        x, y, z → mean of k neighbours
    """

    def __init__(self, k: int = 2):
        self.k = k
        self._X_train: Optional[np.ndarray] = None  # (N, context_window * POSE_DIM)
        self._y_train: Optional[np.ndarray] = None  # (N, POSE_DIM) – normalised
        self._context_window: int = CONTEXT_WINDOW
        # First-brick seeding statistics (computed in fit_from_dir)
        self._first_mean: Optional[np.ndarray] = None  # (POSE_DIM,) raw mean of seq[0]
        self._first_std: Optional[np.ndarray] = None  # (POSE_DIM,) std  of seq[0]

    # ------------------------------------------------------------------
    # Dataset loading (self-contained)
    # ------------------------------------------------------------------

    def fit_from_dir(
        self,
        data_dir: str,
        context_window: int = CONTEXT_WINDOW,
        allowed_demos: Optional[List[str]] = None,
    ) -> None:
        """
        Load all validated 5D sequences from *data_dir*, build the dataset
        with masking and relative normalisation, then store for inference.

        Args
        ----
        allowed_demos : optional list of demo folder names to include
                        (e.g. ['demo_1','demo_2','demo_3','demo_4']).
                        If None, all demos under data_dir are loaded.
        """
        self._context_window = context_window
        sequences = load_sequences(data_dir, allowed_demos=allowed_demos)
        if not sequences:
            raise FileNotFoundError(
                f"No 5D sequences found under '{data_dir}'"
                + (f" matching {allowed_demos}" if allowed_demos else "") + "."
            )

        total = sum(len(s) for s in sequences)
        print(f"[KNNModel] Loaded {len(sequences)} sequences, {total} bricks total.")

        # Compute first-brick seeding statistics (raw, un-normalised)
        first_bricks = np.array(
            [s[0] for s in sequences], dtype=float
        )  # (n_seqs, POSE_DIM)
        self._first_mean = first_bricks.mean(axis=0)
        self._first_std = first_bricks.std(axis=0)
        print(
            f"[KNNModel] First-brick seed: mean={np.round(self._first_mean, 4)}  "
            f"std={np.round(self._first_std, 4)}"
        )

        X, y, _masks, _anchors = build_dataset(sequences, context_window)
        self._X_train = X
        self._y_train = y
        print(f"[KNNModel] Dataset built: X{X.shape}, y{y.shape}  k={self.k}")

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Directly supply pre-built (normalised) training arrays.
        Use fit_from_dir for the full pipeline.
        """
        self._X_train = np.array(X, dtype=float)
        self._y_train = np.array(y, dtype=float)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, history: List[np.ndarray], verbose: bool = False) -> np.ndarray:
        """
        Predict the next brick's 5D pose given the full history of placed bricks.

        Args
        ----
        history : list of np.ndarray, each shape (POSE_DIM,).
                  Raw (global) 5D poses in placement order. May be empty.
        verbose : if True, print the normalised context window as a table.

        Returns
        -------
        np.ndarray of shape (POSE_DIM,) — regulated global 5D pose.
        """
        if self._X_train is None:
            raise RuntimeError("Model not fitted. Call fit_from_dir() or fit() first.")

        # ── Special case: no history yet ──────────────────────────────────
        # With an empty mask all KNN distances collapse to zero → neighbour
        # selection becomes arbitrary.  Instead, seed the first brick from
        # the dataset's first-brick distribution plus Gaussian jitter.
        if len(history) == 0:
            pred = self._sample_first_brick()
            if verbose:
                print(
                    "[KNNModel] Context: (empty — first brick seeded from dataset distribution)"
                )
                print(f"[KNNModel] Anchor : (none)")
                print(
                    f"[KNNModel] Output : x={pred[0]:.4f}  y={pred[1]:.4f}  "
                    f"z={pred[2]:.4f}  b={int(pred[3])}  r={pred[4]:.4f}"
                )
            return pred

        cw = self._context_window

        # --- 1. Anchor & normalisation ------------------------------------
        if history:
            anchor = np.array(history[-1], dtype=float)
        else:
            anchor = np.zeros(POSE_DIM)

        recent = list(history[-cw:]) if len(history) >= cw else list(history)
        pad_len = cw - len(recent)

        mask = np.array([0.0] * pad_len + [1.0] * len(recent))  # (cw,)

        if recent:
            norm_recent = np.array(
                [_normalize_pose(np.array(p), anchor) for p in recent]
            )
        else:
            norm_recent = np.empty((0, POSE_DIM))

        if pad_len > 0:
            context_poses = np.vstack([np.zeros((pad_len, POSE_DIM)), norm_recent])
        else:
            context_poses = norm_recent

        if verbose:
            print(
                f"[KNNModel] Anchor (global): x={anchor[0]:.4f}  y={anchor[1]:.4f}  "
                f"z={anchor[2]:.4f}  b={int(anchor[3])}  r={anchor[4]:.4f}  "
                f"[absolute x,y appended to query]"
            )
            print(f"[KNNModel] Context window (normalised, {cw} slots × 5 dims):")
            print(
                f"  {'slot':>4}  {'mask':>4}  {'dx':>8}  {'dy':>8}  {'dz':>8}  {'b':>4}  {'dr':>8}"
            )
            print(
                f"  {'-' * 4}  {'-' * 4}  {'-' * 8}  {'-' * 8}  {'-' * 8}  {'-' * 4}  {'-' * 8}"
            )
            for slot_i, (row, m) in enumerate(zip(context_poses, mask)):
                label = "PAD" if m == 0 else f"t-{cw - slot_i}"
                print(
                    f"  {label:>4}  {'1' if m else '0':>4}  "
                    f"{row[0]:>8.4f}  {row[1]:>8.4f}  {row[2]:>8.4f}  "
                    f"{int(row[3]):>4}  {row[4]:>8.4f}"
                )

        query = np.concatenate(
            [
                context_poses.flatten(),  # (cw * POSE_DIM,) relative
                [
                    anchor[IDX_X],
                    anchor[IDX_Y],
                ],  # (ABS_DIM=2,) absolute x,y only
            ]
        )  # total: (cw * POSE_DIM + ABS_DIM,)

        # --- 2. Masked distance -------------------------------------------
        # Relative slots use the query mask; absolute anchor features are always valid.
        expanded_mask = np.concatenate(
            [
                np.repeat(mask, POSE_DIM),  # (cw * POSE_DIM,)
                np.ones(ABS_DIM),  # (ABS_DIM,)  — never padded
            ]
        )

        diffs = (self._X_train - query) * expanded_mask  # (N, cw * POSE_DIM)
        dists = np.linalg.norm(diffs, axis=1)  # (N,)

        # --- 3 & 4. First-group-to-k race (multimodal-safe aggregation) --------
        # Walk neighbours in distance order. Assign each to the laying (b=0) or
        # standing (b=1) group based on its b value.  The first group that
        # accumulates self.k members wins; we average only within that group.
        # This avoids blending across orientation modes.
        sorted_idx = np.argsort(dists)  # all N neighbours, closest first
        groups: dict = {0: [], 1: []}  # 0 = laying, 1 = standing
        winning_group: Optional[int] = None

        for idx in sorted_idx:
            pose = self._y_train[idx]
            b = int(round(pose[IDX_B]))
            groups[b].append(pose)
            if len(groups[b]) >= self.k:
                winning_group = b
                break

        # Safety fallback: if dataset is tiny and neither group reached k
        if winning_group is None:
            winning_group = max(groups, key=lambda g: len(groups[g]))

        winner_poses = np.array(groups[winning_group])  # (k, POSE_DIM)
        b_regulated = float(winning_group)
        avg_xyz = winner_poses[:, :3].mean(axis=0)
        avg_r = float(winner_poses[:, IDX_R].mean())
        r_regulated = round(avg_r / 0.01) * 0.01  # 0.01 resolution snap

        if verbose:
            other = 1 - winning_group
            label = "standing" if winning_group == 1 else "laying"
            print(
                f"[KNNModel] Race: {label} group reached {self.k} first  "
                f"(other group had {len(groups[other])} before cutoff)"
            )

        pred_normalised = np.array([*avg_xyz, b_regulated, r_regulated])

        # --- 5. Denormalise back to global frame --------------------------
        pred = _denormalize_pose(pred_normalised, anchor)

        if verbose:
            print(
                f"[KNNModel] Pred (normalised): dx={pred_normalised[0]:.4f}  "
                f"dy={pred_normalised[1]:.4f}  dz={pred_normalised[2]:.4f}  "
                f"b={int(pred_normalised[3])}  dr={pred_normalised[4]:.4f}"
            )
            print(
                f"[KNNModel] Pred (global)    : x={pred[0]:.4f}  y={pred[1]:.4f}  "
                f"z={pred[2]:.4f}  b={int(pred[3])}  r={pred[4]:.4f}"
            )

        return pred

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_first_brick(self) -> np.ndarray:
        """
        Sample a starting pose for the very first brick from the dataset's
        first-brick distribution: mean ± Gaussian noise scaled by the
        per-field standard deviation.

        b is majority-voted (all training first-bricks have b=0 or b=1).
        r is snapped to 0.01 resolution after adding noise.
        """
        if self._first_mean is None:
            raise RuntimeError(
                "First-brick statistics not available. Use fit_from_dir()."
            )

        noise = np.random.normal(0.0, self._first_std)
        pose = self._first_mean + noise

        # Regulate b and r just like in the KNN aggregation
        pose[IDX_B] = float(
            round(self._first_mean[IDX_B])
        )  # use mean's majority, not noisy b
        pose[IDX_R] = round(pose[IDX_R] / 0.01) * 0.01
        return pose

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        n = len(self._X_train) if self._X_train is not None else 0
        cw = self._context_window
        feat_dim = cw * POSE_DIM + ABS_DIM
        return f"KNNModel(k={self.k}, n_train={n}, context_window={cw}, feat_dim={feat_dim})"
