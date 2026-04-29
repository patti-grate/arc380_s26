# Robotic Construction Orchestrator

This repository contains the architecture for robotic brick-stacking, integrating Rhino-based design, Gazebo physics validation, MoveIt motion planning, real-time camera-based perception, and physical ABB robot execution.

---

## 1. Environment Setup (Docker)

The project runs in a containerized ROS 2 Jazzy environment.

### Simulation Environment
Starts Gazebo, MoveIt 2, and the simulation drivers.
```bash
docker compose up ros2_sim
```
*Access the Gazebo GUI at `http://localhost:6080/vnc.html`.*

### Enter Simulation Terminal
```bash
docker exec -it arc380_ros2 bash
```

### Real Robot Environment
Starts the EGM controller interface for communication with the physical ABB IRB 120.
```bash
docker compose up ros2_real
```
*(Note: Ensure the physical robot is in 'Motors On' state and the RAPID EGM program is running).*

### Enter Real Robot Terminal
```bash
docker exec -it arc380_ros2_real bash
```

### UDP Relay (For Real Robot Networking)
If your host machine is not correctly routing packets between Docker and the physical ABB controller (e.g., JG's laptop requires this to send commands to the real robot), you must run the UDP relay script on the Windows host machine:
```cmd
cd c:\dev\arc380_s26
python scripts\udp_relay.py
```

---

## 2. Stability Check (Validating Demos)

Before a design can be built by the robot, it must be validated for structural stability using the physics engine.

```bash
# Inside the docker container
python3 scripts/demo_validation.py --batch batch1 --demo demo_0
```

- **Arguments**:
    - `--batch`: Specify the batch folder (e.g. `batch1`).
    - `--demo`: Specify a single demo name to validate. If omitted, all `demo_*.3dm` files in the batch are processed.

---

## 3. Construction & Motion Planning

Two construction scripts are provided depending on whether the supply brick pose is camera-detected or manually specified.

### 3a. Scripted Construction (`construct_using_validated.py`)

Loads a validated 7D sequence and executes each brick with a fixed (manually specified) supply pose.

#### Execution Modes

| Mode | Flag | Description |
| :--- | :--- | :--- |
| **Dry-Run** | (default) | Plans trajectories and reports pass/fail in console. No motion. |
| **Sim** | `--sim` | Plans trajectories and executes them in Gazebo. |
| **Real** | `--real` | Plans trajectories and executes them on the physical robot. |
| **Replay** | `--replay` | Loads a pre-planned `planned_sequence.json` and executes it directly. |

#### Common Commands

**Plan and Execute in Simulation:**
```bash
python3 scripts/construct_using_validated.py --batch batch1 --demo demo_0 --sim --structure-z-offset 0.004
```

**Replay a Pre-planned Sequence (Fast):**
```bash
python3 scripts/construct_using_validated.py --replay demo_0 --sim --speed-replay 2.0
```

**Dry-Run for Planning Validation:**
```bash
python3 scripts/construct_using_validated.py --batch batch1 --demo demo_0 --structure-z-offset 0.004
```

#### Adjustable Parameters
- `--batch`: Switch between data batches (e.g., `batch0`, `batch1`).
- `--structure-z-offset`: Vertical offset (meters) to apply to the entire target structure (e.g. `--structure-z-offset 0.01`).
- `--hover-z`: Additional Z height for pre/post grasp hover in metres (default: `0.12`).
- `--supply-xyz`: Override the fixed supply brick position as `X,Y,Z` (default: `-0.20,0.40,0.030`).
- `--speed-sim`: Velocity scaling for planning in simulation (default `0.5`).
- `--speed-real`: Velocity scaling for real robot execution (default `0.13`).
- `--speed-replay`: Multiplier for execution speed during replay (e.g., `1.5`).
- `--grasp-id`: Force a specific grasp (e.g. `grasp1`). Default tries all available grasps.

---

### 3b. Perception-Integrated Construction (`construct_validated_perception.py`)

Operator-guided construction where the robot stops between each brick and uses the RealSense camera to detect the supply brick's exact pose before planning. Supports three execution modes.

#### Execution Modes

| Mode | Flag | Description |
| :--- | :--- | :--- |
| **Dry-Run** | (default) | Plans trajectories using perceived pose, reports pass/fail. No motion. |
| **Real** | `--real` | Perceives supply pose, plans with MoveIt (collision/joint checking against the planning scene), executes on the real robot via EGM. |
| **Hybrid** | `--hybrid` | Perceives supply pose, plans each brick in **Gazebo simulation** (full collision/joint checking with physics), then immediately executes the validated trajectory on the real robot via EGM. |
| **Sim** | `--sim` | Plans and executes entirely in Gazebo. Exports trajectories for later replay. |
| **Replay** | `--replay` | Loads a pre-planned `planned_sequence.json` and replays it on the real robot. |

#### Per-Brick Workflow (Real / Hybrid modes)

1. **Home** — robot moves to `SAFE_HOME` between every brick.
2. **Prompt** — operator places the next brick in the supply area and presses Enter.
3. **Detect** — `perception_simple.py` captures a frame, runs ArUco-based perspective correction + k-means colour segmentation + `minAreaRect`, and writes `supply.json` with the detected pose.
4. **Plan** — MoveIt plans all 7 trajectory phases (hover → grasp → lift → transit → place → retract → home) using the detected pose. In `--hybrid`, planning runs inside Gazebo for full physics-backed collision checking.
5. **Confirm** — plan summary (grasp id, supply xyz, goal xyz) is shown; operator types `y / skip / abort`.
6. **Execute** — robot executes the validated trajectory. In `--hybrid`, the sim-planned trajectory is sent to the real robot via EGM with the gripper j6 mounting offset applied automatically.
7. **Register** — the placed brick is added to the MoveIt planning scene so subsequent bricks plan around it.

#### Commands

**Hybrid mode — perceive + plan in sim + execute on real (recommended for best collision checking):**
```bash
python3 scripts/construct_validated_perception.py --demo demo_0 --hybrid
```

**Real mode — perceive + plan with MoveIt scene + execute on real:**
```bash
python3 scripts/construct_validated_perception.py --demo demo_0 --real
```

**Dry-run with camera (plan only, no motion):**
```bash
python3 scripts/construct_validated_perception.py --demo demo_0
```

**Sim mode — plan in Gazebo + export trajectories for later replay:**
```bash
python3 scripts/construct_validated_perception.py --demo demo_0 --sim
```

**Replay pre-planned sim trajectories on real robot:**
```bash
python3 scripts/construct_validated_perception.py --replay demo_0 --real
```

**Skip camera — use a fixed supply pose instead:**
```bash
python3 scripts/construct_validated_perception.py --demo demo_0 --real \
    --skip-perception --supply-xyz -0.20,0.40,0.030
```

#### Parameters

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--demo` | `demo_0` | Demo name in `validated_simPhysics/` |
| `--batch` | `batch1` | Batch folder inside `training_data/` |
| `--real` | off | Execute on the physical robot (MoveIt planning scene only) |
| `--hybrid` | off | Plan in Gazebo simulation, execute on real robot via EGM |
| `--sim` | off | Plan and execute entirely in Gazebo |
| `--replay` | off | Replay a pre-planned JSON sequence (accepts file path or demo name) |
| `--skip-perception` | off | Use `--supply-xyz` instead of camera detection |
| `--supply-xyz` | `-0.20,0.40,0.030` | Fallback supply XYZ when `--skip-perception` is set |
| `--hover-z` | `0.12` | Hover height above pick/place poses (metres) |
| `--grasp-id` | auto | Force a specific grasp variant |
| `--structure-z-offset` | `0.0` | Vertical offset applied to all goal poses (metres) |
| `--speed-sim` | `0.5` | Velocity scaling for simulation planning (used as planning speed in `--hybrid`) |
| `--speed-real` | `0.13` | Velocity scaling for real robot EGM execution |
| `--speed-replay` | `1.0` | Execution speed multiplier for `--replay` mode |
| `--no-export` | off | Disable automatic trajectory export after a successful sim run |
| `--export-dir` | auto | Override the export directory for planned trajectories |

> **`--hybrid` vs `--real`:** Both modes run full MoveIt collision and joint-limit checking. `--hybrid` additionally runs planning inside Gazebo with physics-settled collision geometry and provides visual confirmation before real execution — use it when collision checking fidelity matters. `--real` is faster (no Gazebo overhead) and suitable when the planning scene is trusted to be accurate.

---

## 4. Camera Perception (`perception_simple.py`)

Detects a flat brick's pose in the robot base frame from a single RealSense colour frame.

### Pipeline

1. **Capture** — calls `RealSenseCaptureServer` to grab a `color.png` from the shared directory.
2. **ArUco detection** — finds 4 × DICT_6X6 markers (IDs 0–3) that define the working-area corners.
3. **Perspective correction** — maps the 4 marker outer corners to a `10 × 7.5 inch` rectified image at 96 ppi.
4. **K-means segmentation** — clusters the corrected image into 5 colour groups; selects the cluster closest to the target brick RGB.
5. **Contour selection** — picks the contour whose area is closest to `1 × 2 in²` (expected flat brick footprint).
6. **`minAreaRect`** — fits a rotated rectangle to give pixel-space centre `(col, row)` and yaw angle.
7. **Coordinate mapping** — converts pixel coordinates to robot-frame XY using the known ArUco corner positions (from `aruco_info.py`), with Z hardcoded to `0.030 m` (flat brick on table-top, matching `REAL_SUPPLY_Z`).
8. **Quaternion** — orientation is encoded as a pure Z-axis rotation: `quaternion_from_euler(0, 0, angle_rad)`.
9. **Export** — writes `supply.json` to `SHARED_DIR`:

```json
{
  "supply_xyz": [x, y, 0.030],
  "supply_quat_xyzw": [qx, qy, qz, qw]
}
```

### ArUco Marker Layout (`aruco_info.py`)

Markers define a rectangular working area in the robot base frame (metres):

| Marker ID | Role | Corner 0 (TL) approx. |
| :--- | :--- | :--- |
| 0 | Image origin (pixel 0, 0) | `[-0.068, 0.271, 0.021]` |
| 1 | Bottom-right of image | `[-0.233, 0.271, 0.021]` |
| 2 | Top-right of image | `[-0.233, 0.500, 0.021]` |
| 3 | Top-left of image | `[-0.068, 0.500, 0.021]` |

The pixel → robot-frame conversion derives scale factors from the marker span:
```
x_m_per_px = (marker3.x - marker0.x) / (width_in × ppi)
y_m_per_px = (marker1.y - marker0.y) / (height_in × ppi)
```

---

## 5. Data & Sequence Storage

Sequences are stored hierarchically in `training_data/`:

1.  **Validated Poses (5D)**: `training_data/<batch>/validated_simPhysics/<demo>/5d_sequence/sequence.json`
    - Compact `[x, y, z, b, r]` format: XY position, Z height, binary state (0=laying, 1=standing), yaw rotation.
2.  **Validated Poses (7D)**: `training_data/<batch>/validated_simPhysics/<demo>/7d_sequence/sequence.json`
    - Contains the 7D poses (XYZ + Quat) that have passed the physics stability check.
3.  **Planned Trajectories**: `training_data/<batch>/validated_simPhysics_robot/<demo>/planned_sequence.json`
    - Contains the full joint-space trajectories generated by the MoveIt planner. These files are used by `--replay`.
4.  **Trained Models**: `training_data/trained_models/`
    - `best_model.pth` — standard SST checkpoint
    - `best_model_z_refactored.pth` — legacy rich-feature, support-relative MDN checkpoint
    - `best_model_cog_aware.pth` — COG-aware staged model checkpoint (current)

---

## 6. ML Training Pipeline

A Small Transformer predicts the next brick pose from construction history. Two notebooks are provided:

### Standard Model (`training_data/training_SST.ipynb`)

Predicts full 5D next-brick pose `[x, y, z, b, r]` from an encoded history of placed bricks.

- **Input**: 8-dim feature per brick `[x, y, z, b, sin_r, cos_r, layer_id_norm, time_norm]`
- **Architecture**: 2-layer pre-norm Transformer encoder with CLS token pooling → binary head (b) + MDN (K=5 Gaussian mixtures over 5D pose)
- **Augmentation**: 50× SE(2) augmentation per training pair (random rotation + translation)
- **Training data**: `batch2` and `batch3` validated sequences; sequence-level 75/15/15% train/val/test split
- **Outputs**: `training_data/trained_models/best_model.pth`

```bash
# Run all cells in Jupyter
jupyter notebook training_data/training_SST.ipynb
```

### COG-Aware Staged Model (`training_data/training_SST_z_refactoring.ipynb`)

Makes a staged structural decision rather than predicting global (x, y) directly. All training labels are SE(2)-invariant, so augmentation only transforms `history_raw`.

- **Input**: 17-dim feature per brick, computed causally from history:
    - Base pose (7): `x_n, y_n, b, sin_r, cos_r, layer_norm, time_norm`
    - Layer state (3): `rel_to_top, is_top, is_second_top`
    - Same-layer occupancy (5): `count_norm, ndx, ndy, ndist, is_frontier`
    - Geometry (1): `crit_span_norm` (critical-point half-span / std_xy)
    - Support context (1): `num_support_layer_norm`
- **Architecture**: 2-layer input projection → 2-layer pre-norm Transformer encoder with CLS token pooling + per-token embeddings →
    - `b_head` (binary: laying / standing)
    - `layer_head` (discrete layer class)
    - `ss_head` (support state: 0=ground / 1=one-support / 2=two-support)
    - `s1_score_head` (per-token score → argmax over support-layer tokens → s1 brick)
    - `s2_score_head` (per-token score conditioned on s1 → s2 brick)
    - `proj_head` → `[alpha_A, perp_A, alpha_B, perp_B]` (critical-point projections onto support bricks)
- **Pose recovery**: projections decoded geometrically to world `(x, y, r)` via critical-line inversion
- **Z snapping**: z looked up from `{layer_id → mean_z}` table built from training data
- **Augmentation**: 50× SE(2) augmentation on `history_raw` only; all support labels reused unchanged
- **Inference**: 3-pass — predict b/layer/ss → select s1/s2 → predict projection mean + Gaussian noise
- **Outputs**: `training_data/trained_models/best_model_cog_aware.pth`

```bash
jupyter notebook training_data/training_SST_z_refactoring.ipynb
```

---

## 7. Model Evaluation in Simulation

Two evaluation scripts run a trained model as a live construction agent in Gazebo. The model proposes brick poses one at a time; each candidate is validated for reachability (MoveIt IK) and structural stability (physics) before being accepted.

PyTorch is pre-installed in the Docker image (CPU-only build).

### Standard Evaluator (`scripts/model_evaluation.py`)

Uses the standard SST checkpoint (`best_model.pth`).

```bash
# Inside the docker container
python3 scripts/model_evaluation.py \
    --checkpoint training_data/trained_models/best_model.pth \
    --max-bricks 30 --n-candidates 100 \
    --output-dir training_data/model_generated
```

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--checkpoint` | `training_data/trained_models/best_model.pth` | Path to trained model checkpoint |
| `--max-bricks` | `30` | Maximum bricks to place |
| `--n-candidates` | `100` | MDN samples per brick per round |
| `--max-rounds` | `5` | Re-sample attempts before giving up on a brick |
| `--no-reachability-check` | off | Skip MoveIt IK check (faster, sim-only) |
| `--output-dir` | `training_data/model_generated` | Directory for output JSON sequences |
| `--batch-dirs` | `batch2,batch3` | Training batches used to build the z-level grid |

### COG-Aware Evaluator (`scripts/model_evaluation_rich_feature.py`)

Uses the COG-aware staged checkpoint (`best_model_cog_aware.pth`) with 17-dim features and critical-point projection output. Requires a seed layer (layer-0 bricks from a validated sequence) to provide initial support context.

```bash
# Inside the docker container
python3 scripts/model_evaluation_rich_feature.py \
    --checkpoint training_data/trained_models/best_model_cog_aware.pth \
    --max-bricks 30 --n-candidates 100 \
    --seed-sequence training_data/batch2/validated_simPhysics/demo_1/5d_sequence/sequence.json \
    --output-dir training_data/model_generated_cog
```

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--checkpoint` | `training_data/trained_models/best_model_cog_aware.pth` | Path to trained model checkpoint |
| `--max-bricks` | `30` | Maximum bricks to place |
| `--n-candidates` | `100` | Projection samples per brick per round |
| `--max-rounds` | `5` | Re-sample attempts before giving up on a brick |
| `--no-reachability-check` | off | Skip MoveIt IK check (faster, sim-only) |
| `--output-dir` | `training_data/model_generated_cog` | Directory for output JSON sequences |
| `--seed-sequence` | `batch2/.../demo_1/5d_sequence/sequence.json` | 5D sequence to use as the fixed seed layer |

### Evaluation Loop (both scripts)

1. Seed bricks from `--seed-sequence` (layer 0) are placed without sampling
2. **COG-aware evaluator** uses 3-pass staged inference per brick:
   - Pass 1: predict `b`, `layer_id`, `support_state` from scene encoding
   - Pass 2: score each history token → argmax → select `s1`, `s2` support bricks
   - Pass 3: predict projection mean → add Gaussian noise → decode `(x, y, r)` from critical-line projections
3. For each candidate:
   - Z is snapped from `z_lookup[layer_id]`
   - **Reachability check**: MoveIt IK is tested across grasp/flip variants (skip with `--no-reachability-check`)
   - Brick is spawned in Gazebo and allowed to settle
   - **Stability check**: existing structure is checked for displacement > 1 cm
   - If valid: brick is accepted, history is updated, model moves to the next brick
   - If invalid: brick is removed and the next candidate is tried
4. If all candidates fail, a new sample batch is drawn (up to `--max-rounds` rounds)
5. Accepted sequences are exported as 5D and 7D JSON on completion

---

## 8. Kinematic Principles

1.  **Phased Planning**: Pick-and-place is divided into `hover_supply`, `grasp_supply`, `lift_supply`, `hover_goal`, `place_goal`, `retract_goal`, and `return_home`.
2.  **Wrist Locking**: Joint 4 and 6 are locked during vertical plunges and lifts (`lock_wrist=True`) to ensure stability.
3.  **Grasp Fallbacks**: If a preferred grasp fails due to collision, the orchestrator automatically iterates through all 3 grasp orientations and 4 brick pose flips.
4.  **Safe Home Baseline**: The robot returns to a consistent `SAFE_HOME` configuration between every brick to ensure deterministic planning starting points.
5.  **Gripper j6 Offset**: The physical gripper is mounted +45° around joint 6 relative to the URDF model. Trajectories planned in simulation are automatically corrected by −π/4 rad on joint 6 before being sent to the real robot. In `--real` mode this is handled by the `gripper_tcp_calibrated` frame; in `--hybrid` and `--replay --real` modes it is applied in software per-trajectory.

---

## 9. Dynamic CAD & Physics Integration

The pipeline features deep synchronization between design and simulation:

1.  **Dynamic Grasp Loading**: Grasping poses are parsed at runtime directly from `src/grasping_poses/grasping_poses.3dm` using `rhino3dm`. This enables rapid iteration of robot toolpath geometry without code changes.
2.  **Physics-Aware Planning**: In `--sim` and `--hybrid` modes, supply bricks are spawned and allowed to settle under gravity. The orchestrator queries the **exact 7D resting pose** from Gazebo to anchor the pick-up trajectory, ensuring high-fidelity alignment despite simulation settlement or jitter.
3.  **Linear Cartesian Lead-ins**: All pick-and-place approaches and retracts use **Linear Cartesian path planning** to ensure vertical plunges and consistent brick seating.
4.  **Automatic Synchronization**: The TCP grasp offset is dynamically calculated relative to the `brick_pose` point extracted from the Rhino file, ensuring planning truth matches CAD truth.
5.  **Static Structure Consolidation** (sim mode): After each brick is placed, all previously placed bricks are merged into a single static Gazebo model (`placed_structure`). This keeps physics simulation performance constant regardless of brick count.
