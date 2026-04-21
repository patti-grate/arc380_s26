# Construction Orchestrator & Motion Planning Architecture

This document serves as a knowledge base for other agents and developers working on the brick-stacking kinematics and MoveIt trajectory validation integration.

## Core Principles

1. **Phased Trajectory Assembly**: The pick-and-place operation is divided into rigid chronological phases (`hover_supply`, `grasp_supply`, `lift_supply`, `hover_goal`, `place_goal`, `retract_goal`, `return_home`). 
2. **Deterministic Start Anchoring**: Trajectories MUST natively read the exact fractional `/joint_states` of the physically rested robot to seed `points[0]` bounds, otherwise MoveIt's `/execute_trajectory` strict validation throws `-4 (INVALID_TRAJECTORY)`.
3. **Wrist Locking During Linear Action**: Joint 4 and 6 (the twisting axes) are locked to their start configurations during downward plunges and upward lifts (`lock_wrist=True`). Joint 5 (wrist pitch) is explicitly left free to swing, as the elbow requires orthogonal pitch freedom to maintain a strictly vertical TCP vector.

## Key Files
- `scripts/construct_using_validated.py`: The main orchestrator. It controls execution mode (`dry_run`, `sim`, `real`), iterates over target arrays, dynamically generates fallbacks, tracking `current_start_state` explicitly between sequence phases.
- `scripts/trajectory_planner_draft_JG.py`: The low-level OMPL interface. Handles `AttachedCollisionObject` detaching, `MotionPlanRequest` bounding boxes, and IK tolerance injections.

## Fallback Generation Engine
When dense placement clusters block a standard IK angle, `generate_fallback_poses()` iterates through safe geometric permutations to un-choke OMPL:
- Target (Goal) frames are rotated (`X: 180, Z: 180`)
- Supply pick orientation frames are flipped (`sz: 180`) allowing the robot to "grab from the opposite side" while maintaining standard pick angles.

## The SAFE_HOME Baseline
Between each brick placement sequence, the robot mathematically and physically routes to the `SAFE_HOME_POSITIONS` natively pitched slightly higher than the initial Gazebo span: `[1.57, 0.30, -0.18, 0.0, 0.94, -1.57]`.

- **Why?** It ensures every fallback permutation roots from an identical geometric baseline, rather than the chaotic, variable end-state of the previous brick.
- **Why not natural Gazebo spawn?** The static spawn frame mathematically clips the floor inside the `table_surface` bounding box, causing instantaneous Goal Collision rejections (MoveIt Error `99999`) across OMPL paths. 
- **Tolerance Safety**: Returning the 6D arm to `SAFE_HOME` utilizes a loose algorithmic tolerance (`0.08` radians) during `init_traj` to skirt Random Sampling timeouts. When planning the very next Brick, the `current_start_state` is dropped to `None` so the mathematical timeline maps to the exact fractional resting footprint, negating Phase-Anchor mismatch (`-4`) bounds.
