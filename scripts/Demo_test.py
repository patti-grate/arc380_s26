"""
Trajectory planning + grasp optimization module.

Takes in a set of structure positions and quaternions, then for each brick:
  1. Generates 3 grasp candidates for the supply brick (straight down, tilt from X-, tilt from X+)
  2. Tries each candidate via MoveIt in scored order — first valid one is used
  3. Executes pick from supply, place to structure position
  4. Adds placed brick to MoveIt planning scene so future plans avoid it

Input:
  structure_positions   — np.array (N, 3) of [x, y, z] placement poses
  structure_quaternions — np.array (N, 4) of [x, y, z, w] placement orientations

"""

from typing import Optional

import rclpy
import numpy as np
from rclpy.node import Node
from rclpy.action import ActionClient

from sensor_msgs.msg import JointState

from moveit_msgs.srv import GetMotionPlan
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.msg import (
    MotionPlanRequest,
    Constraints,
    PositionConstraint,
    OrientationConstraint,
    BoundingVolume,
    RobotState,
    RobotTrajectory,
    JointConstraint,
    CollisionObject,
)

from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose, PoseStamped
from control_msgs.action import FollowJointTrajectory, ParallelGripperCommand


# ---------------------------------------------------------------------------
# Global config
# ---------------------------------------------------------------------------

# Brick dimensions in metres — match your model.sdf
BRICK_SIZE_XYZ = (0.051, 0.023, 0.014)

# How far above a pose to hover before moving down (supply side)
PAUSE_OFFSET_Z = 0.1
# How far above the structure position to retreat after placing — larger than
# supply hover to clear placed bricks as the stack grows
STRUCTURE_RETREAT_Z = 0.15

# Supply brick grid origin and spacing
BRICK_OG       = [0.000,  0.480, 0.032]
BRICK_DELTA_X  =  0.06
BRICK_DELTA_Y  = -0.06

# If planning fails: retry with this world-frame offset + fallback planner
PLAN_RETRY_OFFSET_WORLD_M = (-0.01, 0.0, 0.0)
PLAN_FALLBACK_PLANNER_ID  = "RRTConnect"

# Gripper positions
GRIPPER_OPEN   = 0.0
GRIPPER_CLOSED = 0.01

# Grasp optimizer constants
ROBOT_BASE_XYZ    = np.array([0.0, 0.0, 0.0])
APPROACH_TILT_DEG = 45.0
GRASP_STANDOFF_X_M = 0.02   # lateral offset for tilted approaches (tune to avoid neighbours)
GRASP_EDGE_OFFSET_M = 0.020  # X offset toward long edge for top-down standing grasps

# Joint 6 path constraint — prevents multi-revolution spin.
# Needs to be >= pi (~3.14) to allow full 180-deg wrist reorientation
# between supply grasp and structure placement quaternions.
JOINT_6_HALF_WIDTH_RAD = 3.5

JOINT_NAMES = ["joint_1","joint_2","joint_3","joint_4","joint_5","joint_6"]
JOINT_UPPER = [2.87979,  1.91986,  1.22173,  2.79253,  2.09440,  6.98132]
JOINT_LOWER = [-2.87979, -1.91986, -1.91986, -2.79253, -2.09440, -6.98132]

# ---------------------------------------------------------------------------
# Input: structure poses
# structure_positions:   (N, 3) array of [x, y, z]
# structure_quaternions: (N, 4) array of [x, y, z, w]
# ---------------------------------------------------------------------------

structure_positions = np.array([
                                [0.350319, -0.262939, 0.030500],
                                [0.355875, -0.262939, 0.075744],
                                [0.350319, -0.262939, 0.095588],
                                [0.319363, -0.262939, 0.075744],
                                [0.346041, -0.253434, 0.109875],
                                [0.349726, -0.250732, 0.155119],
                                [0.325511, -0.278059, 0.155119],
                                [0.329196, -0.272444, 0.174963]
])

structure_quaternions = np.array([
                                [-8.09525e-12 , 0.00000e+00 , 1.00000e+00 , 0.00000e+00],
                                [0.70710678 , 0.    ,     0.70710678, 0.               ],
                                [0.70710678 ,0.   ,      0.70710678, 0.                ],
                                [-8.09525e-12 , 0.00000e+00 , 1.00000e+00 , 0.00000e+00],
                                [5.6025e-12, 0.0000e+00 ,1.0000e+00 ,        0.0000e+00],
                                [ 0.    ,     -0.41036472 , 0.91192148 , 0.            ],
                                [ 0.64482587, -0.29017168,  0.64482587 ,     0.29017168],
                                [0.     ,    0.91192148, 0.41036472 ,0.                ]
], dtype=float)

# Validate shapes match
assert structure_positions.shape[0] == structure_quaternions.shape[0], \
    "structure_positions and structure_quaternions must have the same number of rows"
assert structure_quaternions.shape[1] == 4, \
    "structure_quaternions must be (N, 4)"


# ---------------------------------------------------------------------------
# Grasp optimizer helper functions for rotation matrices and quaternions
# ---------------------------------------------------------------------------

def _rot_x_180() -> np.ndarray:
    """180 deg around X: flips +Z to -Z."""
    return np.array([
        [1,  0,  0],
        [0, -1,  0],
        [0,  0, -1],
    ], dtype=float)


def _rot_x_90() -> np.ndarray:
    """90 deg around X: for standing brick roll."""
    return np.array([
        [1, 0,  0],
        [0, 0, -1],
        [0, 1,  0],
    ], dtype=float)


def _rot_y(deg: float) -> np.ndarray:
    """Rotation around Y axis by deg degrees."""
    rad = np.radians(deg)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([
        [ c, 0, s],
        [ 0, 1, 0],
        [-s, 0, c],
    ], dtype=float)


def rotation_matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    quaternion = False

    if matrix.shape == (3, 3):
        trace = np.trace(matrix)
        m00, m11, m22 = matrix[0, 0], matrix[1, 1], matrix[2, 2]

        p_squares = np.array([
        1 + trace,             
        1 + 2*m00 - trace,   
        1 + 2*m11 - trace,     
        1 + 2*m22 - trace    
        ])

        # Find the index of the largest p_square value
        i = np.argmax(p_squares)
        p_i = np.sqrt(p_squares[i])

        # Compute the quaternion components based on the largest p_square index
        if i == 0:
            w = 0.5 * p_i
            x = (matrix[2, 1] - matrix[1, 2]) / (2*p_i)
            y = (matrix[0, 2] - matrix[2, 0]) / (2*p_i)
            z = (matrix[1, 0] - matrix[0, 1]) / (2*p_i)

        elif i == 1:
            x = 0.5 * p_i
            w = (matrix[2, 1] - matrix[1, 2]) / (2*p_i)
            y = (matrix[0, 1] + matrix[1, 0]) / (2*p_i)
            z = (matrix[0, 2] + matrix[2, 0]) / (2*p_i)

        elif i == 2:
            y = 0.5 * p_i
            w = (matrix[0, 2] - matrix[2, 0]) / (2*p_i)
            x = (matrix[0, 1] + matrix[1, 0]) / (2*p_i)
            z = (matrix[1, 2] + matrix[2, 1]) / (2*p_i)

        else:  # i == 3
            z = 0.5 * p_i
            w = (matrix[1, 0] - matrix[0, 1]) / (2*p_i)
            x = (matrix[0, 2] + matrix[2, 0]) / (2*p_i)
            y = (matrix[1, 2] + matrix[2, 1]) / (2*p_i)

        quaternion = np.array([x, y, z, w])

        # Normalize the quaternion to ensure it's a unit quaternion
        quaternion = quaternion / np.linalg.norm(quaternion)

    return quaternion

def quaternion_to_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:

    rotation_matrix = False

    if quaternion.shape == (4,):
        quaternion = quaternion / np.linalg.norm(quaternion)
        x, y, z, w = quaternion

        rotation_matrix = np.array([
            [w**2 + x**2 - y**2 - z**2, 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), w**2 - x**2 + y**2 - z**2, 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), w**2 - x**2 - y**2 + z**2]
        ])

    return rotation_matrix
# ---------------------------------------------------------------------------
# Grasp optimizer 
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

    Flat — 3 candidates:
      Grasp 1: straight down         (  0 deg tilt around Y)
      Grasp 2: 45 deg lean from X-   (+45 deg around Y)
      Grasp 3: 45 deg lean from X+   (-45 deg around Y)

    Standing — 6 candidates: same 3 with 90 deg roll around X.

    All candidates:
      - Z flipped to -Z (gripper points into brick)
      - Z component of quaternion negated (TCP frame convention)
      - TCP offset from brick centre along approach direction
    """
    brick_rot  = quaternion_to_rotation_matrix(brick_quat_xyzw)
    base_down  = _rot_x_180()

    tilt_angles = [0.0, APPROACH_TILT_DEG, -APPROACH_TILT_DEG]
    # For standing bricks, only use flat tilt=0 here; the 90-deg roll candidates
    # are commented out until perception is integrated to judge stack height —
    # they require vertical clearance to swing the brick upright mid-trajectory.
    # roll_rots = [np.eye(3), _rot_x_90()] if is_standing else [np.eye(3)]
    roll_rots = [np.eye(3)]
    candidates = []

    tilt_labels = {0.0: "straight down", APPROACH_TILT_DEG: "45deg tilt from X-", -APPROACH_TILT_DEG: "45deg tilt from X+"}

    for roll_rot in roll_rots:
        for tilt_deg in tilt_angles:
            tilt_rot = _rot_y(tilt_deg)

            # Build gripper orientation: roll -> tilt -> flip Z down
            gripper_rot_local = roll_rot @ tilt_rot @ base_down
            gripper_rot_world = brick_rot @ gripper_rot_local

            grasp_quat = rotation_matrix_to_quaternion(gripper_rot_world)
            grasp_quat[2] = -grasp_quat[2]

            # TCP offset from brick centre
            tilt_rad     = np.radians(tilt_deg)
            offset_local = np.array([
                -np.sin(tilt_rad) * GRASP_STANDOFF_X_M,
                 0.0,
                 0.0,
            ])
            grasp_pos = brick_pos + brick_rot @ offset_local

            candidates.append((grasp_pos, grasp_quat, tilt_labels[tilt_deg]))

    if is_standing:
        # Top-edge grasps: straight down but offset toward each long edge of the
        # brick. Avoids the 45-deg tilt candidates which need vertical clearance
        # to swing the brick upright — these work near the floor or as first brick.
        down_quat = rotation_matrix_to_quaternion(brick_rot @ base_down)
        down_quat[2] = -down_quat[2]
        for sign, label in [(+1.0, "top-edge near (+X)"), (-1.0, "top-edge far (-X)")]:
            edge_offset = brick_rot @ np.array([sign * GRASP_EDGE_OFFSET_M, 0.0, 0.0])
            candidates.append((brick_pos + edge_offset, down_quat.copy(), label))

    return candidates


def score_grasp(grasp_pos, grasp_quat, brick_pos) -> float:
    dist      = float(np.linalg.norm(grasp_pos - ROBOT_BASE_XYZ))       # distance from robot base (prefer closer)
    to_b      = brick_pos - ROBOT_BASE_XYZ                              # vector from robot base to brick
    b_dir     = to_b / (np.linalg.norm(to_b) + 1e-9)                    # unit vector from robot base to brick
    overshoot = float(np.dot(grasp_pos - ROBOT_BASE_XYZ, b_dir))        # prefer grasps that are slightly beyond the brick along the approach direction (helps with collision-free planning)
    return dist + 0.5 * overshoot


def get_best_grasp(candidates, brick_pos, node=None) -> Optional[tuple]:
    """
    Returns (position, quaternion_xyzw) for the best valid grasp.

    node=None  -> offline: returns top scored candidate without IK check.
    node=...   -> live: tries each via MoveIt, returns first that succeeds.
    """
    if not candidates:
        print("[grasp] No candidates provided.")
        return None

    scored = sorted(candidates, key=lambda c: score_grasp(c[0], c[1], brick_pos))

    if node is None:
        pos, quat, label = scored[0]
        print(f"[grasp] Selected: {label}")
        return pos, quat

    for i, (pos, quat, label) in enumerate(scored):
        print(f"[grasp] Trying candidate {i+1}/{len(scored)}: {label}")
        traj = node.plan_arm_to_pose_constraints(
            group_name="arm",
            link_name="gripper_tcp",
            frame_id="world",
            goal_xyz=tuple(pos),
            goal_quat_xyzw=tuple(quat),
        )
        if traj is not None:
            print(f"[grasp] Selected: {label}")
            return pos, quat
        print(f"[grasp] Candidate {i+1} failed, trying next")

    print("[grasp] All candidates failed")
    return None


# ---------------------------------------------------------------------------
# ROS / MoveIt client
# ---------------------------------------------------------------------------

class PlanAndExecuteClient(Node):

    def __init__(self):
        super().__init__("plan_and_execute_client")

        self.plan_cli = self.create_client(GetMotionPlan, "/plan_kinematic_path")
        while not self.plan_cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().info("Waiting for /plan_kinematic_path ...")
        self.get_logger().info("/plan_kinematic_path ready.")

        self.execute_moveit_ac = ActionClient(self, ExecuteTrajectory, "/execute_trajectory")
        self.arm_traj_ac       = ActionClient(self, FollowJointTrajectory,
                                              "/arm_controller/follow_joint_trajectory")
        self.gripper_cmd_ac    = ActionClient(self, ParallelGripperCommand,
                                              "/gripper_controller/gripper_cmd")
        self._collision_pub    = self.create_publisher(CollisionObject, "/collision_object", 10)
        self._joint_state      = None
        self.create_subscription(JointState, "/joint_states",
                                 lambda msg: setattr(self, "_joint_state", msg), 10)

    def get_all_joint_positions(self) -> dict:
        """Return current positions for all arm joints as {name: position}."""
        for _ in range(50):
            rclpy.spin_once(self, timeout_sec=0.05)
            if self._joint_state is not None:
                break
        if self._joint_state is None:
            self.get_logger().warn("No /joint_states received; all joints defaulting to 0.0")
            return {name: 0.0 for name in JOINT_NAMES}
        result = {}
        for name in JOINT_NAMES:
            try:
                idx = self._joint_state.name.index(name)
                result[name] = float(self._joint_state.position[idx])
            except ValueError:
                result[name] = 0.0
        return result

    # ------------------------------------------------------------------
    # Planning scene
    # ------------------------------------------------------------------

    def publish_scene_box(self, object_id, frame_id, size_xyz, position_xyz, quat_xyzw=(0,0,0,1)):
        """Add a box collision object to the MoveIt planning scene."""
        co = CollisionObject()
        co.header.stamp    = self.get_clock().now().to_msg()
        co.header.frame_id = frame_id
        co.id              = object_id
        co.operation       = CollisionObject.ADD

        box            = SolidPrimitive()
        box.type       = SolidPrimitive.BOX
        box.dimensions = [float(v) for v in size_xyz]

        pose = Pose()
        pose.position.x    = float(position_xyz[0])
        pose.position.y    = float(position_xyz[1])
        pose.position.z    = float(position_xyz[2])
        qx, qy, qz, qw    = quat_xyzw
        pose.orientation.x = float(qx)
        pose.orientation.y = float(qy)
        pose.orientation.z = float(qz)
        pose.orientation.w = float(qw)

        co.primitives      = [box]
        co.primitive_poses = [pose]

        # Wait for subscriber before publishing
        for _ in range(50):
            if self._collision_pub.get_subscription_count() > 0:
                break
            rclpy.spin_once(self, timeout_sec=0.1)

        self._collision_pub.publish(co)
        self.get_logger().info(f"Scene box added: {object_id}")

    # ------------------------------------------------------------------
    # Constraint builders
    # ------------------------------------------------------------------

    @staticmethod
    def _make_start_state(joint_names, joint_positions) -> RobotState:
        rs = RobotState()
        js = JointState()
        js.name     = list(joint_names)
        js.position = list(joint_positions)
        rs.joint_state = js
        return rs

    @staticmethod
    def _make_position_constraint(link_name, frame_id, target_xyz,
                                   tolerance_xyz=(0.001, 0.001, 0.001),
                                   weight=1.0) -> PositionConstraint:
        pc              = PositionConstraint()
        pc.header.frame_id = frame_id
        pc.link_name    = link_name
        pc.weight       = float(weight)

        box            = SolidPrimitive()
        box.type       = SolidPrimitive.BOX
        box.dimensions = [float(2.0 * t) for t in tolerance_xyz]

        bv = BoundingVolume()
        bv.primitives = [box]

        pose = PoseStamped()
        pose.header.frame_id   = frame_id
        pose.pose.position.x   = float(target_xyz[0])
        pose.pose.position.y   = float(target_xyz[1])
        pose.pose.position.z   = float(target_xyz[2])
        pose.pose.orientation.w = 1.0
        bv.primitive_poses = [pose.pose]

        pc.constraint_region = bv
        return pc

    @staticmethod
    def _make_orientation_constraint(link_name, frame_id, target_quat_xyzw,
                                      tolerance_rpy=(0.001, 0.001, 0.001),
                                      weight=1.0) -> OrientationConstraint:
        oc              = OrientationConstraint()
        oc.header.frame_id = frame_id
        oc.link_name    = link_name
        oc.weight       = float(weight)

        qx, qy, qz, qw   = target_quat_xyzw
        oc.orientation.x  = float(qx)
        oc.orientation.y  = float(qy)
        oc.orientation.z  = float(qz)
        oc.orientation.w  = float(qw)

        oc.absolute_x_axis_tolerance = float(tolerance_rpy[0])
        oc.absolute_y_axis_tolerance = float(tolerance_rpy[1])
        oc.absolute_z_axis_tolerance = float(tolerance_rpy[2])
        return oc

    @staticmethod
    def _make_joint_constraint(joint_name, position,
                                tolerance_above=1e-3, tolerance_below=1e-3,
                                weight=1.0) -> JointConstraint:
        jc                = JointConstraint()
        jc.joint_name     = joint_name
        jc.position       = float(position)
        jc.tolerance_above = float(tolerance_above)
        jc.tolerance_below = float(tolerance_below)
        jc.weight         = float(weight)
        return jc

    # ------------------------------------------------------------------
    # Core planning
    # ------------------------------------------------------------------

    def _call_motion_plan(self, mpr: MotionPlanRequest) -> Optional[RobotTrajectory]:
        req = GetMotionPlan.Request()
        req.motion_plan_request = mpr

        future = self.plan_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is None:
            self.get_logger().error("Motion planning service call failed.")
            return None

        mres = future.result().motion_plan_response
        if mres.error_code.val != mres.error_code.SUCCESS:
            self.get_logger().error(f"Planning failed. Code={mres.error_code.val}")
            return None

        jt = mres.trajectory.joint_trajectory
        self.get_logger().info(f"Plan OK — {len(jt.points)} points for {list(jt.joint_names)}")
        return mres.trajectory

    def plan_arm_to_pose_constraints(
        self,
        group_name: str,
        link_name: str,
        frame_id: str,
        goal_xyz: tuple,
        goal_quat_xyzw: tuple = (0.0, 0.0, 0.0, 1.0),
        start_joint_names=None,
        start_joint_positions=None,
        pos_tolerance_xyz: tuple = (0.001, 0.001, 0.001),
        ori_tolerance_rpy: tuple = (0.001, 0.001, 0.001),
        allowed_planning_time: float = 5.0,
        num_attempts: int = 5,
        max_velocity_scaling: float = 0.2,
        max_acceleration_scaling: float = 0.2,
        planner_id: str = "",
    ) -> Optional[RobotTrajectory]:

        mpr            = MotionPlanRequest()
        mpr.group_name = group_name
        if planner_id:
            mpr.planner_id = planner_id

        if start_joint_names and start_joint_positions:
            mpr.start_state = self._make_start_state(
                start_joint_names, 
                start_joint_positions)

        # Goal: position + orientation
        goal_c = Constraints()
        goal_c.position_constraints = [
            self._make_position_constraint(
                link_name, 
                frame_id, 
                goal_xyz, 
                pos_tolerance_xyz)
        ]
        goal_c.orientation_constraints = [
            self._make_orientation_constraint
            (link_name, 
             frame_id, 
             goal_quat_xyzw, 
             ori_tolerance_rpy)
        ]
        mpr.goal_constraints = [goal_c]

        # Path: constrain joint_6 around its current value to prevent multi-revolution spin
        j6_now = self.get_all_joint_positions().get("joint_6", 0.0)
        path_c = Constraints()
        path_c.joint_constraints = [
            self._make_joint_constraint(
                joint_name="joint_6",
                position=j6_now,
                tolerance_above=JOINT_6_HALF_WIDTH_RAD,
                tolerance_below=JOINT_6_HALF_WIDTH_RAD,
            )
        ]
        mpr.path_constraints = path_c

        mpr.allowed_planning_time          = float(allowed_planning_time)
        mpr.num_planning_attempts          = int(num_attempts)
        mpr.max_velocity_scaling_factor    = float(max_velocity_scaling)
        mpr.max_acceleration_scaling_factor = float(max_acceleration_scaling)

        return self._call_motion_plan(mpr)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute_moveit_trajectory(self, traj: RobotTrajectory) -> bool:
        if not self.execute_moveit_ac.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("/execute_trajectory not available.")
            return False

        goal = ExecuteTrajectory.Goal()
        goal.trajectory = traj

        send_future = self.execute_moveit_ac.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future)
        goal_handle = send_future.result()

        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error("Execution goal rejected.")
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result()

        if result is None:
            self.get_logger().error("No execution result.")
            return False

        if result.result.error_code.val != result.result.error_code.SUCCESS:
            self.get_logger().error(f"Execution failed. Code={result.result.error_code.val}")
            return False

        self.get_logger().info("Execution succeeded.")
        return True

    def send_gripper_command(self, position, max_velocity=0.05,
                              max_effort=0.0, joint_name="left_finger_joint") -> bool:
        if not self.gripper_cmd_ac.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Gripper action not available.")
            return False

        goal = ParallelGripperCommand.Goal()
        goal.command.name     = [joint_name]
        goal.command.position = [float(position)]
        if max_velocity > 0.0:
            goal.command.velocity = [float(max_velocity)]
        if max_effort > 0.0:
            goal.command.effort = [float(max_effort)]

        send_future = self.gripper_cmd_ac.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future)
        goal_handle = send_future.result()

        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error("Gripper goal rejected.")
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        self.get_logger().info("Gripper command done.")
        return True


# ---------------------------------------------------------------------------
# High-level motion helpers
# ---------------------------------------------------------------------------

def move(position, quat_xyzw, node, enable_retry=True) -> bool:
    """
    Plan and execute gripper_tcp to (position, quat_xyzw) in world frame.
    Retry 1: small position offset + RRTConnect (same tight tolerances).
    Retry 2: original position, looser orientation tolerance, more time.
    """
    pos = np.asarray(position, dtype=float).ravel()

    traj = node.plan_arm_to_pose_constraints(
        group_name="arm",
        link_name="gripper_tcp",
        frame_id="world",
        goal_xyz=tuple(pos),
        goal_quat_xyzw=tuple(quat_xyzw),
    )

    if traj is None and enable_retry:
        off  = np.asarray(PLAN_RETRY_OFFSET_WORLD_M, dtype=float)
        pos2 = pos + off
        node.get_logger().warn(f"Planning failed — retrying with offset {off}")
        traj = node.plan_arm_to_pose_constraints(
            group_name="arm",
            link_name="gripper_tcp",
            frame_id="world",
            goal_xyz=tuple(pos2),
            goal_quat_xyzw=tuple(quat_xyzw),
            planner_id=PLAN_FALLBACK_PLANNER_ID,
        )

    if traj is None and enable_retry:
        node.get_logger().warn("Planning failed — retrying with relaxed orientation tolerance")
        traj = node.plan_arm_to_pose_constraints(
            group_name="arm",
            link_name="gripper_tcp",
            frame_id="world",
            goal_xyz=tuple(pos),
            goal_quat_xyzw=tuple(quat_xyzw),
            ori_tolerance_rpy=(0.05, 0.05, 0.05),
            allowed_planning_time=10.0,
            num_attempts=10,
            planner_id=PLAN_FALLBACK_PLANNER_ID,
        )

    if traj is not None:
        return node.execute_moveit_trajectory(traj)

    node.get_logger().error("Planning failed after all retries.")
    return False


def grip(state: bool, node) -> bool:
    """True = close gripper, False = open gripper."""
    pos = GRIPPER_CLOSED if state else GRIPPER_OPEN
    return node.send_gripper_command(position=pos, max_velocity=0.05)


def brick_grab_pos(step: int) -> list:
    """Return world position of supply brick at grid index step."""
    if step < 0:
        return None
    if step > 19:
        return None
    # given brick values 0-19:
    X = step % 5 # 4 rows
    Y = step // 4 # 5 columns
    next_brick_X = X * BRICK_DELTA_X + BRICK_OG[0]
    next_brick_Y = Y * BRICK_DELTA_Y + BRICK_OG[1]
    next_brick_Z = BRICK_OG[2]
    return [next_brick_X, next_brick_Y, next_brick_Z]
  


# ---------------------------------------------------------------------------
# Main sequence
# ---------------------------------------------------------------------------

def _is_flat_brick(q_xyzw: np.ndarray, threshold: float = 0.9) -> bool:
    """True if the brick's +Z axis points mostly upward — i.e. the brick is lying flat."""
    rot = quaternion_to_rotation_matrix(q_xyzw)
    return float(rot[2, 2]) > threshold


def placement_tcp_quat(structure_quat_xyzw: np.ndarray) -> np.ndarray:
    """
    Derive gripper TCP orientation for placing a brick whose body frame is
    described by structure_quat_xyzw. Mirrors the grasp optimizer: approach
    along the brick's +Z axis with the gripper pointing down (_rot_x_180).
    """
    brick_rot = quaternion_to_rotation_matrix(structure_quat_xyzw)
    tcp_rot   = brick_rot @ _rot_x_180()
    tcp_quat  = rotation_matrix_to_quaternion(tcp_rot)
    tcp_quat[2] = -tcp_quat[2]   # TCP frame convention (same as grasp optimizer)
    return tcp_quat


def sequence(node):
    """
    For each brick in structure_positions / structure_quaternions:
      - Run grasp optimization on supply brick (3 candidates, best valid wins)
      - Pick from supply, place at structure position
      - Add placed brick to MoveIt planning scene
    """
    for step in range(len(structure_positions)):

        p_structure    = structure_positions[step]
        q_structure    = structure_quaternions[step]          # brick body [x, y, z, w]
        q_place_tcp    = placement_tcp_quat(q_structure) if _is_flat_brick(q_structure) else q_structure
        p_struct_above = [p_structure[0], p_structure[1],
                          p_structure[2] + STRUCTURE_RETREAT_Z]

        # --- Grasp optimization for supply brick ---
        p_supply_raw  = brick_grab_pos(step)
        supply_pos    = np.array(p_supply_raw)
        supply_quat   = np.array([0.0, 0.0, 0.0, 1.0])     # flat, no rotation

        placing_standing = not _is_flat_brick(q_structure)
        candidates    = generate_grasp_candidates(supply_pos, supply_quat, is_standing=placing_standing)
        result        = get_best_grasp(candidates, brick_pos=supply_pos, node=None)

        if result is None:
            node.get_logger().error(f"Step {step}: no valid supply grasp — skipping.")
            continue

        p_supply, q_supply = result                          # both in [x,y,z,w]
        p_supply_above = [p_supply[0], p_supply[1], p_supply[2] + PAUSE_OFFSET_Z]

        node.get_logger().info(f"Step {step}: using grasp pos={np.round(p_supply,4)} "
                               f"quat={np.round(q_supply,4)}")

        # 1. Move above supply
        move(p_supply_above, q_supply, node)
        # 2. Move down to supply
        move(p_supply, q_supply, node)
        # 3. Grip
        grip(True, node)
        # 4. Move back above supply
        move(p_supply_above, q_supply, node)
        # 5. Move above structure position
        move(p_struct_above, q_place_tcp, node)
        # 6. Move down to structure position
        move(p_structure, q_place_tcp, node)
        # 7. Ungrip
        grip(False, node)
        # 8. Register placed brick in scene BEFORE retreating so MoveIt plans around it
        node.publish_scene_box(
            object_id=f"placed_brick_{step}",
            frame_id="world",
            size_xyz=BRICK_SIZE_XYZ,
            position_xyz=tuple(p_structure),
            quat_xyzw=tuple(q_structure),
        )
        # 9. Retreat above structure — brick is now in scene so path routes around it
        move(p_struct_above, q_place_tcp, node)

    return True


def main():
    rclpy.init()
    node = PlanAndExecuteClient()
    sequence(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()