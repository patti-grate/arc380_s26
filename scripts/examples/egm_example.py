import time
from typing import Optional

import numpy as np
import rclpy
from example_interfaces.srv import SetBool
from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.msg import (
    AttachedCollisionObject,
    BoundingVolume,
    CollisionObject,
    Constraints,
    JointConstraint,
    MotionPlanRequest,
    OrientationConstraint,
    PlanningSceneComponents,
    PositionConstraint,
    RobotState,
    RobotTrajectory,
)
from moveit_msgs.srv import GetCartesianPath, GetMotionPlan, GetPlanningScene, GetPositionIK
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive

from abb_egm_interfaces.action import ExecuteTrajectory

_COLLISION_OBJECT_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
)


class EGMClient(Node):
    def __init__(
        self,
        max_velocity_scaling: float = 0.2,
        max_acceleration_scaling: float = 0.2,
        use_sim_time: bool = False,
    ):
        super().__init__("egm_client")
        self._max_velocity_scaling = max_velocity_scaling
        self._max_acceleration_scaling = max_acceleration_scaling
        # use_sim_time=True is only correct when Gazebo is publishing /clock.
        # Real-robot deployments must leave this False (default) so system clock
        # is used and spin_until_future_complete timeouts behave correctly.
        if use_sim_time:
            self.set_parameters([Parameter("use_sim_time", value=True)])

        # MoveIt planning service
        self.plan_cli = self.create_client(GetMotionPlan, "/plan_kinematic_path")
        while not self.plan_cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().info("Waiting for /plan_kinematic_path service...")
        self.get_logger().info("/plan_kinematic_path service available.")

        # EGM execution action
        self.execute_traj_ac = ActionClient(self, ExecuteTrajectory, "/execute_trajectory")

        self.gripper_client = self.create_client(SetBool, "/egm_controller/control/set_gripper")

        # MoveIt planning scene publisher (used to register collision objects)
        self._collision_object_pub = self.create_publisher(
            CollisionObject, "/collision_object", _COLLISION_OBJECT_QOS
        )
        self._attached_object_pub = self.create_publisher(
            AttachedCollisionObject, "/attached_collision_object", _COLLISION_OBJECT_QOS
        )

        # Planning scene query service (used by remove_all_world_collision_objects)
        self._get_planning_scene_cli = self.create_client(
            GetPlanningScene, "/get_planning_scene"
        )
        # IK feasibility check service
        self._ik_cli = self.create_client(GetPositionIK, "/compute_ik")
        # Cartesian path service
        self._cartesian_cli = self.create_client(GetCartesianPath, "/compute_cartesian_path")

    @staticmethod
    def _make_start_state(joint_names, joint_positions) -> RobotState:
        rs = RobotState()
        js = JointState()
        js.name = list(joint_names)
        js.position = list(joint_positions)
        rs.joint_state = js
        return rs

    @staticmethod
    def _make_position_constraint(
        link_name: str,
        frame_id: str,
        target_xyz,
        tolerance_xyz=(0.005, 0.005, 0.005),
        weight: float = 1.0,
    ) -> PositionConstraint:
        pc = PositionConstraint()
        pc.header.frame_id = frame_id
        pc.link_name = link_name
        pc.weight = float(weight)

        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [
            float(2.0 * tolerance_xyz[0]),
            float(2.0 * tolerance_xyz[1]),
            float(2.0 * tolerance_xyz[2]),
        ]

        bv = BoundingVolume()
        bv.primitives = [box]

        pose = PoseStamped()
        pose.header.frame_id = frame_id
        pose.pose.position.x = float(target_xyz[0])
        pose.pose.position.y = float(target_xyz[1])
        pose.pose.position.z = float(target_xyz[2])
        pose.pose.orientation.w = 1.0
        bv.primitive_poses = [pose.pose]

        pc.constraint_region = bv
        return pc

    @staticmethod
    def _make_orientation_constraint(
        link_name: str,
        frame_id: str,
        target_quat_xyzw: tuple[float, float, float, float],
        tolerance_rpy=(0.05, 0.05, 0.05),
        weight: float = 1.0,
    ) -> OrientationConstraint:
        oc = OrientationConstraint()
        oc.header.frame_id = frame_id
        oc.link_name = link_name
        oc.weight = float(weight)

        qx, qy, qz, qw = target_quat_xyzw
        oc.orientation.x = float(qx)
        oc.orientation.y = float(qy)
        oc.orientation.z = float(qz)
        oc.orientation.w = float(qw)


        oc.absolute_x_axis_tolerance = float(tolerance_rpy[0])
        oc.absolute_y_axis_tolerance = float(tolerance_rpy[1])
        oc.absolute_z_axis_tolerance = float(tolerance_rpy[2])
        return oc

    @staticmethod
    def _make_joint_constraint(
        joint_name: str,
        position: float,
        tolerance_above: float = 1e-3,
        tolerance_below: float = 1e-3,
        weight: float = 1.0,
    ) -> JointConstraint:
        jc = JointConstraint()
        jc.joint_name = joint_name
        jc.position = float(position)
        jc.tolerance_above = float(tolerance_above)
        jc.tolerance_below = float(tolerance_below)
        jc.weight = float(weight)
        return jc

    def _call_motion_plan(self, mpr: MotionPlanRequest) -> Optional[RobotTrajectory]:
        req = GetMotionPlan.Request()
        req.motion_plan_request = mpr

        future = self.plan_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is None:
            self.get_logger().error("Motion planning service call failed.")
            return None

        resp = future.result()
        mres = resp.motion_plan_response

        if mres.error_code.val != mres.error_code.SUCCESS:
            self.get_logger().error(f"Planning failed. MoveItErrorCodes.val = {mres.error_code.val}")
            return None

        traj = mres.trajectory
        jt = traj.joint_trajectory
        self.get_logger().info(
            f"Planning succeeded. JointTrajectory has {len(jt.points)} points for joints: {list(jt.joint_names)}"
        )
        if jt.points:
            last = jt.points[-1]
            self.get_logger().info("Last configuration:")
            for n, p in zip(jt.joint_names, last.positions):
                self.get_logger().info(f"  {n}: {p:.6f}")
        return traj

    def plan_arm_to_pose_constraints(
        self,
        group_name: str,
        link_name: str,
        frame_id: str,
        goal_xyz: tuple[float, float, float],
        start_joint_names: list[str] | None = None,
        start_joint_positions: list[float] | None = None,
        goal_quat_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
        pos_tolerance_xyz: tuple[float, float, float] = (0.001, 0.001, 0.001),
        ori_tolerance_rpy: tuple[float, float, float] = (0.001, 0.001, 0.001),
        allowed_planning_time: float = 5.0,
        num_attempts: int = 5,
        max_velocity_scaling: float | None = None,
        max_acceleration_scaling: float | None = None,
        planner_id: str = "",
        joint_1_constraints: float | None = None,
        joint_2_constraints: float | None = None,
        joint_3_constraints: float | None = None,
        joint_4_constraints: float | None = None,
        joint_5_constraints: float | None = None,
        joint_6_constraints: float | None = None,
        lock_wrist_to_start: bool = False,
        lock_wrist_tolerance: float = 0.4,
    ) -> Optional[RobotTrajectory]:
        mpr = MotionPlanRequest()

        mpr.group_name = group_name
        if planner_id:
            mpr.planner_id = planner_id

        if start_joint_names is not None and start_joint_positions is not None:
            mpr.start_state = self._make_start_state(start_joint_names, start_joint_positions)

        constraints = Constraints()
        constraints.position_constraints = [
            self._make_position_constraint(
                link_name=link_name,
                frame_id=frame_id,
                target_xyz=goal_xyz,
                tolerance_xyz=pos_tolerance_xyz,
            )
        ]
        constraints.orientation_constraints = [
            self._make_orientation_constraint(
                link_name=link_name,
                frame_id=frame_id,
                target_quat_xyzw=goal_quat_xyzw,
                tolerance_rpy=ori_tolerance_rpy,
            )
        ]

        # Add joint constraints if provided
        joint_constraints = []
        c_map = {
            "joint_1": joint_1_constraints,
            "joint_2": joint_2_constraints,
            "joint_3": joint_3_constraints,
            "joint_4": joint_4_constraints,
            "joint_5": joint_5_constraints,
            "joint_6": joint_6_constraints,
        }
        for j_name, j_val in c_map.items():
            if j_val is not None:
                joint_constraints.append(
                    self._make_joint_constraint(j_name, j_val, tolerance_above=0.1, tolerance_below=0.1)
                )
        
        if lock_wrist_to_start and start_joint_names and start_joint_positions:
            s_map = dict(zip(start_joint_names, start_joint_positions))
            for j_name in ["joint_4", "joint_5", "joint_6"]:
                if j_name in s_map:
                    joint_constraints.append(
                        self._make_joint_constraint(j_name, s_map[j_name], tolerance_above=lock_wrist_tolerance, tolerance_below=lock_wrist_tolerance)
                    )

        if joint_constraints:
            constraints.joint_constraints = joint_constraints

        mpr.goal_constraints = [constraints]


        mpr.allowed_planning_time = float(allowed_planning_time)
        mpr.num_planning_attempts = int(num_attempts)

        v_scale = max_velocity_scaling if max_velocity_scaling is not None else self._max_velocity_scaling
        a_scale = max_acceleration_scaling if max_acceleration_scaling is not None else self._max_acceleration_scaling

        mpr.max_velocity_scaling_factor = float(v_scale)
        mpr.max_acceleration_scaling_factor = float(a_scale)

        return self._call_motion_plan(mpr)

    def plan_gripper_to_joint_positions(
        self,
        group_name: str,
        goal_joint_names: list[str],
        goal_joint_positions: list[float],
        start_joint_names: list[str] | None = None,
        start_joint_positions: list[float] | None = None,
        tolerance: float = 1e-3,
        allowed_planning_time: float = 2.0,
        num_attempts: int = 3,
        max_velocity_scaling: float | None = None,
        max_acceleration_scaling: float | None = None,
        planner_id: str = "",
    ) -> Optional[RobotTrajectory]:
        """
        Plans a joint-space motion for the gripper group.
        For a non-mimic 2-finger gripper, pass both finger joints.
        Example:
          goal_joint_names = ["left_finger_joint", "right_finger_joint"]
          goal_joint_positions = [0.01, 0.01]
        """
        if len(goal_joint_names) != len(goal_joint_positions):
            self.get_logger().error("goal_joint_names and goal_joint_positions must match.")
            return None

        mpr = MotionPlanRequest()
        mpr.group_name = group_name
        if planner_id:
            mpr.planner_id = planner_id

        if start_joint_names is not None and start_joint_positions is not None:
            mpr.start_state = self._make_start_state(start_joint_names, start_joint_positions)

        constraints = Constraints()
        constraints.joint_constraints = [
            self._make_joint_constraint(
                joint_name=n,
                position=p,
                tolerance_above=tolerance,
                tolerance_below=tolerance,
            )
            for n, p in zip(goal_joint_names, goal_joint_positions)
        ]
        mpr.goal_constraints = [constraints]

        mpr.allowed_planning_time = float(allowed_planning_time)
        mpr.num_planning_attempts = int(num_attempts)

        v_scale = max_velocity_scaling if max_velocity_scaling is not None else self._max_velocity_scaling
        a_scale = max_acceleration_scaling if max_acceleration_scaling is not None else self._max_acceleration_scaling

        mpr.max_velocity_scaling_factor = float(v_scale)
        mpr.max_acceleration_scaling_factor = float(a_scale)

        return self._call_motion_plan(mpr)

    @property
    def tcp_link(self) -> str:
        return "gripper_tcp_calibrated"

    def publish_scene_box(
        self,
        object_id: str,
        frame_id: str,
        size_xyz: tuple,
        position_xyz: tuple,
        quat_xyzw: tuple = (0.0, 0.0, 0.0, 1.0),
    ) -> None:
        co = CollisionObject()
        co.header.frame_id = frame_id
        co.id = object_id

        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [float(size_xyz[0]), float(size_xyz[1]), float(size_xyz[2])]

        pos = np.asarray(position_xyz, dtype=float).ravel()
        q = np.asarray(quat_xyzw, dtype=float).ravel()

        co.pose.position.x = float(pos[0])
        co.pose.position.y = float(pos[1])
        co.pose.position.z = float(pos[2])
        co.pose.orientation.x = float(q[0])
        co.pose.orientation.y = float(q[1])
        co.pose.orientation.z = float(q[2])
        co.pose.orientation.w = float(q[3])

        prim_pose = Pose()
        prim_pose.orientation.w = 1.0
        co.primitives = [box]
        co.primitive_poses = [prim_pose]
        co.operation = CollisionObject.ADD

        for _ in range(300):
            if self._collision_object_pub.get_subscription_count() > 0:
                break
            rclpy.spin_once(self, timeout_sec=0.1)

        for _ in range(25):
            co.header.stamp = self.get_clock().now().to_msg()
            self._collision_object_pub.publish(co)
            for _ in range(3):
                rclpy.spin_once(self, timeout_sec=0.02)
            time.sleep(0.04)

    def remove_scene_object(self, object_id: str, frame_id: str = "world") -> None:
        """Remove a single planning-scene collision object by id."""
        co = CollisionObject()
        co.header.frame_id = frame_id
        co.id = object_id
        co.operation = CollisionObject.REMOVE

        for _ in range(300):
            if self._collision_object_pub.get_subscription_count() > 0:
                break
            rclpy.spin_once(self, timeout_sec=0.1)

        for _ in range(25):
            co.header.stamp = self.get_clock().now().to_msg()
            self._collision_object_pub.publish(co)
            for _ in range(3):
                rclpy.spin_once(self, timeout_sec=0.02)
            time.sleep(0.04)

        self.get_logger().info(f"CollisionObject REMOVE id={object_id!r}")

    def remove_all_world_collision_objects(
        self, service_timeout_sec: float = 5.0
    ) -> int:
        """
        Query MoveIt for all world collision objects and publish REMOVE for each.
        Returns the number of objects removed.
        """
        if not self._get_planning_scene_cli.wait_for_service(
            timeout_sec=service_timeout_sec
        ):
            self.get_logger().warn(
                "/get_planning_scene not available; skipping collision-object clear."
            )
            return 0

        req = GetPlanningScene.Request()
        req.components = PlanningSceneComponents()
        req.components.components = (
            PlanningSceneComponents.WORLD_OBJECT_NAMES
            | PlanningSceneComponents.WORLD_OBJECT_GEOMETRY
        )

        fut = self._get_planning_scene_cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut)
        if fut.result() is None:
            self.get_logger().error("GetPlanningScene service call returned no result.")
            return 0

        scene = fut.result().scene
        id_to_frame: dict[str, str] = {}
        for co in scene.world.collision_objects:
            oid = (co.id or "").strip()
            if not oid:
                continue
            fid = (co.header.frame_id or "").strip() or "world"
            id_to_frame.setdefault(oid, fid)

        for oid, fid in id_to_frame.items():
            self.remove_scene_object(oid, frame_id=fid)

        self.get_logger().info(
            f"Cleared {len(id_to_frame)} world collision object(s)."
        )
        return len(id_to_frame)

    def attach_box_to_gripper(
        self,
        object_id: str,
        size_xyz: tuple,
        link_name: str = "gripper_tcp_calibrated",
        touch_links: list | None = None,
    ) -> None:
        """Attach a collision box to the gripper TCP so MoveIt avoids it during planning."""
        aco = AttachedCollisionObject()
        aco.link_name = link_name
        aco.object.header.frame_id = link_name
        aco.object.id = object_id
        aco.object.operation = CollisionObject.ADD

        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [float(size_xyz[0]), float(size_xyz[1]), float(size_xyz[2])]

        prim_pose = Pose()
        prim_pose.orientation.w = 1.0
        aco.object.primitives = [box]
        aco.object.primitive_poses = [prim_pose]

        if touch_links is None:
            aco.touch_links = [
                "gripper_tcp", "gripper_tcp_calibrated", "link_6", "link_5",
                "gripper_base", "left_finger", "right_finger",
            ]
        else:
            aco.touch_links = touch_links

        for _ in range(5):
            self._attached_object_pub.publish(aco)
            time.sleep(0.02)

    def detach_box_from_gripper(self, object_id: str) -> None:
        """Detach and remove an attached collision object from the gripper."""
        aco = AttachedCollisionObject()
        aco.object.id = object_id
        aco.object.operation = CollisionObject.REMOVE
        for _ in range(5):
            self._attached_object_pub.publish(aco)
            time.sleep(0.02)

    def check_ik(
        self,
        group_name: str,
        link_name: str,
        frame_id: str,
        target_xyz: tuple,
        target_quat_xyzw: tuple,
        start_joint_names: list | None = None,
        start_joint_positions: list | None = None,
        timeout_sec: float = 0.5,
        avoid_collisions: bool = False,
    ) -> bool:
        """
        Fast IK feasibility check via /compute_ik.
        Returns True if a valid IK solution exists; falls back to True if the
        service is unavailable so OMPL remains the gate.
        """
        from moveit_msgs.msg import PositionIKRequest

        if not self._ik_cli.wait_for_service(timeout_sec=0.5):
            return True  # service not up -- let OMPL decide

        ikr = PositionIKRequest()
        ikr.group_name = group_name
        ikr.ik_link_name = link_name
        ikr.avoid_collisions = avoid_collisions

        ps = PoseStamped()
        ps.header.frame_id = frame_id
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x = float(target_xyz[0])
        ps.pose.position.y = float(target_xyz[1])
        ps.pose.position.z = float(target_xyz[2])
        qx, qy, qz, qw = target_quat_xyzw
        ps.pose.orientation.x = float(qx)
        ps.pose.orientation.y = float(qy)
        ps.pose.orientation.z = float(qz)
        ps.pose.orientation.w = float(qw)
        ikr.pose_stamped = ps

        if start_joint_names and start_joint_positions:
            ikr.robot_state = self._make_start_state(start_joint_names, start_joint_positions)
        else:
            zero_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
            ikr.robot_state = self._make_start_state(zero_names, [0.0] * 6)

        ikr.timeout.nanosec = int(timeout_sec * 1e9)
        req = GetPositionIK.Request()
        req.ik_request = ikr

        fut = self._ik_cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=timeout_sec + 0.5)

        if fut.result() is None:
            return True  # timeout -- optimistically pass through
        return fut.result().error_code.val == 1  # SUCCESS == 1

    def plan_cartesian_path(
        self,
        group_name: str,
        link_name: str,
        frame_id: str,
        waypoints: list,
        start_joint_names: list | None = None,
        start_joint_positions: list | None = None,
        max_step: float = 0.01,
        jump_threshold: float = 0.0,
        avoid_collisions: bool = True,
    ):
        """Plan a linear Cartesian path through a list of Pose waypoints."""
        if not self._cartesian_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().error("/compute_cartesian_path service not available.")
            return None

        from moveit_msgs.srv import GetCartesianPath as _GCP
        req = _GCP.Request()
        req.header.frame_id = frame_id
        req.header.stamp = self.get_clock().now().to_msg()
        if start_joint_names and start_joint_positions:
            req.start_state = self._make_start_state(start_joint_names, start_joint_positions)
        req.group_name = group_name
        req.link_name = link_name
        req.waypoints = waypoints
        req.max_step = max_step
        req.jump_threshold = jump_threshold
        req.avoid_collisions = avoid_collisions

        fut = self._cartesian_cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=10.0)
        if fut.result() is None:
            self.get_logger().error("Cartesian planning service call failed.")
            return None
        res = fut.result()
        if res.fraction < 1.0:
            self.get_logger().warn(f"Cartesian path only {res.fraction * 100:.1f}% complete.")
            return None
        return res.solution

    def reset_gazebo_simulation(self, **kwargs) -> bool:
        """No-op on real robot (Gazebo is sim-only). Always returns False."""
        self.get_logger().warn("reset_gazebo_simulation() called in real mode — ignored.")
        return False

    def replay_arm_trajectory(self, robot_traj: RobotTrajectory) -> bool:
        """Replay a pre-planned RobotTrajectory via EGM (same path as live execution)."""
        return self.execute_moveit_trajectory(robot_traj)

    def execute_moveit_trajectory(self, traj: RobotTrajectory) -> bool:

        if not self.execute_traj_ac.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("/execute_trajectory action not available.")
            return False

        goal = ExecuteTrajectory.Goal()
        goal.trajectory = traj.joint_trajectory
        goal.stop_active_motion = True

        send_future = self.execute_traj_ac.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future)
        goal_handle = send_future.result()

        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error("EGM execution goal rejected.")
            return False

        self.get_logger().info("EGM execution goal accepted.")

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=120.0)
        if not result_future.done():
            self.get_logger().error(
                "EGM execution timed out after 120s. "
                "Ensure the robot RAPID program has activated the EGM motion segment."
            )
            goal_handle.cancel_goal_async()
            return False
        result = result_future.result()

        if result is None:
            self.get_logger().error("Failed to get EGM execution result.")
            return False

        if not result.result.success:
            self.get_logger().error(f"EGM execution failed. Message: {result.result.message}")
            return False

        self.get_logger().info("EGM execution succeeded.")
        return True

    def send_gripper_command(
        self,
        position: float,
        max_velocity: float = 0.02,
        max_effort: float = 0.0,
        joint_name: str = "left_finger_joint",
    ) -> bool:
        if not self.gripper_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("Gripper control service not available.")
            return False

        req = SetBool.Request()
        req.data = position > 1e-6
        future = self.gripper_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is None:
            self.get_logger().error("Failed to call gripper control service.")
            return False

        response = future.result()
        if not response.success:
            self.get_logger().error(f"Gripper control failed: {response.message}")
            return False
        self.get_logger().info(f"Gripper control succeeded: {response.message}")
        time.sleep(2.0)
        return True


def main():
    rclpy.init()
    node = EGMClient()

    gripper_open = 0.00
    gripper_closed = 0.01

    node.send_gripper_command(
        position=gripper_open,
        max_velocity=0.05,
    )

    arm_traj = node.plan_arm_to_pose_constraints(
        group_name="arm",
        link_name="gripper_tcp_calibrated",
        frame_id="world",
        goal_xyz=(0.0, 0.480, 0.1),
        goal_quat_wxyz=(0.0, 1.0, 0.0, 0.0),
        max_velocity_scaling=0.5,            
    )
    if arm_traj is not None:
        node.execute_moveit_trajectory(arm_traj)

    arm_traj = node.plan_arm_to_pose_constraints(
        group_name="arm",
        link_name="gripper_tcp_calibrated",
        frame_id="world",
        goal_xyz=(0.0, 0.480, 0.032),
        goal_quat_wxyz=(0.0, 1.0, 0.0, 0.0),
        max_velocity_scaling=0.5,
    )
    if arm_traj is not None:
        node.execute_moveit_trajectory(arm_traj)

    node.send_gripper_command(
        position=gripper_closed,
        max_velocity=0.05,
    )

    arm_traj = node.plan_arm_to_pose_constraints(
        group_name="arm",
        link_name="gripper_tcp_calibrated",
        frame_id="world",
        goal_xyz=(0.0, 0.480, 0.1),
        goal_quat_wxyz=(0.0, 1.0, 0.0, 0.0),
        max_velocity_scaling=0.5,
    )
    if arm_traj is not None:
        node.execute_moveit_trajectory(arm_traj)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
