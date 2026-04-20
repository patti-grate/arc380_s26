from typing import Optional

import rclpy
import numpy as np
from scipy.spatial.transform import Rotation as R
import sys
import time
import tty
import termios
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from builtin_interfaces.msg import Duration
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from moveit_msgs.srv import GetMotionPlan, GetPlanningScene
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
    AttachedCollisionObject,
    PlanningSceneComponents,
)

from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose, PoseStamped
from control_msgs.action import FollowJointTrajectory, ParallelGripperCommand
from std_srvs.srv import Trigger

# MoveIt's planning scene subscriber often uses transient-local durability; matching
# avoids silent QoS mismatch and helps late-joining publishers.
_COLLISION_OBJECT_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
)


class PlanAndExecuteClient(Node):
    """
    Extends the user's plan_kinematic_path client with:
      - arm planning to pose constraints
      - gripper planning to joint targets
      - MoveIt trajectory execution
      - direct controller execution for arm/gripper
    """

    def __init__(self):
        super().__init__("trajectory_planner_draft")

        # MoveIt planning service
        self.plan_cli = self.create_client(GetMotionPlan, "/plan_kinematic_path")
        while not self.plan_cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().info("Waiting for /plan_kinematic_path service...")
        self.get_logger().info("/plan_kinematic_path service available.")

        self._get_planning_scene_cli = self.create_client(
            GetPlanningScene, "/get_planning_scene"
        )

        # abb_irb120_gazebo simulation_reset_node (gz_moveit.launch.py)
        self._reset_simulation_cli = self.create_client(Trigger, "/reset_simulation")

        # MoveIt execution action
        self.execute_moveit_ac = ActionClient(
            self, ExecuteTrajectory, "/execute_trajectory"
        )

        # ros2_control trajectory actions
        self.arm_traj_ac = ActionClient(
            self,
            FollowJointTrajectory,
            "/arm_controller/follow_joint_trajectory",
        )
        self.gripper_cmd_ac = ActionClient(
            self,
            ParallelGripperCommand,
            "/gripper_controller/gripper_cmd",
        )

        # MoveIt planning_scene_monitor subscribes here (default name; check with
        # `ros2 topic list | grep collision` if your launch uses a namespace).
        self._collision_object_pub = self.create_publisher(
            CollisionObject, "/collision_object", _COLLISION_OBJECT_QOS
        )
        self._attached_object_pub = self.create_publisher(
            AttachedCollisionObject, "/attached_collision_object", _COLLISION_OBJECT_QOS
        )

    def attach_box_to_gripper(
        self,
        object_id: str,
        size_xyz: tuple[float, float, float],
        link_name: str = "gripper_tcp",
        touch_links: list[str] | None = None
    ) -> None:
        """
        Attaches a generated collision box to the specified link (default: gripper_tcp).
        MoveIt will then avoid collisions between this box and the scene.
        (Called BEFORE planning the transit phases.)
        """
        aco = AttachedCollisionObject()
        aco.link_name = link_name
        aco.object.header.frame_id = link_name
        aco.object.id = object_id
        aco.object.operation = CollisionObject.ADD

        # Box centered at the TCP, adjust position if the offset isn't strictly 0
        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [float(size_xyz[0]), float(size_xyz[1]), float(size_xyz[2])]

        prim_pose = Pose()
        prim_pose.orientation.w = 1.0  # Identity against tcp

        aco.object.primitives = [box]
        aco.object.primitive_poses = [prim_pose]
        
        if touch_links is None:
            # Let it ignore collisions with the gripper fingers
            aco.touch_links = [
                "gripper_tcp", "link_6", "link_5", 
                "gripper_base", "left_finger", "right_finger"
            ]
        else:
            aco.touch_links = touch_links

        for _ in range(5):
            self._attached_object_pub.publish(aco)
            time.sleep(0.02)
            
    def detach_box_from_gripper(self, object_id: str) -> None:
        """
        Detaches and removes the object.
        """
        aco = AttachedCollisionObject()
        aco.object.id = object_id
        aco.object.operation = CollisionObject.REMOVE
        for _ in range(5):
            self._attached_object_pub.publish(aco)
            time.sleep(0.02)

    def publish_scene_box(
        self,
        object_id: str,
        frame_id: str,
        size_xyz: tuple[float, float, float],
        position_xyz: tuple[float, float, float],
        quat_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    ) -> None:
        """
        Add a box to the MoveIt planning scene. Use the same frame_id as your
        motion goals (this script uses \"world\" in move()).

        ``quat_xyzw`` is (x, y, z, w) matching geometry_msgs/Quaternion.

        MoveIt expects ``CollisionObject.pose`` (object frame in ``header``) to be a
        valid transform; ``primitive_poses`` are relative to that frame. Putting the
        full box pose on ``co.pose`` and an identity primitive pose avoids an
        all-zero default quaternion on ``co.pose`` (which RViz / MoveIt may drop).
        """
        co = CollisionObject()
        co.header.stamp = self.get_clock().now().to_msg()
        co.header.frame_id = frame_id
        co.id = object_id

        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [float(size_xyz[0]), float(size_xyz[1]), float(size_xyz[2])]

        pos = np.asarray(position_xyz, dtype=float).ravel()
        if pos.size != 3:
            raise ValueError(f"position_xyz must have 3 components; got shape {pos.shape}.")

        q = np.asarray(quat_xyzw, dtype=float).ravel()
        if q.size != 4:
            raise ValueError(f"quat_xyzw must have 4 components; got {q.size}.")
        qx, qy, qz, qw = float(q[0]), float(q[1]), float(q[2]), float(q[3])

        # Object pose in header frame (box center + orientation).
        co.pose.position.x = float(pos[0])
        co.pose.position.y = float(pos[1])
        co.pose.position.z = float(pos[2])
        co.pose.orientation.x = float(qx)
        co.pose.orientation.y = float(qy)
        co.pose.orientation.z = float(qz)
        co.pose.orientation.w = float(qw)

        # One box primitive centered on the object frame origin.
        prim_pose = Pose()
        prim_pose.orientation.w = 1.0

        co.primitives = [box]
        co.primitive_poses = [prim_pose]
        co.operation = CollisionObject.ADD

        # Wait for move_group / planning_scene_monitor to subscribe (discovery can be slow).
        for _ in range(300):
            if self._collision_object_pub.get_subscription_count() > 0:
                break
            rclpy.spin_once(self, timeout_sec=0.1)
        subs = self._collision_object_pub.get_subscription_count()
        if subs == 0:
            self.get_logger().warn(
                "No subscribers on /collision_object yet — publishing anyway; "
                "ensure move_group is running (e.g. gz_moveit.launch.py)."
            )

        # Repeat publish + spin so the message is not lost to DDS timing / graph churn.
        for k in range(25):
            co.header.stamp = self.get_clock().now().to_msg()
            self._collision_object_pub.publish(co)
            for _ in range(3):
                rclpy.spin_once(self, timeout_sec=0.02)
            time.sleep(0.04)

        self.get_logger().info(
            f"CollisionObject ADD id={object_id!r} frame={frame_id!r} "
            f"on /collision_object (subscribers={self._collision_object_pub.get_subscription_count()})"
        )

    def remove_scene_object(self, object_id: str, frame_id: str = "world") -> None:
        """
        Remove a planning-scene object by id so a later run does not leave the robot
        in collision with geometry published on a previous run (e.g. placed brick box).
        """
        co = CollisionObject()
        co.header.frame_id = frame_id
        co.id = object_id
        co.operation = CollisionObject.REMOVE

        for _ in range(300):
            if self._collision_object_pub.get_subscription_count() > 0:
                break
            rclpy.spin_once(self, timeout_sec=0.1)

        for k in range(25):
            co.header.stamp = self.get_clock().now().to_msg()
            self._collision_object_pub.publish(co)
            for _ in range(3):
                rclpy.spin_once(self, timeout_sec=0.02)
            time.sleep(0.04)

        self.get_logger().info(
            f"CollisionObject REMOVE id={object_id!r} frame={frame_id!r} "
            f"(subscribers={self._collision_object_pub.get_subscription_count()})"
        )

    def remove_all_world_collision_objects(self, service_timeout_sec: float = 5.0) -> int:
        """
        Query MoveIt for world ``collision_objects``, then publish REMOVE for each id.
        Does not clear octomap or attached bodies (only ``PlanningSceneWorld.collision_objects``).
        """
        if not self._get_planning_scene_cli.wait_for_service(timeout_sec=service_timeout_sec):
            self.get_logger().warn(
                "/get_planning_scene not available; skipping world collision-object clear."
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
            f"Cleared {len(id_to_frame)} world collision object(s) (GetPlanningScene + REMOVE)."
        )
        return len(id_to_frame)

    def reset_gazebo_simulation(
        self,
        service_wait_sec: float = 15.0,
        reset_callback_timeout_sec: float = 300.0,
    ) -> bool:
        """
        Full Gazebo + ros2_control reset via ``/reset_simulation`` (same as
        ``ros2 service call /reset_simulation std_srvs/srv/Trigger {}``).

        Requires ``simulation_reset_node`` from ``gz_moveit.launch.py``. The
        service callback can take tens of seconds (unload controllers, reset
        world, respawn robot, respawn controllers).
        """
        if not self._reset_simulation_cli.wait_for_service(timeout_sec=service_wait_sec):
            self.get_logger().warn(
                f"/reset_simulation not available within {service_wait_sec}s "
                "(start gz_moveit with simulation_reset_node, or disable reset)."
            )
            return False

        fut = self._reset_simulation_cli.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, fut, timeout_sec=reset_callback_timeout_sec)
        if fut.result() is None:
            self.get_logger().error(
                "Reset service call did not complete "
                f"(timeout {reset_callback_timeout_sec}s or executor error)."
            )
            return False

        resp = fut.result()
        if resp.success:
            self.get_logger().info(f"Gazebo reset: {resp.message}")
            return True

        self.get_logger().error(f"Gazebo reset failed: {resp.message}")
        return False

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
            extra = ""
            if getattr(mres.error_code, "message", ""):
                extra = f" message={mres.error_code.message!r}"
            if getattr(mres.error_code, "source", ""):
                extra += f" source={mres.error_code.source!r}"
            self.get_logger().error(
                f"Planning failed. MoveItErrorCodes.val = {mres.error_code.val}{extra}"
            )
            return None

        traj = mres.trajectory
        jt = traj.joint_trajectory
        self.get_logger().info(
            f"Planning succeeded. JointTrajectory has {len(jt.points)} points "
            f"for joints: {list(jt.joint_names)}"
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
        max_velocity_scaling: float = 0.2,
        max_acceleration_scaling: float = 0.2,
        planner_id: str = "",
        joint_4_constraints: float | None = None,
        joint_6_constraints: float | None = None,
        lock_wrist_to_start: bool = False,
    ) -> Optional[RobotTrajectory]:
        mpr = MotionPlanRequest()
        mpr.group_name = group_name
        if planner_id:
            mpr.planner_id = planner_id

        if start_joint_names is not None and start_joint_positions is not None:
            mpr.start_state = self._make_start_state(
                start_joint_names, start_joint_positions
            )

        if joint_4_constraints is not None or joint_6_constraints is not None:
            path_c = Constraints()
            if joint_4_constraints is not None:
                w4 = float(joint_4_constraints)
                path_c.joint_constraints.append(
                    self._make_joint_constraint(
                        joint_name="joint_4",
                        position=0.0,
                        tolerance_above=w4,
                        tolerance_below=w4,
                    )
                )
            if joint_6_constraints is not None:
                w6 = float(joint_6_constraints)
                path_c.joint_constraints.append(
                    self._make_joint_constraint(
                        joint_name="joint_6",
                        position=0.0,
                        tolerance_above=w6,
                        tolerance_below=w6,
                    )
                )
            mpr.path_constraints = path_c

        if lock_wrist_to_start and start_joint_names is not None and start_joint_positions is not None:
            path_c = mpr.path_constraints
            if not path_c.joint_constraints:  # if empty
                mpr.path_constraints = Constraints()
                path_c = mpr.path_constraints
            
            for jname in ["joint_4", "joint_5", "joint_6"]:
                if jname in start_joint_names:
                    idx = start_joint_names.index(jname)
                    pos = start_joint_positions[idx]
                    path_c.joint_constraints.append(
                        self._make_joint_constraint(
                            joint_name=jname,
                            position=pos,
                            tolerance_above=0.4,
                            tolerance_below=0.4,
                        )
                    )

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
        mpr.goal_constraints = [constraints]

        mpr.allowed_planning_time = float(allowed_planning_time)
        mpr.num_planning_attempts = int(num_attempts)
        mpr.max_velocity_scaling_factor = float(max_velocity_scaling)
        mpr.max_acceleration_scaling_factor = float(max_acceleration_scaling)

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
        max_velocity_scaling: float = 1.0,
        max_acceleration_scaling: float = 1.0,
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
            mpr.start_state = self._make_start_state(
                start_joint_names, start_joint_positions
            )

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
        mpr.max_velocity_scaling_factor = float(max_velocity_scaling)
        mpr.max_acceleration_scaling_factor = float(max_acceleration_scaling)

        return self._call_motion_plan(mpr)

    def execute_moveit_trajectory(self, traj: RobotTrajectory) -> bool:
        """
        Executes a RobotTrajectory through MoveIt's /execute_trajectory action.
        This is the simplest path when MoveIt is already configured with both
        arm_controller and gripper_controller.
        """
        if not self.execute_moveit_ac.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("/execute_trajectory action not available.")
            return False

        goal = ExecuteTrajectory.Goal()
        goal.trajectory = traj

        send_future = self.execute_moveit_ac.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future)
        goal_handle = send_future.result()

        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error("MoveIt execution goal rejected.")
            return False

        self.get_logger().info("MoveIt execution goal accepted.")

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result()

        if result is None:
            self.get_logger().error("Failed to get MoveIt execution result.")
            return False

        error_code = result.result.error_code.val
        if error_code != result.result.error_code.SUCCESS:
            self.get_logger().error(
                f"MoveIt execution failed. MoveItErrorCodes.val = {error_code}"
            )
            return False

        self.get_logger().info("MoveIt execution succeeded.")
        return True

    def _send_follow_joint_trajectory(
        self,
        action_client: ActionClient,
        joint_names: list[str],
        positions: list[float],
        duration_sec: float = 2.0,
    ) -> bool:
        if len(joint_names) != len(positions):
            self.get_logger().error("joint_names and positions length mismatch.")
            return False

        if not action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("FollowJointTrajectory action server not available.")
            return False

        traj = JointTrajectory()
        traj.joint_names = list(joint_names)

        point = JointTrajectoryPoint()
        point.positions = list(positions)
        point.time_from_start = Duration(sec=int(duration_sec), nanosec=int((duration_sec % 1.0) * 1e9))
        traj.points = [point]

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj

        send_future = action_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future)
        goal_handle = send_future.result()

        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error("Controller execution goal rejected.")
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result()

        if result is None:
            self.get_logger().error("Failed to get controller execution result.")
            return False

        if result.result.error_code != 0:
            self.get_logger().error(
                f"Controller execution failed with error_code={result.result.error_code}"
            )
            return False

        self.get_logger().info("Controller execution succeeded.")
        return True

    def send_arm_trajectory(
        self,
        joint_names: list[str],
        positions: list[float],
        duration_sec: float = 3.0,
    ) -> bool:
        return self._send_follow_joint_trajectory(
            self.arm_traj_ac, joint_names, positions, duration_sec
        )

    def send_gripper_command(
        self,
        position: float,
        max_velocity: float = 0.02,
        max_effort: float = 0.0,
        joint_name: str = "left_finger_joint",
    ) -> bool:
        if not self.gripper_cmd_ac.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("ParallelGripperCommand action server not available.")
            return False

        goal = ParallelGripperCommand.Goal()
        goal.command.name = [joint_name]
        goal.command.position = [float(position)]

        if max_velocity > 0.0:
            goal.command.velocity = [float(max_velocity)]

        if max_effort > 0.0:
            goal.command.effort = [float(max_effort)]

        send_future = self.gripper_cmd_ac.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future)
        goal_handle = send_future.result()

        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error("Gripper command goal rejected.")
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result()

        if result is None:
            self.get_logger().error("Failed to get gripper command result.")
            return False

        self.get_logger().info("Gripper command completed.")
        self.get_logger().info(
            f"stalled={result.result.stalled}, reached_goal={result.result.reached_goal}"
        )
        return True

def grip(state,node):
    # TODO: gripper activate. and throw false if it cant?
    gripper_open = 0.0
    gripper_closed = 0.01
    if state:
        # TODO grip
        node.send_gripper_command(
        position=gripper_closed,
        max_velocity=0.05,
    ) 
    else:
        # TODO ungrip 
        node.send_gripper_command(
        position=gripper_open,
        max_velocity=0.05,
    ) 

    return True

# Some helper functions!!!!!!

def place_obstacles(node):
    for i in range(len(goal_pos)):
        node.publish_scene_box(
            object_id=f"obstacle_{i}",
            frame_id="world",
            size_xyz=BRICK_SIZE_XYZ,
            position_xyz=goal_pos[i],
            quat_xyzw=goal_quat[i])

def move(position, quaternion, node):
    # Move to the position with the quaternion
    # the actual moveit code or whatever
    # TODO: move to location. and throw false if it cant?
    
    arm_traj = node.plan_arm_to_pose_constraints(
        group_name="arm",
        link_name="gripper_tcp",
        frame_id="world",
        goal_xyz=tuple(position),
        goal_quat_xyzw=tuple(quaternion),
        joint_6_constraints=4.0
    )
    if arm_traj is not None:
        node.execute_moveit_trajectory(arm_traj)

    return True

def grip(state,node):
    # TODO: gripper activate. and throw false if it cant?
    gripper_open = 0.0
    gripper_closed = 0.01
    if state:
        # TODO grip
        node.send_gripper_command(
        position=gripper_closed,
        max_velocity=0.05,
    ) 
    else:
        # TODO ungrip 
        node.send_gripper_command(
        position=gripper_open,
        max_velocity=0.05,
    ) 

    return True 

# inputs: 
#   - Supply brick location + orientation
#   - Goal brick location + orientation 
# Outputs: 
#   - Trajectory to lead in, grasp brick, lead out, lead in to goal location, place brick 
#  

BRICK_SIZE_XYZ = (0.051, 0.023, 0.014)# just test the obstacle placement 

goal_pos = np.array([[0.35, -0.26, 0.04],
                         [0.35, -0.26, 0.075]])

# quats are in xyzw format 
goal_quat = np.array([[0,1,0, 0],
                      [ 0, 0, 0.3826834, 0.9238796 ]])

def plan_brick_placement(
    node,
    supply_xyz,
    supply_quat_xyzw,
    goal_xyz,
    goal_quat_xyzw,
    *,
    reset_simulation: bool = True,
):
    # Parameter names avoid shadowing module-level ``goal_pos`` / ``goal_quat`` arrays.
    s = np.asarray(supply_xyz, dtype=float).ravel()
    g = np.asarray(goal_xyz, dtype=float).ravel()
    qg = np.asarray(goal_quat_xyzw, dtype=float).ravel()
    if s.size != 3 or g.size != 3:
        raise ValueError("supply_xyz and goal_xyz must each have 3 components.")
    if qg.size != 4:
        raise ValueError("goal_quat_xyzw must have 4 components (x,y,z,w).")

    if reset_simulation:
        node.get_logger().info("Resetting Gazebo (via /reset_simulation)…")
        if not node.reset_gazebo_simulation():
            node.get_logger().error("Aborting plan_brick_placement after failed simulation reset.")
            return

    # Drop all user-added world objects from a previous run (ids unknown).
    node.remove_all_world_collision_objects()

    # 1. lead into supply
    node.get_logger().info("1. supply")
    move(supply_xyz, supply_quat_xyzw, node)
    # 2. grasp brick
    node.get_logger().info("2. grasp brick")
    grip(True, node)
    # 3. lead out  
    node.get_logger().info("3. lead out")
    move([float(s[0]), float(s[1]), float(s[2] + 0.1)], supply_quat_xyzw, node)
    # 4. lead in to goal location
    node.get_logger().info("4. lead in to goal location")
    move(goal_xyz, goal_quat_xyzw, node)
    # 5. place brick
    node.get_logger().info("5. place brick")
    grip(False, node)

    node.publish_scene_box(
        object_id="obstacle",
        frame_id="world",
        size_xyz=BRICK_SIZE_XYZ,
        position_xyz=(float(g[0]), float(g[1]), float(g[2])),
        quat_xyzw=(float(qg[0]), float(qg[1]), float(qg[2]), float(qg[3])),
    )

def main():
    rclpy.init()
    node = PlanAndExecuteClient()
    # brick_og = [0.000, 0.480, 0.032]
    brick_og = [0.000, 0.480, 0.030]

    plan_brick_placement(node, brick_og, [0,1,0,0], 
                               [0.35, -0.26, 0.075], [0,1,0,0])

    # Keep the node alive briefly so discovery + transient-local delivery can finish.
    for _ in range(80):
        rclpy.spin_once(node, timeout_sec=0.05)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
