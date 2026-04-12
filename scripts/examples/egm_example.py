import time
from typing import Optional

import rclpy
from example_interfaces.srv import SetBool
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import (
    BoundingVolume,
    Constraints,
    JointConstraint,
    MotionPlanRequest,
    OrientationConstraint,
    PositionConstraint,
    RobotState,
    RobotTrajectory,
)
from moveit_msgs.srv import GetMotionPlan
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive

from abb_egm_interfaces.action import ExecuteTrajectory


class EGMClient(Node):
    def __init__(self):
        super().__init__("egm_client")
        self.set_parameters([Parameter("use_sim_time", value=True)])

        # MoveIt planning service
        self.plan_cli = self.create_client(GetMotionPlan, "/plan_kinematic_path")
        while not self.plan_cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().info("Waiting for /plan_kinematic_path service...")
        self.get_logger().info("/plan_kinematic_path service available.")

        # EGM execution action
        self.execute_traj_ac = ActionClient(self, ExecuteTrajectory, "/execute_trajectory")

        self.gripper_client = self.create_client(SetBool, "/egm_controller/control/set_gripper")

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
        target_quat_wxyz,
        tolerance_rpy=(0.05, 0.05, 0.05),
        weight: float = 1.0,
    ) -> OrientationConstraint:
        oc = OrientationConstraint()
        oc.header.frame_id = frame_id
        oc.link_name = link_name
        oc.weight = float(weight)

        oc.orientation.w = float(target_quat_wxyz[0])
        oc.orientation.x = float(target_quat_wxyz[1])
        oc.orientation.y = float(target_quat_wxyz[2])
        oc.orientation.z = float(target_quat_wxyz[3])

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
        goal_quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
        pos_tolerance_xyz: tuple[float, float, float] = (0.001, 0.001, 0.001),
        ori_tolerance_rpy: tuple[float, float, float] = (0.001, 0.001, 0.001),
        allowed_planning_time: float = 5.0,
        num_attempts: int = 5,
        max_velocity_scaling: float = 0.2,
        max_acceleration_scaling: float = 0.2,
        planner_id: str = "",
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
                target_quat_wxyz=goal_quat_wxyz,
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
        mpr.max_velocity_scaling_factor = float(max_velocity_scaling)
        mpr.max_acceleration_scaling_factor = float(max_acceleration_scaling)

        return self._call_motion_plan(mpr)

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
        rclpy.spin_until_future_complete(self, result_future)
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
