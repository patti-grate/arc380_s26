#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from moveit_msgs.srv import GetMotionPlan
from moveit_msgs.msg import (
    MotionPlanRequest,
    Constraints,
    PositionConstraint,
    OrientationConstraint,
    BoundingVolume,
    RobotState,
)
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState


class PlanKinematicPathClient(Node):
    """
    Calls MoveIt's /plan_kinematic_path service (moveit_msgs/srv/GetMotionPlan)
    to get a plan (RobotTrajectory) from a start state to goal constraints.
    """

    def __init__(self):
        super().__init__("plan_kinematic_path_client")

        self.cli = self.create_client(GetMotionPlan, "/plan_kinematic_path")
        while not self.cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().info("Waiting for /plan_kinematic_path service...")
        self.get_logger().info("/plan_kinematic_path service available.")

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
        """
        Creates a box-shaped PositionConstraint centered at target_xyz with given tolerances.
        """
        pc = PositionConstraint()
        pc.header.frame_id = frame_id
        pc.link_name = link_name
        pc.weight = float(weight)

        # AABB-style tolerance box around the target point
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
        pose.pose.orientation.w = 1.0  # identity; not used for position constraint
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

    def plan_to_pose_constraints(
        self,
        group_name: str,
        link_name: str,
        frame_id: str,
        start_joint_names: list[str],
        start_joint_positions: list[float],
        goal_xyz: tuple[float, float, float],
        goal_quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
        pos_tolerance_xyz: tuple[float, float, float] = (0.005, 0.005, 0.005),
        ori_tolerance_rpy: tuple[float, float, float] = (0.1, 0.1, 0.1),
        allowed_planning_time: float = 5.0,
        num_attempts: int = 5,
        max_velocity_scaling: float = 0.2,
        max_acceleration_scaling: float = 0.2,
        planner_id: str = "",
    ):
        req = GetMotionPlan.Request()
        mpr = MotionPlanRequest()

        mpr.group_name = group_name
        if planner_id:
            mpr.planner_id = planner_id

        # Start state
        mpr.start_state = self._make_start_state(start_joint_names, start_joint_positions)

        # Goal constraints (position + orientation for the given link)
        constraints = Constraints()
        constraints.position_constraints = [
            self._make_position_constraint(
                link_name=link_name,
                frame_id=frame_id,
                target_xyz=goal_xyz,
                tolerance_xyz=pos_tolerance_xyz,
                weight=1.0,
            )
        ]
        constraints.orientation_constraints = [
            self._make_orientation_constraint(
                link_name=link_name,
                frame_id=frame_id,
                target_quat_wxyz=goal_quat_wxyz,
                tolerance_rpy=ori_tolerance_rpy,
                weight=1.0,
            )
        ]
        mpr.goal_constraints = [constraints]

        # Planning settings
        mpr.allowed_planning_time = float(allowed_planning_time)
        mpr.num_planning_attempts = int(num_attempts)
        mpr.max_velocity_scaling_factor = float(max_velocity_scaling)
        mpr.max_acceleration_scaling_factor = float(max_acceleration_scaling)

        req.motion_plan_request = mpr

        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is None:
            self.get_logger().error("Service call failed.")
            return None

        resp = future.result()
        mres = resp.motion_plan_response

        if mres.error_code.val != mres.error_code.SUCCESS:
            self.get_logger().error(f"Planning failed. MoveItErrorCodes.val = {mres.error_code.val}")
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


def main():
    rclpy.init()
    node = PlanKinematicPathClient()

    group_name = "arm"
    link_name = "tool0"
    frame_id = "world"

    start_joint_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
    start_joint_positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    goal_xyz = (0.138, 0.00, 0.4714)
    goal_quat_wxyz = (0.5102, 0.0, 0.8601, 0.0)
    # ------------------------------------------------------

    node.plan_to_pose_constraints(
        group_name=group_name,
        link_name=link_name,
        frame_id=frame_id,
        start_joint_names=start_joint_names,
        start_joint_positions=start_joint_positions,
        goal_xyz=goal_xyz,
        goal_quat_wxyz=goal_quat_wxyz,
        allowed_planning_time=5.0,
        num_attempts=5,
        max_velocity_scaling=0.2,
        max_acceleration_scaling=0.2,
        planner_id="",
    )

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()