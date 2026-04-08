from typing import Optional

import rclpy
import numpy as np
import sys
import tty
import termios
from rclpy.node import Node
from rclpy.action import ActionClient

from builtin_interfaces.msg import Duration
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

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
)

from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import PoseStamped
from control_msgs.action import FollowJointTrajectory, ParallelGripperCommand


class PlanAndExecuteClient(Node):
    """
    Extends the user's plan_kinematic_path client with:
      - arm planning to pose constraints
      - gripper planning to joint targets
      - MoveIt trajectory execution
      - direct controller execution for arm/gripper
    """

    def __init__(self):
        super().__init__("plan_and_execute_client")

        # MoveIt planning service
        self.plan_cli = self.create_client(GetMotionPlan, "/plan_kinematic_path")
        while not self.plan_cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().info("Waiting for /plan_kinematic_path service...")
        self.get_logger().info("/plan_kinematic_path service available.")

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
            self.get_logger().error(
                f"Planning failed. MoveItErrorCodes.val = {mres.error_code.val}"
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
            mpr.start_state = self._make_start_state(
                start_joint_names, start_joint_positions
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
    
# Function

brick_og = [0.000, 0.420, 0.032]
brick_offsetX = -0.02 # how far from the center to do it
# it is supposed to be 1 cm from the left edge 
brick_deltaX = 0.06
brick_deltaY = -0.06

structure_positions = [] # array containing the brick origin within the structure
strcuture_quaternions = [] # array continaing the brick quats within the struct

pause_offsetZ = 0.1 # how much to offset before moving down with gripper 
strcuture_quaternions = np.array([
[-8.09525e-12 , 0.00000e+00 , 1.00000e+00 , 0.00000e+00],
[0.70710678 , 0.    ,     0.70710678, 0.        ],
[-8.09525e-12 , 0.00000e+00 , 1.00000e+00 , 0.00000e+00],
[0.70710678 ,0.   ,      0.70710678, 0.        ],
[5.6025e-12, 0.0000e+00 ,1.0000e+00 ,0.0000e+00],
[ 0.    ,     -0.41036472 , 0.91192148 , 0.        ],
[ 0.64482587, -0.29017168,  0.64482587 , 0.29017168],
[ 0.64482587, -0.29017168 , 0.64482587 , 0.29017168],
[0.     ,    0.91192148, 0.41036472 ,0.        ]
])
structure_positions = np.array([[0.350319, -0.262939, 0.0305],
                                [0.355875, -0.262939, 0.075744],
                                [0.319363, -0.262939, 0.075744],
                                [0.350319, -0.262939, 0.095588],
                                [0.346041, -0.253434, 0.109875],
                                [0.349726, -0.250732, 0.155119],
                                [0.325511, -0.278059, 0.155119],
                                [0.329196, -0.272444, 0.174963]])
# for each brick in the sequence (steps 0-7): 

    # 1. move to [supply_position + z offset]
    # 2. move down to [supply_position]
    # 3. grip 
    # 4. move to [supply_position + z offset]
    # 5. move to [structure_position + z offset]
    # 6. move down to [structure_position] 
    # 7. ungrip
    # 8. move to [structure_position + z offset] 

def sequence(node):
    for step in range(8):
        # position and quaternion of the supply brick
        p_supply = brick_grab_pos(step)
        q_supply = [0,1,0,0] # TODO whatever the default grabbing orientiation is
        p_supply_above = [p_supply[0], p_supply[1], p_supply[2] + pause_offsetZ]

        #position and quaternion of the structure brick
        p_structure = structure_positions[step]
        q_structure = strcuture_quaternions[step]
        p_structure_above = [p_structure[0], p_structure[1], p_structure[2] + pause_offsetZ]

        # 1. move to [supply_position + z offset]
        move(p_supply_above, q_supply,node) 
        # 2. move down to [supply_position]
        move(p_supply, q_supply,node)
        # 3. grip 
        grip(True,node)
        # 4. move to [supply_position + z offset]
        move(p_supply_above, q_supply,node) 
        # 5. move to [structure_position + z offset]
        move(p_structure_above, q_structure,node)
        # 6. move to [structure_position]
        move(p_structure, q_structure,node)
        # 7. ungrip
        grip(False,node)
        # 8. move to [structure_position + z offset]
        move(p_structure_above, q_structure,node)

        # and do it again for all the other bricks 

    return True

def move(position, quaternion, node):
    # Move to the position with the quaternion
    # the actual moveit code or whatever
    # TODO: move to location. and throw false if it cant?
    
    arm_traj = node.plan_arm_to_pose_constraints(
        group_name="arm",
        link_name="gripper_tcp",
        frame_id="world",
        goal_xyz=tuple(position),
        goal_quat_wxyz=tuple(quaternion),
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

def brick_grab_pos(step: int):
    
    # step: which step in sequence we are in 

    # brick value must be between 0 and 19
    if step < 0:
        return None
    if step > 19:
        return None
    # given brick values 0-19:
    X = step % 4 # 4 rows
    Y = step // 5 # 5 columns
    next_brick_X = X * brick_deltaX + brick_og[0]
    next_brick_Y = Y * brick_deltaY + brick_og[1]
    next_brick_Z = brick_og[2]  # Z positions are same
    return [next_brick_X, next_brick_Y, next_brick_Z]

# Quartenions 


def main():
    rclpy.init()
    node = PlanAndExecuteClient()

    sequence(node)
    
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
