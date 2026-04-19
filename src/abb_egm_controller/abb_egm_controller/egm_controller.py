from __future__ import annotations

import math
import socket
import time
from dataclasses import dataclass, field, replace
from enum import Enum
from threading import Event, Lock, Thread

import rclpy
from abb_egm_interfaces.action import ExecuteTrajectory
from abb_egm_interfaces.msg import EgmRobot, EgmSensor, RobotJointState, RobotPoseState
from abb_egm_interfaces.srv import SetControlMode
from builtin_interfaces.msg import Duration
from example_interfaces.srv import SetBool
from rcl_interfaces.msg import SetParametersResult
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.action.server import ServerGoalHandle
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint

from abb_egm_controller import egm_pb2 as egm
from abb_egm_controller.buffer import Buffer
from abb_egm_controller.controller_state import ControllerState
from abb_egm_controller.egm_config import PARAM_SPECS, ControllerConfig, ControlSpace
from abb_egm_controller.msg_conversion import egm_robot_to_ros, ros_sensor_to_egm


@dataclass
class TrajExecutionState:
    points: list[JointTrajectoryPoint] = field(default_factory=list)
    joint_indices: list[int] = field(default_factory=list)  # trajectory joint order -> robot joint index
    start_time: float = 0.0
    current_idx: int = 0
    segment_idx: int = 0
    last_setpoint: list[float] = field(default_factory=list)
    start_positions: list[float] = field(default_factory=list)
    done_event: Event = field(default_factory=Event)
    cancelled: bool = False

    def reset(self):
        self.points.clear()
        self.joint_indices.clear()
        self.start_time = 0.0
        self.current_idx = 0
        self.segment_idx = 0
        self.last_setpoint.clear()
        self.start_positions.clear()
        self.done_event.clear()
        self.cancelled = False


class EGMController(Node):
    # region Initialization

    def __init__(self, node_name: str = "egm_controller"):
        super().__init__(node_name)

        self.egm_connected = False
        self.state = ControllerState.IDLE

        self.stream_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST, depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT
        )

        self._stream_buffer: Buffer[RobotPoseState | RobotJointState | None] = Buffer(None)

        # Trajectory execution state (shared between EGM loop and action execute callback)
        self._traj_lock = Lock()
        self._traj_state = TrajExecutionState()
        self._traj_final_goal_tolerance: float = 0.001  # radians

        self._gripper_lock = Lock()
        self._gripper_cmd = False

        # Latest robot joint feedback (written by EGM loop, read by action callback)
        self._latest_feedback_joints: list[float] = []

        self._declare_parameters()
        self.add_on_set_parameters_callback(self._on_param_set)
        self._load_config()
        self._init_topics()
        self._init_services()
        self._init_egm()

    def _declare_parameters(self):
        defaults = ControllerConfig()
        self.declare_parameters(
            namespace="",
            parameters=[
                (
                    name,
                    getattr(defaults, name).name
                    if isinstance(getattr(defaults, name), Enum)
                    else getattr(defaults, name),
                    spec.descriptor,
                )
                for name, spec in PARAM_SPECS.items()
            ],
        )

    def _init_topics(self):
        self._pub_cb_group = ReentrantCallbackGroup()
        self.feedback_pub = self.create_publisher(
            EgmRobot, f"/{self.get_name()}/feedback", 10, callback_group=self._pub_cb_group
        )
        self.joint_state_pub = self.create_publisher(JointState, "/joint_states", 10, callback_group=self._pub_cb_group)

        self._sub_cb_group = ReentrantCallbackGroup()
        self.create_subscription(
            RobotPoseState,
            f"/{self.get_name()}/control/pose_stream",
            self._stream_callback,
            self.stream_qos,
            callback_group=self._sub_cb_group,
        )
        self.create_subscription(
            RobotJointState,
            f"/{self.get_name()}/control/joints_stream",
            self._stream_callback,
            self.stream_qos,
            callback_group=self._sub_cb_group,
        )

    def _init_services(self):
        self._srv_cb_group = ReentrantCallbackGroup()
        self.create_service(
            SetControlMode,
            "set_control_mode",
            self._set_control_mode_callback,
            callback_group=self._srv_cb_group,
        )
        self.create_service(
            SetBool,
            f"/{self.get_name()}/control/set_gripper",
            self._set_gripper_callback,
            callback_group=self._srv_cb_group,
        )

        self._action_cb_group = ReentrantCallbackGroup()
        self._action_server = ActionServer(
            self,
            ExecuteTrajectory,
            "execute_trajectory",
            execute_callback=self._execute_trajectory,
            goal_callback=self._trajectory_goal_callback,
            cancel_callback=self._trajectory_cancel_callback,
            callback_group=self._action_cb_group,
        )

    def _load_config(self):
        cfg = ControllerConfig()

        for name in PARAM_SPECS.keys():
            setattr(cfg, name, self.get_parameter(name).value)

        cfg.validate()
        self.config = cfg

    def _init_egm(self):
        self.egm_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.egm_socket.bind(("0.0.0.0", self.config.udp_port))
        self.egm_socket.settimeout(1.0)

        self.egm_thread = Thread(target=self._update_loop, daemon=True)
        self.egm_thread.start()

    # endregion Initialization

    # region Motion control

    def _update_loop(self) -> None:
        addr = None

        self.get_logger().info(f"EGM controller started, listening for robot data on UDP port {self.config.udp_port}")

        while rclpy.ok():
            try:
                data, addr = self.egm_socket.recvfrom(1024)
            except socket.timeout:
                if self.egm_connected:
                    self.egm_connected = False
                    self.get_logger().warning("EGM connection lost, no data received")
                continue

            if not self.egm_connected:
                self.egm_connected = True
                self.get_logger().info(f"EGM connection established with {addr}")

            egm_robot_msg = egm.EgmRobot()
            egm_robot_msg.ParseFromString(data)
            ros_robot_msg = egm_robot_to_ros(
                egm_robot_msg,
                robot_joint_names=self.config.robot_joint_names,
                ext_joint_names=self.config.ext_joint_names,
                ext_joint_types=self.config.ext_joint_types,
            )
            self.feedback_pub.publish(ros_robot_msg)

            # TODO: refactor this. temporary for gripper state
            fb_joints_with_gripper = JointState()
            fb_joints_with_gripper.name = list(ros_robot_msg.feedback_joints.name)
            fb_joints_with_gripper.position = list(ros_robot_msg.feedback_joints.position)
            fb_joints_with_gripper.name.append("left_finger_joint")
            if ros_robot_msg.rapid_dig_val:
                fb_joints_with_gripper.position.append(0.01)  # closed
            else:
                fb_joints_with_gripper.position.append(0.0)  # open

            # self.joint_state_pub.publish(ros_robot_msg.feedback_joints)  # TODO: also publish ext joints
            self.joint_state_pub.publish(fb_joints_with_gripper)
            self._latest_feedback_joints = list(ros_robot_msg.feedback_joints.position)

            try:
                ros_sensor_msg = self._form_ros_sensor_msg(ros_robot_msg)
            except ValueError as e:
                self.get_logger().error(f"Error forming ROS sensor message: {e}")
                continue
            egm_sensor_msg = ros_sensor_to_egm(ros_sensor_msg, ext_joint_types=self.config.ext_joint_types)

            if self.config.docker_mode:
                addr = ("host.docker.internal", self.config.relay_port_out)
            self.egm_socket.sendto(egm_sensor_msg.SerializeToString(), addr)

    def _form_ros_sensor_msg(self, ros_robot_msg: EgmRobot) -> EgmSensor:
        ros_sensor_msg = EgmSensor()
        ros_sensor_msg.msg_type = EgmSensor.MSGTYPE_CORRECTION

        ros_sensor_msg.send_rapid_data = True
        with self._gripper_lock:
            ros_sensor_msg.rapid_dnum = [float(self._gripper_cmd)]  # 0.0 for open, 1.0 for close

        if self.config.control_space == ControlSpace.JOINT:
            ros_sensor_msg.mode = EgmSensor.MODE_JOINTS
        elif self.config.control_space == ControlSpace.CARTESIAN:
            ros_sensor_msg.mode = EgmSensor.MODE_CARTESIAN

        # TODO: address header timestamp
        if self.state == ControllerState.IDLE:
            if self.config.control_space == ControlSpace.JOINT:
                ros_sensor_msg.planned_joints = ros_robot_msg.feedback_joints
            if self.config.control_space == ControlSpace.CARTESIAN:
                ros_sensor_msg.planned_pose = ros_robot_msg.feedback_pose
            ros_sensor_msg.planned_ext_joints = ros_robot_msg.feedback_ext_joints
            return ros_sensor_msg

        if self.state == ControllerState.STREAMING:
            if not self._stream_buffer.is_updated():
                if self.config.control_space == ControlSpace.JOINT:
                    ros_sensor_msg.planned_joints = ros_robot_msg.feedback_joints
                if self.config.control_space == ControlSpace.CARTESIAN:
                    ros_sensor_msg.planned_pose = ros_robot_msg.feedback_pose
                ros_sensor_msg.planned_ext_joints = ros_robot_msg.feedback_ext_joints
                return ros_sensor_msg

            stream_msg = self._stream_buffer.get_value()
            if stream_msg is None:
                raise ValueError("No stream message available for STREAMING control state")

            if self.config.control_space == ControlSpace.CARTESIAN:
                if not isinstance(stream_msg, RobotPoseState):
                    raise ValueError("RobotPoseState message not available for cartesian control stream")
                ros_sensor_msg.planned_pose.header = stream_msg.header
                ros_sensor_msg.planned_pose.pose = stream_msg.pose

            if self.config.control_space == ControlSpace.JOINT:
                if not isinstance(stream_msg, RobotJointState):
                    raise ValueError("RobotJointState message not available for joint control stream")
                ros_sensor_msg.planned_joints.header = stream_msg.header
                ros_sensor_msg.planned_joints.position = stream_msg.joints

            ros_sensor_msg.planned_ext_joints.header = stream_msg.header
            ros_sensor_msg.planned_ext_joints.position = stream_msg.ext_joints
            return ros_sensor_msg

        if self.state == ControllerState.TRAJECTORY:
            if not self._traj_state.points or self._traj_state.cancelled:
                ros_sensor_msg.planned_joints = ros_robot_msg.feedback_joints
                ros_sensor_msg.planned_ext_joints = ros_robot_msg.feedback_ext_joints
                return ros_sensor_msg

            actual = ros_robot_msg.feedback_joints.position
            setpoint = list(actual)

            elapsed = time.monotonic() - self._traj_state.start_time
            final_time = (
                self._traj_state.points[-1].time_from_start.sec
                + 1e-9 * self._traj_state.points[-1].time_from_start.nanosec
            )

            # Advance cached segment index only forward
            last_seg = len(self._traj_state.points) - 1
            while (
                self._traj_state.segment_idx < last_seg
                and (
                    self._traj_state.points[self._traj_state.segment_idx].time_from_start.sec
                    + 1e-9 * self._traj_state.points[self._traj_state.segment_idx].time_from_start.nanosec
                )
                < elapsed
            ):
                self._traj_state.segment_idx += 1

            # Determine interpolation endpoints
            if self._traj_state.segment_idx == 0:
                t0 = 0.0
                p0 = self._traj_state.start_positions
                p1 = self._traj_state.points[0].positions
                t1 = (
                    self._traj_state.points[0].time_from_start.sec
                    + 1e-9 * self._traj_state.points[0].time_from_start.nanosec
                )
                current_idx = 0
            else:
                prev_pt = self._traj_state.points[self._traj_state.segment_idx - 1]
                next_pt = self._traj_state.points[self._traj_state.segment_idx]
                t0 = prev_pt.time_from_start.sec + 1e-9 * prev_pt.time_from_start.nanosec
                t1 = next_pt.time_from_start.sec + 1e-9 * next_pt.time_from_start.nanosec
                p0 = prev_pt.positions
                p1 = next_pt.positions
                current_idx = self._traj_state.segment_idx

            # Interpolate according to elapsed time
            if elapsed >= final_time:
                alpha = 1.0
            elif t1 <= t0:
                alpha = 1.0
            else:
                alpha = (elapsed - t0) / (t1 - t0)
                alpha = max(0.0, min(1.0, alpha))

            sq_final_err = 0.0
            final_positions = self._traj_state.points[-1].positions

            for i, traj_i in enumerate(self._traj_state.joint_indices):
                q0 = p0[traj_i] if self._traj_state.segment_idx == 0 else p0[i]
                q1 = p1[i]
                desired_i = q0 + alpha * (q1 - q0)
                setpoint[traj_i] = desired_i

                final_err = final_positions[i] - actual[traj_i]
                sq_final_err += final_err * final_err

            final_err_norm = math.sqrt(sq_final_err)

            with self._traj_lock:
                self._traj_state.segment_idx = self._traj_state.segment_idx
                self._traj_state.current_idx = current_idx
                self._traj_state.last_setpoint = setpoint
                if elapsed >= final_time and final_err_norm <= self._traj_final_goal_tolerance:
                    self._traj_state.done_event.set()

            ros_sensor_msg.planned_joints.header = ros_robot_msg.feedback_joints.header
            ros_sensor_msg.planned_joints.position = setpoint
            ros_sensor_msg.planned_ext_joints = ros_robot_msg.feedback_ext_joints
            return ros_sensor_msg

    def _stop_streaming_motion(self):
        pass

    def _cancel_trajectory_motion(self):
        with self._traj_lock:
            self._traj_state.cancelled = True
        self._traj_state.done_event.set()

    def _enter_fault_state(self):
        pass

    # endregion Motion control

    # region Action server

    def _trajectory_goal_callback(self, goal_request: ExecuteTrajectory.Goal) -> GoalResponse:
        trajectory = goal_request.trajectory
        stop_active_motion = goal_request.stop_active_motion
        self.get_logger().info(
            f"Received trajectory goal request: {len(trajectory.points)} points, stop_active_motion={stop_active_motion}"
        )
        if not trajectory.points:
            self.get_logger().warn("Rejecting trajectory goal: trajectory is empty")
            return GoalResponse.REJECT

        robot_joint_names = self.config.robot_joint_names
        for name in trajectory.joint_names:
            if name not in robot_joint_names:
                self.get_logger().warn(f"Rejecting trajectory goal: unknown joint '{name}'")
                return GoalResponse.REJECT

        self.get_logger().info("Trajectory goal accepted")
        return GoalResponse.ACCEPT

    def _trajectory_cancel_callback(self, _goal_handle: ServerGoalHandle) -> CancelResponse:
        self.get_logger().info("Trajectory cancellation requested")
        return CancelResponse.ACCEPT

    def _execute_trajectory(self, goal_handle: ServerGoalHandle) -> ExecuteTrajectory.Result:
        request: ExecuteTrajectory.Goal = goal_handle.request
        trajectory = request.trajectory
        stop_active = request.stop_active_motion

        result = ExecuteTrajectory.Result()

        valid, message = self.state.validate_transition(ControllerState.TRAJECTORY, stop_active)
        if not valid:
            self.get_logger().warn(f"Cannot execute trajectory: {message}")
            result.success = False
            result.message = message
            result.error_code = 1
            goal_handle.abort()
            return result

        if self.state == ControllerState.STREAMING and stop_active:
            self._stop_streaming_motion()
        if self.state == ControllerState.TRAJECTORY and stop_active:
            self._cancel_trajectory_motion()
            self._traj_state.done_event.wait()

        robot_joint_names = self.config.robot_joint_names
        joint_indices = [robot_joint_names.index(name) for name in trajectory.joint_names]

        initial_feedback = self._latest_feedback_joints.copy()
        if not initial_feedback:
            initial_feedback = [0.0] * len(robot_joint_names)

        with self._traj_lock:
            self._traj_state.reset()  # TODO: instead of making this a state tracker, refactor into a trajectory executable, and each call instantiates a new instance
            self._traj_state.points = list(trajectory.points)
            self._traj_state.joint_indices = joint_indices
            self._traj_state.start_positions = list(initial_feedback)

        # Capture start_time as late as possible, immediately before starting execution,
        # so that elapsed time in the EGM loop reflects actual robot motion time.
        start_time = time.monotonic()
        with self._traj_lock:
            self._traj_state.start_time = start_time

        self.state = ControllerState.TRAJECTORY
        self.get_logger().info(f"Executing time-parameterized trajectory with {len(trajectory.points)} points")

        feedback = ExecuteTrajectory.Feedback()

        while rclpy.ok():
            if goal_handle.is_cancel_requested:
                self._cancel_trajectory_motion()
                self.state = ControllerState.IDLE
                goal_handle.canceled()
                result.success = False
                result.message = "Trajectory cancelled"
                result.error_code = 2
                return result

            if self._traj_state.done_event.is_set():
                self.get_logger().info("Trajectory execution completed successfully")
                self.state = ControllerState.IDLE
                goal_handle.succeed()
                result.success = True
                result.message = "Trajectory completed successfully"
                result.error_code = 0
                return result

            elapsed = time.monotonic() - start_time
            with self._traj_lock:
                current_idx = self._traj_state.current_idx
                desired_all = self._traj_state.last_setpoint

            feedback.elapsed_time = Duration(
                sec=int(elapsed),
                nanosec=int((elapsed - int(elapsed)) * 1e9),
            )
            feedback.current_point_index = current_idx
            feedback.desired_positions = [desired_all[j] for j in joint_indices] if desired_all else []

            actual_all = self._latest_feedback_joints
            if actual_all and len(actual_all) >= max(joint_indices, default=-1) + 1:
                actual_positions = [actual_all[j] for j in joint_indices]
                feedback.actual_positions = actual_positions

                if feedback.desired_positions:
                    sq_err = 0.0
                    for d, a in zip(feedback.desired_positions, actual_positions):
                        e = d - a
                        sq_err += e * e
                    feedback.position_error_norm = math.sqrt(sq_err)
                else:
                    feedback.position_error_norm = 0.0
            else:
                feedback.actual_positions = []
                feedback.position_error_norm = 0.0

            goal_handle.publish_feedback(feedback)
            time.sleep(0.02)

        self.state = ControllerState.IDLE
        goal_handle.abort()
        result.success = False
        result.message = "Node shutdown during trajectory execution"
        result.error_code = 3
        return result

    # endregion Action server

    # region Callbacks

    def _on_param_set(self, params: list[Parameter]) -> SetParametersResult:
        try:
            new_cfg = replace(self.config)
            for param in params:
                spec = PARAM_SPECS.get(param.name)
                if spec is None:
                    continue
                setattr(new_cfg, param.name, param.value)
            new_cfg.validate()
        except ValueError as e:
            return SetParametersResult(successful=False, reason=str(e))

        self.config = new_cfg
        return SetParametersResult(successful=True)

    def _stream_callback(self, msg: RobotPoseState | RobotJointState):
        if self.state != ControllerState.STREAMING:
            return
        self._stream_buffer.set_value(msg)

    def _set_control_mode_callback(self, request: SetControlMode.Request, response: SetControlMode.Response):
        response.previous_mode = self.state.to_srv()

        try:
            requested_mode = ControllerState.from_srv(request.mode)
        except ValueError as e:
            response.success = False
            response.current_mode = self.state.to_srv()
            response.message = str(e)
            self.get_logger().warn(f"Failed to set control mode: {response.message}")
            return response

        valid, message = self.state.validate_transition(requested_mode, request.stop_active_motion)

        if not valid:
            response.success = False
            response.current_mode = self.state.to_srv()
            response.message = message
            self.get_logger().warn(f"Failed to set control mode: {response.message}")
            return response

        if self.state == ControllerState.STREAMING and request.stop_active_motion:
            self._stop_streaming_motion()

        if self.state == ControllerState.TRAJECTORY and request.stop_active_motion:
            self._cancel_trajectory_motion()

        if requested_mode == ControllerState.FAULT:
            self._enter_fault_state()

        self.state = requested_mode
        response.success = True
        response.current_mode = self.state.to_srv()
        response.message = message
        self.get_logger().info(f"Control mode changed: {message}")
        return response

    def _set_gripper_callback(self, request: SetBool.Request, response: SetBool.Response):
        with self._gripper_lock:
            self._gripper_cmd = request.data
        response.success = True
        response.message = f"Gripper command set to {self._gripper_cmd}"
        self.get_logger().info(response.message)
        return response

    # endregion Callbacks


def main():

    rclpy.init()
    node = EGMController()
    executor = MultiThreadedExecutor(num_threads=8)
    executor.add_node(node)
    executor.spin()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
