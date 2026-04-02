import time
from math import pi

import rclpy
from builtin_interfaces.msg import Duration
from example_interfaces.srv import SetBool
from rclpy.action import ActionClient
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from abb_egm_interfaces.action import ExecuteTrajectory

trajectory = JointTrajectory()
trajectory.joint_names = [f"joint_{i + 1}" for i in range(6)]
point = JointTrajectoryPoint()
point.positions = [0.0, 0.0, 0.0, 0.0, pi / 2, 0.0]
point.time_from_start = Duration(sec=0, nanosec=0)
trajectory.points.append(point)
point = JointTrajectoryPoint()
point.positions = [0.5, 0.1, 0.1, 0.1, pi / 2 + 0.1, 0.1]
point.time_from_start = Duration(sec=2, nanosec=0)
trajectory.points.append(point)


def feedback_callback(feedback_msg):
    fb = feedback_msg.feedback
    elapsed = fb.elapsed_time.sec + fb.elapsed_time.nanosec * 1e-9
    desired = [f"{v:.3f}" for v in fb.desired_positions]
    actual = [f"{v:.3f}" for v in fb.actual_positions]
    print(
        f"  t={elapsed:.2f}s  idx={fb.current_point_index}"
        f"  err_norm={fb.position_error_norm:.4f}"
        f"  desired={desired}"
        f"  actual={actual}"
    )


def main():
    rclpy.init()
    node = Node("egm_test_node")

    action_client = ActionClient(node, ExecuteTrajectory, "execute_trajectory")
    gripper_client = node.create_client(SetBool, "/egm_controller/control/set_gripper")

    print("Waiting for execute_trajectory action server...")
    while not action_client.wait_for_server(timeout_sec=0.1):
        rclpy.spin_once(node, timeout_sec=0.1)
    print("Connected.")

    goal = ExecuteTrajectory.Goal()
    goal.trajectory = trajectory
    goal.stop_active_motion = True

    print("Sending trajectory goal...")
    send_goal_future = action_client.send_goal_async(goal, feedback_callback=feedback_callback)
    while not send_goal_future.done():
        rclpy.spin_once(node)
    goal_handle = send_goal_future.result()

    if not goal_handle or not goal_handle.accepted:
        print("Goal rejected.")
        rclpy.shutdown()
        return

    print("Goal accepted, waiting for result...")
    result_future = goal_handle.get_result_async()
    while not result_future.done():
        rclpy.spin_once(node)
    result = result_future.result()

    print(
        f"Done. success={result.result.success}  error_code={result.result.error_code}  message='{result.result.message}'"
    )

    while not gripper_client.wait_for_service(timeout_sec=1.0):
        print("Waiting for gripper control service...")

    print("Closing gripper...")
    req = SetBool.Request()
    req.data = True  # Close gripper
    future = gripper_client.call_async(req)
    rclpy.spin_until_future_complete(node, future)
    resp = future.result()
    print(f"Gripper control response: success={resp.success}  message='{resp.message}'")
    time.sleep(1.0)

    print("Opening gripper...")
    req.data = False  # Open gripper
    future = gripper_client.call_async(req)
    rclpy.spin_until_future_complete(node, future)
    resp = future.result()
    print(f"Gripper control response: success={resp.success}  message='{resp.message}'")
    time.sleep(1.0)

    rclpy.shutdown()


if __name__ == "__main__":
    main()
