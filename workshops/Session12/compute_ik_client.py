import rclpy
from rclpy.node import Node

from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import RobotState, PositionIKRequest
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState


class ComputeIKClient(Node):
    def __init__(self):
        super().__init__("compute_ik_client")

        pass

    def call_ik(
        self,
        group_name: str,
        link_name: str,
        target_pose: PoseStamped,
        seed_joint_names: list[str] = None,
        seed_joint_positions: list[float] = None,
        frame_id: str = "world",
        timeout_sec: float = 1.0,
    ):
        pass


def main():
    rclpy.init()
    node = ComputeIKClient()

    group_name = "arm"
    link_name = "tool0"
    frame_id = "world"

    # Desired pose (example)
    target = PoseStamped()
    target.header.frame_id = frame_id
    target.pose.position.x = 0.138
    target.pose.position.y = 0.0
    target.pose.position.z = 0.4714
    target.pose.orientation.w = 0.5102
    target.pose.orientation.x = 0.0
    target.pose.orientation.y = 0.8601
    target.pose.orientation.z = 0.0

    # Optional seed (recommended)
    seed_joint_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
    seed_joint_positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # ------------------------------------------------------

    node.call_ik(
        group_name=group_name,
        link_name=link_name,
        target_pose=target,
        seed_joint_names=seed_joint_names,
        seed_joint_positions=seed_joint_positions,
        frame_id=frame_id,
        timeout_sec=1.0,
    )

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
