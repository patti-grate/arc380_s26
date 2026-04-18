#!/usr/bin/env python3
"""
Publish simple box collision objects to /collision_object (MoveIt planning scene).

Run while move_group is up, e.g. after gz_moveit:

  source /opt/ros/jazzy/setup.bash
  source /ros2_ws/install/setup.bash
  python3 /ros2_ws/scripts/publish_collision_boxes.py

Edit BOXES below (frame, size, position, quaternion x,y,z,w).
"""

from __future__ import annotations

import time

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import Pose
from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import SolidPrimitive

# Match MoveIt planning scene expectations (reliable + transient local).
_COLLISION_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
)

# (object_id, frame_id, size_xyz_m, position_xyz_m, quat_xyzw)
BOXES: list[tuple[str, str, tuple[float, float, float], tuple[float, float, float], tuple[float, float, float, float]]] = [
    ("demo_box_a", "world", (0.08, 0.08, 0.08), (0.45, 0.0, 0.15), (0.0, 0.0, 0.0, 1.0)),
    ("demo_box_b", "world", (0.05, 0.12, 0.04), (0.35, -0.20, 0.05), (0.0, 0.0, 0.0, 1.0)),
]


def make_box(
    object_id: str,
    frame_id: str,
    size_xyz: tuple[float, float, float],
    position_xyz: tuple[float, float, float],
    quat_xyzw: tuple[float, float, float, float],
) -> CollisionObject:
    """Build a CollisionObject: pose = box in header frame; primitive at object origin."""
    co = CollisionObject()
    co.header.frame_id = frame_id
    co.id = object_id

    prim = SolidPrimitive()
    prim.type = SolidPrimitive.BOX
    prim.dimensions = [float(size_xyz[0]), float(size_xyz[1]), float(size_xyz[2])]

    qx, qy, qz, qw = quat_xyzw
    co.pose.position.x = float(position_xyz[0])
    co.pose.position.y = float(position_xyz[1])
    co.pose.position.z = float(position_xyz[2])
    co.pose.orientation.x = float(qx)
    co.pose.orientation.y = float(qy)
    co.pose.orientation.z = float(qz)
    co.pose.orientation.w = float(qw)

    prim_pose = Pose()
    prim_pose.orientation.w = 1.0

    co.primitives = [prim]
    co.primitive_poses = [prim_pose]
    co.operation = CollisionObject.ADD
    return co


def main() -> None:
    rclpy.init()
    node = Node("publish_collision_boxes")
    pub = node.create_publisher(CollisionObject, "/collision_object", _COLLISION_QOS)

    node.get_logger().info("Waiting for a subscriber on /collision_object …")
    for _ in range(300):
        if pub.get_subscription_count() > 0:
            break
        rclpy.spin_once(node, timeout_sec=0.1)
    subs = pub.get_subscription_count()
    if subs == 0:
        node.get_logger().warn(
            "No subscribers yet; publishing anyway (start move_group first for best results)."
        )

    for spec in BOXES:
        co = make_box(*spec)
        for _ in range(10):
            co.header.stamp = node.get_clock().now().to_msg()
            pub.publish(co)
            for _ in range(2):
                rclpy.spin_once(node, timeout_sec=0.02)
            time.sleep(0.05)
        node.get_logger().info(f"Published ADD id={co.id!r} frame={co.header.frame_id!r}")

    for _ in range(40):
        rclpy.spin_once(node, timeout_sec=0.05)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
