import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    pkg_moveit = get_package_share_directory("abb_irb120_moveit")

    moveit_config = (
        MoveItConfigsBuilder("abb_irb120", package_name="abb_irb120_moveit")
        .robot_description(file_path="config/abb_irb120_3_60.urdf.xacro")
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .planning_scene_monitor(
            publish_robot_description=True,
            publish_robot_description_semantic=True,
        )
        .to_moveit_configs()
    )

    rviz_config_file = os.path.join(pkg_moveit, "config", "moveit.rviz")

    egm_controller = Node(
        package="abb_egm_controller",
        executable="egm",
        name="egm_controller",
        parameters=[
            {
                "udp_port": 6511,
                "docker_mode": True,
                "relay_port_out": 6512,
            }
        ],
        output="screen",
    )

    rsp = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[
            moveit_config.robot_description,
        ],
        output="screen",
    )

    move_group = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
        ],
    )

    # rviz = Node(
    #     package="rviz2",
    #     executable="rviz2",
    #     name="rviz2",
    #     output="screen",
    #     arguments=["-d", rviz_config_file],
    #     parameters=[
    #         moveit_config.robot_description,
    #         moveit_config.robot_description_semantic,
    #         moveit_config.robot_description_kinematics,
    #         moveit_config.planning_pipelines,
    #         moveit_config.joint_limits,
    #     ],
    # )

    return LaunchDescription(
        [
            rsp,
            egm_controller,
            move_group,
            # rviz,
        ]
    )
