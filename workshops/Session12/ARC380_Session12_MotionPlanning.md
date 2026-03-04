# ARC 380 / CEE 380 / ROB 380

## Session 12 Workshop: Forward / Inverse Kinematics and Motion Planning with MoveIt

Princeton University, Spring 2026\
Professor: Arash Adel | TA: Daniel Ruan

------------------------------------------------------------------------

# Overview

By the end of this workshop you should be able to:

1.  Explain what **MoveIt** is and how it fits into the ROS 2 ecosystem.
2.  Describe the components inside a typical MoveIt configuration
    package.
3.  Use the **RViz MotionPlanning GUI** to:
    -   Manipulate the end effector
    -   Plan a trajectory
    -   Preview a motion plan
4.  Call MoveIt services directly using **rclpy**:
    -   `/compute_fk`
    -   `/compute_ik`
    -   `/plan_kinematic_path`

We will need some additional packages:
```bash
conda install -y -c robostack-jazzy -c conda-forge ros-jazzy-ros2-control ros-jazzy-gz-ros2-control ros-jazzy-ros2-controllers ros-jazzy-moveit filelock
```


# 1. What is MoveIt?

MoveIt is a motion planning framework built on top of ROS.

It provides:

-   Robot model loading (URDF + SRDF)
-   Forward kinematics
-   Inverse kinematics
-   Collision checking
-   Planning scene management
-   Sampling-based motion planners (OMPL)
-   Trajectory generation

Conceptually:

    URDF → RobotModel → IK / Collision Checking → Planner → Trajectory

MoveIt does **not** replace your URDF.\
It consumes your robot description and adds motion intelligence.


# 2. MoveIt Package Structure

Contains:

-   SRDF (semantic groups, end effectors)
-   Kinematics configuration
-   OMPL planner configuration
-   MoveIt launch files
-   Controller configuration (optional)

MoveIt reads the URDF and augments it with planning capabilities.


# 3. Launching MoveIt

``` bash
ros2 launch abb_irb120_moveit demo.launch.py
```


# 4. Using the GUI: End-Effector Manipulation and Planning

## Interactive Marker (IK)

1.  Open the **MotionPlanning** panel.
2.  Drag the end-effector marker.
3.  Observe joint updates.

This performs pose → joint mapping using IK.

## Planning a Motion

1.  Set a goal pose.
2.  Click **Plan**.
3.  Preview the trajectory.

Planning happens in joint space with collision checking enabled.


# 5. Code-along (rclpy)

## `/compute_fk`

JointState → Pose

- https://github.com/moveit/moveit_msgs/blob/ros2/srv/GetPositionFK.srv

## `/compute_ik`

Pose → JointState

- https://github.com/moveit/moveit_msgs/blob/ros2/srv/GetPositionIK.srv

## `/plan_kinematic_path`

Start State + Goal Constraints → Joint Trajectory

- https://github.com/moveit/moveit_msgs/blob/ros2/srv/GetMotionPlan.srv
- https://github.com/moveit/moveit_msgs/blob/ros2/msg/MotionPlanRequest.msg
- https://github.com/moveit/moveit_msgs/blob/ros2/msg/MotionPlanResponse.msg
- https://github.com/moveit/moveit_msgs/blob/ros2/msg/MoveItErrorCodes.msg


# 6. Conceptual Comparison

  Operation   Input          Output         Search?   Collision Checking?
  ----------- -------------- -------------- --------- ---------------------
  FK          Joint values   Pose           No        No
  IK          Pose           Joint values   Yes       Optional
  Planning    Start + Goal   Trajectory     Yes       Yes


# 7. Additional Resources

- MoveIt 2 Documentation: https://moveit.picknik.ai/main/index.html
- The Open Motion Planning Library (OMPL): https://ompl.kavrakilab.org/
