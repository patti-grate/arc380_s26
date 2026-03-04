## ARC 380 / CEE 380 / ROB 380 – Introduction to Robotics for Digital Fabrication  
### Session 10 Workshop: Simulation in ROS
Princeton University, Spring 2026  
Professor: Arash Adel | TA: Daniel Ruan

---

## Overview

In the previous workshop, we visualized the ABB IRB 120 robot in RViz using URDF and `robot_state_publisher`.

In this session, we transition from **visualization** to **physics simulation** using Gazebo.

Gazebo allows us to simulate gravity, contact, collision, and environment interaction, essential for digital fabrication workflows.

By the end of this workshop, you should be able to:

-   Explain the difference between RViz and Gazebo.
-   Launch a working Gazebo simulation of the IRB 120.
-   Understand the high-level Gazebo-ROS 2 architecture.
-   Add objects to a Gazebo world.
-   Spawn objects dynamically using ROS 2.
-   Inspect simulation topics.

---

# 1. Visualization vs. Simulation

## RViz

-   Visualization only
-   No gravity
-   No collision
-   No contact physics
-   Idealized motion

## Gazebo

-   Physics engine
-   Gravity
-   Collision detection
-   Contact forces
-   Dynamic objects

**Key Idea:**\
RViz shows what the robot *should* look like.

Gazebo simulates how the robot behaves in a physical environment.

---

# 2. Gazebo + ROS 2 Architecture

When running Gazebo with ROS 2:

    URDF → Gazebo (physics engine)
            ↕
       ros2_control
            ↕
       ROS 2 topics

### Core Components

-   **URDF** -- Robot model description
-   **Gazebo World** -- Environment definition
-   **ros2_control** -- Interface between physics and ROS
-   **Controllers** -- Drive joint motion
-   **/joint_states + TF** -- Feedback and transforms

Gazebo simulates physics, ROS 2 manages communication and control interfaces.

---



Run:

``` bash
ros2 launch abb_irb120_gazebo simulation.launch.py
```

---

## Inspect ROS Topics

Open a new terminal:

``` bash
ros2 topic list
```

Look for:

-   `/clock`
-   `/tf`
-   `/robot_description`
-   (Later, once controllers are configured) `/joint_states` and controller-related topics

Check joint states:

``` bash
ros2 topic echo /joint_states
```

Notice:

In a full setup with **ros2_control** configured, `/joint_states` will be driven by the physics simulation.

In this workshop’s baseline launch, the robot is spawned and TF is published, but joint controllers may be added in the next steps.

---

# 4. Exploring the Gazebo Interface

Inside Gazebo:

### Panels to Use

-   **Entity Tree**
-   **World Control**
-   **Insert Tool**

### Try the Following

1.  Pause and unpause simulation.
2.  Reset the world.
3.  Adjust the camera view.

Observe:

-   Real-time factor
-   Physics stepping
-   Gravity effects

---

# 5. Adding Objects via the GUI

Using the **Insert** tool:

1.  Add a box.
2.  Place it above the ground.
3.  Press play.

Observe:

-   The object falls due to gravity.
-   It collides with the ground.
-   It can collide with the robot.

---

# 6. Spawning Objects with ROS 2

Objects can also be added programmatically.

Run (Gazebo Sim / `ros_gz_sim`):

```bash
# Spawn an SDF model file into the running Gazebo Sim instance
ros2 run ros_gz_sim create -name brick -file brick.sdf -x 0.4 -y 0.0 -z 0.2
```

Observe:

-   The object appears in the world.
-   It falls due to gravity.
-   It interacts with the environment.

This method enables:

-   Automated scene setup
-   Reproducible experiments
-   Robotic assembly workflows

---

# 7. Understanding the World File

Locate the world file:

    abb_irb120_gazebo/worlds/empty.sdf

Open it and identify:

-   Ground plane
-   Lighting
-   Physics properties
-   Predefined objects

Modify:

-   Add one static object.
-   Relaunch the simulation.

**Key Idea:**
World design is part of robotic system design.

