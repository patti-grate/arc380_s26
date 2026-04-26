"""
Simulation Reset Node

Provides the /reset_simulation service (std_srvs/Trigger) that:
  1. Deactivates then unloads all ros2_control controllers
     (unloading releases hardware-interface bindings so they can be
     re-bound to the new plugin instance after respawn)
  2. Resets the Gazebo world to its initial state — restores brick
     positions/velocities and removes the dynamically-spawned robot
  3. Respawns the robot from the robot_description topic
  4. Polls list_hardware_interfaces until the robot joints are available
  5. Spawns the controllers fresh (load + configure + activate) exactly
     as the launch file does, so they bind to the new plugin instance

Usage (after building and sourcing):
  ros2 run abb_irb120_gazebo simulation_reset_node
  ros2 service call /reset_simulation std_srvs/srv/Trigger "{}"
"""

import subprocess
import time

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger


# Activation order; deactivation/unload uses reversed order.
CONTROLLERS = [
    "joint_state_broadcaster",
    "arm_controller",
    "gripper_controller",
]

# A joint interface that only exists while the robot plugin is alive.
# Used to detect when the new plugin has registered its hardware interfaces.
_HW_READY_PROBE = "joint_1/position"


class SimulationResetNode(Node):
    def __init__(self):
        super().__init__("simulation_reset")

        self.declare_parameter("world_name", "irb120_workcell")
        self.declare_parameter("robot_name", "abb_irb120")
        self.declare_parameter("robot_description_topic", "robot_description")
        self.declare_parameter("hw_ready_timeout_sec", 45.0)

        self.world_name = self.get_parameter("world_name").value
        self.robot_name = self.get_parameter("robot_name").value
        self.robot_description_topic = self.get_parameter("robot_description_topic").value
        self.hw_ready_timeout = self.get_parameter("hw_ready_timeout_sec").value

        self.create_service(Trigger, "/reset_simulation", self._reset_callback)

        self.get_logger().info(
            "Simulation reset node ready — call /reset_simulation to reset."
        )

    # ------------------------------------------------------------------
    # Service callback
    # ------------------------------------------------------------------

    def _reset_callback(self, _request, response):
        self.get_logger().info("=== Simulation reset requested ===")
        errors = []

        # 1. Deactivate then unload controllers (reverse dependency order).
        #    Deactivation must precede unloading; both steps are needed so
        #    the controllers fully release their hardware interface pointers.
        #    When the robot entity respawns the gz_ros2_control plugin creates
        #    *new* interface objects — stale pointers cause silent failures.
        self.get_logger().info("[1/5] Unloading controllers...")
        for ctrl in reversed(CONTROLLERS):
            # Deactivate first (no-op if already inactive)
            self._run(
                ["ros2", "control", "set_controller_state", ctrl, "inactive"],
                timeout=10,
            )
            # Unload to fully release hardware interface bindings
            ok, _, err = self._run(
                ["ros2", "control", "unload_controller", ctrl],
                timeout=10,
            )
            if ok:
                self.get_logger().info(f"  {ctrl}: unloaded")
            else:
                self.get_logger().warn(f"  Could not unload {ctrl}: {err}")

        # 2. Reset Gazebo world.
        self.get_logger().info("[2/5] Resetting Gazebo world...")
        ok, out, err = self._run(
            [
                "gz", "service",
                "-s", f"/world/{self.world_name}/control",
                "--reqtype", "gz.msgs.WorldControl",
                "--reptype", "gz.msgs.Boolean",
                "--timeout", "5000",
                "--req", "reset: {all: true}",
            ],
            timeout=10,
        )
        if ok:
            self.get_logger().info(f"  World reset: {out}")
        else:
            msg = f"World reset failed: {err}"
            self.get_logger().error(f"  {msg}")
            errors.append(msg)

        time.sleep(2.0)

        # 3. Respawn the robot.
        self.get_logger().info("[3/5] Respawning robot...")
        ok, out, err = self._run(
            [
                "ros2", "run", "ros_gz_sim", "create",
                "-name", self.robot_name,
                "-topic", self.robot_description_topic,
            ],
            timeout=45,
        )
        if ok:
            self.get_logger().info(f"  Robot respawned: {out}")
        else:
            msg = f"Robot respawn failed: {err}"
            self.get_logger().error(f"  {msg}")
            errors.append(msg)

        # 4. Poll until the gz_ros2_control plugin has registered hardware
        #    interfaces for the new robot entity.  A fixed sleep is fragile;
        #    polling lets us proceed as soon as the hardware is actually ready.
        self.get_logger().info("[4/5] Waiting for hardware interfaces...")
        if not self._wait_for_hardware(timeout=self.hw_ready_timeout):
            msg = f"Hardware interfaces did not appear within {self.hw_ready_timeout}s"
            self.get_logger().error(f"  {msg}")
            errors.append(msg)
        else:
            self.get_logger().info("  Hardware interfaces ready.")

        # 5. Spawn (load + configure + activate) each controller in order,
        #    exactly as the launch file does.  This creates fresh bindings to
        #    the new hardware interface objects, fixing MoveIt/RViz state.
        self.get_logger().info("[5/5] Spawning controllers...")
        for ctrl in CONTROLLERS:
            ok, _, err = self._run(
                [
                    "ros2", "run", "controller_manager", "spawner", ctrl,
                    "--controller-manager-timeout", "30",
                    "--switch-timeout", "30",
                ],
                timeout=45,
            )
            if ok:
                self.get_logger().info(f"  {ctrl}: active")
            else:
                msg = f"{ctrl} spawn failed: {err}"
                self.get_logger().error(f"  {msg}")
                errors.append(msg)

        if errors:
            response.success = False
            response.message = "Reset completed with errors: " + "; ".join(errors)
        else:
            response.success = True
            response.message = "Simulation reset complete."

        self.get_logger().info(f"=== {response.message} ===")
        return response

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _wait_for_hardware(self, timeout=15.0):
        """Block until the robot's hardware interfaces are [available].

        Returns True if they appeared before *timeout* seconds, False otherwise.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            ok, out, _ = self._run(
                ["ros2", "control", "list_hardware_interfaces"],
                timeout=5,
            )
            if ok and _HW_READY_PROBE in out and "[available]" in out:
                return True
            time.sleep(0.5)
        return False

    def _run(self, cmd, timeout=15):
        """Run a subprocess; return (success, stdout, stderr)."""
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout
            )
            return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
        except subprocess.TimeoutExpired:
            return False, "", f"Command timed out after {timeout}s"
        except Exception as exc:
            return False, "", str(exc)


def main(args=None):
    rclpy.init(args=args)
    node = SimulationResetNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
