import os
import glob
import json
import time
import random
import numpy as np

import rhino3dm

# Import the Brick class we created
from pose_conversion import Brick

try:
    import rclpy
    from rclpy.node import Node
    from ros_gz_interfaces.srv import SpawnEntity, ControlWorld
    from ros_gz_interfaces.msg import WorldControl, WorldReset
    from std_srvs.srv import Empty

    # TF listener for model states equivalent
    from tf2_msgs.msg import TFMessage

    HAVE_ROS2 = True
except ImportError:
    HAVE_ROS2 = False
    print(
        "WARNING: rclpy or ROS 2 packages not found. Script will run in dry-run/mock mode."
    )


class DemoValidator:
    """Mock wrapper if ROS isn't available"""

    def __init__(self):
        self.current_model_states = {}

    def reset_world(self):
        self.current_model_states.clear()
        return True

    def spawn_brick(self, name, pose_7d):
        self.current_model_states[name] = type(
            "obj",
            (object,),
            {
                "position": type(
                    "pos",
                    (object,),
                    {"x": pose_7d[0], "y": pose_7d[1], "z": pose_7d[2]},
                )
            },
        )
        time.sleep(0.01)  # Mock wait
        return True

    def check_stability(self, initial_states):
        return True

    def destroy_node(self):
        pass


if HAVE_ROS2:

    class DemoValidatorNode(Node, DemoValidator):
        def __init__(self):
            Node.__init__(self, "demo_validator")
            DemoValidator.__init__(self)

            self.sdf_path = os.path.join(
                os.getcwd(), "src", "abb_irb120_gazebo", "models", "brick", "model.sdf"
            )
            self.active_bricks = {}  # tracked by name

            # Spin up internal temporary ros_gz_bridges
            import subprocess

            self.control_bridge = subprocess.Popen(
                "ros2 run ros_gz_bridge parameter_bridge /world/irb120_workcell/control@ros_gz_interfaces/srv/ControlWorld",
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            self.reset_client = self.create_client(
                ControlWorld, "/world/irb120_workcell/control"
            )
            sdf_path = os.path.join(
                os.getcwd(), "src", "abb_irb120_gazebo", "models", "brick", "model.sdf"
            )
            try:
                with open(sdf_path, "r") as f:
                    self.sdf_xml = f.read()
            except Exception as e:
                self.get_logger().error(f"Could not load SDF from {sdf_path}: {e}")

        def fetch_latest_poses_from_gz(self):
            import subprocess
            import re

            try:
                output = subprocess.check_output(
                    "gz topic -e -n 1 -t /world/irb120_workcell/pose/info",
                    shell=True,
                    text=True,
                    timeout=2.0,
                )
                poses = output.split("pose {")
                for pose_str in poses:
                    if "name:" not in pose_str:
                        continue
                    name_match = re.search(r'name:\s+"([^"]+)"', pose_str)
                    if not name_match:
                        continue
                    name = name_match.group(1)
                    if "brick" not in name:
                        continue

                    def get_val(regex, s):
                        m = re.search(regex, s)
                        return float(m.group(1)) if m else 0.0

                    x_val = get_val(r"x:\s+([^\n]+)", pose_str)
                    y_val = get_val(r"y:\s+([^\n]+)", pose_str)
                    z_val = get_val(r"z:\s+([^\n]+)", pose_str)

                    pose_mock = type(
                        "pos",
                        (object,),
                        {
                            "position": type(
                                "p", (object,), {"x": x_val, "y": y_val, "z": z_val}
                            )
                        },
                    )
                    self.current_model_states[name] = pose_mock
            except Exception as e:
                pass

        def reset_world(self):
            # Use bridged control service to natively reset Gazebo scenes
            self.get_logger().info("Simulation layer refresh (Resetting states...)")

            if self.reset_client.wait_for_service(timeout_sec=2.0):
                req = ControlWorld.Request()
                req.world_control.reset.all = True
                future = self.reset_client.call_async(req)
                rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
            else:
                self.get_logger().warn(
                    "Could not find bridged ControlWorld service to reset simulation."
                )

            self.get_logger().info(
                "Purging populated simulation components to prepare a clean stage..."
            )
            import subprocess

            # Purge the original 20 bricks that respawn upon reset
            for i in range(25):
                name = f"brick_{i:02d}"
                cmd = f"gz service -s /world/irb120_workcell/remove --reqtype gz.msgs.Entity --reptype gz.msgs.Boolean --timeout 500 --req 'name: \"{name}\" type: MODEL'"
                subprocess.run(
                    cmd,
                    shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

            # Purge any remaining tracked demo bricks locally
            for name in list(self.current_model_states.keys()):
                cmd = f"gz service -s /world/irb120_workcell/remove --reqtype gz.msgs.Entity --reptype gz.msgs.Boolean --timeout 500 --req 'name: \"{name}\" type: MODEL'"
                subprocess.run(
                    cmd,
                    shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

            self.clear_collapse_message()
            self.current_model_states.clear()
            time.sleep(1)
            return True

        def spawn_brick(self, name, pose_7d):
            import subprocess

            x, y, z = pose_7d[0], pose_7d[1], pose_7d[2] + 0.001
            qx, qy, qz, qw = pose_7d[3], pose_7d[4], pose_7d[5], pose_7d[6]

            req_str = f'sdf_filename: \\"{self.sdf_path}\\" name: \\"{name}\\" pose: {{ position: {{x: {x} y: {y} z: {z}}} orientation: {{x: {qx} y: {qy} z: {qz} w: {qw}}} }}'
            cmd = f'gz service -s /world/irb120_workcell/create --reqtype gz.msgs.EntityFactory --reptype gz.msgs.Boolean --timeout 2000 --req "{req_str}"'

            try:
                # Bypass ROS 2 node-spin execution entirely and directly inject into simulation memory using Ignite's internal protocol
                subprocess.run(
                    cmd,
                    shell=True,
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                self.get_logger().info(f"Spawned {name} in simulation")
                return True
            except subprocess.CalledProcessError:
                self.get_logger().warn(f"Failed to natively spawn {name}")
                return False

        def display_collapse_message(self, text="COLLAPSE DETECTED!"):
            import subprocess

            msg = f"ns: 'val', id: 100, action: ADD_MODIFY, type: TEXT, text: '{text}', pose: {{position: {{x: 0.15, y: 0.35, z: 0.5}}}}, scale: {{x: 0.08, y: 0.08, z: 0.08}}, material: {{ambient: {{r: 1.0, g: 0.0, b: 0.0, a: 1.0}}}}"
            cmd = f'gz topic -t /marker -m gz.msgs.Marker -p "{msg}"'
            subprocess.run(
                cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )

        def clear_collapse_message(self):
            import subprocess

            msg = f"ns: 'val', id: 100, action: DELETE_MARKER"
            cmd = f'gz topic -t /marker -m gz.msgs.Marker -p "{msg}"'
            subprocess.run(
                cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )

        def check_stability(self, initial_states):
            self.fetch_latest_poses_from_gz()

            if not self.current_model_states:
                self.get_logger().warn(
                    "CRITICAL: Poses unavilable from Gazebo! Collapse detection is currently blind."
                )
            for name, initial_pose in initial_states.items():
                if name not in self.current_model_states:
                    continue
                curr_pose = self.current_model_states[name]

                dz = curr_pose.position.z - initial_pose.position.z

                # Check for structural collapse (fell > 1cm)
                if dz < -0.01:
                    self.get_logger().warn(f"{name} collapse detected! (dZ: {dz:.3f}m)")
                    self.display_collapse_message()
                    return False
            return True

        def destroy_node(self):
            if hasattr(self, "control_bridge"):
                self.control_bridge.terminate()
            super().destroy_node()


def extract_poses_from_3dm(filepath):
    model = rhino3dm.File3dm.Read(filepath)
    if not model:
        print(f"Failed to read {filepath}")
        return []

    bricks = []
    layer_index = -1
    for i, layer in enumerate(model.Layers):
        if layer.Name == "bricks":
            layer_index = i
            break

    objects = []
    for obj in model.Objects:
        # Fallback to grabbing all geometries if layer wasn't strictly found, otherwise strict match.
        if layer_index == -1 or obj.Attributes.LayerIndex == layer_index:
            objects.append(obj.Geometry)

    for geom in objects:
        brep = None
        if isinstance(geom, rhino3dm.Brep):
            brep = geom
        elif isinstance(geom, rhino3dm.Extrusion):
            brep = geom.ToBrep(True)

        if brep:
            corners = []
            for v in brep.Vertices:
                corners.append([v.Location.X, v.Location.Y, v.Location.Z])
            if len(corners) >= 8:
                brick = Brick(corners=corners[:8])
                bricks.append(brick)

    return bricks


def bucket_and_sort_bricks(bricks, z_tolerance=0.005):
    """Buckets bricks into layers by near-identical Z, sorts bottom-to-top"""
    if not bricks:
        return []

    bricks = sorted(bricks, key=lambda b: b.get_7d_pose()[2])
    layers = []
    current_layer = [bricks[0]]
    current_z = bricks[0].get_7d_pose()[2]

    for b in bricks[1:]:
        z = b.get_7d_pose()[2]
        if abs(z - current_z) <= z_tolerance:
            current_layer.append(b)
        else:
            layers.append(current_layer)
            current_layer = [b]
            current_z = z
    layers.append(current_layer)
    return layers


def shuffle_layers(layers):
    """Randomize order within each layer chunk"""
    shuffled = []
    for layer in layers:
        layer_copy = layer.copy()
        random.shuffle(layer_copy)
        shuffled.extend(layer_copy)
    return shuffled


def main(args=None):
    if HAVE_ROS2:
        rclpy.init(args=args)
        validator = DemoValidatorNode()
    else:
        validator = DemoValidator()

    batch_dir = os.path.join(os.getcwd(), "training_data", "batch0")
    rhino_dir = os.path.join(batch_dir, "rhino")
    val_dir = os.path.join(batch_dir, "validated_simPhysics")

    if not os.path.exists(rhino_dir):
        print(f"Directory {rhino_dir} not found. Ensure we run at project root.")
        # create mock to run through logic if user just tests
        os.makedirs(rhino_dir, exist_ok=True)

    demo_files = glob.glob(os.path.join(rhino_dir, "demo_*.3dm"))
    if not demo_files:
        print("No demo_*.3dm files found to process.")

    for demo_file in demo_files:
        demo_name = os.path.splitext(os.path.basename(demo_file))[0]
        print(f"\n== Processing {demo_name} ==")

        bricks = extract_poses_from_3dm(demo_file)
        if not bricks:
            print(f"No bricks geometry found in {demo_name}.")
            continue

        print(f"Extracted {len(bricks)} bricks.")
        layers = bucket_and_sort_bricks(bricks)

        max_attempts = 4  # 1 original + 3 resamples
        success = False
        final_sequence = []

        for attempt in range(max_attempts):
            print(f"--> Attempt {attempt + 1}/{max_attempts} ...")
            sequence = shuffle_layers(layers) if attempt > 0 else sum(layers, [])

            validator.reset_world()

            stable = True
            initial_states = {}
            for i, brick in enumerate(sequence):
                name = f"{demo_name}_brick_{i}"
                pose_7d = brick.get_7d_pose()

                if not validator.spawn_brick(name, pose_7d):
                    print(f"Failed to physically spawn {name}")

                if HAVE_ROS2:
                    # Sleep for 2 seconds to simulate physics settle (we no longer spin ros bridges)
                    time.sleep(2.0)
                else:
                    time.sleep(0.1)  # Mock wait

                # Snapshot initial resting pose
                validator.fetch_latest_poses_from_gz()
                if name in validator.current_model_states:
                    initial_states[name] = validator.current_model_states[name]

                # Check for structural failure
                if not validator.check_stability(initial_states):
                    stable = False
                    print(f"Collapse detected upon triggering {name}!")
                    break

            if stable:
                success = True
                final_sequence = sequence
                print(f"Successfully stabilized and sequenced {demo_name}.")
                break

        # Export files
        out_base = os.path.join(val_dir, demo_name)
        os.makedirs(os.path.join(out_base, "7d_sequence"), exist_ok=True)
        os.makedirs(os.path.join(out_base, "5d_sequence"), exist_ok=True)

        if success:
            seq_7d = [b.get_7d_pose().tolist() for b in final_sequence]
            seq_5d = []
            for b in final_sequence:
                p5 = b.to_5d_pose()
                seq_5d.append({"error": p5} if isinstance(p5, str) else p5.tolist())

            with open(os.path.join(out_base, "7d_sequence", "sequence.json"), "w") as f:
                json.dump(seq_7d, f, indent=2)
            with open(os.path.join(out_base, "5d_sequence", "sequence.json"), "w") as f:
                json.dump(seq_5d, f, indent=2)
        else:
            with open(os.path.join(out_base, "failure.json"), "w") as f:
                json.dump(
                    {"error": "Structure collapsed after 3 resampling attempts."}, f
                )

    validator.destroy_node()
    if HAVE_ROS2:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
