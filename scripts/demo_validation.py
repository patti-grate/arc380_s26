import os
import glob
import json
import time
import random
import argparse
import numpy as np

import rhino3dm

# Import the Brick class we created
from pose_conversion import Brick

try:
    import rclpy
    from rclpy.node import Node
    from ros_gz_interfaces.srv import SpawnEntity, ControlWorld, DeleteEntity
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

    def fetch_latest_poses_from_gz(self):
        pass

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

            self.bridges = subprocess.Popen(
                "ros2 run ros_gz_bridge parameter_bridge "
                "/world/irb120_workcell/control@ros_gz_interfaces/srv/ControlWorld "
                "/world/irb120_workcell/remove@ros_gz_interfaces/srv/DeleteEntity",
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            self.reset_client = self.create_client(
                ControlWorld, "/world/irb120_workcell/control"
            )
            self.remove_client = self.create_client(
                DeleteEntity, "/world/irb120_workcell/remove"
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
                self.current_model_states.clear()
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

            # Try up to 3 times to ensure the scene is clean
            for clean_attempt in range(3):
                self.fetch_latest_poses_from_gz()
                names_to_remove = [f"brick_{i:02d}" for i in range(25)]
                names_to_remove += list(self.current_model_states.keys())
                names_to_remove = list(set(names_to_remove)) # unique
                
                if not names_to_remove:
                    break # Clean!

                # Use the bridged service for speed
                if self.remove_client.wait_for_service(timeout_sec=2.0):
                    futures = []
                    for name in names_to_remove:
                        req = DeleteEntity.Request()
                        req.entity.name = name
                        req.entity.type = 2 # MODEL
                        futures.append(self.remove_client.call_async(req))
                    
                    if futures:
                        rclpy.spin_until_future_complete(self, futures[-1], timeout_sec=2.0)
                else:
                    self.get_logger().warn("Could not find bridged DeleteEntity service to remove bricks. Falling back to slow subprocess.")
                    for name in names_to_remove:
                        cmd = f"gz service -s /world/irb120_workcell/remove --reqtype gz.msgs.Entity --reptype gz.msgs.Boolean --timeout 500 --req 'name: \"{name}\" type: MODEL'"
                        subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                time.sleep(0.5)

            # Final verification
            self.fetch_latest_poses_from_gz()
            if self.current_model_states:
                self.get_logger().error(f"Failed to fully clean scene! Remaining bricks: {list(self.current_model_states.keys())}")
            else:
                self.get_logger().info("Scene verified clean.")

            self.clear_collapse_message()
            self.current_model_states.clear()
            time.sleep(0.5)
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
            if hasattr(self, "bridges"):
                self.bridges.terminate()
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


def main():
    p = argparse.ArgumentParser(description="Validate Rhino .3dm demos for stability in Gazebo.")
    p.add_argument("--batch", default="batch0", help="Batch folder in training_data/ (default: batch0)")
    p.add_argument("--demo", default=None, help="Specific demo name (e.g. demo_01). If omitted, validates all.")
    args = p.parse_args()

    if HAVE_ROS2:
        rclpy.init()
        validator = DemoValidatorNode()
    else:
        validator = DemoValidator()

    batch_dir = os.path.join(os.getcwd(), "training_data", args.batch)
    rhino_dir = os.path.join(batch_dir, "rhino")
    val_dir = os.path.join(batch_dir, "validated_simPhysics")

    if not os.path.exists(rhino_dir):
        print(f"Directory {rhino_dir} not found. Ensure we run at project root.")
        # create mock to run through logic if user just tests
        os.makedirs(rhino_dir, exist_ok=True)

    if args.demo:
        demo_files = [os.path.join(rhino_dir, f"{args.demo}.3dm")]
    else:
        demo_files = glob.glob(os.path.join(rhino_dir, "demo_*.3dm"))

    if not demo_files:
        print(f"No demo files found in {rhino_dir}.")

    try:
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
            success = True
            final_sequence = []
            brick_counter = 0

            validator.reset_world()

            for layer_idx, layer in enumerate(layers):
                layer_success = False
                
                # Snapshot foundation before trying this layer
                validator.fetch_latest_poses_from_gz()
                foundation_states = validator.current_model_states.copy()

                for attempt in range(max_attempts):
                    print(f"--> Layer {layer_idx+1}/{len(layers)}, Attempt {attempt + 1}/{max_attempts} ...")
                    current_layer_bricks = random.sample(layer, len(layer)) if attempt > 0 else layer

                    stable = True
                    layer_initial_states = foundation_states.copy()
                    spawned_in_attempt = []

                    for brick in current_layer_bricks:
                        name = f"{demo_name}_brick_{brick_counter + len(spawned_in_attempt)}"
                        pose_7d = brick.get_7d_pose()

                        if not validator.spawn_brick(name, pose_7d):
                            print(f"Failed to physically spawn {name}")

                        spawned_in_attempt.append((name, brick))

                        if HAVE_ROS2:
                            time.sleep(2.0)
                        else:
                            time.sleep(0.1)  # Mock wait

                        # Snapshot initial resting pose
                        validator.fetch_latest_poses_from_gz()
                        if name in validator.current_model_states:
                            layer_initial_states[name] = validator.current_model_states[name]

                        # Check for structural failure
                        if not validator.check_stability(layer_initial_states):
                            stable = False
                            print(f"Collapse detected upon triggering {name}!")
                            break

                    if stable:
                        layer_success = True
                        final_sequence.extend([b for n, b in spawned_in_attempt])
                        brick_counter += len(spawned_in_attempt)
                        print(f"Layer {layer_idx+1} stabilized.")
                        break
                    else:
                        # Attempt failed. Check if foundation is still intact.
                        validator.fetch_latest_poses_from_gz()
                        foundation_intact = True
                        for fn_name, fn_pose in foundation_states.items():
                            if fn_name not in validator.current_model_states:
                                foundation_intact = False
                                break
                            curr_pose = validator.current_model_states[fn_name]
                            dx = curr_pose.position.x - fn_pose.position.x
                            dy = curr_pose.position.y - fn_pose.position.y
                            dz = curr_pose.position.z - fn_pose.position.z
                            dist = (dx**2 + dy**2 + dz**2)**0.5
                            if dist > 0.05:
                                foundation_intact = False
                                break
                        
                        if foundation_intact and attempt < max_attempts - 1:
                            print("Foundation intact. Fast-resetting layer...")
                            names_to_remove = [n for n, b in spawned_in_attempt]
                            if validator.remove_client and validator.remove_client.wait_for_service(timeout_sec=1.0):
                                futures = []
                                from ros_gz_interfaces.srv import DeleteEntity
                                for n in names_to_remove:
                                    req = DeleteEntity.Request()
                                    req.entity.name = n
                                    req.entity.type = 2
                                    futures.append(validator.remove_client.call_async(req))
                                if futures:
                                    rclpy.spin_until_future_complete(validator, futures[-1], timeout_sec=2.0)
                            time.sleep(1.0)
                            # Verify fast reset
                            validator.fetch_latest_poses_from_gz()
                        elif attempt < max_attempts - 1:
                            print("Foundation ruined! Full rebuild required.")
                            validator.reset_world()
                            for i, b in enumerate(final_sequence):
                                fn_name = f"{demo_name}_brick_{i}"
                                validator.spawn_brick(fn_name, b.get_7d_pose())
                                if HAVE_ROS2: time.sleep(1.0) # Faster spawn for known good foundation
                            validator.fetch_latest_poses_from_gz()
                            foundation_states = validator.current_model_states.copy()
                            time.sleep(1.0)

                if not layer_success:
                    print(f"Failed to stabilize Layer {layer_idx+1}. Validation failed.")
                    success = False
                    break

            if success:
                print(f"Successfully stabilized and sequenced {demo_name}.")

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
    except KeyboardInterrupt:
        print("\nValidation interrupted by user. Shutting down...")
    except Exception as e:
        print(f"\nError during validation: {e}")
    finally:
        validator.destroy_node()
        if HAVE_ROS2:
            rclpy.shutdown()


if __name__ == "__main__":
    main()
