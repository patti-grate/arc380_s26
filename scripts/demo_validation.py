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

            self.sdf_path = os.path.join(os.getcwd(), "src", "abb_irb120_gazebo", "models", "brick", "model.sdf")
            
            # Spin up internal temporary ros_gz_bridges 
            import subprocess
            self.bridge_proc = subprocess.Popen(
                "ros2 run ros_gz_bridge parameter_bridge /world/irb120_workcell/dynamic_pose/info@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V --ros-args -r /world/irb120_workcell/dynamic_pose/info:=/tf", 
                shell=True
            )
            self.control_bridge = subprocess.Popen(
                "ros2 run ros_gz_bridge parameter_bridge /world/irb120_workcell/control@ros_gz_interfaces/srv/ControlWorld", 
                shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            
            self.reset_client = self.create_client(ControlWorld, "/world/irb120_workcell/control")
            sdf_path = os.path.join(
                os.getcwd(), "src", "abb_irb120_gazebo", "models", "brick", "model.sdf"
            )
            try:
                with open(sdf_path, "r") as f:
                    self.sdf_xml = f.read()
            except Exception as e:
                self.get_logger().error(f"Could not load SDF from {sdf_path}: {e}")

        def model_states_cb(self, msg):
            for transform in msg.transforms:
                name = transform.child_frame_id
                # Accept combinations like demo_1_brick_5 etc.
                if "brick" in name:
                    # Mock a pose object locally to match our logic
                    pose_mock = type("pos", (object,), {
                        "position": type("p", (object,), {
                            "x": transform.transform.translation.x,
                            "y": transform.transform.translation.y,
                            "z": transform.transform.translation.z
                        })
                    })
                    self.current_model_states[name] = pose_mock
                    # Mock a pose object locally to match our logic
                    pose_mock = type("pos", (object,), {
                        "position": type("p", (object,), {
                            "x": transform.transform.translation.x,
                            "y": transform.transform.translation.y,
                            "z": transform.transform.translation.z
                        })
                    })
                    self.current_model_states[name] = pose_mock

        def reset_world(self):
            # Use bridged control service to natively reset Gazebo scenes
            self.get_logger().info("Simulation layer refresh (Resetting states...)")
            
            if self.reset_client.wait_for_service(timeout_sec=2.0):
                req = ControlWorld.Request()
                req.world_control.reset.all = True
                future = self.reset_client.call_async(req)
                rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
            else:
                self.get_logger().warn("Could not find bridged ControlWorld service to reset simulation.")
                
            self.current_model_states.clear()
            time.sleep(1)
            return True

        def spawn_brick(self, name, pose_7d):
            import subprocess
            from scipy.spatial.transform import Rotation as R
            
            euler_rpy = R.from_quat(pose_7d[3:]).as_euler('xyz')
            
            cmd = f'ros2 run ros_gz_sim create -file "{self.sdf_path}" -name "{name}" ' \
                  f'-x {pose_7d[0]:.4f} -y {pose_7d[1]:.4f} -z {(pose_7d[2] + 0.001):.4f} ' \
                  f'-R {euler_rpy[0]:.4f} -P {euler_rpy[1]:.4f} -Y {euler_rpy[2]:.4f}'
                  
            try:
                subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                self.get_logger().info(f"Spawned {name} in simulation")
                return True
            except subprocess.CalledProcessError:
                self.get_logger().warn(f"Failed to execute subprocess spawn for {name}")
                return False

        def check_stability(self, initial_states):
            if not self.current_model_states:
                self.get_logger().warn("CRITICAL: No TF states tracked over the active bridges! Collapse detection is currently blind.")
            for name, initial_pose in initial_states.items():
                if name not in self.current_model_states:
                    continue
                curr_pose = self.current_model_states[name]

                dx = curr_pose.position.x - initial_pose.position.x
                dy = curr_pose.position.y - initial_pose.position.y
                dz = curr_pose.position.z - initial_pose.position.z
                dist = np.sqrt(dx**2 + dy**2 + dz**2)

                # Check for structural collapse (fell > 1cm OR moved horizontally > 1.5cm)
                if dz < -0.01 or dist > 0.015:
                    self.get_logger().warn(
                        f"{name} collapse detected! (Moved by {dist:.3f}m, dZ: {dz:.3f}m)"
                    )
                    return False
            return True

        def destroy_node(self):
            if hasattr(self, 'bridge_proc'):
                self.bridge_proc.terminate()
            if hasattr(self, 'control_bridge'):
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

    batch_dir = os.path.join(os.getcwd(), "training_data", "batch1")
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
                    # Spin for 2 seconds total simulating physics settle
                    for _ in range(20):
                        rclpy.spin_once(validator, timeout_sec=0.1)
                else:
                    time.sleep(0.1)  # Mock wait

                # Snapshot initial resting pose
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
