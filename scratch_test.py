import subprocess
import time

print("--- ROS2 TOPIC LIST ---")
print(subprocess.run('ros2 topic list', shell=True, capture_output=True, text=True).stdout)

print("\n--- LAUNCHING BRIDGE ---")
bridge = subprocess.Popen('ros2 run ros_gz_bridge parameter_bridge /world/empty/dynamic_pose/info@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V --ros-args -r /world/empty/dynamic_pose/info:=/tf', shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

time.sleep(3)
print("\n--- ROS2 TOPIC LIST (WITH BRIDGE) ---")
print(subprocess.run('ros2 topic list', shell=True, capture_output=True, text=True).stdout)

bridge.terminate()
