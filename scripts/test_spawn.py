import subprocess
sdf_path = "/ros2_ws/src/abb_irb120_gazebo/models/brick/model.sdf"
name = "test_brick"
x, y, z = 0.2, 0.1, 0.03
qx, qy, qz, qw = 0, 0, 0, 1

req_str = f'sdf_filename: "{sdf_path}" name: "{name}" pose: {{ position: {{x: {x} y: {y} z: {z}}} orientation: {{x: {qx} y: {qy} z: {qz} w: {qw}}} }}'
cmd = f"gz service -s /world/irb120_workcell/create --reqtype gz.msgs.EntityFactory --reptype gz.msgs.Boolean --timeout 2000 --req '{req_str}'"
print("Command:", cmd)
res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
print('STDOUT:', res.stdout)
print('STDERR:', res.stderr)
