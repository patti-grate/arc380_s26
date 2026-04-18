import subprocess
msg = "ns: 'val', id: 1, action: ADD_MODIFY, type: TEXT, text: 'COLLAPSE DETECTED!', pose: {position: {z: 0.5}}, scale: {x: 0.1, y: 0.1, z: 0.1}, material: {ambient: {r: 1.0, g: 0.0, b: 0.0, a: 1.0}}"
cmd = f"gz topic -t /marker -m gz.msgs.Marker -p \"{msg}\""
res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
print('STDOUT:', res.stdout)
print('STDERR:', res.stderr)
