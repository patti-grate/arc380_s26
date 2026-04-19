import sys; sys.path.insert(0, 'scripts')
import numpy as np
from scipy.spatial.transform import Rotation as R
from pose_conversion import Brick

labels = ['X', 'Y', 'Z']

print('=== State=0 (laying flat) ===')
b = Brick(); b.from_5d_pose([0.05, 0.35, 0.026, 0.0, 0.0])
p7 = b.get_7d_pose()
rot = R.from_quat(p7[3:]).as_matrix()
dots = np.abs(rot.T @ [0,0,1])
print(f'  Vertical axis: {labels[np.argmax(dots)]} (expect Z)')

print()
print('=== State=1 (standing) ===')
for yaw in [0.0, 1.134, 2.705]:
    b = Brick(); b.from_5d_pose([0.05, 0.35, 0.059, 1.0, yaw])
    p7 = b.get_7d_pose()
    rot = R.from_quat(p7[3:]).as_matrix()
    dots = np.abs(rot.T @ [0,0,1])
    print(f'  yaw={yaw:.3f}: Vertical axis={labels[np.argmax(dots)]} (expect X)  dots={np.round(dots,3)}')
