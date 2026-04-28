import numpy as np
import cv2
from cv2 import aruco
from matplotlib import pyplot as plt
#import rclpy
#from rclpy import geometry_msgs
import numpy as np 
#from geometry_msgs.msg import Pose, PoseStamped
#from geometry_msgs.msg import Quaternion
from aruco_info import aruco_corners
#import math
from camera import RealSenseCaptureServer
from camera import SHARED_DIR
import os
import json

export_dir = SHARED_DIR


    
def quaternion_from_euler(ai, aj, ak):
    ai /= 2.0
    aj /= 2.0
    ak /= 2.0
    ci = np.cos(ai)
    si = np.sin(ai)
    cj = np.cos(aj)
    sj = np.sin(aj)
    ck = np.cos(ak)
    sk = np.sin(ak)
    cc = ci*ck
    cs = ci*sk
    sc = si*ck
    ss = si*sk

    q = np.empty((4, ))
    q[0] = cj*sc - sj*cs
    q[1] = cj*ss + sj*cc
    q[2] = cj*cs - sj*sc
    q[3] = cj*cc + sj*ss

    return q


dict6x6 = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
# dict4x4 = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)

detector_params = aruco.DetectorParameters()

detector6x6 = aruco.ArucoDetector(dict6x6, detector_params)
# detector4x4 = aruco.ArucoDetector(dict4x4, detector_params)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# idea is to use the printed sheet with aruco codes (6x6) and then create the 4x4 tags 
# which will need to be larger 

#then you can 

# Define the dimensions of the output image
width = 10      # inches
height = 7.5    # inches
ppi = 96        # pixels per inch
inches2metres = 0.0254 

#calib to the four corners 

#get the image from a server call instead 
#video = cv2.VideoCapture(1)
node = RealSenseCaptureServer()
node.start()
cap = RealSenseCaptureServer.capture_frame(node)
node.write_response(1, cap[0], cap[1])
pathname = os.path.join(SHARED_DIR, "color.png")
frame = cv2.imread(pathname)




#frame to RGB
img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
corners, ids, rejected = detector6x6.detectMarkers(frame)
ids = ids.flatten()
markers_img = img_rgb.copy()
aruco.drawDetectedMarkers(markers_img, corners, ids)

print(corners)


#process the frame's corners
corners = np.array([corners[i] for i in np.argsort(ids)])
corners = np.squeeze(corners)
ids = np.sort(ids)
src_pts = np.array([corners[0][0], corners[1][1], corners[2][2], corners[3][3]], dtype='float32')
dst_pts = np.array([[0, 0], [0, height*ppi], [width*ppi, height*ppi], [width*ppi, 0]], dtype='float32')


M = cv2.getPerspectiveTransform(src_pts, dst_pts)

# Apply the perspective transformation to the input image
# print(img_rgb.shape[1])
corrected_img = cv2.warpPerspective(markers_img, M, (img_rgb.shape[1], img_rgb.shape[0]))


_ids = ids.flatten()

corners = np.array([corners[i] for i in np.argsort(ids)])
corners = np.squeeze(corners)
ids = np.sort(ids)
src_pts = np.array([corners[0][0], corners[1][1], corners[2][2], corners[3][3]], dtype='float32')
dst_pts = np.array([[0, 0], [0, height*ppi], [width*ppi, height*ppi], [width*ppi, 0]], dtype='float32')

#cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if rgb is wanted
markers_img = frame.copy()
#aruco.drawDetectedMarkers(markers_img, corners, ids)
# Crop the output image to the specified dimensions
corrected_img = corrected_img[:int(height*ppi), :int(width*ppi)]

corr_img_rgb = cv2.cvtColor(corrected_img, cv2.COLOR_BGR2RGB)

#k-means
img_data = corrected_img.reshape((-1, 3))
img_data = np.float32(img_data)

k = 5 # maybe fix 

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

_, labels, centers = cv2.kmeans(img_data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
centers = np.uint8(centers)

# Rebuild the image using the labels and centers
kmeans_data = centers[labels.flatten()]
kmeans_img = kmeans_data.reshape(corrected_img.shape)
labels = labels.reshape(corrected_img.shape[:2])
# print(centers)

color_centers = []
# color_centers = img_data[centers[0], centers[1]]
# print(centers)

block = np.array([54,59,29])
distances = np.linalg.norm(centers - block, axis=1)
block_cluster_label = np.argmin(distances)

mask_img = np.zeros(kmeans_img.shape[:2], dtype='uint8')
mask_img[labels == block_cluster_label] = 255


contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
areas = [cv2.contourArea(contour) for contour in contours]
expected_area = 1*2 * ppi**2

perimeters = [cv2.arcLength(contour, closed=True) for contour in contours]

closest_area_idx = np.argmin(np.abs(np.array(areas) - expected_area))

selected_contour_img = corrected_img.copy()

selected_contour = contours[closest_area_idx]


# Get the center using a bounding box
# x, y, w, h = cv2.boundingRect(selected_contour)
# u_c = x + w//2
# v_c = y + h//2
cv2.drawContours(selected_contour_img, contours, -1, (0,255,0), 3)


plt.figure()
plt.imshow(selected_contour_img)
plt.show()

rect = cv2.minAreaRect(selected_contour) # minAreaRect returns a Box2D structure. A Box2D structure is a tuple of ((x, y), (w, h), angle).
center = rect[0]
center = (int(center[0]), int(center[1])) # Convert to integer
angle = rect[2]
angle_rad = np.deg2rad(angle) 
#length = 50

# end_x = (int(center[0] + length * np.cos(angle_rad)), int(center[1] + length * np.sin(angle_rad)))
# end_y = (int(center[0] + length * np.cos(angle_rad + np.pi/2)), int(center[1] + length * np.sin(angle_rad + np.pi/2)))
#7.25in + y
#2in + x

# Coordinate mapping (verified against physical setup):
#   Image +col (rightward)  → World +Y direction
#   Image +row (downward)   → World −X direction
#
# Origin in robot frame is marker 0, corner 0.
# Scale factors derived from the span between opposite markers.
x_origin = aruco_corners[0][0][0]  # robot X at image row=0  (marker 0 TL)
y_origin = aruco_corners[0][0][1]  # robot Y at image col=0  (marker 0 TL)

# World-X span is driven by image rows (marker1 is bottom-left → larger row)
# World-Y span is driven by image cols (marker3 is top-right → larger col)
x_span_m = aruco_corners[1][0][0] - aruco_corners[0][0][0]  # marker1_TL.x - marker0_TL.x
y_span_m = aruco_corners[3][0][1] - aruco_corners[0][0][1]  # marker3_TL.y - marker0_TL.y

# px_col increases → world +Y; px_row increases → world -X (hence the negation)
x_m_per_px = -x_span_m / (height * ppi)   # row↑ = X↓, so negate
y_m_per_px =  y_span_m / (width  * ppi)   # col→ = Y+

# rect[0] = (pixel_col, pixel_row) of the brick centre
px_col, px_row = rect[0]
x_center = x_origin + px_row * x_m_per_px   # row drives world X
y_center = y_origin + px_col * y_m_per_px   # col drives world Y
z = 0.030  # flat brick on table-top (matches REAL_SUPPLY_Z in construct_using_validated.py)

# Brick orientation: minAreaRect angle is in image space (X=col, Y=row).
# Rotating by +pi/2 converts from image-frame angle to world-frame yaw
# (because image +col = world +Y, i.e. image frame is 90° CCW from world frame).
angle_rad_world = angle_rad + np.pi / 2


# Brick orientation: rotation about Z-axis only (angle already converted to world frame)
block_orientation_tuple = quaternion_from_euler(0, 0, angle_rad_world)
block_orientation_tuple = block_orientation_tuple / np.linalg.norm(block_orientation_tuple)
block_position = [x_center, y_center, z]
print(block_orientation_tuple)
print(block_position)

plt.imshow(cv2.cvtColor(selected_contour_img, cv2.COLOR_BGR2RGB))
plt.title(f'Pose of the blue square')
plt.gca().invert_yaxis()
plt.show()

print(_ids)
print(np.argsort(_ids))


if export_dir:
    # os.makedirs(export_dir, exist_ok=True)
    # Use demo name derived from the data path or a generic timestamp
    export_path = os.path.join(export_dir, "supply.json")
    payload = {
        "supply_xyz": list(block_position),
        "supply_quat_xyzw": list(block_orientation_tuple),
    }
    with open(export_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[construct] Trajectories exported to {export_path}")
elif export_dir is not None:
    print("[construct] Export skipped: construction was not fully successful.")

