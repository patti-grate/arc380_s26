import numpy as np
import cv2
from cv2 import aruco
try:
    from matplotlib import pyplot as plt
    _HAVE_PLT = True
except Exception:
    _HAVE_PLT = False

import os
import json
import time
from typing import Optional

from aruco_info import aruco_corners
from camera import RealSenseCaptureServer, SHARED_DIR


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


# ── Detector setup (module-level, no camera involved) ──────────────────────
dict6x6 = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
detector_params = aruco.DetectorParameters()
detector6x6 = aruco.ArucoDetector(dict6x6, detector_params)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Output image dimensions (inches / ppi)
width = 10      # inches
height = 7.5    # inches
ppi = 96        # pixels per inch


def run_perception() -> Optional[dict]:
    """
    Capture one RealSense frame, detect the supply brick pose, and return a
    dict with keys ``supply_xyz`` and ``supply_quat_xyzw``, or None on failure.

    This is the canonical camera-to-pose pipeline.  Call this function directly
    rather than running the script as a subprocess so that the camera is opened
    and closed cleanly inside the same process.
    """
    # ── Camera capture ──────────────────────────────────────────────────────
    cam = RealSenseCaptureServer()
    try:
        cam.start()
        time.sleep(3)
        cap = cam.capture_frame()
        cam.write_response(1, cap[0], cap[1])
    finally:
        cam.stop()

    pathname = os.path.join(SHARED_DIR, "color.png")
    frame = cv2.imread(pathname)
    if frame is None:
        print(f"[perception] ERROR: could not read {pathname}")
        return None

    # ── ArUco detection ─────────────────────────────────────────────────────
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    corners, ids, rejected = detector6x6.detectMarkers(frame)
    if ids is None or len(ids) < 4:
        print(f"[perception] ERROR: expected 4 ArUco markers, found {0 if ids is None else len(ids)}")
        return None

    ids = ids.flatten()
    markers_img = img_rgb.copy()
    aruco.drawDetectedMarkers(markers_img, corners, ids)

    print(corners)

    # ── Perspective correction ──────────────────────────────────────────────
    corners = np.array([corners[i] for i in np.argsort(ids)])
    corners = np.squeeze(corners)
    ids = np.sort(ids)
    src_pts = np.array([corners[0][0], corners[1][1], corners[2][2], corners[3][3]], dtype='float32')
    dst_pts = np.array([[0, 0], [0, height*ppi], [width*ppi, height*ppi], [width*ppi, 0]], dtype='float32')

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    corrected_img = cv2.warpPerspective(markers_img, M, (img_rgb.shape[1], img_rgb.shape[0]))

    _ids = ids.flatten()

    corners = np.array([corners[i] for i in np.argsort(ids)])
    corners = np.squeeze(corners)
    ids = np.sort(ids)
    src_pts = np.array([corners[0][0], corners[1][1], corners[2][2], corners[3][3]], dtype='float32')
    dst_pts = np.array([[0, 0], [0, height*ppi], [width*ppi, height*ppi], [width*ppi, 0]], dtype='float32')

    markers_img = frame.copy()
    corrected_img = corrected_img[:int(height*ppi), :int(width*ppi)]
    corr_img_rgb = cv2.cvtColor(corrected_img, cv2.COLOR_BGR2RGB)

    # ── K-means segmentation ────────────────────────────────────────────────
    img_data = corrected_img.reshape((-1, 3))
    img_data = np.float32(img_data)

    k = 5
    kmeans_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(img_data, k, None, kmeans_criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)

    kmeans_data = centers[labels.flatten()]
    kmeans_img = kmeans_data.reshape(corrected_img.shape)
    labels = labels.reshape(corrected_img.shape[:2])

    block = np.array([0, 0, 255])
    distances = np.linalg.norm(centers - block, axis=1)
    block_cluster_label = np.argmin(distances)

    mask_img = np.zeros(kmeans_img.shape[:2], dtype='uint8')
    mask_img[labels == block_cluster_label] = 255

    # Exclude ArUco marker regions from the mask
    _aruco_margin = int(1.5 * ppi)
    _H, _W = mask_img.shape
    mask_img[:_aruco_margin,  :_aruco_margin]   = 0
    mask_img[:_aruco_margin,  _W-_aruco_margin:] = 0
    mask_img[_H-_aruco_margin:, :_aruco_margin]  = 0
    mask_img[_H-_aruco_margin:, _W-_aruco_margin:] = 0

    # ── Contour selection ───────────────────────────────────────────────────
    contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        print("[perception] ERROR: no contours found after segmentation")
        return None

    areas = [cv2.contourArea(c) for c in contours]
    expected_area = 2.0 * 0.80 * ppi**2
    closest_area_idx = np.argmin(np.abs(np.array(areas) - expected_area))
    selected_contour = contours[closest_area_idx]

    selected_contour_img = corrected_img.copy()
    cv2.drawContours(selected_contour_img, contours, -1, (0, 255, 0), 3)

    if _HAVE_PLT:
        try:
            plt.figure()
            plt.imshow(selected_contour_img)
            plt.show()
        except Exception:
            pass

    # ── minAreaRect → pose ──────────────────────────────────────────────────
    rect = cv2.minAreaRect(selected_contour)
    center = rect[0]
    center = (int(center[0]), int(center[1]))
    angle = rect[2]
    angle_rad = np.deg2rad(angle)

    # Image pixel (0,0) = marker 0 corner 0 (top-left of working area)
    x_origin = aruco_corners[1][3][0]
    y_origin = aruco_corners[1][3][1]

    x_span_m = aruco_corners[0][2][0] - aruco_corners[1][3][0]
    y_span_m = aruco_corners[3][1][1] - aruco_corners[0][2][1]

    x_m_per_px = x_span_m / (width * ppi)
    y_m_per_px = y_span_m / (height * ppi)

    px_col, px_row = rect[0]
    print(px_col, px_row)
    print(x_m_per_px, y_m_per_px)
    x_center = x_origin + px_col * x_m_per_px
    y_center = y_origin + px_row * y_m_per_px
    z = 0.030  # flat brick on table-top

    # minAreaRect angle is in image space; +pi/2 converts to world-frame yaw
    angle_rad_world = angle_rad + np.pi / 2
    block_orientation_tuple = quaternion_from_euler(0, 0, angle_rad_world)
    block_orientation_tuple = block_orientation_tuple / np.linalg.norm(block_orientation_tuple)
    block_position = [x_center, y_center, z]

    print(block_orientation_tuple)
    print(block_position)
    print(_ids)
    print(np.argsort(_ids))

    if _HAVE_PLT:
        try:
            plt.imshow(cv2.cvtColor(selected_contour_img, cv2.COLOR_BGR2RGB))
            plt.title('Pose of the blue square')
            plt.gca().invert_yaxis()
            plt.show()
        except Exception:
            pass

    return {
        "supply_xyz": block_position,
        "supply_quat_xyzw": list(block_orientation_tuple),
    }


if __name__ == "__main__":
    result = run_perception()
    if result is not None:
        export_path = os.path.join(SHARED_DIR, "supply.json")
        with open(export_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[perception] Exported to {export_path}")
    else:
        print("[perception] Detection failed — supply.json not written.")
