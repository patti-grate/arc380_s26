# as-built mesh construction from depth image and brick pose detection by comparing meshes/depth images.

# function that takes two depth images (2d & 3d) and retrieve a single brick pose.

# from depth point cloud images --> line up both frames and find the point cloud of the new block
# then fit the block to the point cloud (icp) --> say that one side will always be flat
# then extract pose and position from that block


# iterate to see if the block is reachable

# use joint constaints

# returns this all to the sequencer

# remembering function -- cases for recognizing it if the original structure is rotated + translated

# use depth image to see what is there --> turn it into a mesh

#starting with 2d:::
import numpy as np
import cv2
from cv2 import aruco
from matplotlib import pyplot as plt


# # Load the predefined dictionary where our markers are printed from
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)

# Load the default detector parameters
detector_params = aruco.DetectorParameters()

# Create an ArucoDetector using the dictionary and detector parameters
detector = aruco.ArucoDetector(dictionary, detector_params)

#load the webcam image
video = cv2.VideoCapture(1)


#adapted from "ArUCo-Markers-Pose-Estimation-Generation-Python" by GSNCodes
def aruco_display(corners, ids, rejected, image):
	if len(corners) > 0:
		# flatten the ArUco IDs list
		ids = ids.flatten()
		# loop over the detected ArUCo corners
		for (markerCorner, markerID) in zip(corners, ids):
			# extract the marker corners (which are always returned in
			# top-left, top-right, bottom-right, and bottom-left order)
			corners = markerCorner.reshape((4, 2))
			(topLeft, topRight, bottomRight, bottomLeft) = corners
			# convert each of the (x, y)-coordinate pairs to integers
			topRight = (int(topRight[0]), int(topRight[1]))
			bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
			bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
			topLeft = (int(topLeft[0]), int(topLeft[1]))

			cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
			cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
			cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
			cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)
			# compute and draw the center (x, y)-coordinates of the ArUco
			# marker
			cX = int((topLeft[0] + bottomRight[0]) / 2.0)
			cY = int((topLeft[1] + bottomRight[1]) / 2.0)
			cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
			# draw the ArUco marker ID on the image
			cv2.putText(image, str(markerID),(topLeft[0], topLeft[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
				0.5, (0, 255, 0), 2)
			print("[Inference] ArUco marker ID: {}".format(markerID))

			#draw a point for the center of the block here:

			

			# show the output image
	return image

def estimatePoseSingleMarkers(corners, marker_size, mtx, distortion):
    '''
    This will estimate the rvec and tvec for each of the marker corners detected by:
       corners, ids, rejectedImgPoints = detector.detectMarkers(image)
    corners - is an array of detected corners for each detected marker in the image
    marker_size - is the size of the detected markers
    mtx - is the camera matrix
    distortion - is the camera distortion matrix
    RETURN list of rvecs, tvecs, and trash (so that it corresponds to the old estimatePoseSingleMarkers())
    '''
    marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, -marker_size / 2, 0],
                              [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
    trash = []
    rvecs = []
    tvecs = []
    for c in corners:
        nada, R, t = cv2.solvePnP(marker_points, c, mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE)
        rvecs.append(R)
        tvecs.append(t)
        trash.append(nada)
    return rvecs, tvecs, trash

#ret, frame = video.read()
while True:
	ret, frame = video.read()

	if ret is False:
		break
	
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break

	h, w, _ = frame.shape

	width=1000
	height = int(width*(h/w))
	frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	cv2.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
	parameters = cv2.aruco.DetectorParameters()
	
	corners, ids, rejected = detector.detectMarkers(frame) #, cameraMatrix=matrix_coefficients, distCoeff=distortion_coefficients

	detected_markers = aruco_display(corners, ids, rejected, frame)
	if len(corners) > 0:
		for i in range(0, len(ids)):
			# Estimate pose of each marker and return the values rvec and tvec---(different from those of camera coefficients)
			# rvec, tvec, markerPoints = estimatePoseSingleMarkers(corners[i], 0.02)
			# print(rvec, tvec)
			print(i)
	cv2.imshow("Image", detected_markers)

	
cv2.destroyAllWindows()
video.release()

if ret:
	cv2.imshow("Captured", frame)      
	img_path = "captured_image.png"   
	cv2.imwrite(img_path, frame)  
	cv2.waitKey(0)                      
	img = cv2.imread(img_path)

	corners, ids, rejected = detector.detectMarkers(img)
	
	img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	markers_img = img_rgb.copy()
	aruco.drawDetectedMarkers(markers_img, corners, ids)
	aruco.drawDetectedMarkers(markers_img, rejected)

	plt.figure(figsize=(4,3))
	plt.imshow(markers_img)
	plt.title('Detected ArUco markers')
	plt.show()

	ids = ids.flatten()
	corners = np.array([corners[i] for i in np.argsort(ids)])
	corners = np.squeeze(corners)
	ids = np.sort(ids)   
	print(corners)
	height = 10
	width = 7.5
	ppi = 96

	widarucoin = 1.0

	src_pts = np.array([corners[0][0], corners[1][1], corners[2][2], corners[3][3]], dtype='float32')
	lengthofsidearuco = corners[0][0] - corners[1][1]
	zedmarker = (widarucoin * ppi)/lengthofsidearuco

	x_center = (corners[0][0] - corners[0][1] / 2) + corners[0][0]
	y_center = (corners[0][0] - corners[1][0] / 2) + corners[0][0]

	#now store this info as a quaternion /// pose, stored as meters  
	w = [x_center, y_center, zedmarker] 
	
	cv2.destroyWindow("Captured")   
		
else:
	print("Failed to capture image.")




# img_path = 'raw_image.png'
# img = cv2.imread(img_path)

# # plt.imshow(img)
# # plt.title('Loaded raw image')
# # plt.show()
# 








dst_pts = np.array([[0, 0], [0, height*ppi], [width*ppi, height*ppi], [width*ppi, 0]], dtype='float32')
# Compute the perspective transformation matrix
M = cv2.getPerspectiveTransform(src_pts, dst_pts)
#print(M)

# Apply the perspective transformation to the input image
# print(img_rgb.shape[1], img_rgb.shape[0])
corrected_img = cv2.warpPerspective(img, M, (img_rgb.shape[1], img_rgb.shape[0]))

# Crop the output image to the specified dimensions
corrected_img = corrected_img[:int(height*ppi), :int(width*ppi)]

#take those same points and compute the size of the new aruco images --> size transformation + pose extraction

#use edge detection to see if any of the block's edges are parallel with the aruco's sides
#then make sure that is aligned before using it




# plt.figure(figsize=(16,9))
# plt.imshow(markers_img)
# plt.title('Detected ArUco markers')
# plt.show()

