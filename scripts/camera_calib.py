import numpy as np
import cv2 as cv

 
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
 
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((9*6,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
 
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
 
# images = glob.glob('arc380_s26/snapshot_testing/calib.jpg')


fname = "/Users/gratepatrick/dev/arcgroup5/arc380_s26/scripts/calib.jpg"
img = cv.imread(fname)

# h, w, _ = img.shape

# width=1000
# height = int(width*(h/w))
# frame = cv.resize(img, (width, height), interpolation=cv.INTER_CUBIC)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

cv.imshow('calib', gray)

# Find the chess board corners
ret, corners = cv.findChessboardCorners(gray, (9,6), None)


# If found, add object points, image points (after refining them)
if ret == True:
    objpoints.append(objp)

    corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
    imgpoints.append(corners2)

    # Draw and display the corners
    cv.drawChessboardCorners(img, (9,6), corners2, ret)
    # cv.imshow('img', img)
    # cv.waitKey(0)
    # cv.destroyAllWindows() 
    # img_path = "drawnmarkers.png"   
    # cv.imwrite(img_path, img)
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print(ret, mtx, dist, rvecs, tvecs)
    key = cv.waitKey(1) & 0xFF

    if key == ord("q"):
        cv.destroyAllWindows()

else:
    print('didnt work')

cv.destroyAllWindows()