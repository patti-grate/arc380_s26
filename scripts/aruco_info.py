import numpy as np

# ArUco marker corners in the robot base frame (in meters)
# Corners are ordered clockwise starting from the top-left corner when facing the robot from the pendant side
aruco_corners: dict[int, np.ndarray] = {
    0: np.array([
        [-0.068,0.271,0.021],
        [-0.094,0.271,0.021],
        [-0.094,0.297,0.021],
        [-0.068,0.297,0.021],
    ]),
    1: np.array([
        [-0.233,0.271,0.021],
        [-0.259,0.271,0.021],
        [-0.259,0.297,0.021],
        [-0.233,0.297,0.021],
    ]),
    2: np.array([
        [-0.233,0.500,0.021],
        [-0.259,0.500,0.021],
        [-0.259,0.525,0.021],
        [-0.233,0.525,0.021],
    ]),
    3: np.array([
        [-0.068,0.500,0.021],
        [-0.094,0.500,0.021],
        [-0.094,0.525,0.021],
        [-0.068,0.525,0.021],
    ]),
}
