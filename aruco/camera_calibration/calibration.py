"""
DESCRIPTION
-----------
This script finds the images taken in the directory "checkerboard_images" and produces an output matrix that calibrates
the camera with respect to the distortion produced.

BACKGROUND/THEORY
-----------------
The theory of camera calibration is presented at https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html.
In short, we want to find the distortion coefficients (5 parameters) to correct the distortion. 

To calibrate the camera, we need 
    1) OBJECT POINTS - a set of 3D real world points
    2) IMAGE POINTS  - its corresponding 2D coordinates of these points in the image.
    
Here we use a well-defined checkerboard pattern with known dimensions. We know the physical size of the squares, so
we know the actual object points of the checkerboard. For example:
    We first assume a frame of reference of the checkerboard - the flat checkerboard is the XY plane and z=0. Assuming
    a square size of 13.5mm, the object coordinates of the corners are then (0,0,0), (13.5,0,0), (27,0,0) and so on.

The image points are straightforward to obtain - there are merely the coordinates of the corners within the frame.
With knowledge of both object points and image points, we can find the distortion that caused inconsistencies between
these two sets of points.


Created by: Gai Zhe
"""   

# Standard Imports
import os
from pathlib import Path

# Third-Party Imports
import cv2
import numpy as np

# DEFINITIONS ----------------------------------------------------------------------------------------------------------
CHESS_BOARD_DIM = (9, 6)
SQUARE_SIZE = 13.5          # Physical square size [mm]

# Termination Criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# DEFINE FILES/FOLDERS -------------------------------------------------------------------------------------------------
current_dir = Path(__file__).parent

# Path to checkerboard images for calibration
image_dir = Path(current_dir, "checkerboard_images").resolve()


# FIND OBJECT POINTS ---------------------------------------------------------------------------------------------------
# Prepare a (9X3, 3) matrix. Each row represent a corner. Three columns represent their X, Y, Z coordinates
obj_3D = np.zeros((CHESS_BOARD_DIM[0] * CHESS_BOARD_DIM[1], 3), dtype=np.float32)
# Get the X and Y coordinates of each corner in terms of square size
#   1. Create a mesh grid
#   2. Transpose it
#   3. Squeeze it into two columns. The (-1) is a placeholder to automatically infer the number of rows
obj_3D[:, :2] = np.mgrid[0 : CHESS_BOARD_DIM[0], 0 : CHESS_BOARD_DIM[1]].T.reshape(-1, 2)
# Multiple by square size to get actual size
obj_3D *= SQUARE_SIZE
print(obj_3D)

# Arrays to store object points and image points from all the images.
obj_points_3D = []  # 3d point in real world space
img_points_2D = []  # 2d points in image plane.


# FIND IMAGE POINTS ---------------------------------------------------------------------------------------------------

# Find the set of image points for each image
for file in os.listdir(image_dir):
    image_path = Path(image_dir, file)
    print(f"Calibrating with image file '{file}'")

    image = cv2.imread(str(image_path))
    grayScale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    board_detected, corners = cv2.findChessboardCorners(image, CHESS_BOARD_DIM, None)
    if board_detected:
        # The object points are constant for all image taken
        obj_points_3D.append(obj_3D)

        # Refine the image points and append to img_points_2D
        corners = cv2.cornerSubPix(grayScale, corners, (3, 3), (-1, -1), criteria)
        img_points_2D.append(corners)

        img = cv2.drawChessboardCorners(image, CHESS_BOARD_DIM, corners, board_detected)


# CALIBRATION ----------------------------------------------------------------------------------------------------------
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    obj_points_3D, img_points_2D, grayScale.shape[::-1], None, None
)
print("Calibrated")

# SAVING DATA ----------------------------------------------------------------------------------------------------------
print("Saving camera matrix, distortion coefficeints, radial and tangential vectors as a 'npz' file")
np.savez(
    Path(current_dir, "MultiMatrix.npz"),
    camMatrix=mtx,
    distCoef=dist,
    rVector=rvecs,
    tVector=tvecs,
)

# LOADING DATA ---------------------------------------------------------------------------------------------------------
print("-------------------------------------------")
print("loading data stored using numpy savez function\n \n \n")

data = np.load(Path(current_dir, 'MultiMatrix.npz'))

camMatrix = data["camMatrix"]
distCof = data["distCoef"]
rVector = data["rVector"]
tVector = data["tVector"]

print(camMatrix)

print("Loaded calibration data successfully")
