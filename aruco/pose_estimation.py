# Standard Imports
import time
from pathlib import Path

# Third-Party Imports
import cv2
import numpy as np
import imutils
from imutils.video import VideoStream

# Project-Specific Imports
from aruco.arucoDict import ARUCO_DICT

# DEFINITIONS ----------------------------------------------------------------------------------------------------------
# Marker
MARKER_SIZE = 13.5  # Square size [mm] - allow for pose and distance estimation
arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT["DICT_6X6_50"])
arucoParams = cv2.aruco.DetectorParameters_create()  # Use default parameters

# LOAD CAMERA DATA -----------------------------------------------------------------------------------------------------
current_dir = Path(__file__).parent
data_path = Path(current_dir, "camera_calibration/MultiMatrix.npz").resolve()
print(f"Loading calibration data stored in {data_path}...\n\n")

data = np.load(data_path)
camMatrix = data["camMatrix"]
distCof = data["distCoef"]
rVector = data["rVector"]
tVector = data["tVector"]

print("Loaded calibration data successfully")

# FUNCTIONS ------------------------------------------------------------------------------------------------------------

vs = VideoStream().start()
time.sleep(2)  # Allow camera to warm up

while True:
    
    frame = vs.read()
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    (corners, ids, rejected) = cv2.aruco.detectMarkers(image=gray_frame,
                                                       dictionary=arucoDict,
                                                       parameters=arucoParams)

    # If checkerboard is detected
    if corners:
        rVec, tVec, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners=corners, markerLength=MARKER_SIZE, cameraMatrix=camMatrix, distCoeffs=distCof
        )

        total_markers = range(0, ids.size)

        # ??? Polylines - draw for every marker on screen

        for markerID, corner, i in zip(ids, corners, total_markers):

            topLeft, topRight, btmRight, btmLeft = corner.reshape((4, 2))

            # # Convert the coordinates to integers to be used by OpenCV (it is currently a 'numpy.float32')
            # topLeft = (int(topLeft[0]), int(topLeft[1]))
            # topRight = (int(topRight[0]), int(topRight[1]))
            # btmRight = (int(btmRight[0]), int(btmRight[1]))
            # btmLeft = (int(btmLeft[0]), int(btmLeft[1]))

            cv2.polylines(
                frame, [corner.astype(np.int32)], isClosed=True, color=(0, 255, 255), thickness=4, lineType=cv2.LINE_AA
            )

            # Estimate distance from that particular marker
            distance = np.sqrt(
                tVec[i][0][2] ** 2 + tVec[i][0][0] ** 2 + tVec[i][0][1] ** 2
            )

            # Annotate Pose
            cv2.drawFrameAxes(frame, camMatrix, distCof, rVec[i], tVec[i], length=4, thickness=4)

    cv2.imshow("Coloured Frame", frame)

    # Terminate program and cleanup when 'q' is pressed
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cv2.destroyAllWindows()
vs.stop()
