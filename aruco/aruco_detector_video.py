"""
Perform real-time detection using the camera
"""
# Standard Imports
import time

# Third-Party Imports
import cv2
import imutils
from imutils.video import VideoStream

# Project-Specific Imports
from aruco.arucoDict import ARUCO_DICT
from aruco.aruco_detector import annotate_tags


# DEFINE ARUCO DICTIONARY AND DETECTION PARAMETER ----------------------------------------------------------------------
arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT["DICT_6X6_50"])  # Define what type of aruco markers to look for
arucoParams = cv2.aruco.DetectorParameters_create()              # Use default parameters


# DETECT IMAGE IN VIDEO ------------------------------------------------------------------------------------------------
# Start a VideoStream instance
print("Starting video stream, warming up...")
vs = VideoStream().start()
time.sleep(2)  # Allow camera to warm up
print("Ready for input...")

# Loop over frames from video stream
while True:

    # Obtain the current frame
    frame = vs.read()
    frame = imutils.resize(frame, width=1000, height=1000)

    # Detect markers in the current frame
    start_time = time.time()
    (corners, ids, rejected) = cv2.aruco.detectMarkers(image=frame,
                                                       dictionary=arucoDict,
                                                       parameters=arucoParams)

    detection_time = time.time() - start_time
    print(f"Detection takes {detection_time * 1000} ms")

    # ANALYTICS --------------------------------------------------------------------------------------------------------
    # If at least one marker is detected,
    if len(corners) > 0:

        # Print analytics
        ids = ids.flatten()
        print(f"Within the image of size {frame.shape}:")
        print(f"    {len(ids)} tags are detected, with IDs {ids}.")
        print(f"    {len(rejected)} tags are rejected.")

        for (markerCorners, markerID) in zip(corners, ids):
            # Corner are always in the order: top-left, top-right, bottom-right, bottom-left
            topLeft, topRight, btmRight, btmLeft = markerCorners.reshape((4, 2))

            # Convert the coordinates to integers to be used by OpenCV (it is currently a 'numpy.float32')
            topLeft = (int(topLeft[0]), int(topLeft[1]))
            topRight = (int(topRight[0]), int(topRight[1]))
            btmRight = (int(btmRight[0]), int(btmRight[1]))
            btmLeft = (int(btmLeft[0]), int(btmLeft[1]))

            # Draw information onto the frame
            annotate_tags(frame, markerID, topLeft, topRight, btmRight, btmLeft)

        cv2.imshow("frame", frame)
        key = cv2.waitKey(1) & 0xFF  # Waits for a key event for 1ms, extract the least significant 8 bits of results

        # Break the loop if the key 'q' is pressed
        if key == ord('q'):  # return the integer representation (ASC-II) of q
            break

# Cleanup
cv2.destroyAllWindows()
vs.stop()
