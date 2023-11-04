"""
Created by: Gai Zhe

This file be used in two ways:
1. A library of functions
        Contain functions:
            annotate_tags(image, markerID, topLeft, topRight, btmRight, btmLeft)

        -----
        Example Usage:
            from aruco_detector import annotate_tags

2. Run as a script.
        Takes input image and annotate the image with bounding boxes, centres and marker IDs, if they are found.
        Takes two arguments:
            --image (-i): Specify path to the input image
            --type (-t): Specify ArUco dictionary

        -----
        Example Usage:
            python aruco_detector.py --image example.png --type DICT_ARUCO_ORIGINAL
"""
# Standard Imports
import argparse
from pathlib import Path
from typing import Tuple

# Third-Party Imports
import numpy as np
import cv2

# Project-Specific Imports
from aruco.arucoDict import ARUCO_DICT


# FUNCTIONS ------------------------------------------------------------------------------------------------------------
def annotate_tags(image: np.ndarray, 
                  markerID: int, 
                  topLeft: Tuple[int, int], 
                  topRight: Tuple[int, int], 
                  btmRight: Tuple[int, int], 
                  btmLeft: Tuple[int, int]):

    """
    Draw bounding box, centre and marker ID on the input image.

    :param image:    The image to be drawn on
    :param markerID: The ID to be annotated beside the bounding box
    :param topLeft:  A tuple of the (X, Y) coordinates of the top-left corner
    :param topRight: A tuple of the (X, Y) coordinates of the top-right corner
    :param btmRight: A tuple of the (X, Y) coordinates of the bottom-right corner
    :param btmLeft:  A tuple of the (X, Y) coordinates of the bottom-left corner
    
    :return: The annotated image of the same size as image
    """

    # Draw bounding boxes
    cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
    cv2.line(image, topRight, btmRight, (0, 255, 0), 2)
    cv2.line(image, btmRight, btmLeft, (0, 255, 0), 2)
    cv2.line(image, btmLeft, topLeft, (0, 255, 0), 2)

    # Draw centre
    c_x = int(((topLeft[0]) + btmRight[0]) / 2)
    c_y = int(((topLeft[1]) + btmRight[1]) / 2)
    cv2.circle(image, (c_x, c_y), 4, (255, 0, 0), -1)

    # Draw AruCo marker ID on the image
    cv2.putText(image, str(markerID), (c_x, btmRight[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    return image


# WHEN RAN AS A SCRIPT -------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    # Get arguments
    arg = argparse.ArgumentParser()
    arg.add_argument("-i", "--image", type=str, default="example2.png", help="path to image containing ArUco marker")
    arg.add_argument("-t", "--type", type=str, default="DICT_6X6_50", help="type of ArUco marker to generate")
    args = vars(arg.parse_args())  # Convert argument to dictionary

    # Find the image
    # IMPORTANT: DO NOT GIVE AN IMAGE FULLY OCCUPIED BY THE MARKER ITSELF - IT CANNOT DETECT THE CORNERS
    image_path = Path(Path(__file__).parent, args["image"]).resolve()
    image = cv2.imread(filename=str(image_path))
    print(type(image))

    # Detect the image. Three items are returned:
    #   1) "corners" is a list containing x & y coordinates of detected ArUco markers
    #   2) "ids" is a list containing IDs of detected marker. None if no ID detected
    #   3) "rejected" is a list of potentially found but rejected markers. Useful for debugging.
    arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[args["type"]])  # Define what type of aruco markers to look for
    arucoParams = cv2.aruco.DetectorParameters_create()  # Use default parameters
    (corners, ids, rejected) = cv2.aruco.detectMarkers(image=image,
                                                       dictionary=arucoDict,
                                                       parameters=arucoParams)

    # If at least one marker is detected,
    if len(corners) > 0:

        # Print analytics
        ids = ids.flatten()
        print(f"Within the image of size {image.shape}:")
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

            # Annotate image with bounding box
            image = annotate_tags(image, markerID, topLeft, topRight, btmRight, btmLeft)

        print("Previewing image, waiting for input to terminate ...")
        cv2.imshow("Image", image)
        cv2.waitKey(0)
