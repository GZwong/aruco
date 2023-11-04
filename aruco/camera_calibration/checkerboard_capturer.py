"""
Created by: Gai Zhe

This script can be used to capture images of a checkerboard pattern. Only images where a checkerboard pattern is
detected can be taken by pressing the button "s". Press "q" to terminate the program. 
"""

# Standard Imports
import os
import time
from pathlib import Path

# Third-Party Imports
import cv2
import imutils
from imutils.video import VideoStream


# FUNCTIONS ------------------------------------------------------------------------------------------------------------
def image_dir():
    # If it not yet exist, create a directory "checkerboard_images" to store images
    image_dir_path = Path(Path(__file__).parent, "checkerboard_images")
    
    if not os.path.isdir(image_dir_path):
        os.makedirs(image_dir_path)
        print(f'"{image_dir_path}" Directory is created.')
    else:
        print(f'"{image_dir_path}" Directory already exists.')
        
    return image_dir_path

# MAIN SCRIPT ----------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    
    # Definitions
    CHESS_BOARD_DIM = (9, 6)  # The chessboard has 9x6 image points (where black edges meet)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare folder to store images
    image_dir_path = image_dir()
    n = 0  # image_counter

    # Start a VideoStream instance
    vs = VideoStream().start()
    time.sleep(2)  # Allow camera to warm up

    while True:
        
        # Obtain current frame. 
        frame = vs.read()         # Frame to be annotated
        copyframe = frame.copy()  # Frame to be saved
        
        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Attempt to find the chessboard. 
        #   Note 1: Use Grayscaled image for faster detection
        #   Note 2: This statement runs faster when a chessboard is detected
        board_detected, corners = cv2.findChessboardCorners(gray_frame, CHESS_BOARD_DIM)

        # If a chessboard is found, annotate chessboard pattern at the corners 
        if board_detected:
            print("Board Detected!")
            # Increase accuracy of corner detection using cornerSubPix
            corners = cv2.cornerSubPix(gray_frame, corners, (3, 3), (-1, -1), criteria)
            # Draw the chessboard pattern
            frame = cv2.drawChessboardCorners(frame, CHESS_BOARD_DIM, corners, board_detected)

        # Annotate the frame with number of saved images
        cv2.putText(
            frame,
            f"saved_img : {n}",
            (30, 40),
            cv2.FONT_HERSHEY_PLAIN,
            1.4,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        # Show the image for visual representation
        cv2.imshow("Annotated Frame", frame)
        cv2.imshow("Original Frame", copyframe)

        # Wait for one second for a key press
        key = cv2.waitKey(1)
        
        # Press "q" to end the program
        if key == ord("q"):
            break
        
        # Press "s" to take a snapshot
        if key == ord("s") and board_detected:
            # Save the image with checkerboard to 
            cv2.imwrite(f"{image_dir_path}/image{n}.png", copyframe)
            print(f"saved image number {n}")
            n += 1  # incrementing the image counter


    # Clean up - close windows and stop VideoStream
    cv2.destroyAllWindows()
    vs.stop()

    print("Total saved Images:", n)