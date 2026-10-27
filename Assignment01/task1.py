# Assignment 1 (Group 17)
# Import Libraries
import cv2
import matplotlib.pyplot as plt

# Function For Converting BGR Frames To Gray
def frames(cap: cv2.VideoCapture):
    counter = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret or frame is None:
            # Bonus: Skip corrupted frames and continue reading the next frames
            counter += 1
            print(f"Failed reads: {counter}")

            # Bonus: Terminate processing if the total failed reads reach 5 frames
            if counter >= 5:
                print('Error! Maximum failed reads reached')
                break
            continue

        # Convert each valid BGR frame to Grayscale
        yield cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Read Video
cap = cv2.VideoCapture('test.mp4')
if not cap.isOpened():
    print("Error! Failed To Open The Video")
else:
    try:
        for frame in frames(cap):
            cv2.imshow('Frame', frame)

            if cv2.waitKey(50) == ord('q'):
                break

    finally:
        # Release the video capture and close all OpenCV windows
        cap.release()
        cv2.destroyAllWindows()
