# Assignment 2 (Team 17)
# Import libraries required for video processing and visualization
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load OpenCV's built-in Haar Cascade classifiers for face and eye detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml'
)

# Function to process the video stream frame by frame
def frames(cap: cv2.VideoCapture):
    counter = 0

    while cap.isOpened():
        # Read the next frame from the video stream
        ret, frame = cap.read()

        if not ret or frame is None:
            # Bonus: Skip unreadable frames and continue processing
            counter += 1
            print(f"Failed reads: {counter}")

            # Bonus: Stop processing if 5 frame reads fail
            if counter >= 5:
                print('Error! Maximum failed reads reached')
                break
            continue

        # Convert the current BGR frame to grayscale for Haar Cascade detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect face regions in the current frame
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # Process each detected face separately
        for i, (x, y, w, h) in enumerate(faces):

            # Draw a green rectangle around the detected face
            cv2.rectangle(
                frame, (x, y), (x + w, y + h), (0, 255, 0), 2
            )

            # Extract the grayscale region of interest (ROI) for this face
            # Eye detection will be performed only inside this face region
            roi_gray = gray[y:y + h, x:x + w]

            # Detect eyes inside the current face region using Haar Cascade
            eyes = eye_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.1,
                minNeighbors=8,
                minSize=(20, 15)
            )

            # Draw a blue rectangle around each detected eye
            # Add the face coordinates to convert eye coordinates
            # from the face ROI to the full video frame
            for i, (x_1, y_1, w_1, h_1) in enumerate(eyes):
                cv2.rectangle(
                    frame,
                    (x + x_1, y + y_1),
                    (x + x_1 + w_1, y + y_1 + h_1),
                    (255, 0, 0),
                    2
                )

        # Yield the processed frame with face and eye detections
        yield(frame)


# Open the video stream using VideoCapture
cap = cv2.VideoCapture('test_1.mp4')

if not cap.isOpened():
    print("Error! Failed To Open The Video")
else:
    try:
        # Process and display each frame in real time
        for frame in frames(cap):
            cv2.imshow('Frame', frame)

            # Press 'q' to stop video processing
            if cv2.waitKey(50) == ord('q'):
                break

    finally:
        # Release the video capture resource and close all OpenCV windows
        cap.release()
        cv2.destroyAllWindows()
