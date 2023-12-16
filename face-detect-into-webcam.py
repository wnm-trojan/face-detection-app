import cv2
from random import randrange

trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# to capture video from webcam
# webcam = cv2.VideoCapture('assets/videos/video1.mp4')
webcam = cv2.VideoCapture(0) # parameter 0 is webcam

# Iterate forever over frames
while True:

    #### Read the current frame
    successful_frame_raed, frame = webcam.read()

    # must convert to grayscale
    grayscale_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscale_img)

    # draw rectangles around the faces
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 256, 0), 2)

    #
    cv2.imshow('WNM Face Detector', frame)
    #
    key = cv2.waitKey(1)

    #### Stop if Q key is pressed
    if key == 18  or key == 113:
        break

#### Release the VideoCapture object
webcam.release()



print("Code Completed");