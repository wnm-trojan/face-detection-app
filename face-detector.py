import cv2
from random import randrange

trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# choose an image to detect face in
#img = cv2.imread('assets/images/IMG_0388.jpg')
img = cv2.imread('assets/images/group.jpg')

# must convert to grayscale
grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#detect faces
face_coordinates = trained_face_data.detectMultiScale(grayscale_img)

# draw rectangles around the faces
#cv2.rectangle(img, (307, 115), (307+336, 115+336), (0, 255, 0), 2) #static
#(x, y, w, h) = face_coordinates[0] #single detection
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 256, 0), 2)
    #cv2.rectangle(img, (x, y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 2) #random color

#
cv2.imshow('WNM Face Detector', img)
#
cv2.waitKey()

print("Code Completed");