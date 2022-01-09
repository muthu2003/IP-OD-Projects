import cv2 as cv
import numpy as np

# img = cv.imread('pictures/people.jpg')
# img = cv.resize(img, (400, 300), cv.INTER_AREA, interpolation=2)
# cv.imshow('People', img)

# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('GRAY', gray)

####################################################
haar_cascade = cv.CascadeClassifier('haar_face.xml')
#####################################################
# Be able to access many of Object Detection
# faces_rect = haar_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=7)

# print(f'No. of Faces = {len(faces_rect)}')

# for (x,y,w,h) in faces_rect:
# cv.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2)
# cv.imshow('Detected Faces', img)

capture = cv.VideoCapture(0)
while True:
    isTrue, frame = capture.read()
    faces_rect_2 = haar_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=7)
    # print(len(faces_rect_2))
    for (x, y, w, h) in faces_rect_2:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv.imshow('Detected Faces', frame)

    if cv.waitKey(20) & 0xFF == ord('d'):
        break

capture.release()
cv.waitKey(0)