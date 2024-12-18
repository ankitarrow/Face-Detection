import cv2
import numpy
import sys
import os
import time
detector = 'haarcascade_frontalface_default.xml'
datasets = 'dataset'

subset = input("Enter name of person: ")
path = os.path.join(datasets, subset)

if not os.path.isdir(path):
    os.mkdir(path)


(width, height) = (130, 100)

FacesCascade = cv2.CascadeClassifier(detector)
webcam = cv2.VideoCapture(0)

c = 1
print("Taking pictures, change face angles")
while c < 51:

    retur, image = webcam.read()

    if retur == True:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = FacesCascade.detectMultiScale(gray, 1.3, 4)
        for (x,y,w,h) in faces:
            cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
            face = gray[y:y + h, x:x + w]

            face_resize = cv2.resize(face, (width, height))
            cv2.imwrite('%s/%s.png' % (path,c), face_resize)
        c += 1

        cv2.imshow('OpenCV', image)
        key = cv2.waitKey(20)

        if key == 27:
            break
print("Subset created.")
webcam.release()
cv2.destroyAllWindows()