import cv2
import sys
import numpy
import os

haar_file = 'haarcascade_frontalface_default.xml'

datasets = 'dataset'

print('Training classifier, this may take a few seconds')

(images, labels, names, id) = ([], [], {}, 0)

for (subdirs, dirs, files) in os.walk(datasets):
    for subdir in dirs:
      names[id] = subdir

      subjectpath = os.path.join(datasets, subdir)
      for filename in os.listdir(subjectpath):
            path = subjectpath + '/' + filename
            label = id
            images.append(cv2.imread(path, 0))
            labels.append(int(label))
      id += 1
(width, height) = (130, 100)

(images, labels) = [numpy.array(lists) for lists in [images, labels]]


model = cv2.face.LBPHFaceRecognizer_create()

model.train(images, labels)

face_cascade = cv2.CascadeClassifier(haar_file)
webcam = cv2.VideoCapture(0)
print('Classifier trained, now recognising faces.')

while True:
    (_, im) = webcam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
    

    
        cv2.rectangle(im,(x,y),(x + w,y + h),(0, 255, 255),2)
        face = gray[y:y + h, x:x + w]
        sample = cv2.resize(face, (width, height))

        recognized = model.predict(sample)

        if recognized[1] < 74:
            cv2.putText(im,'%s' % (names[recognized[0]].strip()),(x + 5, (y + 25) + h), cv2.FONT_HERSHEY_PLAIN,1.5,(20,185,20), 2)

            accuracy = (recognized[1]) if recognized[1] <= 100.0 else 100.0


        else:
            cv2.putText(im,'Unknown',(x + 5, (y + 25) + h), cv2.FONT_HERSHEY_PLAIN,1.5,(65,65, 255), 2)



    cv2.imshow('OpenCV Face Recognition -  esc to close', im)
    key = cv2.waitKey(10)

    if key == 27:
        break
