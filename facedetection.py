import cv2
import numpy as np
import time

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades+'/haarcascade_frontalface_default.xml')
cp = cv2.VideoCapture(0)
pTime = 0
while(True):
    i,frame=cp.read()
    if frame is not None:
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.0485258, 6)
        if faces == ():
            print("No faces found")
        else:
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (127, 0, 255), 2)
                cv2.imshow('Face Detection', frame)
                cv2.waitKey(0)
        cv2.destroyAllWindows()