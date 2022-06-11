import cv2
import numpy as np
import face_recognition
import time
def face(frame):
    face = face_recognition.face_locations(frame)[0]
    cv2.rectangle(frame, (face[3], face[0]),(face[1], face[2]), (255,0,255), 2)
    return frame
def faces(frame): 
    faces = face_recognition.face_locations(frame)
    for face in faces:
        cv2.rectangle(frame, (face[3], face[0]),(face[1], face[2]), (255,0,255), 2)
    return frame
cp = cv2.VideoCapture(0)
cp.set(cv2.CAP_PROP_FPS,60)
pTime = 0
while(True):
    i,frame=cp.read()
    if frame is not None:
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        frame=face(frame)
        cv2.putText(frame, f'FPS:{int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Frame", frame)
    q = cv2.waitKey(1)
    if q==ord("q"):
        break
cv2.destroyAllWindows()