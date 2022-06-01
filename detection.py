import cv2
eye_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + '/haarcascade_eye.xml')
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + '/haarcascade_frontalface_default.xml')
car_classifier = cv2.CascadeClassifier(cv2.data.haarcascades+'/haarcascade_car.xml')
body_classifier = cv2.CascadeClassifier(cv2.data.haarcascades+'/haarcascade_fullbody.xml')
def face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.04, 8)
    if faces != ():
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (127, 0, 255), 2)
    return frame

def eye(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eye_classifier.detectMultiScale(frame, 1.1, 6)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (255, 255, 0), 2)
    return frame

def car(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = car_classifier.detectMultiScale(frame, 1.1, 6)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (255, 255, 0), 2)
    return frame

def body(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = body_classifier.detectMultiScale(frame, 1.1, 6)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (255, 255, 0), 2)
    return frame