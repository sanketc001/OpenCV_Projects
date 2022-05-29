import cv2
import numpy as np
import mediapipe as mp

NUM_FACE = 2

class FaceLandMarks():
    def _init_(self, staticMode=False,maxFace=NUM_FACE, minDetectionCon=0.5, minTrackCon=0.5):
        self.staticMode = staticMode
        self.maxFace =  maxFace
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh_connections
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode, self.maxFace, self.minDetectionCon, self.minTrackCon)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)

    def findFaceLandmark(self, img, draw=True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(self.mpFaceMesh())
        self.results = self.mpFaceMesh.FaceMesh(self.staticMode, self.maxFace, self.minDetectionCon, self.minTrackCon).faceMesh.process(self.imgRGB)

        faces = []

        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_TESSELATION, self.drawSpec, self.drawSpec)
                face = []
                for id, lm in enumerate(faceLms.landmark):
                    # print(lm)
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    #cv2.putText(img, str(id), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0), 1)
                    #print(id, x, y)
                    face.append([x,y])
                faces.append(face)
        return img, faces

def hand(img):
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(static_image_mode=False,
                          max_num_hands=2,
                          min_detection_confidence=0.5,
                          min_tracking_confidence=0.5)
    #mpDraw = mp.solutions.drawing_utils
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                # print(id,lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # if id ==0:
                cv2.circle(img, (cx, cy), 3, (255, 0, 255), cv2.FILLED)
            #mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
    return img
def pose(frame):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='Path to image or video. Skip to capture frames from camera')
    parser.add_argument('--thr', default=0.2, type=float, help='Threshold value for pose parts heat map')
    parser.add_argument('--width', default=368, type=int, help='Resize input to specific width.')
    parser.add_argument('--height', default=368, type=int, help='Resize input to specific height.')
    args = parser.parse_args()
    BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
                "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }
    POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
                ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
                ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
                ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
                ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]
    inWidth = args.width
    inHeight = args.height
    net = cv2.dnn.readNetFromTensorflow("graph_opt.pb")
    # cap = cv2.VideoCapture(args.input if args.input else 0)
    # while cv2.waitKey(1) < 0:
    #     hasFrame, frame = cap.read()
    #     if not hasFrame:
    #         cv2.waitKey()
    #         break
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements
    assert(len(BODY_PARTS) == out.shape[1])
    points = []
    for i in range(len(BODY_PARTS)):
           # Slice heatmap of corresponging body's part.
        heatMap = out[0, i, :, :]
            # Originally, we try to find all the local maximums. To simplify a sample
            # we just find a global one. However only a single pose at the same time
            # could be detected this way.
        _, conf, _, point = cv2.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        # Add a point if it's confidence is higher than threshold.
        points.append((int(x), int(y)) if conf > args.thr else None)
    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        assert(partFrom in BODY_PARTS)
        assert(partTo in BODY_PARTS)
        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]
        if points[idFrom] and points[idTo]:
            cv2.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv2.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
            cv2.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
    t, _ = net.getPerfProfile()
    return frame

if __name__ == '__main__':
    # detector = FaceLandMarks()
    # img = cv2.imread('rishab.jpg')
    # img, faces = detector.findFaceLandmark(img)
    # cv2.imshow(img)
    # url = "http://192.168.1.201:4747/video"
    import time
    import fullshortface
    short = False
    full = False
    import meshface
    mesh=False
    import segmentationblurselfie
    blur=False
    import posegraph3dhands
    pose=False
    import mediapipe_holistic
    #cp = cv2.VideoCapture(url)
    cp = cv2.VideoCapture(0)
    pTime = 0
    while(True):
        i,frame=cp.read()
        if frame is not None:
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            frame=mediapipe_holistic.holistic(frame)
            if full:
                frame = fullshortface.full(frame)
            if short:
                frame=fullshortface.short(frame)
            if pose:
                frame=posegraph3dhands.pose(frame)
            if mesh:
                frame=meshface.mesh(frame)
            if blur:
                frame=segmentationblurselfie.blur(frame)
            cv2.putText(frame, f'FPS:{int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Frame", frame)
        q = cv2.waitKey(1)
        if q==ord("1"):
            if pose:
                pose=False
            else:
                pose=True
        if q==ord("2"):
            if blur:
                blur=False
            else:
                blur=True
        if q==ord("3"):
            if mesh:
                mesh=False
            else:
                mesh=True
        if q==ord("4"):
            if short:
                short=False
            else:
                short=True
        if q==ord("5"):
            if full:
                full=False
            else:
                full=True
        if q==ord("q"):
            break
    cv2.destroyAllWindows()