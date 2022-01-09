import cv2 as cv
import mediapipe as mp
import time

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

capture = cv.VideoCapture(0)

while True:
    success, img = capture.read()
    img = cv.resize(img, (500, 400), interpolation=cv.INTER_AREA)
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    # print(results.pose_landmarks)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x*w), int(lm.y*h)
            print([id, cx, cy])

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv.putText(img, str(int(fps)), (40, 40), cv.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2 )
    cv.imshow("Frames", img)
    cv.waitKey(1)