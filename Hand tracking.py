import cv2 as cv
import mediapipe as mp
import time

capture = cv.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    isTrue, frame = capture.read()
    imgRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                # print(id, lm)
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(id, cx, cy)
                if id == 8:
                    cv.circle(frame, (cx, cy), 10, (255, 0, 0), 2)
            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv.putText(frame, str(int(fps)), (20, 70), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
    cv.imshow('Image', frame)
    cv.waitKey(1)
    if 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows()
