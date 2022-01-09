import cv2 as cv
import time
import numpy as np
import HandTrackingModule as htm

#########################
wCam, hCam = 960, 540
wScr, hScr = 1366, 768     # autopy.screen.size()
########################

l = 20
pTime = 0

frameR = 100
smoothen = 7
xp, yp = 0, 0
xc, yc = 0, 0

capture = cv.VideoCapture(0)
capture.set(3, 1080)
capture.set(4, 720)

detector = htm.hand_detector(detectionConf=0.75)

while True:
    success, img = capture.read()
    img = detector.find_hands(img)
    img, lm_list = detector.find_location(img, draw=False)

    if len(lm_list) != 0:
        fingers = detector.fingers_up()

        x1, y1 = lm_list[8][1:]
        x2, y2 = lm_list[12][1:]

        vertex1 = (min(lm_list[20][1] - l, lm_list[18][1]-l), min(lm_list[12][2]-l, lm_list[8][2]-l, lm_list[10][2]-l))
        vertex2 = (max(lm_list[1][1]+l, lm_list[4][1]+l, lm_list[8][1]+l), (lm_list[0][2] + l))
        cv.rectangle(img, vertex2, vertex1, (0, 0, 255), 2)

        # print(img.shape)
        cv.rectangle(img, (frameR, frameR-20), ((wCam-frameR),(hCam-frameR)), (255, 0, 255), 2)

        if fingers[1] and fingers[2] == 0:
            # Moving Mode
            x3 = np.interp(x1, (frameR, wCam-frameR), (0, wScr))
            y3 = np.interp(y1, (frameR-20, hCam-frameR), (0, hScr))

            # smoothening values
            xc = xp + (x3 - xp)/smoothen
            yc = yp + (y3 - yp) / smoothen
            xp, yp = xc, yc

            # autopy.mouse.move(wScr-xc, yc)                   ## use (xc,yc) instead of (x3,y3)
            cv.circle(img, (x1, y1), 15, (255, 0, 255), -1)

        if fingers[1] and fingers[2] and fingers[3] == 0 and fingers[4] == 0:
            # Selection Mode
            img, dist, _ = detector.find_distance(img, 8, 12)
            print(dist)
            if dist < 40:
                cv.circle(img, (int((x1+x2)/2), int((y1+y2)/2)), 15, (0, 255, 0), -1)
                # autopy.mouse.click()

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv.putText(img, f'{int(fps)}/s', (20, 60), cv.FONT_ITALIC, 1, (255, 0, 0), 2)
    cv.imshow("Frames", img)
    cv.waitKey(1)
