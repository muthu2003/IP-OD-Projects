import cv2 as cv
import time
import HandTrackingModule as htm

#########################
wCam, hCam = 1080, 720
########################

pTime = 0

capture = cv.VideoCapture(0)
capture.set(3, wCam)
capture.set(4, hCam)

detector = htm.hand_detector(detectionConf=0.75)
tipID = [8, 12, 16, 20]

while True:
    success, img = capture.read()
    img = detector.find_hands(img)
    img, lm_list = detector.find_location(img, draw=False)

    count = 0

    if len(lm_list) != 0:
        if lm_list[4][1] > lm_list[20][1]:
            if lm_list[4][1] > lm_list[1][1]:
                count += 1
        else:
            if lm_list[4][1] < lm_list[1][1]:
                count += 1

        for tip in tipID:
            if lm_list[tip][2] < lm_list[tip-1][2]:
                count += 1
        # print(count)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv.putText(img, str(count), (20, 120), cv.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 3)
    cv.putText(img, f'{int(fps)}/s', (20, 60), cv.FONT_ITALIC, 1, (255, 0, 0), 2)
    cv.imshow("Frames", img)
    cv.waitKey(1)