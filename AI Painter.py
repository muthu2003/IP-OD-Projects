import cv2 as cv
import time
import os
import numpy as np
import HandTrackingModule as htm

#########################
wCam, hCam = 1280, 720
########################

pTime = 0
color = (0, 255, 255)
thicky = 15

blank = np.zeros((720, 1280, 3), np.uint8)

capture = cv.VideoCapture(0)
capture.set(3, wCam)
capture.set(4, hCam)

detector = htm.hand_detector()

# newList = []
# folder_path = "templates"
# myList = os.listdir(folder_path)
# print(myList)

# for img_path in myList:
#     image = cv.imread(f'{folder_path}/{img_path}')
#     newList.append(image)
# print(len(newList))
# print(newList)
# header = newList[0]

while True:
    success, img = capture.read()
    img = cv.flip(img, 1)
    img = detector.find_hands(img)
    img, lm_list = detector.find_location(img, draw=False)

    r = 70
    x, y = 200, 80
    blue = cv.circle(img, (x, y), r, (0, 255, 255), -1)
    green = cv.circle(img, (x+300, y), r, (0, 255, 0), -1)
    red = cv.circle(img, (x+600, y), r, (0, 0, 255), -1)
    erase = cv.circle(img, (x+900, y), r, (0, 0, 0), 3)

    # h, w, c = header.shape
    # img[0:h, 0:w] = header

    if len(lm_list) != 0:
        x1, y1 = lm_list[8][1:]
        x2, y2 = lm_list[12][1:]

        fingers = detector.fingers_up()
        # print(fingers)
        cx, cy = int((x1+x2)/2), int((y1+y2)/2)

        if fingers[1] and fingers[2]:                 # Anyway just the location of necessary will be drawing
            print('Selection Mode')
            xp, yp = 0, 0
            cv.circle(img, (cx, cy), 30, color, -1)

            if y-r < cy < y+r:
                if x-r < cx < x+r:
                    color = (255, 0, 0)     # blue
                if x+300-r < cx < x+300+r:
                    color = (0, 255, 0)     # green
                if x+600-r < cx < x+600+r:
                    color = (0, 0, 255)     # red
                if x+900-r < cx < x+900+r:
                    color = (0, 0, 0)     # erase

        if fingers[1] and fingers[2] == False:
            print('drawing mode')
            cv.circle(img, (x1, y1), 15, color, -1)
            # Drawing Method
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if color == (0, 0, 0):
                thicky = 50
            else:
                thicky = 15

            cv.line(blank, (xp, yp), (x1, y1), color, thicky)
            cv.line(img, (xp, yp), (x1, y1), color, thicky)
            xp, yp = x1, y1

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv.putText(img, f'{int(fps)}/s', (20, 60), cv.FONT_ITALIC, 1, (255, 0, 0), 2)

    # blank = cv.bitwise_not(blank)               ## because it's not in grayscale
    blankGray = cv.cvtColor(blank, cv.COLOR_BGR2GRAY)
    # blank = cv.bitwise_not(blankGray)
    _, invert = cv.threshold(blankGray, 50, 255, cv.THRESH_BINARY_INV)
    imgInv = cv.cvtColor(invert, cv.COLOR_GRAY2BGR)
    # To avoid those frame dancing
    img = cv.bitwise_and(img, imgInv)             # doesn't work properly for some colors
    img = cv.bitwise_or(img, blank)               # to add the one in blank exactly on img frames

    # img = cv.addWeighted(img, 0.5, blank, 0.5, 0)    # to blend the frames
    cv.imshow("Frames", img)
    # cv.imshow("Canvas", blank)
    if cv.waitKey(1) and 0xFF == ord('q'):
        break

capture.release()
cv.destroyAllWindows()
