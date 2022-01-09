import cv2 as cv
import time
import HandTrackingModule as htm
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

#########################
wCam, hCam = 1080, 720
########################

pTime = 0

capture = cv.VideoCapture(0)
capture.set(3, wCam)
capture.set(4, hCam)

detector = htm.hand_detector()


devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
vol_range = volume.GetVolumeRange()
min_vol, max_vol = vol_range[0], vol_range[1]
volBar = 400

while True:
    success, img = capture.read()
    img = detector.find_hands(img)
    img, lm_list = detector.find_location(img, draw=False)
    if len(lm_list) != 0:
        # print(lm_list[4], lm_list[8])
        x1, y1 = lm_list[4][1], lm_list[4][2]
        x2, y2 = lm_list[8][1], lm_list[8][2]

        r = 10
        cv.circle(img, (x1, y1), r, (255, 255, 0), -1)
        cv.circle(img, (x2, y2), r, (255, 255, 0), -1)
        cv.line(img, (x1, y1), (x2, y2), (255, 255, 0), 3)
        cx, cy = int((x1+x2)/2), int((y1+y2)/2)
        cv.circle(img, (cx, cy), r, (255, 255, 0), -1)

        length = ((x2-x1)**2 + (y2-y1)**2)**(1/2)
        # print(length)
        if length<50 or length>160:
            cv.circle(img, (cx, cy), r, (255, 0, 255), -1)

        vol = np.interp(length, [50, 160], [min_vol, max_vol])
        volBar = np.interp(length, [50, 160], [400, 150])
        print(vol)

        volume.SetMasterVolumeLevel(vol, None)

    cv.rectangle(img, (40, 150), (70, 400), (255, 0, 0), 3)
    cv.rectangle(img, (40, int(volBar)), (70, 400), (255, 0, 0), -1)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv.putText(img, f'{int(fps)}/s', (20, 60), cv.FONT_ITALIC, 1, (255, 0, 0), 2)

    cv.imshow("Frames", img)
    cv.waitKey(1)

