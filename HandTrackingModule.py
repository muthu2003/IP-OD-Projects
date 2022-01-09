import cv2 as cv
import mediapipe as mp
import time


class hand_detector:
    def __init__(self, mode=False, maxHands=2, modelComp=1,
                 detectionConf=0.5, trackingConf=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComp = modelComp
        self.detectionConf = detectionConf
        self.trackingConf = trackingConf

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComp,
                                        self.detectionConf, self.trackingConf)
        self.mpDraw = mp.solutions.drawing_utils

    def find_hands(self, img, draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(len(self.results.multi_hand_landmarks))
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img

    def find_location(self, img, handNo=0, drawId=12, draw=True):

        self.lm_list = []

        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[handNo]

            for iD, lm in enumerate(my_hand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cX, cY = int(lm.x * w), int(lm.y * h)

                self.lm_list.append([iD, cX, cY])

                if draw:
                    if iD == drawId:
                        cv.circle(img, (cX, cY), 10, (255, 0, 0), 2)
                    # if len(self.lm_list) != 0:
                    #     print(self.lm_list)
                        # vertex1 = ((self.lm_list[1][1]-30), max(self.lm_list[12][2], self.lm_list[8][2]-20))
                        # vertex2 = ((self.lm_list[20][1]+20), (self.lm_list[0][2]+20))
                        # cv.rectangle(img, vertex1, vertex2, (0, 0, 255), 2)

        return img, self.lm_list

    def fingers_up(self):
        fingers = []
        tipID = [8, 12, 16, 20]

        # Thumb
        if len(self.lm_list) != 0:
            if self.lm_list[4][1] > self.lm_list[20][1]:
                if self.lm_list[4][1] > self.lm_list[1][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            else:
                if self.lm_list[4][1] < self.lm_list[1][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            # Fingers
            for tip in tipID:
                if self.lm_list[tip][2] < self.lm_list[tip - 1][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)

        return fingers

    def find_distance(self, img, p1, p2, draw=True, radius=15, t=3):

        x1, y1 = self.lm_list[p1][1:]
        x2, y2 = self.lm_list[p2][1:]
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

        if draw:
            cv.circle(img, (x1, y1), radius, (255, 255, 0), -1)
            cv.circle(img, (x2, y2), radius, (255, 255, 0), -1)
            cv.line(img, (x1, y1), (x2, y2), (255, 255, 0), t)
            cv.circle(img, (cx, cy), radius, (255, 0, 0), -1)

        length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** (1 / 2)
        location = [x1, y1, x2, y2, cx, cy]

        return img, length, location


def main():
    pTime = 0

    capture = cv.VideoCapture(0)
    detector = hand_detector(maxHands=3)

    while True:
        success, img = capture.read()

        frame = detector.find_hands(img)
        frame, lmList = detector.find_location(frame)
        if len(lmList) != 0:
            print(lmList)
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


if __name__ == "__main__":
    main()
