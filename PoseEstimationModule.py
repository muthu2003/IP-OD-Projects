import cv2 as cv
import mediapipe as mp
import time


class pose_detector:
    def __init__(self, mode=False, complexity=1, smooth=True, segmentation=False,
                 detection_conf=0.5, tracking_conf=0.5):
        self.mode = mode
        self.complexity = complexity
        self.smooth = smooth
        self.segmentation = segmentation
        self.detection_conf = detection_conf
        self.tracking_conf = tracking_conf

        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose()
        self.mpDraw = mp.solutions.drawing_utils

    def draw_pose(self, img):

        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        # print(results.pose_landmarks)
        if self.results.pose_landmarks:
            self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img

    def find_pose_landmarks(self, img):

        lm_list = []
        for iD, lm in enumerate(self.results.pose_landmarks.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            # print([iD, cx, cy])
            lm_list.append([iD, cx, cy])

        return lm_list


def main():
    pTime = 0

    capture = cv.VideoCapture(0)
    detector = pose_detector()

    while True:
        success, img = capture.read()
        # img = cv.resize(img, (500, 400), interpolation=cv.INTER_AREA)

        frame = detector.draw_pose(img)
        lm_data = detector.find_pose_landmarks(img)
        print(lm_data[13])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv.putText(frame, str(int(fps)), (40, 40), cv.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)
        cv.imshow("Frames", frame)
        cv.waitKey(1)


if __name__ == "__main__":
    main()
