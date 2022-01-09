import cv2 as cv
import mediapipe as mp
import time


class face_detector:
    def __init__(self, detectionConf=0.5, selection=0):
        self.detection_conf = detectionConf
        self.model_selection = selection

        self.mpFaces = mp.solutions.face_detection
        self.face_detect = self.mpFaces.FaceDetection()
        self.mpDraw = mp.solutions.drawing_utils

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.face_m = self.mpFaceMesh.FaceMesh(max_num_faces=2)

    def find_face(self, img, draw_default=False):

        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.face_detect.process(imgRGB)
        if self.results.detections:
            for detect in enumerate(self.results.detections):
                if draw_default:
                    self.mpDraw.draw_detection(img, detect)

        return img

    def find_landmarks(self, img):

        list_1 = []
        list_2 = []

        if self.results.detections:
            for iD, detection in enumerate(self.results.detections):
                # print(iD, detection)
                keyPoints = detection.location_data.relative_keypoints
                boundingBox = detection.location_data.relative_bounding_box
                # print(iD, keyPoints)
                h, w, c = img.shape
                b_box = int(boundingBox.xmin * w), int(boundingBox.ymin * h), int(boundingBox.width * w), int(
                    boundingBox.height * h)
                list_1.append([id, b_box])

                for iD, lm in enumerate(keyPoints):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    list_2.append([iD, cx, cy])

                cv.putText(img, f'{int(detection.score[0] * 100)}', (b_box[0], b_box[1] - 20), cv.FONT_ITALIC, 1,
                           (0, 0, 255), 2)
                cv.rectangle(img, b_box, (0, 0, 255), 2)

        return img, list_1, list_2

    def face_mesh(self, img, draw=False):

        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        drawSpec = self.mpDraw.DrawingSpec((0, 255, 255), 1, 1)
        [cx, cy] = 0, 0

        results = self.face_m.process(imgRGB)
        if results.multi_face_landmarks:
            # print(results.multi_face_landmarks)
            for faceLms in results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS, drawSpec, drawSpec)
                for id, lm in enumerate(faceLms.landmark):
                    # print([lm.x, lm.y])
                    h, w, c = img.shape
                    [cx, cy] = int(lm.x * w), int(lm.y * h)
                    # print([id, cx, cy])
        return img, [cx, cy]


def main():
    pTime = 0

    capture = cv.VideoCapture(0)
    detector = face_detector()

    while True:
        success, img = capture.read()

        frame = detector.find_face(img)
        frame, boundary_box, landmarks = detector.find_landmarks(frame)
        frame, _ = detector.face_mesh(frame, draw=True)

        # print(landmarks)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv.putText(frame, str(int(fps)), (20, 80), cv.FONT_ITALIC, 2, (255, 255, 255), 2)
        cv.imshow("Frames", frame)
        cv.waitKey(1)


if __name__ == "__main__":
    main()
