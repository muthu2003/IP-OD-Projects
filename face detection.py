import cv2 as cv
import mediapipe as mp
import time

mpFaces = mp.solutions.face_detection
face_detect = mpFaces.FaceDetection(model_selection=1)
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

capture = cv.VideoCapture('videos/pose.mp4')

while True:
    success, img = capture.read()

    img = cv.resize(img, (600, 500), interpolation=cv.INTER_AREA)

    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    results = face_detect.process(imgRGB)
    if results.detections:
        for id, detection in enumerate(results.detections):
            print(id, detection)
            keyPoints = detection.location_data.relative_keypoints
            boundingBox = detection.location_data.relative_bounding_box
            # print(id, keyPoints)
            mpDraw.draw_detection(img, detection)

            h, w, c = img.shape
            b_box = int(boundingBox.xmin * w), int(boundingBox.ymin * h), int(boundingBox.width * w), int(
                boundingBox.height * h)
            cv.putText(img, f'{int(detection.score[0] * 100)}', (b_box[0], b_box[1] - 20), cv.FONT_ITALIC, 1,
                       (255, 0, 255), 2)
            cv.rectangle(img, b_box, (0, 0, 255), 2)

    cv.putText(img, str(int(fps)), (20, 80), cv.FONT_ITALIC, 2, (255, 255, 255), 2)
    cv.imshow("Frames", img)
    cv.waitKey(1)
