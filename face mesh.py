import cv2 as cv
import mediapipe as mp
import time

pTime = 0
cTime = 0

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
face_mesh = mpFaceMesh.FaceMesh(max_num_faces=2)
drawSpec = mpDraw.DrawingSpec((0, 255, 255), 1, 1)

capture = cv.VideoCapture(0)

while True:
    success, img = capture.read()

    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    results = face_mesh.process(imgRGB)
    if results.multi_face_landmarks:
        # print(results.multi_face_landmarks)
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS, drawSpec, drawSpec)
            for id, lm in enumerate(faceLms.landmark):
                # print([lm.x, lm.y])
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                # print([id, cx, cy])

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv.putText(img, f'{int(fps)}/s', (20, 80), cv.FONT_ITALIC, 1, (255, 0, 0), 2)
    cv.imshow("Frames", img)
    cv.waitKey(10)
