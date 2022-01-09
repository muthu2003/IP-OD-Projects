import cv2 as cv
import numpy as np

wImg, hImg = 300, 300

capture = cv.VideoCapture(0)
capture.set(10, 150)


def preProcess(image):
    imgGray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    imgBLur = cv.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv.Canny(imgBLur, 190, 200)
    imgDil = cv.dilate(imgCanny, (5, 5), iterations=4)
    imgThresh = cv.erode(imgDil, (5, 5), iterations=1)

    return imgThresh


def getContours(image):
    biggest = np.array([])
    maxArea = 0
    contours, hier = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > 40:                                # should properly be determined
            cv.drawContours(imgContour1, cnt, -1, (0, 0, 255), 3)
            peri = cv.arcLength(cnt, True)
            approx = cv.approxPolyDP(cnt, 0.02 * peri, True)
            # print(approx)
            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area
                cv.drawContours(imgContour2, biggest, -1, (0, 0, 255), 20)

            # x, y, w, h = cv.boundingRect(approx)
            # bbox = [x, y, w, h]

    print(area)
    return biggest

def reorder(myPoints):
        myPoints = myPoints.reshape((4, 2))
        myPointsNew = np.zeros((4, 1, 2), np.int32)
        add = myPoints.sum(1)
        diff = np.diff(myPoints, axis=1)

        myPointsNew[0] = myPoints[np.argmin(add)]
        myPointsNew[1] = myPoints[np.argmin(diff)]
        myPointsNew[2] = myPoints[np.argmax(diff)]
        myPointsNew[3] = myPoints[np.argmax(add)]

        return myPointsNew


def getWarp(img, big_cp):
    big_cp = reorder(big_cp)
    pts1 = np.float32([big_cp])
    pts2 = np.float32([[0, 0], [wImg, 0], [0, hImg], [wImg, hImg]])
    matrix = cv.getPerspectiveTransform(pts1, pts2)
    imgOut = cv.warpPerspective(img, matrix, (wImg, hImg))

    return imgOut


while True:
    success, img = capture.read()
    img = cv.resize(img, (wImg, hImg), interpolation=cv.INTER_AREA)
    imgContour1 = img.copy()
    imgContour2 = img.copy()

    img_thresh = preProcess(img)
    big_cont = getContours(img_thresh)
    print(big_cont)

    if len(big_cont) == 4:
        imgOut = getWarp(img, big_cont)
        cv.imshow("Scanned", imgOut)

    ImageArray = (img, img_thresh, imgContour1, imgContour2)

    # stackedImages = cv.hconcat(ImageArray)
    # stackedImages = np.concatenate(ImageArray, 1)

    cv.imshow("Image", img)
    cv.imshow("Threshold", img_thresh)
    cv.imshow("Contours", imgContour1)
    cv.imshow("Contour Corner", imgContour2)
    # cv.imshow("Stacked", stackedImages)

    cv.waitKey(1)