import cv2 as cv
import numpy as np
from time import sleep

cap = cv.VideoCapture(0)

sleep(3)

ret, frame = cap.read()
if (ret):
    frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    frame_threshold = cv.inRange(frame_hsv, (0, 0, 255), (0, 0, 255))

    frame_contours = frame_threshold
    frame_temp = np.array(frame, copy=True)

    contours, hierarchy = cv.findContours(frame_contours, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    print(contours)