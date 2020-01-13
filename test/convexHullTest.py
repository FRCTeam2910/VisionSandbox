import cv2 as cv
import numpy as np

frame = cv.imread('test\\TestImages\\test1.jpg')

frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
frame_threshold = cv.inRange(frame_hsv, (55, 128, 144), (72, 255, 255))

frame_contours = frame_threshold
contours, hierarchy = cv.findContours(frame_contours, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

