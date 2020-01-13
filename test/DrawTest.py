import cv2 as cv

frame = cv.imread('test\\TestImages\\test1.jpg')

cv.circle(frame, (10000, 10000), 3, (255, 255, 355))

cv.imshow("img", frame)
cv.waitKey(0)
print(frame.shape)