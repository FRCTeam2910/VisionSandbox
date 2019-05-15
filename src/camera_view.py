import cv2 as cv
import numpy as np

cap = cv.VideoCapture(1)

low = 0
high = 255
high_H = 180
high_noise = 40
high_kernel = 50
high_aperture = 20
high_gaussian = 20
high_sigma = 20

kernel_size = 5

aperture_size = 5

gaussian_kernel_size = 5
sigma_x = 0
sigma_y = 0

low_H = low
low_S = low
low_V = low
high_H = high_H
high_S = high
high_V = high

median_blur = False
gaussian_blur = False
smooth = False

window_capture_name = 'Video Capture'
median_blur_button_name = 'Median Blur'
gaussian_blur_button_name = 'Gaussian Blur'
smooth_button_name = 'Image Smoothing'

window_median_blur_name = 'Median Blur'
aperture_name = 'Linear Aperture Size'

window_gaussian_blur_name = 'Gaussian Blur'
gaussian_kernel_size_name = 'Gaussian Kernel Size'
sigma_x_name = 'Sigma X'
sigma_y_name = 'Sigma Y'

window_smooth_name = 'Image Smoothening'
kernel_size_name = 'Kernel Size'

window_threshold_name = 'Threshold View'
low_H_name = 'Low H'
low_S_name = 'Low S'
low_V_name = 'Low V'
high_H_name = 'High H'
high_S_name = 'High S'
high_V_name = 'High V'

window_contour_name = 'Countour View'

def on_kernel_size_trackbar(val):
    global kernel_size
    kernel_size = val
    cv.setTrackbarPos(kernel_size_name, window_smooth_name, kernel_size)

def on_low_H_thresh_trackbar(val):
    global low_H
    global high_H
    low_H = val
    low_H = min(high_H-1, low_H)
    cv.setTrackbarPos(low_H_name, window_threshold_name, low_H)

def on_high_H_thresh_trackbar(val):
    global low_H
    global high_H
    high_H = val
    high_H = max(high_H, low_H+1)
    cv.setTrackbarPos(high_H_name, window_threshold_name, high_H)

def on_low_S_thresh_trackbar(val):
    global low_S
    global high_S
    low_S = val
    low_S = min(high_S-1, low_S)
    cv.setTrackbarPos(low_S_name, window_threshold_name, low_S)

def on_high_S_thresh_trackbar(val):
    global low_S
    global high_S
    high_S = val
    high_S = max(high_S, low_S+1)
    cv.setTrackbarPos(high_S_name, window_threshold_name, high_S)

def on_low_V_thresh_trackbar(val):
    global low_V
    global high_V
    low_V = val
    low_V = min(high_V-1, low_V)
    cv.setTrackbarPos(low_V_name, window_threshold_name, low_V)

def on_high_V_thresh_trackbar(val):
    global low_V
    global high_V
    high_V = val
    high_V = max(high_V, low_V+1)
    cv.setTrackbarPos(high_V_name, window_threshold_name, high_V)

def on_aperture_size_trackbar(val):
    global aperture_size
    if (val % 2 == 0):
        aperture_size = val + 1
    elif (val < 0):
        aperture_size = 1
    cv.setTrackbarPos(aperture_name, window_median_blur_name, aperture_size)

def on_gaussian_kernel_size_trackbar(val):
    global gaussian_kernel_size
    if (val % 2 == 0):
        gaussian_kernel_size = val + 1
    elif (val < 0):
        gaussian_kernel_size = 1
    cv.setTrackbarPos(gaussian_kernel_size_name, window_gaussian_blur_name, gaussian_kernel_size)

def on_sigma_x_trackbar(val):
    global sigma_x
    sigma_x = val
    cv.setTrackbarPos(sigma_x_name, window_gaussian_blur_name, sigma_x)

def on_sigma_y_trackbar(val):
    global sigma_y
    sigma_y = val
    cv.setTrackbarPos(sigma_y_name, window_gaussian_blur_name, sigma_y)

def setupCaptureWindow():
    cv.namedWindow(window_capture_name)

def setupMedianBlurWindow():
    cv.namedWindow(window_median_blur_name)
    cv.createTrackbar(aperture_name, window_median_blur_name, aperture_size, high_aperture, on_aperture_size_trackbar)

def setupGaussianBlurWindow():
    cv.namedWindow(window_gaussian_blur_name)
    cv.createTrackbar(gaussian_kernel_size_name, window_gaussian_blur_name, gaussian_kernel_size, high_gaussian, on_gaussian_kernel_size_trackbar)
    cv.createTrackbar(sigma_x_name, window_gaussian_blur_name, sigma_x, high_sigma, on_sigma_x_trackbar)
    cv.createTrackbar(sigma_y_name, window_gaussian_blur_name, sigma_y, high_sigma, on_sigma_y_trackbar)

def setupImageSmoothingWindow():
    cv.namedWindow(window_smooth_name)
    cv.createTrackbar(kernel_size_name, window_smooth_name, kernel_size, high_kernel, on_kernel_size_trackbar)

def setupHSVThresholdingWindow():
    cv.namedWindow(window_threshold_name)
    cv.createTrackbar(low_H_name, window_threshold_name , low_H, high_H, on_low_H_thresh_trackbar)
    cv.createTrackbar(high_H_name, window_threshold_name , high_H, high_H, on_high_H_thresh_trackbar)
    cv.createTrackbar(low_S_name, window_threshold_name , low_S, high, on_low_S_thresh_trackbar)
    cv.createTrackbar(high_S_name, window_threshold_name , high_S, high, on_high_S_thresh_trackbar)
    cv.createTrackbar(low_V_name, window_threshold_name , low_V, high, on_low_V_thresh_trackbar)
    cv.createTrackbar(high_V_name, window_threshold_name , high_V, high, on_high_V_thresh_trackbar)

def setupContourWindow():
    cv.namedWindow(window_contour_name)

setupCaptureWindow()
setupHSVThresholdingWindow()
setupContourWindow()

while True:
    ret, frame = cap.read()
    if (ret):
        # Perform a median blur to remove salt and pepper noise
        # frame_median = cv.medianBlur(frame, aperture_size)

        # Perform a Gaussian blur to remove Gaussian noise
        # frame_gaussian = cv.GaussianBlur(frame, (gaussian_kernel_size, gaussian_kernel_size), sigma_x, sigma_y)

        # Perform HSV thresholding
        frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        frame_threshold = cv.inRange(frame_hsv, (low_H, low_S, low_V), (high_H, high_S, high_V))

        # Smooth the image
        # frame_smooth = cv.blur(frame_median, (kernel_size, kernel_size))

        # Find contours within the image
        frame_contours = frame_threshold
        frame_temp = np.array(frame, copy=True)

        contours, hierarchy = cv.findContours(frame_contours, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        frame_contours = cv.drawContours(frame_temp, contours, -1, (0, 0, 255), 5)

        # Label the number of contours found
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(frame_contours, str(len(contours)), (10,500), font, 4, (255, 255, 255), 2, cv.LINE_AA)

        # Update the windows with the updated frame
        cv.imshow(window_capture_name, frame)
        # cv.imshow(window_gaussian_blur_name, frame_gaussian)
        # cv.imshow(window_smooth_name, frame_smooth)
        # cv.imshow(window_median_blur_name, frame_median)
        cv.imshow(window_threshold_name, frame_threshold)
        cv.imshow(window_contour_name, frame_contours)
    key = cv.waitKey(30)
    if key == ord("q"):
        break