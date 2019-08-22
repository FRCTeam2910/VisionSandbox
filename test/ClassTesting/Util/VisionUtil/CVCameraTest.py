import cv2 as cv
import src.Util.VisionUtil.CVCamera as CVCamera

CVCamera = CVCamera.CVCamera

camera = CVCamera('test/MiscTestScripts/CameraConfig.cfg', 0)

# Capturing a single frame
# ret, frame = camera.getFrame()
# cv.imshow('stream', frame)
# key = cv.waitKey(0)

# Performing a stream
while True:
    ret, frame = camera.getFrame()
    cv.imshow('stream', frame)
    key = cv.waitKey(30)
    if key == ord("q"):
        cv.destroyAllWindows()
        break