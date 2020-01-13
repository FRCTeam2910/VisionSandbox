import numpy as np
import cv2
import time

points = np.array([
    [[250, 411]],
    [[549, 406]],
    [[290, 286]],
    [[512, 417]],
    [[509, 417]],
    [[468, 296]],
    [[203, 397]],
    [[248, 273]],
    [[505, 290]],
    [[506, 287]],
    [[201, 398]],
], dtype=np.float32)

start = time.time()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
centers = cv2.kmeans(points, 8, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)[2]
end = time.time()
print((end - start) * 1000)
print(centers)