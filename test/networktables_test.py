from time import sleep
from networktables import NetworkTables
import logging
import numpy as np
import cv2

#cap = cv2.VideoCapture('http://10.29.10.100:5800/')

logging.basicConfig(level=logging.DEBUG)
ip = '10.29.10.2'
NetworkTables.initialize(server=ip)
sd = NetworkTables.getTable('limelight')

# corners = np.zeros((4, 2), dtype=np.float32)

while True:
    tv = sd.getNumber('tv', 0)
    if (tv == 1):
        camtran = sd.getNumberArray('camtran', [0, 0])
        print(camtran)
    sleep(.01)

# while True:
#     tv = sd.getNumber('tv', 0)
#     if (tv == 1):
#         ret, img = cap.read()
#         tcornx = sd.getNumberArray('tcornx', [0, 0])
#         tcorny = sd.getNumberArray('tcorny', [0, 0])

#         if (len(tcornx) == 4) and (len(tcorny) == 4):
#             for i in range (4):
#                 corners[i][0] = tcornx[i]
#                 corners[i][1] = tcorny[i]

#             print(corners)
#             #cv.imshow("stream", img)

# # when everyting is done, release the capture
# #cap.release()