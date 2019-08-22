import numpy as np
import cv2 as cv
import src.Util.VisionUtil.VisionUtil as VisionUtil
import src.Util.MathUtil.MathHelper as MathHelper

VisionUtil = VisionUtil.VisionUtil
MathHelper = MathHelper.MathHelper

# Dual target straight
# imgpts = np.array([
#     [256, 135],
#     [228, 236],
#     [258, 244],
#     [286, 143],
#     [347, 143],
#     [375, 245],
#     [405, 237],
#     [377, 135]
# ], dtype=np.float32)

# dual target 30
# imgpts = np.array([
#     [297, 112],
#     [222, 186],
#     [244, 209],
#     [319, 136],
#     [372, 165],
#     [345, 267],
#     [375, 275],
#     [402, 174]
# ], dtype=np.float32)

# dual target 60 - fails
imgpts = np.array([
    [337, 116],
    [235, 142],
    [243, 173],
    [345, 146],
    [375, 199],
    [301, 273],
    [323, 296],
    [398, 221]
], dtype=np.float32)

right, up, midpt = MathHelper.getPrincipalAxes(imgpts)

imgpts = VisionUtil.sortImgPts(imgpts, right, midpt)

print(right)

print(up)

print(np.degrees(MathHelper.getAngle(MathHelper.horizontal, right, True)))