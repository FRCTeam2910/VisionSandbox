import src.Util.VisionUtil.CVCamera as CVCamera
import src.Util.VisionUtil.VisionUtil as VisionUtil
import src.Util.VisionUtil.Contour as Contour
import src.Util.VisionUtil.ContourGroup as ContourGroup
import src.Util.MathUtil.Vector3 as Vector3
import src.Util.MathUtil.Rotation3 as Rotation3
import src.Util.MathUtil.RigidTransform3 as RigidTransform3
import json
from math import atan2, cos, sin, sqrt, pi
import cv2 as cv
import numpy as np
import tkinter as tk
from PIL import ImageTk, Image
import time

CVCamera = CVCamera.CVCamera
VisionUtil = VisionUtil.VisionUtil
Contour = Contour.Contour
ContourGroup = ContourGroup.ContourGroup
Vector3 = Vector3.Vector3
Rotation3 = Rotation3.Rotation3
RigidTransform3 = RigidTransform3.RigidTransform3

# 640x480
resolution = (640, 480)
frameSize = 307200
frameMidpoint = (320, 240)
framerate = 60

distortionCoefficients = []
cameraMatrix = []

PATH_TO_TARGET_FILE = 'test/MiscTestScripts/TargetModel.mdl'
targetModel = open('test/MiscTestScripts/TargetModel.mdl', 'r')
objectPoints = np.array(json.loads(targetModel.readline()), dtype=np.float32)

font = cv.FONT_HERSHEY_SIMPLEX

def findContours(img):
    contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    return contours

# Drawing functions

'''

    Default drawing colors (and there BGR values):
    Contours - pink (203, 192, 255)
    Contour Verticies - red (0, 0, 255)
    Contour Midpoint - (0, 100, 0)
    Reference Vector - green (0, 255, 0)
    Contour Label - blue (255, 0, 0)
    Rotated Bounding Box - cyan (255, 191, 0)
    Straight Bounding Box - yellow (0, 255, 255)

'''
def drawContours(img, contours):
    contours_temp = []
    for contour in contours:
        if (isinstance(contour, ContourGroup)):
            group_contours = contour.contours
            for contour in group_contours:
                contours_temp.append(contour.getContourPoints())
        else:
            contours_temp.append(contour.getContourPoints())

    img_temp = np.array(img, copy=True)
    dst = cv.drawContours(img_temp, contours_temp, -1, (203, 192, 255), 2)
    return dst

def drawBoundingBoxes(img, contours):
    for contour in contours:
        cv.drawContours(img, contour.getRotatedRect(), 0, (255, 191, 0), 2)
        cv.rectangle(img, contour.getBoundingBoxUpperLeftPoint(), contour.getBoundingBoxLowerRightPoint(), (0, 255, 255), 2)

def labelContours(img, contours):
    for i in range(len(contours)):
        anchor = contours[i].getBoundingBoxUpperLeftPoint()
        anchor = (anchor[0] - 20, anchor[1] - 5)
        cv.putText(img, str(i + 1), anchor, cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3, cv.LINE_AA)

def drawReferenceVector(img, contours):
    for contour in contours:
        point = contour.getReferenceVector().getPoint(18, True)
        cv.circle(img, contour.getMidpoint(), 3, (0, 100, 0), 2, cv.LINE_AA)
        cv.line(img, contour.getMidpoint(), point, (0, 255, 0), 2, cv.LINE_AA)

def labelVerticies(img, contours):
    for contour in contours:
        pts = contour.getVerticies()
        for x in range(len(pts)):
            cv.putText(img, str(x + 1), (int(pts[x][0]), int(pts[x][1])), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv.LINE_AA)

def drawPnPAxes(img, imgpts):
    origin = tuple(imgpts[0].ravel())
    img = cv.line(img, origin, tuple(imgpts[1].ravel()), (0,0,255), 3)
    img = cv.line(img, origin, tuple(imgpts[2].ravel()), (0,255,0), 3)
    img = cv.line(img, origin, tuple(imgpts[3].ravel()), (0,255,255), 3)
    return img

# Methods grabbing the specific contours we want
def processContours(contours, frameCenter):
    processed_contours = []
    for contour in contours:
        if (cv.contourArea(contour) > 20):
            cnt = Contour(contour, frameCenter)
            processed_contours.append(cnt)
    return processed_contours

def filterContours(contours, targetAreaRange, targetFullnessRange, aspectRatioRange, frameSize):
    filteredContours = []
    for contour in contours:
        cntTargetArea = contour.getContourArea() / frameSize
        cntTargetFullness = contour.getContourArea() / contour.getRotatedRectArea()
        cntAspectRatio = contour.getBoundingBoxAspectRatio()

        withinTargetAreaRange = withinRange(cntTargetArea, targetAreaRange)
        withinTargetFullnessRange = withinRange(cntTargetFullness, targetFullnessRange)
        withinTargetAspectRatioRange = withinRange(cntAspectRatio, aspectRatioRange)

        if (withinTargetAreaRange and withinTargetFullnessRange and withinTargetAspectRatioRange):
            filteredContours.append(contour)

    if (len(filteredContours) == 0):
        return None
    
    return filteredContours

def sortContours(filteredContours, sortingMode):
    # left to right
    if (sortingMode == 'left'):
        return sorted(filteredContours, key=lambda cnt: cnt.midpoint[0])
    # right to left
    elif (sortingMode == 'right'):
        return sorted(filteredContours, key=lambda cnt: cnt.midpoint[0], reverse=True)
    # top to bottom
    elif (sortingMode == 'top'):
        return sorted(filteredContours, key=lambda cnt: cnt.midpoint[1])
    # bottom to top
    elif (sortingMode == 'bottom'):
        return sorted(filteredContours, key=lambda cnt: cnt.midpoint[1], reverse=True)
    # center outwards
    elif (sortingMode == 'center'):
        return sorted(filteredContours, key=lambda cnt: cnt.distanceToCenter)

def pairContours(sortedContours, intersectionLocation, targetAreaRange, targetFullnessRange, aspectRatioRange, sortingMode, frameCenter, frameSize):
    pairs = []
    if (intersectionLocation == 'neither'):
        for i in range(len(sortedContours)):
            refContour = sortedContours[i]
            j = i + 1
            while (j < len(sortedContours)):
                contour = sortedContours[j]
                _contours = [refContour, contour]
                pair = ContourGroup(_contours, frameCenter)
                pairs.append(pair)
                j = j + 1
    else:
        for i in range(len(sortedContours)):
            refContour = sortedContours[i]
            j = i + 1
            while (j < len(sortedContours)):
                contour = sortedContours[j]
                refContourRefVector = refContour.getContourLine()
                contourRefVector = contour.getContourLine()
                intersectionPoint = refContourRefVector.intersects(contourRefVector)
                if (intersectionPoint is not None):
                    intersectionPoint[1] = frameCenter[1] * 2 - intersectionPoint[1]
                    if (intersectionLocation == 'above' and intersectionPoint[1] < refContour.midpoint[1] and intersectionPoint[1] < contour.midpoint[1]):
                        _contours = [refContour, contour]
                        pair = ContourGroup(_contours, frameCenter)
                        pairs.append(pair)
                    elif (intersectionLocation == 'below' and intersectionPoint[1] > refContour.midpoint[1] and intersectionPoint[1] > contour.midpoint[1]):
                        _contours = [refContour, contour]
                        pair = ContourGroup(_contours, frameCenter)
                        pairs.append(pair)
                    elif (intersectionLocation == 'right' and intersectionPoint[0] > refContour.midpoint[0] and intersectionPoint[0] > contour.midpoint[0]):
                        _contours = [refContour, contour]
                        pair = ContourGroup(_contours, frameCenter)
                        pairs.append(pair)
                    elif (intersectionLocation == 'left' and intersectionPoint[0] < refContour.midpoint[0] and intersectionPoint[0] < contour.midpoint[0]):
                        _contours = [refContour, contour]
                        pair = ContourGroup(_contours, frameCenter)
                        pairs.append(pair)
                j = j + 1
    # Now filter the pairs
    filteredPairs = filterContours(pairs, targetAreaRange, targetFullnessRange, aspectRatioRange, frameSize)

    if filteredPairs is None:
        return None

    # Now sort the pairs
    sortedPairs = sortContours(filteredPairs, sortingMode)

    # Return the first pair in the list, which theoretically is the closest thing to what we want
    return [sortedPairs[0]]

# Other misc. helper methods
def withinRange(val, range):
    if (val > range[0] and val < range[1]):
        return True
    else:
        return False

# Rotates and expands an image to avoid cropping
def rotateImage(img, angle):
    # Get the dimensions and center of the image
    height, width = img.shape[:2]
    imgCenter = (width / 2, height / 2)
    
    # Now get our ratation matrix
    rotationMatrix = cv.getRotationMatrix2D(imgCenter, angle, 1)

    # Take the absolute value of the cos and sin from the rotation matrix
    absoluteCos = abs(rotationMatrix[0, 0])
    absoluteSin = abs(rotationMatrix[0, 1])

    # Find the new width and height bounds
    widthBound = int(height * absoluteSin + width * absoluteCos)
    heightBound = int(height * absoluteCos + width * absoluteSin)

    # Subtract the old image center from the rotation matrix (essentially beringing it back to the origin) and add the new corrdinates
    rotationMatrix[0, 2] += widthBound / 2 - imgCenter[0]
    rotationMatrix[1, 2] += heightBound / 2 - imgCenter[1]

    # Finally rotate the image with our modified rotation matrix
    rotatedImg = cv.warpAffine(img, rotationMatrix, (widthBound, heightBound))
    return rotatedImg

def pipeline(frame):
    pipelineStart = time.time()
    #  Now we threshold
    frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    minThresh = (59, 166, 139)
    maxThresh = (73, 255, 255)
    frame_threshold = cv.inRange(frame_hsv, minThresh, maxThresh)

    performErosion = False
    shapeOfErosionKernel = cv.MORPH_RECT
    sizeOfErosionKernel = (3, 3)
    erosionKernelAnchor = (-1, -1)
    erosionAnchor = (-1, -1)
    erosionIterations = 1

    performDilation = False
    shapeOfDilationKernel = cv.MORPH_RECT
    sizeOfDilationKernel = (3, 3)
    dilationKernelAnchor = (-1, -1)
    dilationAnchor = (-1, -1)
    dilationIterations = 1
    # Check if we should perform a erosion
    if (performErosion):
        erosionKernel = cv.getStructuringElement(shapeOfErosionKernel, sizeOfErosionKernel, anchor=erosionKernelAnchor)
        frame_threshold = cv.erode(frame_threshold, erosionKernel, anchor=erosionAnchor, iterations=erosionIterations)

    # Check if we should perform a dilation
    if (performDilation):
        dilationKernel = cv.getStructuringElement(shapeOfDilationKernel, sizeOfDilationKernel, anchor=dilationKernelAnchor)
        frame_threshold = cv.dilate(frame_threshold, dilationKernel, anchor=dilationAnchor, iterations=dilationIterations)

    # Find the contours in the image
    contours = findContours(frame_threshold)

    # Process the contours
    processedContours = processContours(contours, cap.getFrameMidpoint())

    # Filter the contours
    contourAreaRange = (0.0, 1.0)
    contourFullnessRange = (0.0, 1.0)
    contourAspectRatioRange = (0.0, 10.0)
    filteredContours = filterContours(processedContours, contourAreaRange, contourFullnessRange, contourAspectRatioRange, frameSize)

    target = filteredContours

    if filteredContours is not None:
        # Sort the contours
        sortingMode = 'center'
        sortedContours = sortContours(filteredContours, sortingMode)

        target = [sortedContours[0]]

        # Check if we should pair the contours
        pairContours = True
        intersectionLocation = 'above'
        if (pairContours):
            intersectionLocation = intersectionLocation

            targetAreaRange = (0.0, 1.0)
            targetFullnessRange = (0.0, 1.0)
            targetAspectRatioRange = (0.0, 10.0)

            # Sorting mode for the contours pairs
            targetSortingMode = 'center'

            target = pairContours(sortedContours, intersectionLocation, targetAreaRange, targetFullnessRange, targetAspectRatioRange, targetSortingMode, frameMidpoint, frameSize)

    rigidTransform = None
    imgpts = None
    angleToTarget = None
    performPoseEstimation = False
    if (performPoseEstimation and (target is not None) and (len(objectPoints) == len(target[0].getVerticies()))):
        range = [1, len(target[0].getVerticies())]
        rigidTransform, imgpts = VisionUtil.getTranslation(cameraMatrix, distortionCoefficients, objectPoints, target[0].getVerticies(), range)
        angleToTarget = VisionUtil.getAngleToTarget(rigidTransform.rotation.rotationMatrix)

    pipelineEnd = time.time()
    print(pipelineEnd - pipelineStart)

    cv.putText(frame, str(resolution[0]) + 'x' + str(resolution[1]) + ' @' + str(framerate) + ' fps', (3, 11), font, 0.35, (255, 255, 255), 1, cv.LINE_AA)
    cv.putText(frame, 'Contours found: ' + str(len(contours)), (3, 23), font, 0.35, (255, 255, 255), 1, cv.LINE_AA)
    
    if target is not None:
        # Draw and label the remaining bits
        frame_contours = drawContours(frame, target)
        labelVerticies(frame_contours, target)
        drawBoundingBoxes(frame_contours, target)
        if (rigidTransform is not None):
            cv.putText(frame_contours, str(rigidTransform.translation), (3, 35), font, 0.35, (255, 255, 255), 1, cv.LINE_AA)
            cv.putText(frame_contours, str(rigidTransform.rotation), (3, 47), font, 0.35, (255, 255, 255), 1, cv.LINE_AA)
            cv.putText(frame_contours, str(np.degrees(angleToTarget)), (3, 59), font, 0.35, (255, 255, 255), 1, cv.LINE_AA)
            frame_contours = drawPnPAxes(frame_contours, imgpts)
        else:
            drawReferenceVector(frame_contours, target)
        cv.imshow('Pipeline Output', frame_contours)           
    else:
        cv.imshow('Pipeline Output', frame)

frame = cv.imread('test\\TestImages\\test1.jpg')
