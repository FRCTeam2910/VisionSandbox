import cv2 as cv
import numpy as np
import src.Util.VisionUtil.VisionUtil as VisionUtil
import src.Util.VisionUtil.Contour as Contour
import src.Util.VisionUtil.ContourGroup as ContourGroup
from math import atan2, cos, sin, sqrt, pi
import time

VisionUtil = VisionUtil.VisionUtil
Contour = Contour.Contour
ContourGroup = ContourGroup.ContourGroup

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

# Methods grabbing the specific contours we want
def processContours(contours, frameCenter):
    processed_contours = []
    for contour in contours:
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

def groupContours(sortedContours, numOfContoursToGroup, intersectionLocation, targetAreaRange, targetFullnessRange, aspectRatioRange, sortingMode, frameCenter, frameSize):
    if (numOfContoursToGroup == 1):
        return [sortedContours[0]]
    elif (numOfContoursToGroup == 2):
        if (intersectionLocation == 'neither'):
            contours = []
            count = 0
            while (count != numOfContoursToGroup):
                contours.append(sortedContours[count])
                count = count + 1
            return [ContourGroup(contours, frameCenter)]
        else:
            pairs = []
            for i in range(len(sortedContours)):
                refContour = sortedContours[i]
                j = i + 1
                while(j < len(sortedContours)):
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

### ----- NOW THE ACTUAL TEST ----- ###

# Begin by loading our test image
frame = cv.imread('test\\TestImages\\ContourTestImage.png')

height, width = frame.shape[:2]
frameCenter = np.array([width / 2, height / 2])
frameSize = height * width

# Perform a median blur
frame_median = cv.medianBlur(frame, 5)

# Convert to HSV colorspace and threshold
# maybe put this in it's own method?
frame_hsv = cv.cvtColor(frame_median, cv.COLOR_BGR2HSV)
frame_threshold = cv.inRange(frame_hsv, (0, 0, 255), (0, 0, 255))

# Find the contours in the image
contours = findContours(frame_threshold)

# Process the contours
processedContours = processContours(contours, frameCenter)

# Filter the contours
targetAreaRange = (0.0, 1.0)
targetFullnessRange = (0.0, 1.0)
aspectRatioRange = (0.0, 4.0)

filteredContours = filterContours(processedContours, targetAreaRange, targetFullnessRange, aspectRatioRange, frameSize)

if filteredContours is not None:
    # Sort the contours
    sortedContours = sortContours(filteredContours, 'center')

    # Finally, group the contours
    numOfContoursToGroup = 2
    intersectionLocation = 'below'

    targetAreaRange = (0.0, 1.0)
    targetFullnessRange = (0.29, 0.32)
    aspectRatioRange = (1.9, 2.1)

    # Sorting mode for the contours pairs
    sortingMode = 'center'

    # target = groupContours(sortedContours, numOfContoursToGroup, intersectionLocation, targetAreaRange, targetFullnessRange, aspectRatioRange, sortingMode, frameCenter, frameSize)
    target = pairContours(sortedContours, intersectionLocation, targetAreaRange, targetFullnessRange, aspectRatioRange, sortingMode, frameCenter, frameSize)


    if target is not None:
        # Draw and label the remaining bits
        frame_contours = drawContours(frame, target)
        labelVerticies(frame_contours, target)
        labelContours(frame_contours, target)
        drawReferenceVector(frame_contours, target)
        drawBoundingBoxes(frame_contours, target)

        # Lastly, show the result
        cv.imshow('ContourTest', frame_contours)
        cv.waitKey(0)
    else:
        print('No Target Found!')
else:
    print('No Contours Found!')