import cv2 as cv
import numpy as np
import src.Util.MathUtil.MathHelper as MathHelper
import src.Util.MathUtil.Line as Line
import src.Util.VisionUtil.VisionUtil as VisionUtil

MathHelper = MathHelper.MathHelper
Line = Line.Line
VisionUtil = VisionUtil.VisionUtil

class Contour:
    def __init__(self, contourPoints, frameCenter):
        # First process the contour points
        self.contourPoints = contourPoints
        self.processedContourPoints = Contour.processPoints(contourPoints)

        # Find the area of the contour
        self.contourArea = cv.contourArea(self.contourPoints)

        # Obtain the straight bounding box
        # self.boundingBoxPoints, self.boundingBoxArea, self.boundingBoxAspectRatio = VisionUtil.getBoundingBoxPoints(self.contourPoints)
        x, y, self.boundingBoxWidth, self.boundingBoxHeight = cv.boundingRect(self.contourPoints)
        self.boundingBoxArea = self.boundingBoxHeight * self.boundingBoxWidth
        self.boundingBoxAspectRatio = self.boundingBoxWidth / self.boundingBoxHeight
        self.boundingBoxUpperLeftPoint = (x, y)
        self.boundingBoxLowerRightPoint = (x + self.boundingBoxWidth, y + self.boundingBoxHeight)

        # Find the rotated rect and it's area
        rect = cv.minAreaRect(self.contourPoints)
        box = np.int0(cv.boxPoints(rect))
        self.rotatedRect = [box]
        self.rotatedRectArea = cv.contourArea(box)

        # Compute the verticies of the contour
        self.contourVerticies = cv.approxPolyDP(contourPoints, 0.015 * cv.arcLength(contourPoints, True), True)
        self.processedContourVerticies = Contour.processPoints(self.contourVerticies)

        # Find the midpoint of the contour
        self.midpoint = MathHelper.getMidpoint(self.processedContourVerticies)

        # Find the direction vector using the rotated rect and create a Line instance of it
        directionVector = VisionUtil.getReferenceVector(box)
        self.rotation = MathHelper.getAngle(MathHelper.horizontal, directionVector, True)
        self.referenceVector = Line(directionVector, self.midpoint)
        self.contourLine = Line(directionVector, [self.midpoint[0], frameCenter[1] * 2 - self.midpoint[1]])

        # Finally, sort the image points
        self.processedContourVerticies = VisionUtil.sortImgPts(self.processedContourVerticies, directionVector, self.midpoint)
        
        # Get the distance to the center of the frame
        self.distanceToCenter = np.linalg.norm(np.array([self.midpoint[0] - frameCenter[0], frameCenter[1] - self.midpoint[1]]))

    def processPoints(pts):
        result = np.zeros(shape=(len(pts), 2), dtype=np.float32)
        for i in range(len(pts)):
            result[i][0] = pts[i][0][0]
            result[i][1] = pts[i][0][1]
        return result
    
    def getContourPoints(self):
        return self.contourPoints

    def getProcessedContourPoints(self):
        return self.processedContourPoints

    def getContourVerticies(self):
        return self.contourVerticies

    def getVerticies(self):
        return self.processedContourVerticies

    def getReferenceVector(self):
        return self.referenceVector

    def getContourLine(self):
        return self.contourLine

    def getRotation(self):
        return self.rotation

    def getContourArea(self):
        return self.contourArea

    def getRotatedRect(self):
        return self.rotatedRect

    def getRotatedRectArea(self):
        return self.rotatedRectArea

    def getBoundingBoxPoints(self):
        return self.boundingBoxPoints

    def getBoundingBoxArea(self):
        return self.boundingBoxArea

    def getBoundingBoxAspectRatio(self):
        return self.boundingBoxAspectRatio

    def getBoundingBoxUpperLeftPoint(self):
        return self.boundingBoxUpperLeftPoint

    def getBoundingBoxLowerRightPoint(self):
        return self.boundingBoxLowerRightPoint

    def getBoundingBoxWidth(self):
        return self.boundingBoxWidth

    def getBoundingBoxHeight(self):
        return self.boundingBoxHeight

    def getMidpoint(self):
        return self.midpoint

    def getDistanceToCenter(self):
        return self.distanceToCenter