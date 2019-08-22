import cv2 as cv
import numpy as np
import src.Util.MathUtil.MathHelper as MathHelper
import src.Util.MathUtil.Line as Line
import src.Util.VisionUtil.VisionUtil as VisionUtil

MathHelper = MathHelper.MathHelper
Line = Line.Line
VisionUtil = VisionUtil.VisionUtil

class ContourGroup:
    def __init__(self, contours, frameCenter):
        # Save the contours that comprise the group
        self.contours = contours

        # Combine the points from the contours
        self.points = ContourGroup.combinePoints(contours)

        # Find the area of the contour
        self.contourArea = ContourGroup.combineArea(contours)

        # Obtain the straight bounding box
        self.boundingBoxPoints, self.boundingBoxArea, self.boundingBoxAspectRatio = VisionUtil.getBoundingBoxPoints(self.points)
        x, y, self.boundingBoxWidth, self.boundingBoxHeight = cv.boundingRect(self.points)
        self.boundingBoxUpperLeftPoint = (x, y)
        self.boundingBoxLowerRightPoint = (x + self.boundingBoxWidth, y + self.boundingBoxHeight)
        self.boundingBoxPoints = [self.boundingBoxUpperLeftPoint, [x + self.boundingBoxWidth, y], self.boundingBoxLowerRightPoint, [x, y + self.boundingBoxHeight]]

        # Find the rotated rect and it's area
        rect = cv.minAreaRect(self.points)
        box = np.int0(cv.boxPoints(rect))
        self.rotatedRect = [box]
        self.rotatedRectArea = cv.contourArea(box)
        
        # Get the center of the group
        self.midpoint = MathHelper.getMidpoint(self.boundingBoxPoints)

        # Find the direction vector using the rotated rect and create a Line instance of it
        directionVector = VisionUtil.getReferenceVector(box)
        self.rotation = MathHelper.getAngle(MathHelper.horizontal, directionVector)
        self.referenceVector = Line(directionVector, self.midpoint)
        self.contourLine = Line(directionVector, [self.midpoint[0], frameCenter[1] * 2 - self.midpoint[1]])

        # Finally, sort the image points
        self.points = VisionUtil.sortImgPts(self.points, directionVector, self.midpoint)

        # Get the distance to the center of the frame
        self.distanceToCenter = np.linalg.norm(np.array([self.midpoint[0] - frameCenter[0], frameCenter[1] - self.midpoint[1]]))
    
    def combinePoints(contours):
        points = []
        for contour in contours:
            verticies = contour.getVerticies()
            for vertex in verticies:
                points.append(vertex)
        return np.array(points, dtype=np.float32)

    def combineArea(contours):
        area = 0
        for contour in contours:
            area = area + contour.getContourArea()
        return area

    def getVerticies(self):
        return self.points

    def getContourArea(self):
        return self.contourArea

    def getReferenceVector(self):
        return self.referenceVector

    def getContourLine(self):
        return self.contourLine

    def getRotation(self):
        return self.rotation

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