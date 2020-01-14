import cv2 as cv
import numpy as np
import src.Util.MathUtil.MathHelper as MathHelper
import src.Util.MathUtil.Line as Line
import src.Util.VisionUtil.VisionUtil as VisionUtil

MathHelper = MathHelper.MathHelper
Line = Line.Line
VisionUtil = VisionUtil.VisionUtil

class Contour:
    def __init__(self, contourPoints, frameCenter, useConvexHull, numOfCorners):
        # First process the contour points
        self.points = contourPoints

        self.useConvexHull = useConvexHull

        # Compute the verticies of the contour
        self.vertices = cv.approxPolyDP(self.points, 0.015 * cv.arcLength(self.points, True), True)

        if useConvexHull:
            # Find the convex hull of the contour
            self.vertices = cv.convexHull(self.vertices)

        # Apply k-means if there are duplicates
        if (len(self.vertices) > numOfCorners) and (numOfCorners is not 0):
            criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            self.vertices = cv.kmeans(self.vertices.astype(dtype=np.float32), numOfCorners, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)[2]
            self.vertices = self.vertices.reshape((numOfCorners, 1, 2)).astype(int)

        # Find the area of the contour
        self.area = cv.contourArea(self.vertices)

        # Obtain the straight bounding box
        self.boundingBoxPoints, self.boundingBoxArea, self.boundingBoxAspectRatio = VisionUtil.getBoundingBoxPoints(self.vertices)
        x, y, self.boundingBoxWidth, self.boundingBoxHeight = cv.boundingRect(self.vertices)
        self.boundingBoxArea = self.boundingBoxHeight * self.boundingBoxWidth
        self.boundingBoxAspectRatio = self.boundingBoxWidth / self.boundingBoxHeight
        self.boundingBoxUpperLeftPoint = (x, y)
        self.boundingBoxLowerRightPoint = (x + self.boundingBoxWidth, y + self.boundingBoxHeight)

        # Find the rotated rect and it's area
        rect = cv.minAreaRect(self.vertices)
        box = np.int0(cv.boxPoints(rect))
        self.rotatedRect = [box]
        self.rotatedRectArea = cv.contourArea(box)

        # Find the midpoint of the contour
        self.midpoint = tuple(np.average(self.vertices, axis=0).ravel().astype(int))

        # Find the direction vector using the rotated rect and create a Line instance of it
        self.directionVector = VisionUtil.getReferenceVector(box)
        self.rotation = MathHelper.getAngle(MathHelper.horizontal, self.directionVector, True)
        self.referenceVector = Line(self.directionVector, self.midpoint)
        self.contourLine = Line(self.directionVector, [self.midpoint[0], frameCenter[1] * 2 - self.midpoint[1]])

        # Finally, sort the vertices
        self.vertices = VisionUtil.sortImgPts(self.vertices, self.directionVector, self.midpoint)
        
        # Get the distance to the center of the frame (used for sorting)
        self.distanceToCenter = np.linalg.norm(np.array([self.midpoint[0] - frameCenter[0], frameCenter[1] - self.midpoint[1]]))