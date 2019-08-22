import numpy as np
import json

class CameraConfig:
    def __init__(self, cameraConfigFilePath):
        cameraConfig = open(cameraConfigFilePath, 'r')
        self.resolution = json.loads(cameraConfig.readline())
        self.framerate = json.loads(cameraConfig.readline())
        self.cameraMatrix = np.array(json.loads(cameraConfig.readline()))
        self.distortionCoefficients = np.array(json.loads(cameraConfig.readline()))
        self.frameMidpoint = json.loads(cameraConfig.readline())
        self.frameSize = json.loads(cameraConfig.readline())

    def getResolution(self):
        return self.resolution

    def getFramerate(self):
        return self.framerate

    def getCameraMatrix(self):
        return self.cameraMatrix

    def getDistortionCoefficients(self):
        return self.distortionCoefficients

    def getFrameMidpoint(self):
        return self.frameMidpoint

    def getFrameSize(self):
        return self.frameSize