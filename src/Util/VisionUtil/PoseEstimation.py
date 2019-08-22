class PoseEstimation:
    def __init__(self, cameraConfig, pathToTargetModel):
        self.cameraMatrix = cameraConfig.getCameraMatrix()
        self.distortionCoefficients = cameraConfig.getDistortionCoefficients()
        targetModel = open(pathToTargetModel, 'r')
        self.objectPoints = np.array(json.loads(targetModel.readline()))
        self.numOfObjectPoints = len(self.objectPoints)