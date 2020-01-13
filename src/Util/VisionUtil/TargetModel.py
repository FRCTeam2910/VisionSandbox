import numpy as np
import json
import Util.MathUtil.MathHelper as MathHelper

MathHelper = MathHelper.MathHelper

class TargetModel():
    def __init__(self, pathToObjPts):
        self.objPts = json.loads(open(pathToObjPts, 'r').readline())['points']
        self.polarPts = np.zeros((len(self.objPts), 2), dtype=np.float32)
        for i in range(len(self.objPts)):
            vector = [self.objPts[i][0], self.objPts[i][1]]
            length = MathHelper.getLength(vector)
            angle = MathHelper.getAngle([1, 0], MathHelper.norm(vector), False)
            self.polarPts[i][0] = length
            self.polarPts[i][1] = angle