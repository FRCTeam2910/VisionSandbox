import src.Util.MathUtil.Vector3 as Vector3
import src.Util.MathUtil.Rotation3 as Rotation3
import src.Util.MathUtil.RigidTransform3 as RigidTransform3
import numpy as np

Vector3 = Vector3.Vector3
Rotation3 = Rotation3.Rotation3
RigidTransform3 = RigidTransform3.RigidTransform3

rotMat = np.array([[0.99445978, 0.02089393, 0.1030203],
    [0.05172469, -0.95045264, -0.30653603],
    [0.09151117, 0.31016644, -0.94626766]], dtype=np.float32)

a = np.array([[1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]], dtype=np.float32)

b = np.array([[10, 11, 12],
    [13, 14, 15],
    [16, 17, 18]], dtype=np.float32)

# Comment out the assert line in Rotation3 to test addition
# A = Rotation3(a)

# B = Rotation3(b)

# C = A + B

rotation = Rotation3(rotMat)

inv_rot = rotation.inverse()

vectorA = Vector3(1, 2, 3)
vectorB = Vector3(4, 5, 6)

vectorANorm = vectorA.norm()
vectorBNorm = vectorB.norm()

vectorANegate = vectorA.negate()
vectorBNegate = vectorB.negate()

vectorAScaled = vectorA.scale(2)
vectorBScaled = vectorB.scale(2)

print(str(vectorA))
print(str(vectorB))

product = vectorA * vectorB
sum = vectorA + vectorB
difference = vectorA - vectorB

dotProduct = vectorA.dot(vectorB)
crossProduct = vectorA.cross(vectorB)

print('test')