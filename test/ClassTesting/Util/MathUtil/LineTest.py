import numpy as np
import src.Util.MathUtil.Line as Line

Line = Line.Line

'''

The first test will configure a random line and test the getPoint() method, as well as verifying the initialization occured correctly

We will test three conditions to ensure the functionality of the line library
The first, will configure two lines that intersect above
The second will configure two lines that intersect below
The third will configure two lines that do no intersect (are parallel to one another)

These tests will verify the general functionality of the Line class and that it
performs as expected for the three cases given above for intersection - above, below
and the case of no intersection

'''

# declaration and getPoint() method test
directionVector = [np.sqrt(3) / 2, 1 / 2]
point = [0, 0]

myLine = Line(directionVector, point)

print('a: ' + str(myLine.a) + ', b: ' + str(myLine.b), ', x sub nought: ' + str(myLine.xSubNought) + ', y sub nought: ' + str(myLine.ySubNought))

print('Value of f(x) at t = 10: ' + str(myLine.getPoint(10, False)))

print('Rounded value of f(x) at t = 10: ' + str(myLine.getPoint(10, True)))

# intersect above case test
aDirectionVector = [np.sqrt(2) / 2, np.sqrt(2) / 2]
aPoint = [-1.0, 0.0]
a = Line(aDirectionVector, aPoint)

bDirectionVector = [-1 * (np.sqrt(2) / 2), np.sqrt(2) / 2]
bPoint = [1.0, 0.0]
b = Line(bDirectionVector, bPoint)

intersectionPoint = a.intersects(b)

print(intersectionPoint)

# intersect below case test
aDirectionVector = [np.sqrt(2) / 2, -1 * (np.sqrt(2) / 2)]
aPoint = [-1.0, 0.0]
a = Line(aDirectionVector, aPoint)

bDirectionVector = [np.sqrt(2) / 2, 1 * (np.sqrt(2) / 2)]
bPoint = [1.0, 0.0]
b = Line(bDirectionVector, bPoint)

intersectionPoint = a.intersects(b)

print(intersectionPoint)

# parallel line case test
aDirectionVector = [0, 1]
aPoint = [-1.0, 0.0]
a = Line(aDirectionVector, aPoint)

bDirectionVector = aDirectionVector
bPoint = [1.0, 0.0]
b = Line(bDirectionVector, bPoint)

intersectionPoint = a.intersects(b)

print(intersectionPoint)