import numpy as np
from math import sqrt
import src.Util.MathUtil.MathHelper as MathHelper

MathHelper = MathHelper.MathHelper

# METHODS NOT TESTED:
# getPrincipalAxes() - this will be tested in the contour testing file

NUM_OF_TESTS = 9
TESTS_PASSED = 0

# Test rotating a point

print('-------------------\n' +
    'rotatePoint() tests\n' +
    '-------------------')

point = np.array([1, 0])

rotatedPoint = MathHelper.rotatePoint(point, np.radians(45))

# Basically comparing the result with (√(2)/2, √(2)/2)
if (np.array_equal(rotatedPoint, [sqrt(2) / 2, sqrt(2) / 2])):
    print('rotate point method test succeeded!\n')
    TESTS_PASSED+=1
else:
    print('rotate point method test FAILED!\n')


# Test doting two vectors together

print('-----------\n' +
    'dot() tests\n' +
    '-----------')

a = np.array([1, 2])
b = np.array([3, 4])

result = MathHelper.dot(a, b)

if (result == 11):
    print('dot product method test succeeded!\n')
    TESTS_PASSED+=1
else:
    print('dot product method test FAILED!\n')


# Test normalizing a vector

print('------------\n' +
    'norm() tests\n' +
    '------------')

vector = np.array([3, 4])

normalizedVector = MathHelper.norm(vector)

if (np.array_equal(normalizedVector, [0.6, 0.8])):
    print('norm method test succeeded!\n')
    TESTS_PASSED+=1
else:
    print('norm method test FAILED!\n')


# Test getting the length of a vector

print('--------------\n' +
    'length() tests\n' +
    '--------------')

vectorLength = MathHelper.getLength(vector)

if (vectorLength == 5):
    print('vector length method test succeeded!\n')
    TESTS_PASSED+=1
else:
    print('vector length method test FAILED!\n')


# Test getting the angle between two vectors and getting the relative angle direction between two vectors

print('------------------------------------------------\n' +
    'getAngle() and getRelativeAngleDirection() tests\n' +
    '------------------------------------------------')

# The vector that'll remain the same is the horizontal  - (1, 0)
# We'll test with the second vector both in the first half of the unit circle and in the second half

# First, verify the horizontal declared in MathHelper is correct
if (np.array_equal(MathHelper.horizontal, [1., 0.])):
    print('horizontal vector in MathHelper is correct!')
    TESTS_PASSED+=1
else:
    print('WARNING - incorrect declaration of horizontal in MathHelper!')

# Now declare our vectors, we'll be re-assigning a and b
a = np.array([0, 1])
b = np.array([0, -1])

# First semi-sphere, signed range
result = MathHelper.getAngle(MathHelper.horizontal, a, True)

if (result == np.radians(90)):
    print('first semi-sphere signed range get angle succeeded!')
    TESTS_PASSED+=1
else:
    print('first semi-sphere signed range get angle FAILED!')

# First semi-sphere, unsigned range (should be same result as before)
result = MathHelper.getAngle(MathHelper.horizontal, a, False)

if (result == np.radians(90)):
    print('first semi-sphere unsigned range get angle succeeded!')
    TESTS_PASSED+=1
else:
    print('first semi-sphere unsigned range get angle FAILED!')

# Second semi-sphere, signed range
result = MathHelper.getAngle(MathHelper.horizontal, b, True)

if (result == np.radians(-90)):
    print('second semi-sphere signed range get angle succeeded!')
    TESTS_PASSED+=1
else:
    print('second semi-sphere signed range get angle FAILED!')

# Second semi-sphere, unsigned range
result = MathHelper.getAngle(MathHelper.horizontal, b, False)

if (result == np.radians(270)):
    print('second semi-sphere unsigned range get angle succeeded!\n')
    TESTS_PASSED+=1
else:
    print('second semi-sphere unsigned range get angle FAILED!\n')

print('\n---------------------------------------------------------------------------\n' +
    'Excluding getPrincipalAxes(), ' + str(TESTS_PASSED / NUM_OF_TESTS * 100.) + '% of the methods are performing properly\n' +
    '---------------------------------------------------------------------------\n')