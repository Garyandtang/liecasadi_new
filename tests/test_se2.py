import manifpy
import numpy as np
import pytest

from src.liecasadi.so2 import SO2, SO2Tangent
from src.liecasadi.se2 import SE2, SE2Tangent


def pos_generation():
    x = (np.random.rand(1) - 0.5) * 5
    y = (np.random.rand(1) - 0.5) * 5
    # x = np.array([1])
    # y = np.array([1])
    x = x[0]
    y = y[0]
    return x, y, np.array([x, y])


# theta generation
theta = (np.random.rand(1) - 0.5) * 2 * np.pi
# theta = np.array([1])
theta = theta[0]


# theta1 generation
theta1 = (np.random.rand(1) - 0.5) * 2 * np.pi
theta1 = theta1[0]


# position generation
x, y, pos = pos_generation()

# position1 generation
x1, y1, pos1 = pos_generation()

# SE2 objects
mySE2 = SE2.from_x_y_theta(x, y, theta)
manifSE2 = manifpy.SE2(x, y, theta)

# SE2_1 objects
mySE2_1 = SE2.from_x_y_theta(x1, y1, theta1)
manifSE2_1 = manifpy.SE2(x1, y1, theta1)

# # SE2Tangent objects
# vec = (np.random.rand(3) - 0.5) * 2 * np.pi
# mySE2Tang = SE2Tangent(vec)
# manifSE2Tang = manifpy.SE2Tangent(vec)
#
# SE2 matrix
matrix = np.array([[np.cos(theta), -np.sin(theta), x],
                   [np.sin(theta), np.cos(theta), y],
                   [0, 0, 1]])
mySE2_matrix = SE2.from_matrix(matrix)


def test_SE2_init():
    assert mySE2.as_matrix() - manifSE2.transform() == pytest.approx(0.0, abs=1e-4)
    assert mySE2.translation() - manifSE2.translation() == pytest.approx(0.0, abs=1e-4)
    assert mySE2.rotation().as_matrix() - manifSE2.rotation() == pytest.approx(0.0, abs=1e-4)
    assert mySE2.translation() - manifSE2.translation() == pytest.approx(0.0, abs=1e-4)

def test_SE2_mul():
    assert (mySE2 * mySE2_1).as_matrix() - (manifSE2 * manifSE2_1).transform() == pytest.approx(0.0, abs=1e-4)


def test_SE2_log():
    mySE2_tang = mySE2.log()
    manifSE2_tang = manifSE2.log()
    print("mySE2_tang.vec: ", mySE2_tang.vec)
    print("manifSE2_tang.coeffs(): ", manifSE2_tang.coeffs())
    assert mySE2_tang.vec - manifSE2_tang.coeffs() == pytest.approx(0.0, abs=1e-4)

def test_SE2_sub():
    sub_1 = mySE2 - mySE2_1
    sub_2 = manifSE2 - manifSE2_1
    assert sub_1.vec - sub_2.coeffs() == pytest.approx(0.0, abs=1e-4)

def test_SE2_inverse():
    assert mySE2.inverse().as_matrix() - manifSE2.inverse().transform() == pytest.approx(0.0, abs=1e-4)


def test_SE2_Matrix_init():
    assert mySE2_matrix.as_matrix() - manifSE2.transform() == pytest.approx(0.0, abs=1e-4)
    assert mySE2_matrix.transform() - manifSE2.transform() == pytest.approx(0.0, abs=1e-4)
    assert mySE2_matrix.as_matrix() - matrix == pytest.approx(0.0, abs=1e-4)
