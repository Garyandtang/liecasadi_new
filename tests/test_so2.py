import manifpy
import numpy as np
import pytest

from src.liecasadi.so2 import SO2, SO2Tangent


# theta generation
theta = (np.random.rand(1) - 0.5) * 2 * np.pi
theta = theta[0]

# SO2 objects
mySO2Euler = SO2.from_euler(theta)
manifSO2 = manifpy.SO2(theta)

# SO2Tangent objects
vec = (np.random.rand(1) - 0.5) * 2 * np.pi
vec = vec[0]
mySO2Tang = SO2Tangent(vec)
manifSO2Tang = manifpy.SO2Tangent(vec)

# SO2 matrix
matrix = np.array([[np.cos(theta), -np.sin(theta)],
                   [np.sin(theta), np.cos(theta)]])
mySO2Matrix = SO2.from_matrix(matrix)

def test_SO2_Euler_init():
    assert mySO2Euler.as_euler() - theta == pytest.approx(0.0, abs=1e-4)
    assert mySO2Euler.as_matrix() - matrix == pytest.approx(0.0, abs=1e-4)
    assert mySO2Euler.as_matrix() - manifSO2.rotation() == pytest.approx(0.0, abs=1e-4)
    assert mySO2Euler.as_euler() - manifSO2.angle() == pytest.approx(0.0, abs=1e-4)

def test_SO2_Matrix_init():
    assert mySO2Matrix.as_matrix() - manifSO2.rotation() == pytest.approx(0.0, abs=1e-4)
    assert mySO2Matrix.as_euler() - manifSO2.angle() == pytest.approx(0.0, abs=1e-4)
    assert mySO2Matrix.as_euler() - theta == pytest.approx(0.0, abs=1e-4)
    assert mySO2Matrix.as_matrix() - matrix == pytest.approx(0.0, abs=1e-4)


def test_euler():
    from scipy.spatial.transform import Rotation
    r = np.random.randn(1) * np.pi
    r = r[0]
    assert SO2.from_euler(r).as_matrix() - Rotation.from_euler('z', r).as_matrix()[:2, :2] == pytest.approx(0.0, abs=1e-4)

def test_exp():
    print(mySO2Tang.exp().as_matrix())
    print(manifSO2Tang.exp().rotation())
    assert mySO2Tang.exp().as_matrix() - manifSO2Tang.exp().rotation() == pytest.approx(0.0, abs=1e-4)

def test_log():
    assert mySO2Tang.exp().log().vec - vec == pytest.approx(0.0, abs=1e-4)
    assert mySO2Euler.log().exp().as_matrix() - manifSO2.log().exp().rotation() == pytest.approx(0.0, abs=1e-4)

def test_inv():
    assert mySO2Euler.inverse().as_matrix() - manifSO2.inverse().rotation() == pytest.approx(0.0, abs=1e-4)

def test_right_sum():
    assert (mySO2Euler + mySO2Tang).as_matrix() - (manifSO2 + manifSO2Tang).rotation() == pytest.approx(0.0, abs=1e-4)

def test_left_sum():
    assert (mySO2Tang + mySO2Euler).as_matrix() - (manifSO2Tang + manifSO2).rotation() == pytest.approx(0.0, abs=1e-4)

def test_act():
    vec = np.random.randn(2)
    assert mySO2Euler.act(vec) - manifSO2.act(vec) == pytest.approx(0.0, abs=1e-4)


theta2 = (np.random.rand(1) - 0.5) * 2 * np.pi
theta2 = theta2[0]
mySO2Euler2 = SO2.from_euler(theta2)
manifSO2_2 = manifpy.SO2(theta2)

def test_mul():
    assert (mySO2Euler * mySO2Euler2).as_matrix() - (manifSO2 * manifSO2_2).rotation() == pytest.approx(0.0, abs=1e-4)
    assert (mySO2Euler2 * mySO2Euler).as_matrix() - (manifSO2_2 * manifSO2).rotation() == pytest.approx(0.0, abs=1e-4)