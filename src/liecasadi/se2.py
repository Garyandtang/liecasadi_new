# developer: Garyandtang (HKUST)

import dataclasses
from dataclasses import field
from typing import Union, List

import casadi as cs
import numpy as np

from liecasadi.hints import Angle, Matrix, TangentVector, Vector, Scalar
from src.liecasadi.so2 import SO2, SO2Tangent

@dataclasses.dataclass
class SE2:
    data: Vector  # [x, y, cos(theta), sin(theta)]

    def __repr__(self) -> str:
        return f"Pos: \t {self.data[:2]} \nSO2: \t {self.data[2:]}"

    def from_matrix(H: Matrix) -> "SE2":
        assert H.shape == (3, 3)
        return SE2(data=cs.vertcat(H[:2, 2], SO2.from_matrix(H[:2, :2]).data))

    def rotation(self) -> SO2:
        return SO2(data=self.data[2:])

    def translation(self) -> Vector:
        return self.data[:2]

    def transform(self) -> Matrix:
        return self.as_matrix()

    def as_matrix(self) -> Matrix:
        return cs.vertcat(
            cs.horzcat(SO2(self.data[2:]).as_matrix(), self.data[:2]),
            cs.horzcat([0, 0, 1]).T
        )

    @staticmethod
    def from_position_angle(xy: Vector, theta: Angle) -> "SE2":
        assert xy.shape in [(2,), (2, 1)]
        assert theta.shape in [(1,), (1, 1), ()]
        return SE2(data=cs.vertcat(xy, cs.vertcat(cs.cos(theta), cs.sin(theta))))

    @staticmethod
    def from_x_y_theta(x: Scalar, y: Scalar, theta: Angle) -> "SE2":
        assert x.shape in [(1,), (1, 1), ()]
        assert y.shape in [(1,), (1, 1), ()]
        assert theta.shape in [(1,), (1, 1), ()]
        return SE2(data=cs.vertcat(x, y, cs.vertcat(cs.cos(theta), cs.sin(theta))))

    def inverse(self) -> "SE2":
        return SE2(
            data=cs.vertcat(
                -SO2(self.data[2:]).inverse().act(self.data[:2]),
                SO2(self.data[2:]).inverse().data
            )
        )

    def log(self) -> "SE2Tangent":
        # https://github.com/artivis/manif/blob/devel/include/manif/impl/se2/SE2_base.h
        so2 = SO2(self.data[2:])
        theta = so2.as_euler()
        theta_cos = self.data[2]
        theta_sin = self.data[3]
        theta_sq = theta * theta
        if theta_sq < 1e-10:
            # Taylor expansion
            A = 1. - theta_sq / 6.
            B = 0.5 * theta - (1. / 24.) * theta * theta_sq
        else:
            # Euler's formula
            A = theta_sin / theta
            B = (1 - theta_cos) / theta

        den = 1 / (A * A + B * B)
        A = A * den
        B = B * den
        pos = cs.vertcat(A * self.data[0] + B * self.data[1], -B * self.data[0] + A * self.data[1])
        vec = cs.vertcat(pos, theta)

        return SE2Tangent(vec=vec)

    def __mul__(self, other):
        assert isinstance(other, SE2)
        pos = self.data[:2] + SO2(self.data[2:]).act(other.data[:2])
        theta = (SO2(self.data[2:]) * SO2(other.data[2:])).data
        return SE2(data=cs.vertcat(pos, theta))

    def __rmul__(self, other):
        assert isinstance(other, SE2)
        pos = other.data[:2] + SO2(other.data[2:]).act(self.data[:2])
        theta = (SO2(self.data[2:]) * SO2(other.data[2:])).data
        return SE2(data=cs.vertcat(pos, theta))

    def __sub__(self, other):
        assert isinstance(other, SE2)
        return (other.inverse() * self).log()


@dataclasses.dataclass
class SE2Tangent:
    vec: TangentVector

    def __repr__(self) -> str:
        return f"SE2Tangent: {self.vec}"

    def exp(self) -> SE2:
        theta = self.vec[2]
        so2 = SO2Tangent(vec=self.vec[2:4])
        so2_exp = so2.exp()
        J = cs.vertcat(
            cs.horzcat(so2_exp.data[1], -so2_exp.data[0], 0),
            cs.horzcat(0, 0, 1)
        )
        return SE2(data=J @ self.vec)

    def __add__(self, other) -> "SE2Tangent":
        assert isinstance(other, SE2Tangent)
        return SE2Tangent(vec=self.vec + other.vec)

    def __sub__(self, other) -> "SE2Tangent":
        assert isinstance(other, SE2Tangent)
        return SE2Tangent(vec=self.vec - other.vec)

    def __mul__(self, other) -> "SE2Tangent":
        assert isinstance(other, Scalar)
        return SE2Tangent(vec=self.vec * other)

    def __rmul__(self, other) -> "SE2Tangent":
        assert isinstance(other, Scalar)
        return SE2Tangent(vec=other * self.vec)
