# developer: Garyandtang (HKUST)

import dataclasses
from dataclasses import field
from typing import Union, List

import casadi as cs
import numpy as np

from liecasadi.hints import Angle, Matrix, TangentVector, Vector

@dataclasses.dataclass
class SO2:
    data: Vector  # [cos(theta), sin(theta)]

    def __repr__(self) -> str:
        return f"SO2: {self.data}"

    @staticmethod
    def Identity():
        return SO2(data=cs.vertcat(1, 0))

    @staticmethod
    def from_euler(theta: Angle) -> "SO2":
        assert theta.shape == (1,) or (1, 1)
        return SO2(data=cs.vertcat(cs.cos(theta), cs.sin(theta)))

    @staticmethod
    def from_matrix(matrix: Matrix) -> "SO2":
        # matrix: [cos(theta), -sin(theta); sin(theta), cos(theta)]
        assert matrix.shape == (2, 2) or (2, 2, 1)
        return SO2(data=cs.vertcat(matrix[0, 0], matrix[1, 0]))

    def as_matrix(self) -> Matrix:
        return cs.vertcat(cs.horzcat(self.data[0], -self.data[1]),
                          cs.horzcat(self.data[1], self.data[0]))

    def as_euler(self) -> Vector:
        return cs.atan2(self.data[1], self.data[0])

    def inverse(self) -> "SO2":
        # matrix: [cos(theta), -sin(theta); sin(theta), cos(theta)]
        # inverse: [cos(theta), sin(theta); -sin(theta), cos(theta)]
        return SO2(data=cs.vertcat(self.data[0], -self.data[1]))

    def log(self) -> "SO2Tangent":
        theta = self.as_euler()
        return SO2Tangent(vec=theta)

    def act(self, vec: Vector) -> Vector:
        # matrix: [cos(theta), -sin(theta); sin(theta), cos(theta)]
        # act: [cos(theta), -sin(theta); sin(theta), cos(theta)] * [x; y]
        return cs.vertcat(self.data[0] * vec[0] - self.data[1] * vec[1],
                          self.data[1] * vec[0] + self.data[0] * vec[1])

    def __mul__(self, other) -> "SO2":
        assert isinstance(other, SO2)
        # matrix: [cos(theta), -sin(theta); sin(theta), cos(theta)]
        # other: [cos(alpha), -sin(alpha); sin(alpha), cos(alpha)]
        # mul: [cos(theta), -sin(theta); sin(theta), cos(theta)] * [cos(alpha), -sin(alpha); sin(alpha), cos(alpha)]
        # result: [cos(theta + alpha), -sin(theta + alpha); sin(theta + alpha), cos(theta + alpha)]
        return SO2(data=cs.vertcat(self.data[0] * other.data[0] - self.data[1] * other.data[1],
                                      self.data[1] * other.data[0] + self.data[0] * other.data[1]))

    def __rmul__(self, other) -> "SO2":
        assert isinstance(other, SO2)
        # matrix: [cos(theta), -sin(theta); sin(theta), cos(theta)]
        # other: [cos(alpha), -sin(alpha); sin(alpha), cos(alpha)]
        # mul: [cos(alpha), -sin(alpha); sin(alpha), cos(alpha)] * [cos(theta), -sin(theta); sin(theta), cos(theta)]
        # result: [cos(theta + alpha), -sin(theta + alpha); sin(theta + alpha), cos(theta + alpha)]
        return SO2(data=cs.vertcat(self.data[0] * other.data[0] - self.data[1] * other.data[1],
                                      self.data[1] * other.data[0] + self.data[0] * other.data[1]))

    def __sub__(self, other) -> "SO2Tangent":
        # manifold right minus
        # X - Y = log(X.inverse() * Y)
        assert isinstance(other, SO2)
        temp = other.inverse() * self
        return temp.log()



@dataclasses.dataclass
class SO2Tangent:
    vec: TangentVector  # [theta]

    def __repr__(self) -> str:
        return f"SO2Tangent: {self.vec}"

    def exp(self) -> SO2:
        theta = self.vec
        return SO2(data=cs.vertcat(cs.cos(theta), cs.sin(theta)))

    def __add__(self, other: SO2) -> "SO2":
        assert isinstance(other, SO2)
        return self.exp() * other

    def __radd__(self, other: SO2) -> "SO2":
        assert isinstance(other, SO2)
        return other * self.exp()

    def __mul__(self, other):
        if type(other) is float:
            return SO2Tangent(vec=self.vec * other)












