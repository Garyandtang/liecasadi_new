import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from mpl_toolkits import mplot3d
import time

from liecasadi import SO3, SO3Tangent, SE3Tangent, SE3

class SE3Optimization:
    def __init__(self):
        self.nPos = 3    # xyz
        self.nQuat = 4   # xyzw
        self.nTwist = 6  # 3 for angular velocity, 3 for linear velocity
        self.dt = 0.02
        self.nTraj = 100

    def solve(self, start_pos, start_quat, end_pos, end_quat):
        start_time = time.time()
        opti = ca.Opti()

        pos = opti.variable(self.nPos, self.nTraj + 1)
        quat = opti.variable(self.nQuat, self.nTraj + 1)
        twist = opti.variable(self.nTwist, self.nTraj + 1)

        for k in range(self.nTraj):
            curr_SE3 = SE3(pos[:, k], quat[:, k])
            curr_se3 = SE3Tangent(twist[:, k] * self.dt)
            forward_SE3 = curr_SE3 * curr_se3.exp()
            next_SE3 = SE3(pos[:, k + 1], quat[:, k + 1])
            opti.subject_to(next_SE3.pos == forward_SE3.pos)
            opti.subject_to(next_SE3.xyzw == forward_SE3.xyzw)

        cost = 0
        for k in range(self.nTraj):
            cost += ca.sumsqr(twist[:, k])

        # set initial pose
        opti.subject_to(quat[:, 0] == start_quat)
        opti.subject_to(pos[:, 0] == start_pos)

        # set final pose
        opti.subject_to(quat[:, self.nTraj] == end_quat)
        opti.subject_to(pos[:, self.nTraj] == end_pos)

        opti.minimize(cost)
        opti.solver("ipopt")
        try:
            sol = opti.solve()
        except:
            print("Can't solve the problem!")

        pos_sol = sol.value(pos)
        quat_sol = sol.value(quat)
        end_time = time.time()
        print("time: ", end_time - start_time)
        return pos_sol, quat_sol

def main():
    start_pos = np.array([0, 0, 0])
    start_quat = SO3.Identity().as_quat()
    end_pos = np.array([1, 1, 0])
    end_quat = SO3.from_euler(np.array([0, 0,  np.pi])).as_quat()
    se3_optimization = SE3Optimization()
    pos_sol, quat_sol = se3_optimization.solve(start_pos, start_quat, end_pos, end_quat)

    # plot 3d xyz position
    fig = plt.figure()
    ax = plt.figure().add_subplot(projection='3d')
    # set x, y, z limit
    ax.set_xlim(-1, 2)
    ax.set_ylim(-1, 2)
    ax.set_zlim(-1, 2)
    ax.plot(pos_sol[0, :], pos_sol[1, :], pos_sol[2, :])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()

    # plot 2d xy position
    fig = plt.figure()
    # x limit and y limit
    plt.xlim(-1, 2)
    plt.ylim(-1, 2)
    plt.plot(pos_sol[0, :], pos_sol[1, :])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


if __name__ == '__main__':
    main()
