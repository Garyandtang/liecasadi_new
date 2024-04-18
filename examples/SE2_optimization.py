import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from mpl_toolkits import mplot3d
import time

from src.liecasadi.so2 import SO2, SO2Tangent
from src.liecasadi.se2 import SE2, SE2Tangent

class SE2Optimization:
    def __init__(self):
        self.nState = 4    # x y cos(theta) sin(theta)
        self.nTwist = 3  # v_x v_y w
        self.dt = 0.05
        self.nTraj = 2
        self.Q = np.eye(3) * 100
        self.R = np.eye(self.nTwist) * 0.1

    def solve(self, start_state, end_state):
        start_time = time.time()
        opti = ca.Opti()

        state = opti.variable(self.nState, self.nTraj)
        twist = opti.variable(self.nTwist, self.nTraj)

        for k in range(self.nTraj-1):
            curr_SE2 = SE2(state[:, k])
            curr_se2 = SE2Tangent(twist[:, k] * self.dt)
            forward_SE2 = curr_SE2 * curr_se2.exp()
            next_SE2 = SE2(state[:, k + 1])
            opti.subject_to(next_SE2.data == forward_SE2.data)

        if end_state.shape[0] == 3:
            end_state = np.array([end_state[0], end_state[1], ca.cos(end_state[2]), ca.sin(end_state[2])])
        # opti.subject_to(state[:, -1] == end_state)
        cost = 0
        for k in range(self.nTraj-1):
            ref_SE2 = SE2(end_state)
            curr_SE2 = SE2(state[:, k])
            error = curr_SE2 - ref_SE2
            cost += ca.mtimes([error.vec.T, self.Q, error.vec])
            twist_d = np.zeros(self.nTwist)
            cost += ca.mtimes([(twist[:, k] - twist_d).T, self.R, (twist[:, k] - twist_d)])

        last_SE2 = SE2(state[:, -1])
        last_ref_SE2 = SE2(end_state)
        last_error = last_SE2 - last_ref_SE2
        cost += ca.mtimes([last_error.vec.T, 100*self.Q, last_error.vec])

        # set initial pose
        if start_state.shape[0] == 3:
            start_state = np.array([start_state[0], start_state[1], np.cos(start_state[2]), np.sin(start_state[2])])
        opti.subject_to(state[:, 0] == start_state)

        # set final pose


        opti.minimize(cost)
        # don't use ipopt, it's too slow
        opti.solver("ipopt")

        try:
            sol = opti.solve()
        except:
            print("Can't solve the problem!")

        state_sol = sol.value(state)
        end_time = time.time()
        print("time: ", end_time - start_time)
        return state_sol

def main():
    start_state = np.array([0, 0, 0])
    end_state = np.array([1, 1, 0])
    se2_optimization = SE2Optimization()
    state_sol = se2_optimization.solve(start_state, end_state)

    # plot x y
    plt.figure()
    plt.plot(state_sol[0, :], state_sol[1, :])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    # plot theta
    plt.figure()
    plt.plot(state_sol[2, :])
    plt.xlabel('N')
    plt.ylabel('theta')
    plt.show()


if __name__ == '__main__':
    main()
