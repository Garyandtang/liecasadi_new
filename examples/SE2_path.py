import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from mpl_toolkits import mplot3d
import time

from src.liecasadi.so2 import SO2, SO2Tangent
from src.liecasadi.se2 import SE2, SE2Tangent
class TrajGenerator:
    def __init__(self, config):
        self.nState = 4
        self.nControl = 2
        if not config:
            config = {'param': {'start_state': np.array([0, 0, 0]),
                                'linear_vel': 0.5,
                                'angular_vel': 0.5,
                                'dt': 0.02,
                                'nTraj': 170}}
        self.generate_circle_traj(config['param'])

    def generate_circle_traj(self, config):
        # example
        # config = {'type': TrajType.CIRCLE,
        #           'param': {'start_state': np.array([0, 0, 0]),
        #                     'linear_vel': 0.5,
        #                     'angular_vel': 0.5,
        #                     'dt': 0.02,
        #                     'nTraj': 170}}
        self.dt = config['dt']
        self.nTraj = config['nTraj']
        self.ref_state = np.zeros((self.nState, self.nTraj))  # [x, y, theta]
        self.ref_control = np.zeros((self.nControl, self.nTraj))  # [v, w]
        state = config['start_state']
        state = np.array([state[0], state[1], ca.cos(state[2]), ca.sin(state[2])])
        self.ref_state[:, 0] = state
        vel_cmd = np.array([config['linear_vel'], config['angular_vel']])
        self.ref_control[:, 0] = vel_cmd
        v = self.vel_cmd_to_local_vel(vel_cmd)  # constant velocity
        for i in range(self.nTraj - 1):  # 0 to nTraj-2
            curr_state = self.ref_state[:, i]
            X = SE2(curr_state)
            X_next = X * SE2Tangent(v * self.dt).exp()  # X * SE2Tangent(xi * self.dt).exp()
            self.ref_state[:, i + 1] = X_next.data.full().flatten()
            self.ref_control[:, i + 1] = vel_cmd

    def vel_cmd_to_local_vel(self, vel_cmd):
        # non-holonomic constraint
        # vel_cmd: [v, w]
        # return: [v, 0, w]
        return np.array([vel_cmd[0], 0, vel_cmd[1]])

    def get_traj(self):
        return self.ref_state, self.ref_control, self.dt

def main():
    traj_config = {'param': {'start_state': np.array([0, 0, 0]),
                             'linear_vel': 0.5,
                             'angular_vel': 0.5,
                             'dt': 0.02,
                             'nTraj': 1700}}
    traj_gen = TrajGenerator(traj_config)
    ref_state, ref_control, dt = traj_gen.get_traj()
    # plot
    plt.figure()
    plt.plot(ref_state[0, :], ref_state[1, :])
    plt.title("trajectory")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


if __name__ == '__main__':
    main()