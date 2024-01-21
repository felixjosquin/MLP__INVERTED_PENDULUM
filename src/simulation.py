import os
import csv

import numpy as np
from scipy.integrate import odeint

import torch
import torch.nn as nn

from utils import csv_header

################ Model parameters ###################

## Physical parameters ##
g = 9.8  # [m/s^2]
fair = 0.0  # friction coefficient air [N/m.s^-1 ]
fplexi = 0.5  # friction coefficient Plexiglas x Steel [N/N.m.s^-1 ]

## Bar parameters ##
l_bar = 0.5  # length of bar [m]
m_bar = 0.5  # weight of bar[kg]

fb = fair * l_bar**3 / 3.0  # Moment from friction force [N.m/rad.s^-1]
Jb = m_bar * l_bar**2 / 12.0  # Moment of inertia of bar in G [kg.m^2]

## Cart parameters ##
m_cart = 0.3  # [kg]
fcc = m_cart * g * fplexi  # friction of cart with grown [N/m.s^-1]

## Courroie parameters ##
R_roue_courroie = 0.5
reduc = 0.01
K_motor = 0.0184
R_motor = 5.5
#####################################################

################ simaulation parametre ##############
dt_simu = 0.01  # time tick [s]

angle_max = 20  # angle max
x_max = 0.2  # angle max
time_max = 10  # time max
#####################################################


################# differential function ###################

# X'=F(X,t)
# X=[θ, dθ/dt, x, dx/dt]

# d²θ/dt² = a + alpha * d²x/dt²  ==> d²θ/dt² = (a + alpha * b) / (1 - alpha * beta)
# d²x/dt² = b + beta  * d²θ/dt²  ==> d²x/dt² = (b + beta  * a) / (1 - alpha * beta)


def F(X, _, U):
    theta, dtheta, _, dx = X
    w_motor = dx / (R_roue_courroie * reduc)
    U_m = w_motor * K_motor
    I = (U - U_m) / R_motor
    F = I * K_motor / (reduc * R_roue_courroie)
    a = (3.0 * g * np.sin(theta) / (2 * l_bar)) - (fb * dtheta / (4 * Jb))
    b = (F - fcc * dx - 0.5 * m_bar * l_bar * np.sin(theta) * dtheta**2) / (
        m_cart + m_bar
    )
    alpha = 3 * np.cos(theta) / (2 * l_bar)
    beta = 0.5 * l_bar * m_bar * np.cos(theta) / (m_bar + m_cart)

    d2tehta = (a + alpha * b) / (1 - alpha * beta)
    d2x = (b + beta * a) / (1 - alpha * beta)
    return [dtheta, d2tehta, dx, d2x]


#####################################################


class Simulation:
    def __init__(self, X0):
        self.X = X0
        self.file_path = self._init_file()
        self.time = 0.0
        self.episode = 0

    def step(self, U_command, dt_command):
        T = np.arange(self.time, self.time + dt_command + dt_simu, dt_simu)
        sol = odeint(F, self.X, T, args=(U_command,))
        self._register_step(sol[1:], T[1:], U_command)
        self.X = sol[-1]
        self.time = T[-1]
        return (self.X, self._get_reward(), self._is_termined(), self._is_truncated())

    def get_state(self):
        return self.X

    def reset(self, X0):
        self.X = X0
        self.time = 0.0
        self.episode += 1

    def _register_step(self, X, T, U_command):
        lines_to_add = [
            {
                "EPISODE": self.episode,
                "TIME": T[index],
                "THETA": x[0],
                "dTHETA/dt": x[1],
                "X": x[2],
                "dX/dt": x[3],
                "U_command": U_command,
            }
            for (index, x) in enumerate(X)
        ]
        with open(self.file_path, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_header)
            for line in lines_to_add:
                writer.writerow(line)

    def _is_termined(self):
        return abs(self.X[0]) > np.radians(angle_max) and abs(self.X[2]) > x_max

    def _is_truncated(self):
        return self.time > time_max

    def _get_reward(self):
        return 1 - int(self._is_termined())  # +1 if pendulum don't fall

    def _init_file(self):
        i = 0
        file_path = f"./data/simu_0.csv"
        while os.path.exists(file_path):
            i += 1
            file_path = f"./data/simu_{i}.csv"

        with open(file_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_header)
            writer.writeheader()

        return file_path


X0 = np.array([0.01, 0.0, 0.0, 0.0])
simu = Simulation(X0=X0)
simu.step(5.0, 10 * dt_simu)
simu.step(0.0, 3.0)
