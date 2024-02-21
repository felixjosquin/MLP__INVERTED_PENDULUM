import os
import csv

import numpy as np
from scipy.integrate import odeint

from src.utils import ACTIONS, DT_COMMAND, CSV_HEADER

################ Model parameters ###################

## Physical parameters ##
g = 9.8  # [m/s^2]
fair = 0.0  # friction coefficient air [N/m.s^-1 ]
fplexi = 0.5  # friction coefficient Plexiglas x Steel [N/N.m.s^-1 ]

## Bar parameters ##
l_bar = 0.20  # length of bar [m]
m_bar = 0.072  # weight of bar[kg]

fb = fair * l_bar**3 / 3.0  # Moment from friction force [N.m/rad.s^-1]
Jb = m_bar * l_bar**2 / 12.0  # Moment of inertia of bar in G [kg.m^2]

## Cart parameters ##
m_cart = 0.230  # [kg]
fcc = m_cart * g * fplexi  # friction of cart with grown [N/m.s^-1]

## Courroie parameters ##
R_roue_courroie = 0.00635
reduc = 1.0
K_motor = 0.38197
R_motor = 7.5
U_max = 12.0
#####################################################

################ simaulation parametre ##############
dt_simu = 0.005  # time tick [s]
Theta_0_range = (3.0, 5.0)
dTheta_0_range = (0.0, 1.0)
#####################################################


################ DNQ parametre ##############
x_good = 0.2

angle_max = 20  # angle max
x_max = 0.3  # x max
time_max = 10.0  # time max
#####################################################


################# differential function ###################

# X'=F(X,t)
# X=[θ, dθ/dt, x, dx/dt]

# d²θ/dt² = a + alpha * d²x/dt²  ==> d²θ/dt² = (a + alpha * b) / (1 - alpha * beta)
# d²x/dt² = b + beta  * d²θ/dt²  ==> d²x/dt² = (b + beta  * a) / (1 - alpha * beta)


# def F(X, _, U):
#     theta, dtheta, _, dx = X
#     w_motor = dx / (R_roue_courroie * reduc)
#     U_m = w_motor * K_motor
#     I = (U - U_m) / R_motor
#     F = I * K_motor / (reduc * R_roue_courroie)
#     a = (3.0 * g * np.sin(theta) / (2 * l_bar)) - (fb * dtheta / (4 * Jb))
#     b = (F - fcc * dx - 0.5 * m_bar * l_bar * np.sin(theta) * dtheta**2) / (
#         m_cart + m_bar
#     )
#     alpha = 3 * np.cos(theta) / (2 * l_bar)
#     beta = 0.5 * l_bar * m_bar * np.cos(theta) / (m_bar + m_cart)

#     d2tehta = (a + alpha * b) / (1 - alpha * beta)
#     d2x = (b + beta * a) / (1 - alpha * beta)
#     return [dtheta, d2tehta, dx, d2x]


#####################################################

tau = 0.0434
p_positif = np.poly1d([9.93417956e-3, 49.50761959e-3])
p_negatif = np.poly1d([8.07755163e-3, -64.7761631e-3])


def F(X, _, U):
    theta, dtheta, _, dx = X
    dx_c = p_positif(U) if U > 0 else p_negatif(U) if U < 0 else 0.0
    d2x = (dx_c - dx) / tau
    d2theta = (3 * g * np.sin(theta) + 3 * d2x * np.cos(theta)) / (2 * l_bar)
    return [dtheta, d2theta, dx, d2x]


#####################################################


class Simulation:
    def __init__(self):
        self.X = np.multiply(
            np.radians(
                [
                    np.random.uniform(*Theta_0_range),
                    np.random.uniform(*dTheta_0_range),
                    0,
                    0,
                ]
            ),
            np.random.choice([1, -1], (4,)),
        )  # X=[θ, dθ/dt, x, dx/dt]
        self.simu_number = 0
        self.file_path = self._init_file()
        self.time = 0.0
        self.episode = 0

    def step(self, action):
        U_command = ACTIONS[action]
        T = np.arange(self.time, self.time + DT_COMMAND + dt_simu, dt_simu)
        sol = odeint(F, self.X, T, args=(U_command,))
        self._register_step(sol[1:], T[1:], U_command)
        self.X = sol[-1]
        self.time = T[-1]
        return (self.X, self._get_reward(), self._is_termined(), self._is_truncated())

    def get_state(self):
        return self.X

    def reset(self):
        self.X = np.multiply(
            np.radians(
                [
                    np.random.uniform(*Theta_0_range),
                    np.random.uniform(*dTheta_0_range),
                    0,
                    0,
                ]
            ),
            np.random.choice([1, -1], (4,)),
        )
        self.time = 0.0
        self.episode += 1

    def _register_step(self, X, T, U_command):
        lines_to_add = [
            {
                CSV_HEADER.EPISODE: self.episode,
                CSV_HEADER.TIME: T[index],
                CSV_HEADER.THETA: x[0],
                CSV_HEADER.dTHETA: x[1],
                CSV_HEADER.X: x[2],
                CSV_HEADER.dX: x[3],
                CSV_HEADER.U_command: U_command,
            }
            for (index, x) in enumerate(X)
        ]
        with open(self.file_path, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=[el.value for el in CSV_HEADER])
            for line in lines_to_add:
                writer.writerow(line)

    def _is_termined(self):
        return abs(self.X[0]) > np.radians(angle_max) or abs(self.X[2]) > x_max

    def _is_truncated(self):
        return self.time > time_max

    def _get_reward(self):
        return (
            5.0 * ((np.radians(angle_max) - abs(self.X[0])) / np.radians(angle_max))
            + 1.0 * int(abs(self.X[2]) < x_good)
            - 5.0 * int(self._is_termined())
        )

    def _init_file(self):
        i = 0
        file_path = f"./data/simulations/simu_0.csv"
        while os.path.exists(file_path):
            i += 1
            file_path = f"./data/simulations/simu_{i}.csv"
        self.simu_number = i
        with open(file_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[el.value for el in CSV_HEADER])
            writer.writeheader()
        return file_path


def get_reward(state, termined):
    return (
        5.0 * ((np.radians(angle_max) - abs(state[0])) / np.radians(angle_max))
        + 1.0 * ((x_max - abs(state[2])) / x_max)
        - 5.0 * int(termined)
    )
