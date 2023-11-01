import math
import numpy as np
from scipy.integrate import odeint

import torch
import torch.nn as nn

################ Model parameters ###################

## Physical parameters ##
g = 9.8  # [m/s^2]
fair = 0.0  # friction coefficient air [N/m.s^-1 ]
fplexi = 0.5  # friction coefficient Plexiglas x Steel [N/N.m.s^-1 ]

## Bar parameters ##
l_bar = 0.5  # length of bar [m]
m_bar = 0.5  # [kg]

fb = fair * l_bar**3 / 3.0  # Moment from friction force [N.m/rad.s^-1 ]
Jb = m_bar * l_bar**2 / 12.0  # Moment of inertia of bar in G [kg.m^2]

## Cart parameters ##
m_cart = 0.3  # [kg]

fcc = m_cart * g * fplexi  # friction of cart with grown [N/m.s^-1 ]

#####################################################

################ simaulation parametre ##############
dt = 0.001  # time tick [s]
sim_max = 20.0  # simulation time
angle_max = 20
#####################################################


################# differential function ###################

# X'=F(X,t)
# X=[θ, dθ/dt, x, dx/dt]

# d²θ/dt² = a + alpha * d²x/dt²  ==> d²θ/dt² = (a + alpha * b) / (1 - alpha * beta)
# d²x/dt² = b + beta  * d²θ/dt²  ==> d²x/dt² = (b + beta  * a) / (1 - alpha * beta)


def F(X, t, F):
    theta, dtheta, x, dx = X

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
        self.X = np.array([X0])
        self.forces = []
        self.time = 0.0

    def step(self, force, dt_command):
        T = np.arange(self.time, self.time + dt_command + dt, dt)
        sol = odeint(F, self.X[-1], T, args=(force,))
        self.X = np.concatenate((self.X, sol[1:]), axis=0)
        self.forces.append((force, dt_command))
        self.time = T[-1]
        return math.degrees(abs(self.X[-1, 0])) < angle_max

    def get_last(self):
        return self.X[-1]


def loss_simultation(command, X_t, dt_command):
    N = 10
    for _ in range(N):
        theta, dtehta, _, dx = X_t
        a = (3.0 * g * torch.sin(theta) / (2 * l_bar)) - (fb * dtehta / (4 * Jb))
        b = (
            command[0] - fcc * dx - 0.5 * m_bar * l_bar * torch.sin(theta) * dtehta**2
        ) / (m_cart + m_bar)

        alpha = 3 * torch.cos(theta) / (2 * l_bar)
        beta = 0.5 * l_bar * m_bar * torch.cos(theta) / (m_bar + m_cart)

        d2tehta = (a + alpha * b) / (1 - alpha * beta)
        d2x = (b + beta * a) / (1 - alpha * beta)

        dX_t = torch.stack([dtehta, d2tehta, dx, d2x])
        X_t = X_t + torch.mul(dX_t, dt_command / N)
    loss = X_t[0] ** 2
    return loss
