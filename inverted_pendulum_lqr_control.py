"""
Inverted Pendulum LQR control
author: Trung Kien - letrungkien.k53.hut@gmail.com
"""

import math
import time

import matplotlib.pyplot as plt
import numpy as np

###### Model parameters ######
l_bar = 2.0  # length of bar
m = 0.3  # [kg]
g = 9.8  # [m/s^2]
##############################


###### simaulation parametre ###########
dt = 0.1  # time tick [s]
sim_time = 5.0  # simulation time
angle_max = 45 
X0 = np.array([
        0.0, # x
        0.0, # dx/dt
        0.3, # tehta
        0.0  # dtheta/dt
    ])
########################################

################# fonction dérivé ###################
def F (t,X,d2x):
    d2theta=(g*math.sin(X[2])+d2x*math.cos(X[2]))/l_bar
    return np.array([X[1],d2x,X[3],d2theta])

#####################################################



show_animation = True
def main():
    X = np.copy(X0)
    time = 0.0

    while sim_time > time and math.degrees(X[2])<angle_max and math.degrees(X[2])>-angle_max:
        time += dt
        X=calcul_Runge_Kutta(time,X)
        if show_animation:
            plt.clf()
            px = float(X[0])
            theta = float(X[2])
            plot_cart(px, theta)
            plt.xlim([-5.0, 2.0])
            plt.pause(0.001)

    print("Finish")
    print(f"x={float(X[0]):.2f} [m] , theta={math.degrees(X[2]):.2f} [deg]")
    if show_animation:
        plt.show()

def calcul_Runge_Kutta (time,X):
    d2x=calcul_d2x()
    k1=F(time,X,d2x)*dt
    k2=F(time+dt/2,X+k1/2,d2x)*dt
    k3=F(time+dt/2,X+k2/2,d2x)*dt
    k4=F(time+dt,X+k3,d2x)*dt
    return X+(k1+2*k2+2*k3+k4)/6


def calcul_d2x():
    return -4.

def plot_cart(xt, theta):
    cart_w = 1.0 # longeur du card
    cart_h = 0.5 # hauteur du card
    radius = 0.1

    # Cordonée des point de la card
    cx = np.array([-cart_w / 2.0, cart_w / 2.0, cart_w /2.0, -cart_w / 2.0, -cart_w / 2.0])
    cy = np.array([0.0, 0.0, cart_h, cart_h, 0.0])
    cy += radius * 2.0

    cx = cx + xt

    bx = np.array([0.0, l_bar * math.sin(-theta)])
    bx += xt
    by = np.array([cart_h, l_bar * math.cos(-theta) + cart_h])
    by += radius * 2.0

    angles = np.arange(0.0, math.pi * 2.0, math.radians(3.0))
    ox = np.array([radius * math.cos(a) for a in angles])
    oy = np.array([radius * math.sin(a) for a in angles])

    rwx = np.copy(ox) + cart_w / 4.0 + xt
    rwy = np.copy(oy) + radius
    lwx = np.copy(ox) - cart_w / 4.0 + xt
    lwy = np.copy(oy) + radius

    wx = np.copy(ox) + bx[-1]
    wy = np.copy(oy) + by[-1]

    plt.plot(cx, cy, "-b")
    plt.plot(bx, by, "-k")
    plt.plot(rwx, rwy, "-k")
    plt.plot(lwx, lwy, "-k")
    plt.plot(wx, wy, "-k")
    plt.title(f"x: {xt:.2f} , theta: {math.degrees(theta):.2f}")

    # for stopping simulation with the esc key.
    plt.gcf().canvas.mpl_connect(
        'key_release_event',
        lambda event: [exit(0) if event.key == 'escape' else None])

    plt.axis("equal")


if __name__ == '__main__':
    main()
