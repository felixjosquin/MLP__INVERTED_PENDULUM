import numpy as np
import matplotlib.pyplot as plt
import math
from simulation import l_bar,dt

class Draw:
    def __init__(self,X_historique):
        self.x=X_historique[0,:]
        self.dx=X_historique[1,:]
        self.theta=X_historique[2,:]
        self.dtheta=X_historique[3,:]        
        self.time=X_historique[4,:]

    
    def draw_cart(self,i=0):
        plt.clf()
        plot_cart(self.x[i], self.theta[i],self.time[i])
        plt.pause(dt/6)
        if i+1<len(self.time):
            self.draw_cart(i+1)
    
    def draw_graph(self):
        figure, axis = plt.subplots(2, 2)
        axis[0, 0].plot(self.time, np.rad2deg(self.theta))
        axis[0, 0].set_title("Theta [deg]")
        axis[0, 1].plot(self.time, np.rad2deg(self.dtheta))
        axis[0, 1].set_title("dTheta/dt [deg/s]")
        axis[1, 0].plot(self.time, self.x)
        axis[1, 0].set_title("x Function [m]")
        axis[1, 1].plot(self.time, self.dx)
        axis[1, 1].set_title("dx/dt [m/s]")
        plt.show()




def plot_cart(xt, theta,time):
    cart_w = 1.0 # longeur du card
    cart_h = 0.5 # hauteur du card
    radius = 0.1

    plt.xlim([-2.0, 2.0])
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
    plt.title(f"x: {xt:.2f} , theta: {math.degrees(theta):.2f}, time: {time:.2f}")

      # for stopping simulation with the esc key.
    plt.gcf().canvas.mpl_connect(
        'key_release_event',
        lambda event: [exit(0) if event.key == 'escape' else None])

    plt.axis("equal")
